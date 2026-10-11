"""
app/dguard_app.py - DERMA-GUARD care platform.

    streamlit run app/dguard_app.py

Views
  Patient      intake (photo, optional dermoscopy, details, symptoms), result,
               dermoscopy requests, questions, follow-ups
  Doctor       prioritised queue, evidence, specialist team, gatekeeper, similar
               cases, report, verdict
  Stress Lab   damage a real lesion's photo and watch the system respond
  Research     every experiment result with confidence intervals
  System       platform activity and audit log

Research prototype - not clinically validated.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from src.care import CareService, CareStore, CaseStatus  # noqa: E402
from src.constants import MILK_CLASS_FULL_NAMES  # noqa: E402

st.set_page_config(page_title="DERMA-GUARD", page_icon="🩺", layout="wide")

STYLE = {
    "PROCEED": ("✅", "#1b7f3b", "Answered - HIGH confidence"),
    "PROCEED_LOW": ("⚠️", "#b54708", "Answered - LOW confidence, a doctor will confirm"),
    "ACQUIRE": ("📷", "#9a6700", "Dermoscopy photo requested"),
    "DEFER": ("🧑‍⚕️", "#b42318", "Sent to a doctor - the evidence contradicts itself"),
}


# ------------------------------------------------------------------ backend
@st.cache_resource(show_spinner="Loading the specialist team (first time only)...")
def backend():
    from src.agents.pipeline import DGuardPipeline
    engine = DGuardPipeline.build()
    db = os.environ.get("DGUARD_DB", str(ROOT / "data" / "care" / "dguard_care.sqlite"))
    svc = CareService(engine, CareStore(db), upload_dir=ROOT / "data" / "care" / "uploads")
    samples = engine.sample_lesions("test", n=30, seed=7)
    return engine, svc, samples


try:
    engine, svc, samples = backend()
except FileNotFoundError as e:
    st.error(f"Missing artifact: {e}")
    st.code("python scripts/milk_native_infer.py\npython scripts/build_specialists.py\n"
            "python scripts/run_team.py\n"
            "python scripts/run_eade_milk.py --native outputs/milk_team --missed-cost 5 "
            "--out outputs/eade_team/cost_5")
    st.stop()

store = svc.store
train = engine.splits["train"]
SITES = sorted(x for x in train["site"].dropna().unique() if x)
GENERAL_SITES = sorted(x for x in train["anatom_site_general"].dropna().unique() if x)


def load_json(rel):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ widgets
def badge(action, urgency=None, tier=None, big=True):
    key = "PROCEED_LOW" if action == "PROCEED" and tier == "LOW" else action
    icon, colour, text = STYLE.get(key, ("•", "#555", action))
    urg = (" &nbsp;<span style='background:#b42318;color:white;padding:2px 8px;"
           "border-radius:10px;font-size:0.8em'>URGENT</span>" if urgency == "URGENT" else "")
    size = "1.25em" if big else "1em"
    st.markdown(f"<div style='border-left:6px solid {colour};padding:8px 14px;"
                f"background:rgba(0,0,0,0.03);border-radius:4px'>"
                f"<b style='color:{colour};font-size:{size}'>{icon} {key.replace('_', ' ')}</b>"
                f" - {text}{urg}</div>", unsafe_allow_html=True)


def gatekeeper(result: dict):
    """Every safety check, pass or fired - why the case got its outcome."""
    d = result["decision"]
    g = d.get("gates") or {}
    risks = d.get("risks") or {}
    team = result.get("team") or {}
    rows = [
        ("Sources contradict each other?", "Dempster-Shafer conflict above the validation p90",
         g.get("conflict", False), "DEFER"),
        ("Cancer ruled out before saying benign?", "conformal set over benign/malignant",
         g.get("ambiguous_set", False), "blocks HIGH"),
        ("Certified risk threshold met?", "Learn-then-Test bound on missed cancers",
         g.get("below_risk_threshold", False), "blocks HIGH"),
        ("Photo quality in calibrated range?", "image-quality agent (1st-99th percentile)",
         g.get("photo_quality", False), "blocks HIGH"),
        ("Case similar to training data?", "Mahalanobis OOD (validation p99)",
         g.get("out_of_distribution", False) or team.get("ood", False), "blocks HIGH"),
        ("Answering cheaper than a review?", "expected cost of answering vs 1.5",
         (risks.get("proceed", 0) > risks.get("defer", 9)), "blocks HIGH"),
    ]
    st.markdown("**Gatekeeper** - every safety check on this case")
    for q, how, fired, effect in rows:
        c1, c2 = st.columns([0.08, 0.92])
        c1.markdown("⛔" if fired else "✅")
        c2.markdown(f"**{q}** {'- FIRED, ' + effect if fired else ''}  \n"
                    f"<span style='color:#666;font-size:0.85em'>{how}</span>",
                    unsafe_allow_html=True)
    if risks:
        st.caption("Expected cost - answer: {:.2f} · doctor review: {:.2f}{}".format(
            risks.get("proceed", 0), risks.get("defer", 0),
            f" · dermoscopy: {risks['acquire']:.2f}" if "acquire" in risks else ""))


def team_panel(result: dict):
    team = result.get("team") or {}
    if not team:
        return
    from src.fusion.runtime import CONSENSUS_TEXT
    st.markdown("**Specialist team** - the Devil's Advocate's evidence profile")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Consensus", f"Level {team.get('consensus_level', '?')}",
              help=CONSENSUS_TEXT.get(team.get("consensus_level"), ""))
    t2.metric("Agree", f"{team['agreeing']} / {team['n_specialists']}")
    t3.metric("Consistency", f"{team['consistency']:.2f}",
              help="1 - mean pairwise Jensen-Shannon divergence")
    t4.metric("Unlike training data", "YES" if team["ood"] else "no")
    rows = [{"specialist": n, "says": v,
             "quality Q": team["quality"].get(n),
             "reliability R": team["reliability"].get(n),
             "uncertainty U": team["uncertainty"].get(n),
             "weight": team["weights"].get(n)} for n, v in team["votes"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    w = team.get("weights") or {}
    if w:
        st.bar_chart(pd.DataFrame({"fusion weight": w}), horizontal=True, height=180)
    st.caption(f"Iterative diagnosis-adaptive fusion: {team['iterations']} refinement "
               f"step(s). Weights are set per case by learned reliability, quality and the "
               f"suspected diagnosis.")


def evidence_panel(result: dict):
    d = result["decision"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence (calibrated)", f"{(d.get('tentative_confidence') or 0):.0%}",
              help="Probability that the stated diagnosis is correct, calibrated on "
                   "validation data")
    c2.metric("P(malignant)", f"{(d.get('p_malignant') or 0):.0%}")
    c3.metric("Ignorance u", f"{result['ignorance']:.3f}",
              help="How much the system does NOT know (1 = nothing)")
    c4.metric("Conflict C", f"{result['conflict']:.3f}",
              help="How strongly the inputs contradict each other")
    diff = d.get("differential") or []
    if diff:
        st.markdown("**Differential**")
        st.bar_chart(pd.DataFrame({"probability": {
            engine.classes[int(c)]: float(p) for c, p in diff}}), horizontal=True, height=150)
    rel = result.get("reliability_for_tentative") or {}
    if rel:
        st.caption("Measured reliability of each input for this diagnosis (validation): "
                   + ", ".join(f"{k} {v:.2f}" for k, v in rel.items() if v is not None))


def show_images(cid):
    cols = st.columns(2)
    for col, kind, title in ((cols[0], "clinical_photo", "Clinical photo"),
                             (cols[1], "dermoscopy", "Dermoscopy")):
        e = store.latest_evidence(cid, kind)
        with col:
            if e and e["path"] and Path(e["path"]).exists():
                st.image(e["path"], caption=title, width="stretch")
            else:
                st.caption(f"{title}: not provided")


def similar_gallery(result: dict):
    sim = (result.get("similar") or {}).get("similar_cases") or []
    if not sim:
        return
    agree = result["similar"].get("agreement")
    st.markdown("**Similar confirmed past cases** (training lesions; context only, never "
                "used to decide)" + (f" - {agree:.0%} share the diagnosis" if agree is not None else ""))
    cols = st.columns(len(sim))
    for col, c in zip(cols, sim):
        with col:
            if c.get("clinical_path") and Path(c["clinical_path"]).exists():
                st.image(c["clinical_path"], width="stretch")
            st.caption(f"**{c['diagnosis']}** · {c['similarity']:.2f}")


def as_table(rows):
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: json.dumps(v, default=str)
                              if isinstance(v, (dict, list)) else ("" if v is None else str(v)))
    return df


def messages_panel(cid, audience):
    for m in store.messages(cid):
        who = {"system": "🤖 DERMA-GUARD", "doctor": "🧑‍⚕️ Doctor", "patient": "🙂 Patient"}[m["sender"]]
        with st.chat_message("assistant" if m["sender"] != "patient" else "user"):
            st.markdown(f"**{who}** · _{m['kind']}_ · {m['created_at'][:16].replace('T', ' ')}")
            st.write(m["body"])


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🩺 DERMA-GUARD")
st.sidebar.caption("Evidence-aware specialist consensus with risk-controlled acquisition "
                   "and deferral")
role = st.sidebar.radio("View", ["Patient", "Doctor", "Stress Lab", "Research", "System"])
st.sidebar.divider()
s_ = store.stats()
st.sidebar.metric("Cases", s_["cases"])
st.sidebar.metric("Waiting for a doctor", s_["by_status"].get(CaseStatus.AWAITING_DOCTOR, 0))
st.sidebar.caption("Research prototype - not clinically validated.")

# ======================================================================
# PATIENT
# ======================================================================
if role == "Patient":
    st.title("Patient")
    patients = store.patients()
    with st.expander("Register a new patient", expanded=not patients):
        with st.form("reg"):
            name = st.text_input("Name")
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("Age", 1, 110, 55)
            sex = c2.selectbox("Sex", ["female", "male"])
            tone = c3.selectbox("Skin tone (0 lightest - 5 darkest)", list("012345"), index=3)
            if st.form_submit_button("Register") and name:
                st.session_state["pid"] = svc.register_patient(name, age, sex, tone)
                st.rerun()
    if not patients:
        st.info("Register a patient to begin.")
        st.stop()

    labels = {p["id"]: f"#{p['id']} {p['name']}" for p in patients}
    keys = list(labels)
    pid = st.selectbox("Patient", keys, format_func=labels.get,
                       index=keys.index(st.session_state["pid"])
                       if st.session_state.get("pid") in labels else 0)
    patient = store.patient(pid)
    cases = store.cases(patient_id=pid)
    with st.expander("Open a new case (a new spot)", expanded=not cases):
        with st.form("case"):
            site = st.selectbox("Where is the spot?", SITES)
            complaint = st.text_input("What made you notice it?", "a spot that looks different")
            if st.form_submit_button("Open case"):
                st.session_state["cid"] = svc.open_case(pid, site, complaint)
                st.rerun()
    if not cases:
        st.stop()

    clabels = {c["id"]: f"Case #{c['id']} · {c['site'] or '?'} · {c['status']}" for c in cases}
    ckeys = list(clabels)
    cid = st.selectbox("Case", ckeys, format_func=clabels.get,
                       index=ckeys.index(st.session_state["cid"])
                       if st.session_state.get("cid") in clabels else 0)
    case = store.case(cid)

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("1 · Evidence")
        src = st.radio("Photos", ["Real held-out MILK10k lesion (demo)", "Upload my own"],
                       horizontal=True)
        symptoms = st.text_area("Describe the spot in your own words (optional)",
                                placeholder="e.g. it has been bleeding and getting bigger "
                                            "for 3 weeks")
        if src.startswith("Real"):
            names = {i: f"{s['lesion_id']}" for i, s in enumerate(samples)}
            si = st.selectbox("Demo lesion", list(names), format_func=names.get)
            sample = samples[si]
            with_derm = st.checkbox("Include the dermoscopy photo from the start", value=False)
            st.caption("Held-out test lesion - never used for training. "
                       f"True diagnosis (hidden from the system): **{sample['true_label']}**")
            if st.button("Submit and assess", type="primary",
                         disabled=case["status"] == CaseStatus.CLOSED):
                svc.submit_photo(cid, sample["clinical_path"])
                if with_derm:
                    svc.submit_dermoscopy(cid, sample["dermoscopy_path"])
                svc.submit_details(cid, {**sample["metadata"], "symptoms_text": symptoms})
                st.session_state[f"derm_{cid}"] = sample["dermoscopy_path"]
                svc.assess(cid)
                st.rerun()
        else:
            up = st.file_uploader("Clinical photo", type=["jpg", "jpeg", "png"])
            upd = st.file_uploader("Dermoscopy photo (optional)", type=["jpg", "jpeg", "png"])
            m1, m2 = st.columns(2)
            gsite = m1.selectbox("Body area", GENERAL_SITES)
            site = m2.selectbox("Exact site", SITES,
                                index=SITES.index(case["site"]) if case["site"] in SITES else 0)
            if st.button("Submit and assess", type="primary", disabled=up is None):
                svc.submit_photo(cid, Image.open(up))
                if upd is not None:
                    svc.submit_dermoscopy(cid, Image.open(upd))
                svc.submit_details(cid, {"age": patient["age"], "sex": patient["sex"],
                                         "skin_tone_class": patient["skin_tone"],
                                         "site": site, "anatom_site_general": gsite,
                                         "symptoms_text": symptoms})
                svc.assess(cid)
                st.rerun()

        if case["status"] == CaseStatus.AWAITING_PATIENT:
            st.warning("📷 The system asked for a **dermoscopy photo** of this spot.")
            derm_sample = st.session_state.get(f"derm_{cid}")
            if derm_sample and st.button("Upload the dermoscopy photo (demo lesion)"):
                svc.submit_dermoscopy(cid, derm_sample)
                svc.assess(cid)
                st.rerun()
            upd2 = st.file_uploader("…or upload your own dermoscopy photo",
                                    type=["jpg", "jpeg", "png"], key=f"du{cid}")
            if upd2 is not None and st.button("Submit dermoscopy"):
                svc.submit_dermoscopy(cid, Image.open(upd2))
                svc.assess(cid)
                st.rerun()

        if store.followups(cid, status="SCHEDULED"):
            st.info("🗓️ Follow-up scheduled for "
                    + ", ".join(f["due_date"] for f in store.followups(cid, status="SCHEDULED")))
            last = store.latest_evidence(cid, "clinical_photo")
            if last and st.button("Submit follow-up photo (demo: same photo)"):
                out = svc.submit_followup_photo(cid, last["path"])
                st.session_state[f"change_{cid}"] = out["change"]
                st.rerun()
        ch = st.session_state.get(f"change_{cid}")
        if ch:
            (st.error if ch["alert"] else st.success)(
                f"Change check: P(malignant) {ch['previous_p_malignant']:.0%} → "
                f"{ch['p_malignant']:.0%}" + (" - sent to a doctor urgently" if ch["alert"]
                                              else " - no worrying change"))
        show_images(cid)

    with right:
        st.subheader("2 · Result")
        a = store.latest_assessment(cid)
        if a:
            r = a["result"]
            badge(a["action"], a["urgency"], r["decision"].get("tier"))
            st.write("")
            st.markdown(r["patient_message"].replace("\n", "  \n"))
            conf = r["decision"].get("tentative_confidence")
            if conf is not None:
                st.progress(min(max(conf, 0.0), 1.0),
                            text=f"Confidence that this assessment is right: {conf:.0%}")
            sc = r.get("scribe") or {}
            if sc.get("red_flags"):
                st.error("Warning signs from your description: " + ", ".join(sc["red_flags"]))
            rv = store.reviews(cid)
            if rv:
                st.success(f"🧑‍⚕️ Reviewed by Dr {rv[-1]['doctor']}: **{rv[-1]['verdict']}** - "
                           f"{rv[-1]['diagnosis'] or ''}")
        else:
            st.caption("No assessment yet.")
        st.subheader("3 · Messages")
        messages_panel(cid, "patient")
        q = st.chat_input("Ask your doctor a question")
        if q:
            svc.ask(cid, q)
            st.rerun()

# ======================================================================
# DOCTOR
# ======================================================================
elif role == "Doctor":
    st.title("Doctor")
    doctor = st.sidebar.text_input("Your name", "Rao")
    queue = svc.doctor_queue()
    st.subheader(f"Review queue ({len(queue)})")
    if not queue:
        st.success("Nothing waiting. Deferred cases, low-confidence answers to confirm, and "
                   "system answers needing sign-off appear here, most urgent first.")
        st.stop()
    qdf = pd.DataFrame(queue)
    qdf["p_malignant"] = qdf["p_malignant"].map(lambda x: f"{x:.0%}" if x is not None else "-")
    st.dataframe(qdf[[c for c in ("case_id", "patient", "kind", "urgency", "action", "tier",
                                  "tentative", "p_malignant", "unanswered_questions",
                                  "waiting_since") if c in qdf.columns]],
                 hide_index=True, width="stretch")

    cid = st.selectbox("Open case", [r["case_id"] for r in queue],
                       format_func=lambda c: f"Case #{c}")
    case = store.case(cid)
    a = store.latest_assessment(cid)
    p = store.patient(case["patient_id"])
    st.markdown(f"**{p['name']}** · age {p['age'] or '?'} · {p['sex'] or '?'} · "
                f"skin tone {p['skin_tone'] or '?'} · site **{case['site'] or '?'}**")

    tabs = st.tabs(["Assessment", "Specialist team & gates", "Similar cases & report",
                    "Messages", "Timeline"])
    with tabs[0]:
        show_images(cid)
        if a:
            r = a["result"]
            badge(a["action"], a["urgency"], r["decision"].get("tier"))
            st.code(r["doctor_message"], language=None)
            evidence_panel(r)
            sc = r.get("scribe") or {}
            if sc:
                st.info(f"Patient's account: {sc.get('summary', '')}  (informativeness "
                        f"{sc.get('informativeness', 0):.2f})")
            for kind, q in (r.get("quality") or {}).items():
                if q and q.get("issues"):
                    st.warning(f"Image quality - {kind}: " + ", ".join(q["issues"]))
    with tabs[1]:
        if a:
            l, rr = st.columns([1.3, 1])
            with l:
                team_panel(a["result"])
            with rr:
                gatekeeper(a["result"])
    with tabs[2]:
        if a:
            similar_gallery(a["result"])
            rep = a["result"].get("report") or {}
            if rep.get("text"):
                st.markdown(f"**Report** _(source: {rep.get('source')})_")
                st.info(rep["text"])
    with tabs[3]:
        messages_panel(cid, "doctor")
        reply = st.chat_input("Reply to the patient")
        if reply:
            svc.reply(cid, doctor, reply)
            st.rerun()
    with tabs[4]:
        st.dataframe(as_table(svc.timeline(cid)), hide_index=True, width="stretch")
        if len(store.assessments(cid)) > 1:
            st.markdown("**Assessment history**")
            st.dataframe(pd.DataFrame([{k: x[k] for k in ("created_at", "stage", "action",
                                                           "tentative", "p_malignant")}
                                       for x in store.assessments(cid)]), hide_index=True)

    st.subheader("Your decision")
    with st.form("review"):
        verdict = st.radio("Verdict", ["CONFIRM", "OVERRIDE", "NEED_IN_PERSON"], horizontal=True)
        default = engine.classes.index(a["tentative"]) if a and a["tentative"] in engine.classes else 0
        dx = st.selectbox("Diagnosis", engine.classes, index=default,
                          format_func=lambda c: f"{c} - {MILK_CLASS_FULL_NAMES.get(c, c)}")
        note = st.text_area("Clinical note (internal)")
        instr = st.text_area("Instruction to the patient")
        fu = st.number_input("Follow-up in N days (0 = none)", 0, 365, 0)
        if st.form_submit_button("Submit review", type="primary"):
            svc.doctor_review(cid, doctor, verdict,
                              diagnosis=dx if verdict != "NEED_IN_PERSON" else None,
                              note=note or None, instruction=instr or None,
                              followup_days=int(fu) or None)
            st.success("Review recorded.")
            st.rerun()

# ======================================================================
# STRESS LAB
# ======================================================================
elif role == "Stress Lab":
    from src.eval.corruptions import IMAGE_CORRUPTIONS
    st.title("Stress Lab")
    st.markdown("Damage the evidence of a real held-out lesion and watch how the system "
                "responds. A safe system should become **less** confident, ask for more "
                "evidence, or hand the case to a doctor - never become confidently wrong.")
    names = {i: f"{s['lesion_id']}  (true: {s['true_label']})" for i, s in enumerate(samples)}
    c1, c2, c3 = st.columns([1.4, 1, 1])
    si = c1.selectbox("Lesion", list(names), format_func=names.get)
    corr = c2.selectbox("Damage", ["none", "blur", "dark", "noise", "jpeg", "occlusion",
                                   "remove patient details"])
    with_derm = c3.checkbox("With dermoscopy", value=False)
    sample = samples[si]
    clean = Image.open(sample["clinical_path"]).convert("RGB")
    derm = Image.open(sample["dermoscopy_path"]).convert("RGB") if with_derm else None
    damaged = IMAGE_CORRUPTIONS[corr](clean) if corr in IMAGE_CORRUPTIONS else clean
    meta_bad = {} if corr == "remove patient details" else sample["metadata"]

    if st.button("Run both", type="primary"):
        with st.spinner("Assessing..."):
            st.session_state["stress"] = (engine.assess(clean, derm, sample["metadata"]).to_dict(),
                                          engine.assess(damaged, derm, meta_bad).to_dict(),
                                          si, corr, with_derm)
    res = st.session_state.get("stress")
    if res and res[2:] == (si, corr, with_derm):
        L, R = st.columns(2)
        for col, title, img, r in ((L, "Original", clean, res[0]),
                                   (R, f"Damaged: {corr}", damaged, res[1])):
            with col:
                st.subheader(title)
                st.image(img, width=260)
                d = r["decision"]
                badge(r["action"], d.get("urgency"), d.get("tier"), big=False)
                st.markdown(f"Tentative: **{engine.classes[d['tentative']] if d.get('tentative') is not None else '-'}** "
                            f"(truth {sample['true_label']}) · confidence "
                            f"{(d.get('tentative_confidence') or 0):.0%}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Ignorance u", f"{r['ignorance']:.3f}")
                m2.metric("Conflict", f"{r['conflict']:.3f}")
                m3.metric("P(malignant)", f"{(d.get('p_malignant') or 0):.0%}")
                w = r.get("fusion_weights") or {}
                if w:
                    st.bar_chart(pd.DataFrame({"weight": w}), horizontal=True, height=170)
                gatekeeper(r)
    st.caption("The experiment version over 300 lesions: scripts/run_stress_milk.py "
               "(results in the Research view).")

# ======================================================================
# RESEARCH
# ======================================================================
elif role == "Research":
    st.title("Research results")
    st.caption("MILK10k test split (787 lesions), scored once; everything fitted on the 786 "
               "validation lesions.")
    an = load_json("outputs/analysis/analysis.json")
    if an:
        ci = an["ci"]
        k = st.columns(4)
        k[0].metric("Answered", f"{ci['answered']['point']:.0%}",
                    help=f"95% CI {ci['answered']['ci95'][0]:.0%}-{ci['answered']['ci95'][1]:.0%}")
        k[1].metric("HIGH confidence", f"{ci['high_cov']['point']:.0%}")
        k[2].metric("HIGH accuracy", f"{ci['high_acc']['point']:.1%}",
                    help=f"95% CI {ci['high_acc']['ci95'][0]:.1%}-{ci['high_acc']['ci95'][1]:.1%}")
        k[3].metric("Missed cancers (HIGH)", f"{ci['high_miss']['point']:.1%}",
                    delta=f"vs {ci['b3_miss']['point']:.1%} baseline", delta_color="inverse")
        st.markdown("**Paired bootstrap tests**")
        st.dataframe(pd.DataFrame(an["paired"]).T, width="stretch")
    tr = load_json("outputs/milk_team/team_report.json")
    t1, t2, t3, t4, t5 = st.tabs(["Fusion", "Decision vs baselines", "Robustness",
                                  "Distribution shift", "Fairness"])
    with t1:
        if tr:
            st.markdown("**Fusion methods**")
            st.dataframe(pd.DataFrame(tr["fusion"]).T.round(3), width="stretch")
            st.markdown("**Each specialist alone**")
            st.dataframe(pd.DataFrame(tr["specialists"]).T.round(3), width="stretch")
        rel = engine.reliability
        st.markdown("**Which input to trust, per diagnosis** (measured; the adaptive prior)")
        st.dataframe(pd.DataFrame({"photo": rel["image"], "dermoscopy": rel["dermoscopy"],
                                   "details": rel["meta"], "n": rel["n_per_class"]},
                                  index=rel["classes"]), width="stretch")
    with t2:
        for cost in (5, 10):
            r = load_json(f"outputs/eade_team/cost_{cost}/report.json")
            if r:
                st.markdown(f"**Missed-cancer cost {cost}**")
                st.dataframe(pd.DataFrame(r["baselines"]).set_index("policy").round(3),
                             width="stretch")
        if an:
            st.markdown("**Learned-deferral baseline (L2D-style) at the same coverage**")
            st.write(an["learned_deferral"])
    with t3:
        s = load_json("outputs/stress_milk/report_team.json")
        if s:
            st.dataframe(pd.DataFrame(s["conditions"]).T.round(3), width="stretch")
    with t4:
        sh = load_json("outputs/shift_pad/report.json")
        if sh:
            st.markdown("MILK10k-trained system on **PAD-UFES-20** (Brazilian smartphone photos), "
                        "no refitting; the original PAD-UFES model is switched off to avoid leakage.")
            st.dataframe(pd.DataFrame({"value": {k: v for k, v in sh.items()
                                                 if not isinstance(v, list)}}), width="stretch")
        else:
            st.info("Run scripts/run_shift_pad.py")
    with t5:
        if an:
            st.dataframe(pd.DataFrame(an["skin_tone"]).T, width="stretch")
            st.caption("Groups under 30 lesions are too small to judge.")

# ======================================================================
# SYSTEM
# ======================================================================
else:
    st.title("System")
    st.json(store.stats())
    st.dataframe(as_table(store.audit_log(limit=200)), hide_index=True, width="stretch")
    st.markdown("""
**Pipeline:** quality agent → five specialists (3 evidential heads, the original PAD-UFES
Model A, a similar-case vote) → Devil's Advocate (Q, R, U, C, M, OOD) → iterative
diagnosis-adaptive fusion + Dempster–Shafer ignorance/conflict → gatekeeper (conflict,
conformal, Learn-then-Test, quality, OOD, cost) → outcome → librarian, scribe, report +
verifier → care workflow.
""")
