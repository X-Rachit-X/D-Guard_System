"""
app/dguard_app.py - DERMA-GUARD care platform.

    streamlit run app/dguard_app.py

Views
  Patient          intake (photo, optional dermoscopy, details, symptoms), result,
                   "why this result", requests for more evidence, messages,
                   follow-ups, delete my data
  Doctor           prioritised queue, evidence, specialist team, gatekeeper,
                   explanation, similar cases, report, messages, verdict
  Compare systems  one lesion through EVERY system (each specialist, every fusion
                   rule, threshold / learned-rejector baselines, D-GUARD), the
                   test-set comparison tables, and what every metric means
  Stress Lab       damage a real lesion's photo and watch the system respond
  Research         every experiment result with confidence intervals
  Architecture     the pipeline as a diagram, and which file implements what
  System           platform activity, audit log, demo reset

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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from src.care import CareService, CareStore, CaseStatus  # noqa: E402
from src.constants import MILK_CLASS_FULL_NAMES  # noqa: E402

st.set_page_config(page_title="DERMA-GUARD", page_icon="🩺", layout="wide")

STYLE = {
    "PROCEED": ("✅", "#1b7f3b", "Answered - HIGH confidence"),
    "PROCEED_LOW": ("⚠️", "#b54708", "Answered - LOW confidence, a doctor will confirm"),
    "ACQUIRE": ("📷", "#9a6700", "More evidence requested"),
    "DEFER": ("🧑‍⚕️", "#b42318", "Sent to a doctor - the evidence contradicts itself"),
}
SENDER = {"system": ("DERMA-GUARD", "🩺"), "doctor": ("Doctor", "🧑‍⚕️"),
          "patient": ("You", "🙂")}
KIND_CHIP = {"result": ("#1d4ed8", "result"), "instruction": ("#9a6700", "next step"),
             "question": ("#7c3aed", "question"), "reply": ("#1b7f3b", "doctor's reply")}


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


def full(c):
    return f"{MILK_CLASS_FULL_NAMES.get(c, c)} ({c})" if c else "-"


# ------------------------------------------------------------------ widgets
def badge(action, urgency=None, tier=None, big=True, target=None):
    key = "PROCEED_LOW" if action == "PROCEED" and tier == "LOW" else action
    icon, colour, text = STYLE.get(key, ("•", "#555", action))
    if action == "ACQUIRE" and target:
        text = "Details requested" if target == "metadata" else "Dermoscopy photo requested"
    urg = (" &nbsp;<span style='background:#b42318;color:white;padding:2px 8px;"
           "border-radius:10px;font-size:0.8em'>URGENT</span>" if urgency == "URGENT" else "")
    size = "1.25em" if big else "1em"
    st.markdown(f"<div style='border-left:6px solid {colour};padding:8px 14px;"
                f"background:rgba(127,127,127,0.08);border-radius:4px'>"
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
        ("Photo quality in calibrated range?", "quality agent: blur, light, contrast, glare, "
         "covered centre (1st-99th percentile)", g.get("photo_quality", False), "blocks HIGH"),
        ("Patient details given?", "age / sex / site - every calibration case had them",
         g.get("missing_details", False), "asks for details"),
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
                    f"<span style='opacity:0.65;font-size:0.85em'>{how}</span>",
                    unsafe_allow_html=True)
    if risks:
        st.caption("Expected cost - answer: {:.2f} · doctor review: {:.2f}{}".format(
            risks.get("proceed", 0), risks.get("defer", 0),
            f" · dermoscopy: {risks['acquire']:.2f}" if "acquire" in risks else ""))


def team_panel(result: dict):
    team = result.get("team") or {}
    if not team:
        return
    from src.fusion.methods import LABELS
    from src.fusion.runtime import CONSENSUS_TEXT
    st.markdown("**Specialist team** - the Devil's Advocate's evidence profile")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Consensus", f"Level {team.get('consensus_level', '?')}",
              help=CONSENSUS_TEXT.get(team.get("consensus_level"), ""))
    t2.metric("Agree", f"{team['agreeing']} / {team['n_specialists']}")
    t3.metric("Consistency", f"{team['consistency']:.2f}",
              help="1 - mean pairwise Jensen-Shannon divergence")
    t4.metric("Unlike training data", "YES" if team["ood"] else "no",
              help=f"Mahalanobis distance / threshold = {team.get('ood_distance') or 0:.2f}")
    rows = [{"specialist": n, "says": v,
             "quality Q": team["quality"].get(n),
             "reliability R": team["reliability"].get(n),
             "uncertainty U": team["uncertainty"].get(n),
             "track record for this dx": (team.get("utility") or {}).get(n),
             "weight": team["weights"].get(n)} for n, v in team["votes"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    w = team.get("weights") or {}
    if w:
        st.bar_chart(pd.DataFrame({"fusion weight": w}), horizontal=True, height=180)
    imp = team.get("gate_importance") or {}
    st.caption(f"Combination rule: **{LABELS.get(team.get('method'), team.get('method'))}** "
               f"(picked on validation lesions). "
               + ("What the learned gate listens to: " + ", ".join(
                   f"{k} {v:.2f}" for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:5])
                  if imp else ""))


def explanation_panel(result: dict, expanded=False):
    ex = result.get("explanation") or {}
    if not ex.get("text"):
        return
    with st.expander("🔍 Why this result - who said what, why they disagree, who was trusted",
                     expanded=expanded):
        st.markdown(ex["text"].replace("\n", "  \n"))
        st.caption(f"Source: {ex.get('source')}. Every sentence is built from measured "
                   f"numbers; an LLM rewrite is only shown if a verifier finds no invented "
                   f"diagnosis or percentage.")


def evidence_panel(result: dict):
    d = result["decision"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence (calibrated)", f"{(d.get('tentative_confidence') or 0):.0%}",
              help="Of past validation cases given this confidence, this share were right")
    c2.metric("P(malignant)", f"{(d.get('p_malignant') or 0):.0%}")
    c3.metric("Ignorance u", f"{result['ignorance']:.3f}",
              help="How much the evidential heads do NOT know (1 = nothing)")
    c4.metric("Conflict C", f"{result['conflict']:.3f}",
              help="How strongly the inputs contradict each other")
    diff = d.get("differential") or []
    if diff:
        st.markdown("**Differential**")
        st.bar_chart(pd.DataFrame({"probability": {
            engine.classes[int(c)]: float(p) for c, p in diff}}), horizontal=True, height=150)


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
    st.markdown("**Similar confirmed past cases** (training lesions; context only)"
                + (f" - {agree:.0%} share the diagnosis" if agree is not None else ""))
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
    """Conversation for one case: filter, chat bubbles, quick actions, clear."""
    msgs = store.messages(cid)
    n_q = sum(m["kind"] == "question" for m in msgs)
    n_r = sum(m["kind"] == "reply" for m in msgs)
    top = st.columns([2.2, 1])
    flt = top[0].radio("Show", ["All", "Results & next steps", "Questions & replies"],
                       horizontal=True, key=f"flt_{audience}_{cid}", label_visibility="collapsed")
    if n_q > n_r:
        top[1].warning(f"{n_q - n_r} question(s) waiting for a doctor")
    keep = {"All": None, "Results & next steps": {"result", "instruction"},
            "Questions & replies": {"question", "reply"}}[flt]
    box = st.container(height=420)
    with box:
        if not msgs:
            st.caption("No messages yet.")
        for m in msgs:
            if keep and m["kind"] not in keep:
                continue
            who, avatar = SENDER.get(m["sender"], (m["sender"], "•"))
            if audience == "doctor" and m["sender"] == "patient":
                who = "Patient"
            colour, chip = KIND_CHIP.get(m["kind"], ("#555", m["kind"]))
            with st.chat_message("user" if m["sender"] == "patient" else "assistant",
                                 avatar=avatar):
                st.markdown(f"<b>{who}</b> &nbsp;<span style='background:{colour};color:white;"
                            f"padding:1px 8px;border-radius:9px;font-size:0.75em'>{chip}</span>"
                            f" &nbsp;<span style='opacity:0.6;font-size:0.8em'>"
                            f"{m['created_at'][:16].replace('T', ' ')}</span>",
                            unsafe_allow_html=True)
                st.markdown(m["body"].replace("\n", "  \n"))
    if audience == "patient":
        qs = ["What does my result mean?", "How urgent is this?",
              "Do I need a biopsy?", "What should I watch for?"]
        cols = st.columns(len(qs))
        for c, q in zip(cols, qs):
            if c.button(q, key=f"qq_{cid}_{q}", width="stretch"):
                svc.ask(cid, q)
                st.rerun()
        q = st.chat_input("Ask your doctor a question", key=f"ci_p_{cid}")
        if q:
            svc.ask(cid, q)
            st.rerun()
    else:
        tpl = {"-": "", "Book in person": "Please book an in-person appointment so I can "
               "examine the spot.", "Reassure": "This looks harmless. Keep an eye on it and "
               "send a new photo if it changes.", "Biopsy": "I recommend a small biopsy to "
               "confirm the diagnosis. The clinic will contact you."}
        t = st.selectbox("Quick reply", list(tpl), key=f"tpl_{cid}")
        if t != "-" and st.button("Send quick reply", key=f"send_tpl_{cid}"):
            svc.reply(cid, st.session_state.get("doctor_name", "Rao"), tpl[t])
            st.rerun()
        reply = st.chat_input("Reply to the patient", key=f"ci_d_{cid}")
        if reply:
            svc.reply(cid, st.session_state.get("doctor_name", "Rao"), reply)
            st.rerun()
    with st.popover("🗑️ Clear conversation"):
        st.caption("Deletes every message of this case. The assessment itself is kept.")
        if st.button("Yes, clear messages", key=f"clr_{audience}_{cid}", type="primary"):
            svc.clear_messages(cid, actor=audience)
            st.rerun()


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🩺 DERMA-GUARD")
st.sidebar.caption("Evidence-aware specialist consensus with risk-controlled acquisition "
                   "and deferral")
role = st.sidebar.radio("View", ["Patient", "Doctor", "Compare systems", "Stress Lab",
                                 "Research", "Architecture", "System"])
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
    with st.expander("🔒 My data - delete"):
        st.caption("Deleting removes the records and any photos you uploaded from this "
                   "platform. The audit log keeps only the fact that a deletion happened.")
        d1, d2 = st.columns(2)
        if cases:
            dc = d1.selectbox("Case to delete", [c["id"] for c in cases],
                              format_func=lambda c: f"Case #{c}")
            if d1.checkbox("I understand this cannot be undone", key="cdel_ok") and \
                    d1.button("Delete this case", type="primary"):
                n = svc.delete_case(dc)
                st.session_state.pop("cid", None)
                st.toast(f"Case #{dc} deleted ({n} uploaded file(s) removed)")
                st.rerun()
        if d2.checkbox(f"Delete patient {patient['name']} and ALL their cases", key="pdel_ok") \
                and d2.button("Delete patient", type="primary"):
            n = svc.delete_patient(pid)
            st.session_state.pop("pid", None)
            st.session_state.pop("cid", None)
            st.toast(f"Patient deleted ({n} uploaded file(s) removed)")
            st.rerun()
    if not cases:
        st.stop()

    clabels = {c["id"]: f"Case #{c['id']} · {c['site'] or '?'} · {c['status']}" for c in cases}
    ckeys = list(clabels)
    cid = st.selectbox("Case", ckeys, format_func=clabels.get,
                       index=ckeys.index(st.session_state["cid"])
                       if st.session_state.get("cid") in clabels else 0)
    case = store.case(cid)
    last = store.latest_assessment(cid)
    target = (last["result"]["decision"].get("acquire_target") if last else None)

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
            o1, o2 = st.columns(2)
            with_derm = o1.checkbox("Include the dermoscopy photo from the start", value=False,
                                    help="Leave off to see the system ASK for it when needed")
            with_meta = o2.checkbox("Include patient details", value=True,
                                    help="Turn off to see the system ASK for them")
            st.caption("Held-out test lesion - never used for training. "
                       f"True diagnosis (hidden from the system): **{sample['true_label']}**")
            if st.button("Submit and assess", type="primary",
                         disabled=case["status"] == CaseStatus.CLOSED):
                svc.submit_photo(cid, sample["clinical_path"])
                if with_derm:
                    svc.submit_dermoscopy(cid, sample["dermoscopy_path"])
                svc.submit_details(cid, {**(sample["metadata"] if with_meta else {}),
                                         "symptoms_text": symptoms})
                st.session_state[f"derm_{cid}"] = sample["dermoscopy_path"]
                st.session_state[f"meta_{cid}"] = sample["metadata"]
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

        if case["status"] == CaseStatus.AWAITING_PATIENT and target == "metadata":
            st.warning("📝 The system asked for **a few details** before it answers.")
            with st.form(f"details_{cid}"):
                demo = st.session_state.get(f"meta_{cid}") or {}
                f1, f2, f3 = st.columns(3)
                age_ = f1.number_input("Age", 1, 110, int(demo.get("age") or patient["age"] or 50))
                sex_ = f2.selectbox("Sex", ["female", "male"],
                                    index=0 if (demo.get("sex") or patient["sex"]) != "male" else 1)
                gs_ = f3.selectbox("Body area", GENERAL_SITES,
                                   index=GENERAL_SITES.index(demo["anatom_site_general"])
                                   if demo.get("anatom_site_general") in GENERAL_SITES else 0)
                if st.form_submit_button("Send details and reassess", type="primary"):
                    svc.submit_details(cid, {**demo, "age": age_, "sex": sex_,
                                             "anatom_site_general": gs_,
                                             "skin_tone_class": demo.get("skin_tone_class")
                                             or patient["skin_tone"]})
                    svc.assess(cid)
                    st.rerun()
        elif case["status"] == CaseStatus.AWAITING_PATIENT:
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

        if last and (last["result"]["decision"].get("gates") or {}).get("photo_quality"):
            st.info("📸 The photo was outside the quality range the system was checked on. "
                    "A clearer photo may give a confident answer.")
            upr = st.file_uploader("Retake the photo", type=["jpg", "jpeg", "png"],
                                   key=f"retake{cid}")
            if upr is not None and st.button("Submit new photo"):
                svc.submit_photo(cid, Image.open(upr))
                svc.assess(cid)
                st.rerun()

        if store.followups(cid, status="SCHEDULED"):
            st.info("🗓️ Follow-up scheduled for "
                    + ", ".join(f["due_date"] for f in store.followups(cid, status="SCHEDULED")))
            lastp = store.latest_evidence(cid, "clinical_photo")
            if lastp and st.button("Submit follow-up photo (demo: same photo)"):
                out = svc.submit_followup_photo(cid, lastp["path"])
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
        if last:
            r = last["result"]
            badge(last["action"], last["urgency"], r["decision"].get("tier"), target=target)
            st.write("")
            st.markdown(r["patient_message"].replace("\n", "  \n"))
            conf = r["decision"].get("tentative_confidence")
            if conf is not None and last["action"] == "PROCEED":
                st.progress(min(max(conf, 0.0), 1.0),
                            text=f"Confidence that this assessment is right: {conf:.0%}")
            sc = r.get("scribe") or {}
            if sc.get("red_flags"):
                st.error("Warning signs from your description: " + ", ".join(sc["red_flags"]))
            explanation_panel(r)
            rv = store.reviews(cid)
            if rv:
                st.success(f"🧑‍⚕️ Reviewed by Dr {rv[-1]['doctor']}: **{rv[-1]['verdict']}** - "
                           f"{rv[-1]['diagnosis'] or ''}")
        else:
            st.caption("No assessment yet.")
        st.subheader("3 · Messages")
        messages_panel(cid, "patient")

# ======================================================================
# DOCTOR
# ======================================================================
elif role == "Doctor":
    st.title("Doctor")
    doctor = st.sidebar.text_input("Your name", "Rao")
    st.session_state["doctor_name"] = doctor
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

    tabs = st.tabs(["Assessment", "Why - explanation", "Specialist team & gates",
                    "Similar cases & report", "Messages", "Timeline"])
    with tabs[0]:
        show_images(cid)
        if a:
            r = a["result"]
            badge(a["action"], a["urgency"], r["decision"].get("tier"),
                  target=r["decision"].get("acquire_target"))
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
            explanation_panel(a["result"], expanded=True)
    with tabs[2]:
        if a:
            l, rr = st.columns([1.3, 1])
            with l:
                team_panel(a["result"])
            with rr:
                gatekeeper(a["result"])
    with tabs[3]:
        if a:
            similar_gallery(a["result"])
            rep = a["result"].get("report") or {}
            if rep.get("text"):
                st.markdown(f"**Report** _(source: {rep.get('source')})_")
                st.info(rep["text"])
    with tabs[4]:
        messages_panel(cid, "doctor")
    with tabs[5]:
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
# COMPARE SYSTEMS
# ======================================================================
elif role == "Compare systems":
    from src.fusion.methods import LABELS, METHODS
    from src.fusion.runtime import compare_last
    st.title("Compare systems")
    st.markdown("Every system this project built or compares against, on the **same** input. "
                "Pick a held-out lesion and run them all, or look at the test-set tables.")
    tab_one, tab_all, tab_metrics = st.tabs(["One lesion, every system", "Test-set comparison",
                                             "Evaluation methods explained"])

    with tab_one:
        names = {i: f"{s['lesion_id']}  (true: {s['true_label']})" for i, s in enumerate(samples)}
        c1, c2 = st.columns([1.6, 1])
        si = c1.selectbox("Lesion", list(names), format_func=names.get, key="cmp_lesion")
        stage = c2.radio("Evidence", ["Photo + details", "Photo + dermoscopy + details"],
                         horizontal=False, key="cmp_stage")
        sample = samples[si]
        groups = {"Specialists alone": "Specialist alone",
                  "Fusion rules": "fusion", "Decision systems": "decision"}
        show = st.multiselect("Show", list(groups), default=list(groups))
        if st.button("Run every system", type="primary"):
            with st.spinner("Running..."):
                clin = Image.open(sample["clinical_path"]).convert("RGB")
                derm = (Image.open(sample["dermoscopy_path"]).convert("RGB")
                        if "dermoscopy" in stage else None)
                resA = engine.engine.assess(clin, None, sample["metadata"])
                res = engine.assess(clin, derm, sample["metadata"])
                fus = compare_last(engine.engine.team) if engine.engine.team else {}
                # decision-level baselines on stage-A probabilities (as in analyze_milk.py)
                dec = {}
                bl = ROOT / "outputs" / "analysis" / "baselines.joblib"
                pA = np.array(resA.probs)
                predA = engine.classes[int(pA.argmax())]
                dec["Always answer (no abstention)"] = (predA, True, float(pA.max()))
                if bl.exists():
                    import joblib
                    B = joblib.load(bl)
                    dec["B3 confidence threshold"] = (predA, bool(pA.max() >= B["b3_threshold"]),
                                                      float(pA.max()))
                    srt = np.sort(pA)
                    ent = -(pA * np.log(np.clip(pA, 1e-12, 1))).sum() / np.log(len(pA))
                    f = np.array([[pA.max(), srt[-1] - srt[-2], ent,
                                   pA[engine.engine.malignant].sum(), resA.ignorance,
                                   resA.conflict]])
                    risk = float(B["rejector"].predict_proba(f)[0, 1])
                    dec["B4 learned rejector (L2D)"] = (predA, risk <= B["rejector_threshold"],
                                                        1 - risk)
                d = res.decision
                tent = engine.classes[d.tentative] if d.tentative is not None else None
                lab = ("HIGH" if d.action.value == "PROCEED" and d.tier != "LOW"
                       else d.action.value + (f" ({d.tier})" if d.tier else ""))
                dec[f"D-GUARD -> {lab}"] = (tent, d.action.value == "PROCEED" and d.tier != "LOW",
                                            d.tentative_confidence or 0)
                st.session_state["cmp"] = (si, stage, fus, dec, res.to_dict())
        cmp = st.session_state.get("cmp")
        if cmp and cmp[:2] == (si, stage):
            _, _, fus, dec, rd = cmp
            truth = sample["true_label"]
            rows = []
            for k, v in fus.items():
                grp = "Specialists alone" if k.startswith("Specialist alone") else "Fusion rules"
                if grp not in show:
                    continue
                rows.append({"group": grp, "system": k.replace("Specialist alone: ", ""),
                             "says": v["prediction"], "correct": "✅" if v["prediction"] == truth else "❌",
                             "confidence": round(v["confidence"], 3),
                             "P(true class)": round(float(v["prob"][engine.classes.index(truth)]), 3),
                             "answers?": "always",
                             "weights": ", ".join(f"{a}: {b:.2f}" for a, b in (v["weights"] or {}).items())})
            if "Decision systems" in show:
                for k, (pred, answers, conf) in dec.items():
                    rows.append({"group": "Decision systems", "system": k, "says": pred,
                                 "correct": "✅" if pred == truth else "❌",
                                 "confidence": round(float(conf), 3), "P(true class)": None,
                                 "answers?": "HIGH answer" if answers else "abstains / asks / LOW",
                                 "weights": ""})
            st.markdown(f"True diagnosis: **{full(truth)}** (hidden from every system)")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            pt = {r["system"]: r["P(true class)"] for r in rows if r["P(true class)"] is not None}
            if pt:
                st.markdown("**Probability each system gave the TRUE diagnosis**")
                st.bar_chart(pd.DataFrame({"P(true class)": pt}), horizontal=True, height=320)
            explanation_panel(rd, expanded=False)
            st.caption("Specialists alone and fusion rules always answer; decision systems may "
                       "abstain. B3 and B4 use the photo+details probabilities, the same way "
                       "scripts/analyze_milk.py evaluates them, at D-GUARD's test coverage.")

    with tab_all:
        an = load_json("outputs/analysis/analysis.json")
        tr = load_json("outputs/milk_team/team_report.json")
        if an and an.get("systems"):
            st.markdown("**Decision systems on the 787 test lesions**")
            sysdf = pd.DataFrame(an["systems"]).T
            st.dataframe(sysdf, width="stretch")
            metric = st.selectbox("Plot", ["missed_cancer_rate", "accuracy", "coverage"],
                                  key="sys_metric")
            st.bar_chart(sysdf[metric].astype(float), horizontal=True, height=300)
        if tr:
            st.markdown("**Fusion rules** - validation (out-of-fold, used to pick the winner) "
                        "and test")
            stg = st.radio("Stage", ["B (+ dermoscopy)", "A (photo + details)"], horizontal=True)
            sk = stg[0]
            rows = []
            for k, v in tr["fusion"].items():
                if k.startswith(sk + ":"):
                    vv = tr.get("fusion_val_oof", {}).get(k, {})
                    rows.append({"method": k[2:], "val macro-F1": vv.get("macro_f1"),
                                 **{f"test {m}": v[m] for m in ("acc", "macro_f1", "bal_acc",
                                                                "mal_auroc")}})
            fdf = pd.DataFrame(rows).set_index("method").astype(float).round(3)
            st.dataframe(fdf, width="stretch")
            st.bar_chart(fdf[["val macro-F1", "test macro_f1"]], horizontal=True, height=320)
            sel = tr.get("selected", {}).get(sk)
            if sel:
                st.success(f"Picked on validation for stage {sk}: **{LABELS.get(sel, sel)}**")
            orc = tr.get("oracle", {}).get(sk)
            if orc:
                st.caption(f"Ceiling - an oracle that always picks the best specialist (uses the "
                           f"label): accuracy {orc['acc']:.3f}, macro-F1 {orc['macro_f1']:.3f}.")
            subs = {k[2:]: v for k, v in tr.get("subsets", {}).items() if k.startswith(sk + ":")}
            if subs:
                st.markdown("**Which specialists are worth having?** Every subset, fused with "
                            "the selected rule")
                sdf = pd.DataFrame(subs).T[["val_macro_f1", "macro_f1", "acc", "mal_auroc"]]
                sdf["n specialists"] = [k.count("+") + 1 for k in sdf.index]
                st.dataframe(sdf.astype(float).sort_values("val_macro_f1", ascending=False)
                             .round(3), width="stretch")
            st.markdown("**Each specialist alone (test)**")
            st.dataframe(pd.DataFrame({k[2:]: v for k, v in tr["specialists"].items()
                                       if k.startswith(sk + ":")}).T.round(3), width="stretch")

    with tab_metrics:
        st.markdown("""
| Method | Question it answers | Where |
|---|---|---|
| **Coverage** | What share of patients get a (HIGH-confidence) answer? | all tables |
| **Accuracy (11 classes)** | Of those answers, how many name the right diagnosis? | |
| **Missed-cancer rate** | Of answered cancers, how many were called benign? The safety number. | |
| **Macro-F1 / balanced accuracy** | Does it handle rare diagnoses, not just the common ones? | fusion tables |
| **Malignant AUROC** | Ranking cancers above benign lesions, independent of any threshold | |
| **AUGRC** | Area under the generalised risk-coverage curve: abstention quality over *all* coverages (lower = better) | Research > Decision |
| **Acquisition efficiency** | Does asking for dermoscopy go to the patients who benefit? (vs asking the most uncertain) | Research > Decision |
| **Bootstrap 95% CI / paired test** | Is a difference real or noise on 787 lesions? 2,000 resamples | Research |
| **Learn-then-Test** | A *certified* bound: missed-cancer rate below α with probability 1-δ, if test data look like calibration data | Gatekeeper |
| **Conformal sets** | Benign/malignant set with 1-α coverage; a set with both blocks a confident benign answer | Gatekeeper |
| **Cross-fitting** | Fusion rules are fitted on 4/5 of validation and scored on the other 1/5, so the winner and the calibration are not judged on their own training data | Fusion tables |
| **Oracle best head** | Ceiling: picks the specialist that was right, using the label - shows how much better *selection* could be | Fusion tables |
| **Subset study** | Every combination of specialists - is each one pulling its weight? | Fusion tables |
| **DERMA-STRESS** | Blur, darkness, noise, JPEG, occlusion, missing details: does the system become *more careful*? | Stress Lab, Research |
| **Distribution shift** | PAD-UFES-20 (phone photos, Brazil) and Derm7pt (with dermoscopy, Italy): is it still safe elsewhere? | Research |
| **Net benefit** | Decision-curve analysis for "refer as possibly malignant" | Research |
| **Skin-tone fairness** | Error rates per MILK10k skin-tone group | Research |
""")

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
        with st.spinner("Assessing (with every agent and gate)..."):
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
                badge(r["action"], d.get("urgency"), d.get("tier"), big=False,
                      target=d.get("acquire_target"))
                st.markdown(f"Tentative: **{engine.classes[d['tentative']] if d.get('tentative') is not None else '-'}** "
                            f"(truth {sample['true_label']}) · confidence "
                            f"{(d.get('tentative_confidence') or 0):.0%}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Ignorance u", f"{r['ignorance']:.3f}")
                m2.metric("Conflict", f"{r['conflict']:.3f}")
                m3.metric("P(malignant)", f"{(d.get('p_malignant') or 0):.0%}")
                q = (r.get("quality") or {}).get("clinical photo") or {}
                if q.get("severe"):
                    st.error("Quality agent: " + ", ".join(q["severe"]))
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
               "validation lesions (fusion rules cross-fitted).")
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
                    help=f"95% CI {ci['high_miss']['ci95'][0]:.1%}-{ci['high_miss']['ci95'][1]:.1%}")
        st.markdown("**Paired bootstrap tests**")
        st.dataframe(pd.DataFrame(an["paired"]).T, width="stretch")
    tr = load_json("outputs/milk_team/team_report.json")
    t1, t2, t3, t4, t5, t6 = st.tabs(["Fusion", "Decision vs baselines", "Robustness",
                                      "Shift: PAD-UFES-20", "External: Derm7pt", "Fairness"])
    with t1:
        if tr:
            st.markdown("**Fusion methods (test)** - full comparison in *Compare systems*")
            st.dataframe(pd.DataFrame(tr["fusion"]).T.round(3), width="stretch")
            if tr.get("gate_importance"):
                st.markdown("**What the learned Devil's Advocate listens to** (mean |weight| "
                            "per evidence feature, standardised)")
                st.dataframe(pd.DataFrame(tr["gate_importance"]).round(3), width="stretch")
        rel = engine.reliability
        st.markdown("**Which input to trust, per diagnosis** (measured; the adaptive prior)")
        st.dataframe(pd.DataFrame({"photo": rel["image"], "dermoscopy": rel["dermoscopy"],
                                   "details": rel["meta"], "n": rel["n_per_class"]},
                                  index=rel["classes"]), width="stretch")
    with t2:
        for cost in (2, 5, 10):
            r = load_json(f"outputs/eade_team/cost_{cost}/report.json")
            if r:
                st.markdown(f"**Missed-cancer cost {cost}**")
                st.dataframe(pd.DataFrame(r["baselines"]).set_index("policy").round(3),
                             width="stretch")
        if an:
            st.markdown("**Learned-deferral baseline (L2D-style) at the same coverage**")
            st.write(an["learned_deferral"])
            if an.get("v2_vs_v1"):
                st.markdown("**Fusion v2 (learned gate) vs v1 (EM), both cross-fitted**")
                st.write(an["v2_vs_v1"])
    with t3:
        s = load_json("outputs/stress_milk/report_team.json")
        if s:
            st.dataframe(pd.DataFrame(s["conditions"]).T.round(3), width="stretch")
            st.caption("'model only' = without the pipeline's quality and missing-details gates.")
    with t4:
        sh = load_json("outputs/shift_pad/report.json")
        if sh:
            st.markdown("MILK10k-trained system on **PAD-UFES-20** (Brazilian smartphone photos), "
                        "no refitting; the original PAD-UFES model is switched off to avoid leakage.")
            st.dataframe(pd.DataFrame({"value": {k: v for k, v in sh.items()
                                                 if not isinstance(v, list)}}), width="stretch")
        else:
            st.info("Run scripts/run_shift_pad.py")
        ps = load_json("outputs/analysis/pad_symptoms.json")
        if ps:
            st.markdown("**Do the scribe's warning signs mean anything?** PAD-UFES-20 "
                        "questionnaire, one row per lesion (cancer = BCC, SCC, melanoma)")
            dd = ps["definitions"]["PAD-UFES (BCC, SCC, MEL)"]["symptoms"]
            st.dataframe(pd.DataFrame(dd).T[["n_yes", "mal_if_yes", "mal_if_no",
                                             "relative_risk"]].astype(float).round(3),
                         width="stretch")
    with t5:
        d7 = load_json("outputs/shift_derm7pt/report.json")
        if d7:
            st.markdown("MILK10k-trained system on **Derm7pt** (clinical + dermoscopy, Italian "
                        "atlas), no refitting. The full loop runs: dermoscopy is used only when "
                        "the photo stage asks for it.")
            st.dataframe(pd.DataFrame({k: v for k, v in d7.items()
                                       if k in ("dguard_flow", "always_stage_B")}).round(3),
                         width="stretch")
            st.write({k: v for k, v in d7.items() if not isinstance(v, (dict, list))})
            st.dataframe(pd.DataFrame(d7.get("per_class", {})).T, width="stretch")
        else:
            st.info("Run scripts/run_shift_derm7pt.py --data dataset/release_v0")
    with t6:
        if an:
            st.dataframe(pd.DataFrame(an["skin_tone"]).T, width="stretch")
            st.caption("Groups under 30 lesions are too small to judge.")

# ======================================================================
# ARCHITECTURE
# ======================================================================
elif role == "Architecture":
    st.title("Architecture")
    st.graphviz_chart("""
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica", fontsize=11];
  edge [color="#555"];
  subgraph cluster_in { label="Evidence"; style=dashed;
    photo [label="Clinical photo"]; derm [label="Dermoscopy\\n(optional / acquired)"];
    meta [label="Patient details"]; text [label="Symptoms text"]; }
  quality [label="Quality agent\\nblur·light·contrast·glare·covered", fillcolor="#fef3c7"];
  scribe [label="Scribe\\nred flags → urgency only", fillcolor="#fef3c7"];
  subgraph cluster_spec { label="Specialist team"; style=dashed;
    s1 [label="S1 photo head\\n(evidential ResNet50)"]; s2 [label="S2 dermoscopy head"];
    s3 [label="S3 details head"]; s4 [label="S4 PAD-UFES Model A\\n(original system)"];
    s5 [label="S5 similar-case vote\\n(kNN, train only)"]; }
  ds [label="Dempster–Shafer\\nignorance u · conflict C", fillcolor="#e0f2fe"];
  da [label="Devil's Advocate\\nQ · R(learned) · U · C · M · OOD · utility", fillcolor="#fde2e2"];
  fuse [label="Fusion rule picked on validation\\nv2 learned gate | v1 iterative EM | …", fillcolor="#fde2e2"];
  cal [label="Calibrated confidence\\n(isotonic)", fillcolor="#e0f2fe"];
  gate [label="Gatekeeper\\nconformal · Learn-then-Test · cost\\nquality · OOD · missing details", fillcolor="#dcfce7"];
  out [label="HIGH · LOW · ACQUIRE (dermoscopy / details) · DEFER", fillcolor="#dcfce7"];
  agents [label="Librarian · Explainer · Report + verifier", fillcolor="#fef3c7"];
  care [label="Care platform\\npatient · doctor queue · messages · follow-ups · audit", fillcolor="#f3e8ff"];
  photo -> quality; derm -> quality; text -> scribe;
  photo -> s1; derm -> s2; meta -> s3; photo -> s4; photo -> s5; derm -> s5; meta -> s5;
  s1 -> ds; s2 -> ds; s3 -> ds;
  s1 -> da; s2 -> da; s3 -> da; s4 -> da; s5 -> da; quality -> da;
  da -> fuse -> cal -> gate; ds -> gate; quality -> gate; scribe -> out;
  gate -> out -> agents -> care; out -> derm [style=dashed, label="ask", fontsize=9];
}
""", width="stretch")
    st.markdown("""
| Component | What it does | File |
|---|---|---|
| Data loader | MILK10k, lesion-level split (no lesion in two splits) | `src/data/milk10k.py` |
| Evidential heads S1-S3 | Dirichlet evidence → belief + ignorance per input | `notebooks/kaggle_train_dguard.py` (training), `src/native/engine.py` |
| S4 PAD-UFES Model A | The original project's trained model, 6 → 11 classes | `src/specialists/` |
| S5 similar-case vote | 15 nearest training lesions in embedding space | `src/fusion/adaptive.py` (`knn_probs`) |
| Devil's Advocate | Quality, learned reliability, uncertainty, consistency, missingness, OOD, utility | `src/fusion/adaptive.py`, `src/fusion/gating.py` |
| Fusion rules | average, original formula, best head, EM v1, stacking, learned gate v2 | `src/fusion/methods.py` |
| Gatekeeper / policy | Conformal sets, Learn-then-Test, cost matrix, tiers | `src/decision/policy.py`, `src/calibration/` |
| Agents | Quality, scribe, librarian, explainer, report | `src/agents/` |
| Care platform | SQLite store, routing, queue, messages, deletion | `src/care/` |
| Experiments | Team, decision, statistics, stress, shift | `scripts/` |
""")

# ======================================================================
# SYSTEM
# ======================================================================
else:
    st.title("System")
    st.json(store.stats())
    st.dataframe(as_table(store.audit_log(limit=200)), hide_index=True, width="stretch")
    with st.expander("⚠️ Reset the demo database"):
        st.caption("Deletes every patient, case, message and uploaded photo on this platform. "
                   "Dataset images are never touched.")
        if st.checkbox("I understand everything will be deleted") and \
                st.button("Reset database", type="primary"):
            n = svc.reset_all()
            st.session_state.clear()
            st.toast(f"Database reset ({n} uploaded file(s) removed)")
            st.rerun()
