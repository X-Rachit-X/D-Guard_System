"""
app/dguard_app.py - D-GUARD care platform (patient, doctor and system views).

    streamlit run app/dguard_app.py

Runs the Kaggle-trained MILK10k model through the calibrated PROCEED / ACQUIRE /
DEFER policy, and wraps it in the care workflow: intake, dermoscopy requests,
doctor review queue, instructions, questions and follow-ups. Everything is
stored in data/care/dguard_care.sqlite and every step is audited.

Research prototype - not clinically validated.
"""
from __future__ import annotations

import json
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

st.set_page_config(page_title="D-GUARD", page_icon="🩺", layout="wide")

ACTION_STYLE = {
    "PROCEED": ("✅", "#1b7f3b", "Answered - HIGH confidence"),
    "PROCEED_LOW": ("⚠️", "#b54708", "Answered - LOW confidence, a doctor will confirm"),
    "ACQUIRE": ("📷", "#9a6700", "More evidence needed"),
    "DEFER": ("🧑‍⚕️", "#b42318", "Sent to a doctor"),
}


# ------------------------------------------------------------------ backend
@st.cache_resource(show_spinner="Loading the trained model (first time only)...")
def backend():
    from src.agents.pipeline import DGuardPipeline
    engine = DGuardPipeline.build()
    svc = CareService(engine, CareStore(ROOT / "data" / "care" / "dguard_care.sqlite"),
                      upload_dir=ROOT / "data" / "care" / "uploads")
    samples = engine.sample_lesions("test", n=24, seed=7)
    return engine, svc, samples


try:
    engine, svc, samples = backend()
except FileNotFoundError as e:
    st.error(f"Missing artifact: {e}")
    st.code("python scripts/milk_native_infer.py\n"
            "python scripts/run_eade_milk.py --missed-cost 5 --out outputs/eade_milk/cost_5")
    st.stop()

store = svc.store
train = engine.splits["train"]
SITES = sorted(x for x in train["site"].dropna().unique() if x)
GENERAL_SITES = sorted(x for x in train["anatom_site_general"].dropna().unique() if x)


def badge(action, urgency=None, tier=None):
    key = "PROCEED_LOW" if action == "PROCEED" and tier == "LOW" else action
    icon, colour, text = ACTION_STYLE.get(key, ("•", "#555", action))
    urg = (f" &nbsp;<span style='background:#b42318;color:white;padding:2px 8px;"
           f"border-radius:10px;font-size:0.8em'>URGENT</span>" if urgency == "URGENT" else "")
    st.markdown(f"<div style='border-left:6px solid {colour};padding:8px 14px;"
                f"background:rgba(0,0,0,0.03);border-radius:4px'>"
                f"<b style='color:{colour};font-size:1.2em'>{icon} {action}</b> - {text}{urg}"
                f"</div>", unsafe_allow_html=True)


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


def evidence_panel(result: dict):
    d = result["decision"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(malignant)", f"{(d.get('p_malignant') or 0):.0%}")
    c2.metric("Ignorance u", f"{result['ignorance']:.3f}",
              help="How much the system does NOT know. 1.0 = no evidence at all.")
    c3.metric("Conflict C", f"{result['conflict']:.3f}",
              help="How strongly the sources contradict each other.")
    c4.metric("Stage", "with dermoscopy" if result["stage"] == "B" else "photo + details")

    l, r = st.columns(2)
    with l:
        st.markdown("**Dynamic fusion weights** - share of evidence each source "
                    "contributed to *this* case")
        w = result.get("fusion_weights") or {}
        if w:
            st.bar_chart(pd.DataFrame({"weight": w}), horizontal=True, height=160)
        st.markdown("**What each source says on its own**")
        st.write(result.get("modality_votes") or {})
    with r:
        st.markdown("**Differential** (most likely diagnoses)")
        diff = d.get("differential") or []
        if diff:
            st.bar_chart(pd.DataFrame({"probability": {
                engine.classes[int(c)]: float(p) for c, p in diff}}),
                horizontal=True, height=160)
        rel = result.get("reliability_for_tentative") or {}
        if rel:
            st.markdown("**Measured reliability of each source for this diagnosis** "
                        "(validation set)")
            st.write({k: (None if v is None else round(v, 2)) for k, v in rel.items()})
    gates = [k for k, v in (d.get("gates") or {}).items() if v]
    st.caption("Safety gates fired: " + (", ".join(gates) if gates else "none"))


def as_table(rows):
    """DataFrame with every non-scalar cell as text, so mixed columns render cleanly."""
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].map(lambda v: isinstance(v, (dict, list))).any() or df[c].dtype == object:
            df[c] = df[c].map(lambda v: json.dumps(v, default=str)
                              if isinstance(v, (dict, list)) else ("" if v is None else str(v)))
    return df


def messages_panel(cid, audience):
    for m in store.messages(cid):
        who = {"system": "🤖 D-GUARD", "doctor": "🧑‍⚕️ Doctor", "patient": "🙂 Patient"}[m["sender"]]
        if audience == "patient" and m["sender"] == "patient" and m["kind"] == "question":
            who = "🙂 You"
        with st.chat_message("assistant" if m["sender"] != "patient" else "user"):
            st.markdown(f"**{who}** · _{m['kind']}_ · {m['created_at'][:16].replace('T', ' ')}")
            st.write(m["body"])


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🩺 D-GUARD")
st.sidebar.caption("Risk-controlled, cost-aware evidence acquisition and deferral "
                   "for multimodal dermatology AI")
role = st.sidebar.radio("View as", ["Patient", "Doctor", "System"], index=0)
st.sidebar.divider()
s = store.stats()
st.sidebar.metric("Cases", s["cases"])
st.sidebar.metric("Waiting for a doctor", s["by_status"].get(CaseStatus.AWAITING_DOCTOR, 0))
st.sidebar.caption("Research prototype - not clinically validated. "
                   "Do not use for diagnosis.")

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
                pid = svc.register_patient(name, age, sex, tone)
                st.session_state["pid"] = pid
                st.rerun()
    if not patients:
        st.info("Register a patient to begin.")
        st.stop()

    labels = {p["id"]: f"#{p['id']} {p['name']}" for p in patients}
    pid = st.selectbox("Patient", list(labels), format_func=labels.get,
                       index=list(labels).index(st.session_state.get("pid", list(labels)[0]))
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
    cid = st.selectbox("Case", list(clabels), format_func=clabels.get,
                       index=list(clabels).index(st.session_state["cid"])
                       if st.session_state.get("cid") in clabels else 0)
    case = store.case(cid)

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("1 · Evidence")
        sample_names = {i: f"{s_['lesion_id']} (demo lesion)" for i, s_ in enumerate(samples)}
        use_sample = st.toggle("Use a real held-out MILK10k lesion (demo)", value=True)
        if use_sample:
            si = st.selectbox("Demo lesion", list(sample_names), format_func=sample_names.get)
            sample = samples[si]
            st.caption("Held-out test lesion - the model never trained on it. "
                       f"True diagnosis is hidden from the system: {sample['true_label']}")
        else:
            sample = None
            up = st.file_uploader("Clinical photo of the spot", type=["jpg", "jpeg", "png"])
            m1, m2 = st.columns(2)
            gsite = m1.selectbox("Body area", GENERAL_SITES)
            site = m2.selectbox("Exact site", SITES,
                                index=SITES.index(case["site"]) if case["site"] in SITES else 0)

        if st.button("Submit photo + details and assess", type="primary",
                     disabled=case["status"] == CaseStatus.CLOSED):
            if sample:
                svc.submit_photo(cid, sample["clinical_path"])
                svc.submit_details(cid, sample["metadata"])
                st.session_state[f"derm_{cid}"] = sample["dermoscopy_path"]
            elif up is not None:
                svc.submit_photo(cid, Image.open(up))
                svc.submit_details(cid, {"age": patient["age"], "sex": patient["sex"],
                                         "skin_tone_class": patient["skin_tone"],
                                         "site": site, "anatom_site_general": gsite})
            else:
                st.warning("Upload a photo first.")
                st.stop()
            svc.assess(cid)
            st.rerun()

        if case["status"] == CaseStatus.AWAITING_PATIENT:
            st.warning("📷 The system asked for a **dermoscopy photo** of this spot.")
            derm_sample = st.session_state.get(f"derm_{cid}")
            if derm_sample and st.button("Upload the dermoscopy photo (demo lesion)"):
                svc.submit_dermoscopy(cid, derm_sample)
                svc.assess(cid)
                st.rerun()
            upd = st.file_uploader("…or upload your own dermoscopy photo",
                                   type=["jpg", "jpeg", "png"], key=f"du{cid}")
            if upd is not None and st.button("Submit dermoscopy"):
                svc.submit_dermoscopy(cid, Image.open(upd))
                svc.assess(cid)
                st.rerun()

        if case["status"] in (CaseStatus.FOLLOW_UP, CaseStatus.ASSESSED) and \
                store.followups(cid, status="SCHEDULED"):
            st.info("🗓️ Follow-up scheduled for "
                    + ", ".join(f["due_date"] for f in store.followups(cid, status="SCHEDULED")))
            fup = st.file_uploader("Upload your follow-up photo", type=["jpg", "jpeg", "png"],
                                   key=f"fu{cid}")
            fs = st.session_state.get(f"derm_{cid}")
            c_a, c_b = st.columns(2)
            if fup is not None and c_a.button("Submit follow-up photo"):
                out = svc.submit_followup_photo(cid, Image.open(fup))
                st.session_state[f"change_{cid}"] = out["change"]
                st.rerun()
            last = store.latest_evidence(cid, "clinical_photo")
            if last and c_b.button("Re-submit the same photo (demo: no change)"):
                out = svc.submit_followup_photo(cid, last["path"])
                st.session_state[f"change_{cid}"] = out["change"]
                st.rerun()
        ch = st.session_state.get(f"change_{cid}")
        if ch:
            (st.error if ch["alert"] else st.success)(
                f"Change check: P(malignant) {ch['previous_p_malignant']:.0%} → "
                f"{ch['p_malignant']:.0%}" + ("  — sent to a doctor urgently" if ch["alert"]
                                               else "  — no worrying change"))
        show_images(cid)

    with right:
        st.subheader("2 · Result")
        a = store.latest_assessment(cid)
        if a:
            badge(a["action"], a["urgency"], a["result"]["decision"].get("tier"))
            st.write("")
            st.markdown(a["result"]["patient_message"].replace("\n", "  \n"))
            rv = store.reviews(cid)
            if rv:
                st.success(f"🧑‍⚕️ Reviewed by Dr {rv[-1]['doctor']}: "
                           f"**{rv[-1]['verdict']}** - {rv[-1]['diagnosis'] or ''}")
        else:
            st.caption("No assessment yet.")
        st.subheader("3 · Messages & instructions")
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
        st.success("Nothing waiting. Deferred cases and system answers needing sign-off "
                   "appear here, most urgent first.")
        st.stop()
    qdf = pd.DataFrame(queue)
    qdf["p_malignant"] = qdf["p_malignant"].map(lambda x: f"{x:.0%}" if x is not None else "-")
    st.dataframe(qdf[["case_id", "patient", "kind", "urgency", "action", "tentative",
                      "p_malignant", "unanswered_questions", "waiting_since"]],
                 hide_index=True, width="stretch")

    cid = st.selectbox("Open case", [r["case_id"] for r in queue],
                       format_func=lambda c: f"Case #{c}")
    case = store.case(cid)
    a = store.latest_assessment(cid)
    p = store.patient(case["patient_id"])
    st.markdown(f"**{p['name']}** · age {p['age'] or '?'} · {p['sex'] or '?'} · "
                f"skin tone {p['skin_tone'] or '?'} · site **{case['site'] or '?'}**")

    tab1, tab2, tab3 = st.tabs(["Evidence & assessment", "Messages", "Timeline & audit"])
    with tab1:
        show_images(cid)
        if a:
            badge(a["action"], a["urgency"], a["result"]["decision"].get("tier"))
            st.code(a["result"]["doctor_message"], language=None)
            rep = a["result"].get("report") or {}
            if rep.get("text"):
                st.markdown(f"**Report** _(source: {rep.get('source')})_")
                st.info(rep["text"])
            for kind, q in (a["result"].get("quality") or {}).items():
                if q and q.get("issues"):
                    st.warning(f"Image quality - {kind}: " + ", ".join(q["issues"]))
            evidence_panel(a["result"])
            sim = (a["result"].get("similar") or {}).get("similar_cases") or []
            if sim:
                agree = a["result"]["similar"].get("agreement")
                st.markdown("**Similar confirmed past cases** (training lesions, "
                            "histopathology-backed; shown for context, never used to decide)"
                            + (f" - {agree:.0%} share the tentative diagnosis"
                               if agree is not None else ""))
                cols = st.columns(len(sim))
                for col, c in zip(cols, sim):
                    with col:
                        if c.get("clinical_path") and Path(c["clinical_path"]).exists():
                            st.image(c["clinical_path"], width="stretch")
                        st.caption(f"**{c['diagnosis']}** · sim {c['similarity']:.2f}")
            if len(store.assessments(cid)) > 1:
                st.markdown("**Assessment history**")
                st.dataframe(pd.DataFrame([{k: x[k] for k in ("created_at", "stage", "action",
                                                               "tentative", "p_malignant")}
                                           for x in store.assessments(cid)]),
                             hide_index=True)
    with tab2:
        messages_panel(cid, "doctor")
        reply = st.chat_input("Reply to the patient")
        if reply:
            svc.reply(cid, doctor, reply)
            st.rerun()
    with tab3:
        st.dataframe(as_table(svc.timeline(cid)), hide_index=True, width="stretch")
        with st.expander("Raw audit log"):
            st.json(store.audit_log(cid))

    st.subheader("Your decision")
    with st.form("review"):
        verdict = st.radio("Verdict", ["CONFIRM", "OVERRIDE", "NEED_IN_PERSON"],
                           horizontal=True,
                           help="CONFIRM: agree with the system's (tentative) call. "
                                "OVERRIDE: state your own diagnosis. "
                                "NEED_IN_PERSON: book a visit.")
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
# SYSTEM
# ======================================================================
else:
    st.title("System")
    t1, t2, t3 = st.tabs(["Research results", "Platform activity", "How it works"])
    with t1:
        st.markdown("Native evidential model trained on MILK10k (Kaggle, seed 42), "
                    "calibrated on 786 validation lesions, scored once on 787 test lesions.")
        rows = []
        for cost in (5, 10):
            f = ROOT / "outputs" / "eade_milk" / f"cost_{cost}" / "report.json"
            if f.exists():
                r = json.loads(f.read_text())
                for b in r["baselines"]:
                    rows.append({"miss cost": cost, **b})
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df.style.format({c: "{:.3f}" for c in df.columns
                                          if df[c].dtype.kind == "f"}),
                         hide_index=True, width="stretch")
            st.caption("cover = answered alone · acq = dermoscopy requested · miss = cancer "
                       "called benign among answered · FA = benign called cancer · "
                       "saysM = share of answers naming a cancer · cost = mean cost/case")
        rel = engine.reliability
        st.markdown("**Which source to trust for which diagnosis** (measured, validation)")
        st.dataframe(pd.DataFrame({"clinical photo": rel["image"],
                                   "dermoscopy": rel["dermoscopy"],
                                   "patient details": rel["meta"],
                                   "n": rel["n_per_class"]}, index=rel["classes"]),
                     width="stretch")
    with t2:
        st.json(store.stats())
        st.dataframe(as_table(store.audit_log(limit=200)), hide_index=True,
                     width="stretch")
    with t3:
        st.markdown("""
1. **Each source forms its own opinion.** Photo, dermoscopy and patient details each get
   an evidential head: *belief* in each diagnosis plus *ignorance* `u` (1 = no evidence).
2. **Opinions are combined by Dempster–Shafer.** A missing source is total ignorance and
   changes nothing. A weak source has high `u` and little influence - the **dynamic
   fusion weights** shown per case. Disagreement shows up as **conflict `C`**.
3. **Safety gates.** A conformal set over *benign / malignant* blocks false reassurance;
   a Learn-then-Test threshold certifies the risk level; high conflict sends the case
   to a human.
4. **Cheapest action.** Compare the expected cost of answering, asking for dermoscopy, or
   deferring - with a missed cancer priced far above a false alarm.
5. **Care workflow.** ACQUIRE asks the patient for a dermoscopy photo; DEFER goes to the
   doctor queue *with* the system's tentative opinion; PROCEED gives a result and a
   follow-up. Follow-up photos are checked for change.
""")
