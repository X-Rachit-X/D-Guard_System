"""
scripts/demo_care.py - The whole system, end to end, in the terminal.

Real trained model, real held-out MILK10k lesions (never training lesions).
Four patients, one per outcome:

  1  PROCEED (HIGH)  confident answer, with instructions and a follow-up
  2  ACQUIRE         asks for a dermoscopy photo, then decides
  3  PROCEED (LOW)   answers, clearly marked low confidence; a doctor confirms
  4  DEFER           the sources contradict; tentative opinion + doctor review,
               who reviews it and books a follow-up; at the follow-up a new photo
               is checked for change

Use it as the panel-demo backup: it needs no browser and no server.

    python scripts/demo_care.py
    python scripts/demo_care.py --db data/care/demo.sqlite   # keep the database
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.care import CareService, CareStore  # noqa: E402

W = 78


def banner(t):
    print()
    print("=" * W)
    print(t)
    print("=" * W)


def show(label, text):
    print(f"\n[{label}]")
    for line in str(text).splitlines():
        print(textwrap.fill(line, W, initial_indent="  ", subsequent_indent="    "))


def pick_cases(engine, n_scan=787):
    """Find one real test lesion per path. Deterministic order."""
    test = engine.splits["test"]
    found = {}
    for i in range(min(n_scan, len(test))):
        r = test.iloc[i]
        meta = _meta(engine, r)
        cp = str(engine.img_dir / r["clinical_file"])
        dp = str(engine.img_dir / r["dermoscopy_file"])
        a = engine.assess_paths(cp, None, meta)
        act = a.action
        if act == "ACQUIRE" and "ACQUIRE" not in found:
            b = engine.assess_paths(cp, dp, meta)
            if (b.action == "PROCEED" and b.decision.tier != "LOW"
                    and b.decision.prediction == int(r["label_idx"])):
                found["ACQUIRE"] = (r, meta, cp, dp)
        elif act == "DEFER" and "DEFER" not in found and a.decision.tentative is not None:
            found["DEFER"] = (r, meta, cp, dp)
        elif act == "PROCEED" and a.decision.tier == "LOW" and "LOW" not in found \
                and a.decision.urgency == "URGENT":
            found["LOW"] = (r, meta, cp, dp)
        elif (act == "PROCEED" and a.decision.tier != "LOW" and "PROCEED" not in found
              and a.decision.prediction == int(r["label_idx"])):
            found["PROCEED"] = (r, meta, cp, dp)
        if len(found) == 4:
            break
    return found


def _meta(engine, r):
    return {"age": float(r["age"]), "sex": r["sex"], "site": r["site"],
            "anatom_site_general": r["anatom_site_general"],
            "skin_tone_class": str(r["skin_tone_class"]),
            **{c: float(r[c]) for c in engine.monet_median}}


def explain_result(out):
    res = out["result"]
    tier = f" [{res.decision.tier} confidence]" if res.decision.tier else ""
    print(f"\n  ACTION   : {out['action']}{tier}   (stage {res.stage}, urgency {out['urgency']})")
    print(f"  tentative: {out['tentative']}   P(malignant) "
          f"{res.decision.p_malignant:.0%}   ignorance u {res.ignorance:.3f}   "
          f"conflict C {res.conflict:.3f}")
    print(f"  dynamic fusion weights (evidence share, this case): {res.fusion_weights}")
    print(f"  what each source says on its own: {res.modality_votes}")
    sim = (res.similar or {}).get("similar_cases") or []
    if sim:
        print(f"  similar confirmed past cases: "
              f"{[c['diagnosis'] for c in sim]}  (agreement {res.similar.get('agreement') or 0:.0%})")
    q = [f"{k}: {v['issues']}" for k, v in (res.quality or {}).items() if v and v["issues"]]
    if q:
        print(f"  photo quality warnings: {q}")
    if res.report:
        print(f"  report ({res.report['source']}): "
              f"{res.report['text'].splitlines()[0]}")
    if res.reliability_for_tentative:
        rel = {k: (None if v is None else round(v, 2))
               for k, v in res.reliability_for_tentative.items()}
        print(f"  measured reliability of each source for "
              f"{out['tentative']}: {rel}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=":memory:")
    args = ap.parse_args()

    from src.agents.pipeline import DGuardPipeline
    banner("D-GUARD care platform - live demo with the trained MILK10k model")
    print("loading model and calibration ...")
    engine = DGuardPipeline.build()
    svc = CareService(engine, CareStore(args.db), upload_dir=ROOT / "data" / "care" / "demo_uploads")
    print("finding one real held-out lesion for each path ...")
    print("(examples are SELECTED to illustrate each path - the HIGH and ACQUIRE examples")
    print(" are correctly answered cases. Real accuracy, including confident errors, is in")
    print(" docs/05_results.md: HIGH tier 83.8% accurate, 34.8% false alarms on benign.)")
    cases = pick_cases(engine)
    missing = {"PROCEED", "ACQUIRE", "LOW", "DEFER"} - set(cases)
    if missing:
        print(f"note: no example found for {missing} in the scanned lesions")

    # ------------------------------------------------------------ patient 1
    if "PROCEED" in cases:
        r, meta, cp, dp = cases["PROCEED"]
        banner(f"PATIENT 1 - confident case   (lesion {r['lesion_id']}, "
               f"truth {r['label_raw']})")
        pid = svc.register_patient("Patient One", age=meta["age"], sex=meta["sex"])
        cid = svc.open_case(pid, site=meta["site"], complaint="spot on the skin")
        svc.submit_photo(cid, cp)
        svc.submit_details(cid, meta)
        out = svc.assess(cid)
        explain_result(out)
        show("what the patient sees", out["result"].patient_message)
        for m in svc.store.messages(cid):
            if m["kind"] == "instruction":
                show("instruction", m["body"])
        print(f"\n  case status: {svc.store.case(cid)['status']}  "
              f"follow-ups: {[f['due_date'] for f in svc.store.followups(cid)]}")

    # ------------------------------------------------------------ patient 2
    if "ACQUIRE" in cases:
        r, meta, cp, dp = cases["ACQUIRE"]
        banner(f"PATIENT 2 - needs more evidence   (lesion {r['lesion_id']}, "
               f"truth {r['label_raw']})")
        pid = svc.register_patient("Patient Two", age=meta["age"], sex=meta["sex"])
        cid = svc.open_case(pid, site=meta["site"])
        svc.submit_photo(cid, cp)
        svc.submit_details(cid, meta)
        out = svc.assess(cid)
        explain_result(out)
        show("what the patient sees", out["result"].patient_message)
        print("\n  ... the patient gets a dermoscopy photo taken and uploads it ...")
        svc.submit_dermoscopy(cid, dp)
        out2 = svc.assess(cid)
        explain_result(out2)
        show("what the patient sees now", out2["result"].patient_message)
        print(f"\n  ignorance before -> after dermoscopy: "
              f"{out['result'].ignorance:.3f} -> {out2['result'].ignorance:.3f}")

    # ------------------------------------------------------- low confidence
    if "LOW" in cases:
        r, meta, cp, dp = cases["LOW"]
        banner(f"PATIENT 3 - answered, but LOW confidence   (lesion {r['lesion_id']}, "
               f"truth {r['label_raw']})")
        pid = svc.register_patient("Patient Low", age=meta["age"], sex=meta["sex"])
        cid = svc.open_case(pid, site=meta["site"])
        svc.submit_photo(cid, cp)
        svc.submit_details(cid, meta)
        out = svc.assess(cid)
        explain_result(out)
        show("what the patient sees", out["result"].patient_message)
        print(f"\n  case status: {svc.store.case(cid)['status']}  (a doctor confirms it)")

    # ------------------------------------------------------------ patient 3
    if "DEFER" in cases:
        r, meta, cp, dp = cases["DEFER"]
        banner(f"PATIENT 4 - the system will not decide alone   (lesion "
               f"{r['lesion_id']}, truth {r['label_raw']})")
        pid = svc.register_patient("Patient Three", age=meta["age"], sex=meta["sex"])
        cid = svc.open_case(pid, site=meta["site"])
        svc.submit_photo(cid, cp)
        svc.submit_details(cid, meta)
        out = svc.assess(cid)
        explain_result(out)
        show("what the patient sees", out["result"].patient_message)
        show("what the doctor sees", out["result"].doctor_message)

        svc.ask(cid, "Is this serious? Should I be worried?")
        banner("DOCTOR'S QUEUE")
        for q in svc.doctor_queue():
            print(f"  case {q['case_id']:>2}  {q['kind']:8} {q['urgency']:8} "
                  f"{q['action'] or '-':8} tentative {q['tentative'] or '-':8} "
                  f"P(mal) {q['p_malignant'] or 0:.0%}  questions {q['unanswered_questions']}")

        svc.reply(cid, "Rao", "Thanks for asking - I am looking at your photos now.")
        svc.doctor_review(cid, "Rao", "CONFIRM",
                          note="Agree with the tentative call; re-image in 3 months.",
                          instruction="I have reviewed your photos. Please book a routine "
                                      "dermatology visit and send a new photo in 3 months.",
                          followup_days=90)
        print(f"\n  after review: status {svc.store.case(cid)['status']}, "
              f"diagnosis {svc.store.reviews(cid)[-1]['diagnosis']}")

        print("\n  ... 3 months later the patient uploads a new photo ...")
        fu = svc.submit_followup_photo(cid, cp, details=meta)
        ch = fu["change"]
        print(f"  change check: P(malignant) {ch['previous_p_malignant']:.0%} -> "
              f"{ch['p_malignant']:.0%}   alert: {ch['alert']}")
        print(f"  status now: {svc.store.case(cid)['status']}")

        banner("CASE TIMELINE (everything is recorded)")
        for e in svc.timeline(cid):
            print(f"  {e['t'][11:19]}  {e['type']:10} {e['what']}")

    banner("PLATFORM SUMMARY")
    for k, v in svc.store.stats().items():
        print(f"  {k:24} {v}")
    print()
    print("Research prototype - not clinically validated. Do not use for diagnosis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
