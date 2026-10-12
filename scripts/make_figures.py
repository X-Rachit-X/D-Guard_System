"""
scripts/make_figures.py - Every figure in the paper and the report, from saved results.

Reads only outputs/ (no images, no patient data) and writes vector PDFs for LaTeX and
PNGs for Word to paper/figures/. Run after the experiments:

    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

from src.constants import MILK_CLASS_NAMES, milk_malignant_indices  # noqa: E402

MAL = milk_malignant_indices()
K = len(MILK_CLASS_NAMES)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
C_DG, C_B3, C_B4, C_GREY = "#1f4e79", "#c0504d", "#e69f00", "#9e9e9e"


def J(p):
    p = ROOT / p
    return json.loads(p.read_text()) if p.exists() else None


def L(p):
    p = ROOT / p
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("  ", name)


def missed(y, pred):
    return np.isin(y, MAL) & ~np.isin(pred, MAL)


# ------------------------------------------------------------------ 1 fusion rules
def fig_fusion():
    tr = J("outputs/milk_team/team_report.json")
    if not tr:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, st, title in ((axes[0], "A", "Photo + details"), (axes[1], "B", "+ dermoscopy")):
        rows = [(k[2:], v["macro_f1"], tr["fusion_val_oof"][k]["macro_f1"])
                for k, v in tr["fusion"].items() if k.startswith(st + ":")]
        names = [r[0].replace("Devil's Advocate v2 – ", "DA v2 ").replace("Devil's Advocate v2 – ", "DA v2 ")
                 .replace(" (evidential heads)", "").replace("(logistic regression)", "")
                 .replace("Best head per case (highest R)", "Best head (max R)") for r in rows]
        y = np.arange(len(rows))
        ax.barh(y - 0.2, [r[2] for r in rows], 0.4, color=C_GREY, label="validation (out-of-fold)")
        ax.barh(y + 0.2, [r[1] for r in rows], 0.4, color=C_DG, label="test")
        sel = tr["selected"][st]
        from src.fusion.methods import LABELS
        for i, r in enumerate(rows):
            if r[0] == LABELS.get(sel):
                ax.text(r[1] + 0.005, i + 0.2, "selected", va="center", fontsize=7, color=C_DG)
        ax.set_yticks(y, names, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("11-class macro-F1")
        ax.axvline(tr["oracle"][st]["macro_f1"], ls=":", color="k", lw=0.8)
        ax.text(tr["oracle"][st]["macro_f1"], len(rows) - 0.4, " oracle", fontsize=7)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig_fusion_rules")


# ------------------------------------------------------------------ 2 risk-coverage
def fig_risk_coverage():
    c = L("outputs/eade_team/cost_5/cases.npz")
    t = L("outputs/milk_team/test.npz")
    an = J("outputs/analysis/analysis.json")
    if c is None or t is None:
        return
    y = c["y"]
    pA = t["no_dermo_prob"]
    pred = pA.argmax(1)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    covs = np.linspace(0.1, 1.0, 46)
    for score, lab, col in ((pA.max(1), "confidence threshold (B3)", C_B3),
                            (1 - t["no_dermo_u"].ravel(), "1 - ignorance", C_GREY)):
        r = []
        for cv in covs:
            keep = score >= np.quantile(score, 1 - cv)
            r.append(missed(y[keep], pred[keep]).mean())
        ax.plot(covs * 100, np.array(r) * 100, color=col, label=lab, lw=1.4)
    high = (c["action"] == "PROCEED") & (c["tier"] != "LOW")
    ax.scatter([high.mean() * 100], [missed(y[high], c["pred"][high]).mean() * 100], s=45,
               color=C_DG, zorder=5, label="D-GUARD HIGH tier")
    if an and an.get("systems"):
        b4 = an["systems"].get("B4 learned rejector (L2D)")
        if b4:
            ax.scatter([b4["coverage"] * 100], [b4["missed_cancer_rate"] * 100], marker="s", s=35,
                       color=C_B4, zorder=5, label="learned rejector (B4)")
    ax.axhline(5, ls=":", color="k", lw=0.8)
    ax.text(12, 5.2, "target 5%", fontsize=7)
    ax.set_xlabel("patients answered (%)")
    ax.set_ylabel("cancers called benign (%)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    save(fig, "fig_risk_coverage")


# ------------------------------------------------------------------ 3 back-end robustness
def fig_backend():
    rows = []
    for name, d in (("EM v1\n5 specialists", "outputs/eade_team_em_noS6/cost_5"),
                    ("gate v2\n5 specialists", "outputs/eade_team_noS6/cost_5"),
                    ("EM v1\n6 specialists", "outputs/eade_team_em/cost_5"),
                    ("gate v2\n6 specialists", "outputs/eade_team/cost_5")):
        r = J(f"{d}/report.json")
        if not r:
            continue
        bl = {b["policy"]: b for b in r["baselines"]}
        b3 = [b for k, b in bl.items() if k.startswith("B3 max-softmax abstain, photo")][0]
        rows.append((name, r["dguard"]["miss_rate_proceeded"], b3["miss_rate"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r[1] * 100 for r in rows], 0.36, color=C_DG, label="D-GUARD policy")
    ax.bar(x + 0.18, [r[2] * 100 for r in rows], 0.36, color=C_B3, label="confidence threshold")
    for i, r in enumerate(rows):
        ax.text(i - 0.18, r[1] * 100 + 0.2, f"{r[1] * 100:.1f}", ha="center", fontsize=7)
        ax.text(i + 0.18, r[2] * 100 + 0.2, f"{r[2] * 100:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=7)
    ax.set_ylabel("missed cancers, same coverage (%)")
    ax.axhline(5, ls=":", color="k", lw=0.8)
    ax.legend(fontsize=7, frameon=False)
    save(fig, "fig_backend_robustness")


# ------------------------------------------------------------------ 4 stress
def fig_stress():
    s = J("outputs/stress_milk/report_team.json")
    if not s:
        return
    conds = list(s["conditions"])
    m = [s["conditions"][k]["high_miss_rate_model_only"] or 0 for k in conds]
    g = [s["conditions"][k]["high_miss_rate"] or 0 for k in conds]
    hm = [s["conditions"][k]["high_rate_model_only"] for k in conds]
    hg = [s["conditions"][k]["high_rate"] for k in conds]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    x = np.arange(len(conds))
    for ax, a, b, lab in ((axes[0], hm, hg, "high-confidence answers (%)"),
                          (axes[1], m, g, "missed cancers among them (%)")):
        ax.bar(x - 0.2, np.array(a) * 100, 0.4, color=C_GREY, label="model alone")
        ax.bar(x + 0.2, np.array(b) * 100, 0.4, color=C_DG, label="with D-GUARD's checks")
        ax.set_xticks(x, [k.replace("_", " ") for k in conds], rotation=30, fontsize=7)
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=7, frameon=False)
    save(fig, "fig_stress")


# ------------------------------------------------------------------ 5 weights
def fig_weights():
    tr = J("outputs/milk_team/team_report.json")
    if not tr:
        return
    disp = {"clinical": "photo head", "dermoscopy": "dermoscopy head", "metadata": "details head",
            "pad_ufes_A": "old Model A", "knn_cases": "similar cases", "embed_clf": "embedding clf."}
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    cols = ["#4472c4", "#70ad47", "#ffc000", "#a5a5a5", "#ed7d31", "#5b9bd5"]
    for i, st in enumerate(("A", "B")):
        w = tr["mean_weights"][st]
        left = 0
        for j, (k, v) in enumerate(w.items()):
            ax.barh(i, v, left=left, color=cols[list(disp).index(k)],
                    label=disp[k] if i == 1 else None)
            if v > 0.06:
                ax.text(left + v / 2, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white")
            left += v
    ax.set_yticks([0, 1], ["photo + details", "+ dermoscopy"])
    ax.set_xlabel("mean fusion weight (test)")
    ax.legend(fontsize=6, frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.35), loc="upper center")
    save(fig, "fig_weights")


# ------------------------------------------------------------------ 6 calibration
def fig_calibration():
    import joblib
    t = L("outputs/milk_team/test.npz")
    bpath = ROOT / "outputs/milk_team/team_bundle.joblib"
    if t is None or not bpath.exists():
        return
    iso = joblib.load(bpath).confidence_calibrator
    p = t["full_prob"]
    y = t["y"]
    raw = p.max(1)
    cal = iso.predict(raw)
    hit = p.argmax(1) == y
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    bins = np.linspace(0, 1, 11)
    for conf, lab, col in ((raw, "raw fused probability", C_GREY), (cal, "calibrated (shown)", C_DG)):
        xs, ys = [], []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (conf >= a) & (conf < b)
            if m.sum() >= 10:
                xs.append(conf[m].mean())
                ys.append(hit[m].mean())
        ax.plot(xs, ys, "o-", color=col, label=lab, ms=3)
    ax.plot([0, 1], [0, 1], ":", color="k", lw=0.8)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("actual accuracy")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "fig_calibration")


# ------------------------------------------------------------------ 7 legacy
def fig_legacy():
    r = J("outputs/legacy_pad/report.json")
    if not r:
        return
    codes = list("ABCDEFG")
    rep = [r["old_reported_accuracy"].get(c, np.nan) for c in codes]
    old = [r["models"][c]["old test (as reported)"]["accuracy"] for c in codes]
    unseen = [r["models"][c]["patient never in training"]["accuracy"] for c in codes]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    x = np.arange(len(codes))
    ax.bar(x - 0.27, np.array(rep) * 100, 0.27, color="#d9d9d9", label="reported by the old system")
    ax.bar(x, np.array(old) * 100, 0.27, color=C_GREY, label="same weights, same test photos")
    ax.bar(x + 0.27, np.array(unseen) * 100, 0.27, color=C_DG, label="photos of unseen patients")
    ax.set_xticks(x, [f"{c}\n{r['models'][c]['inputs']}" for c in codes], fontsize=6.5)
    ax.set_ylabel("accuracy, PAD-UFES-20 (%)")
    fu = r.get("fusion_patient_unseen", {})
    if "gate" in fu:
        ax.axhline(fu["gate"]["accuracy"] * 100, ls="--", color=C_DG, lw=0.9)
        ax.text(6.6, fu["gate"]["accuracy"] * 100 + 1, "A-G fused by\nlearned gate", fontsize=6,
                ha="right", color=C_DG)
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_ylim(0, 100)
    save(fig, "fig_legacy_models")


# ------------------------------------------------------------------ 8 outcomes + cost
def fig_outcomes_cost():
    rows = []
    for cm in (2, 3, 5, 10):
        r = J(f"outputs/eade_team/cost_{cm}/report.json")
        if r:
            rows.append((cm, r["tiers"]["high_rate"], r["tiers"]["low_rate"], r["tiers"]["defer_rate"],
                         r["dguard"]["miss_rate_proceeded"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    x = np.arange(len(rows))
    h = np.array([r[1] for r in rows]) * 100
    lo = np.array([r[2] for r in rows]) * 100
    d = np.array([r[3] for r in rows]) * 100
    ax.bar(x, h, color=C_DG, label="HIGH answer")
    ax.bar(x, lo, bottom=h, color="#f4b183", label="LOW answer (doctor confirms)")
    ax.bar(x, d, bottom=h + lo, color=C_B3, label="defer")
    ax.set_xticks(x, [f"c_miss={r[0]}" for r in rows])
    ax.set_ylabel("patients (%)")
    ax2 = ax.twinx()
    ax2.plot(x, [r[4] * 100 for r in rows], "ko-", ms=3, lw=1)
    ax2.set_ylabel("missed cancers, HIGH (%)")
    ax2.set_ylim(0, 8)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    save(fig, "fig_outcomes_cost")


# ------------------------------------------------------------------ 9 confusion
def fig_confusion():
    t = L("outputs/milk_team/test.npz")
    if t is None:
        return
    y, pred = t["y"], t["full_prob"].argmax(1)
    M = np.zeros((K, K))
    for a, b in zip(y, pred):
        M[a, b] += 1
    R = M / np.maximum(M.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K), MILK_CLASS_NAMES, rotation=60, fontsize=7)
    ax.set_yticks(range(K), [f"{c} ({int(M[i].sum())})" for i, c in enumerate(MILK_CLASS_NAMES)],
                  fontsize=7)
    for i in range(K):
        for j in range(K):
            if M[i, j]:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=5.5,
                        color="white" if R[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (n)")
    save(fig, "fig_confusion")


# ------------------------------------------------------------------ 10 shift
def fig_shift():
    an = J("outputs/analysis/analysis.json")
    pad = J("outputs/shift_pad/report.json")
    d7 = J("outputs/shift_derm7pt/report.json")
    if not (an and pad and d7):
        return
    lab = ["MILK10k test", "PAD-UFES-20", "Derm7pt"]
    high = [an["ci"]["high_cov"]["point"], pad["high"], d7["dguard_flow"]["high"]]
    acc = [an["ci"]["high_acc"]["point"], pad["high_accuracy"], d7["dguard_flow"]["high_accuracy"]]
    miss = [an["ci"]["high_miss"]["point"], pad["high_miss_milk_def"], d7["dguard_flow"]["high_miss"]]
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    x = np.arange(3)
    ax.bar(x - 0.27, np.array(high) * 100, 0.27, color=C_GREY, label="answered with HIGH confidence")
    ax.bar(x, np.array(acc) * 100, 0.27, color=C_DG, label="accuracy of HIGH answers")
    ax.bar(x + 0.27, np.array(miss) * 100, 0.27, color=C_B3, label="missed cancers among them")
    for i in range(3):
        ax.text(i + 0.27, miss[i] * 100 + 1, f"{miss[i] * 100:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, lab)
    ax.set_ylabel("%")
    ax.legend(fontsize=6.5, frameon=False)
    save(fig, "fig_shift")


# ------------------------------------------------------------------ 11 gate importance
def fig_gate():
    tr = J("outputs/milk_team/team_report.json")
    if not tr or not tr.get("gate_importance"):
        return
    gi = tr["gate_importance"]
    feats = list(gi["B"])
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    y = np.arange(len(feats))
    ax.barh(y - 0.2, [gi["A"][f] for f in feats], 0.4, color=C_GREY, label="photo + details")
    ax.barh(y + 0.2, [gi["B"][f] for f in feats], 0.4, color=C_DG, label="+ dermoscopy")
    ax.set_yticks(y, feats, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("mean |coefficient| (standardised)")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "fig_gate_importance")


# ------------------------------------------------------------------ 12 symptoms
def fig_symptoms():
    s = J("outputs/analysis/pad_symptoms.json")
    if not s:
        return
    d = s["definitions"]["PAD-UFES (BCC, SCC, MEL)"]["symptoms"]
    names = list(d)
    rr = [d[n]["relative_risk"] for n in names]
    lo = [d[n]["relative_risk"] - d[n]["ci95"][0] for n in names]
    hi = [d[n]["ci95"][1] - d[n]["relative_risk"] for n in names]
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    y = np.arange(len(names))
    ax.errorbar(rr, y, xerr=[lo, hi], fmt="o", color=C_DG, ms=4, capsize=2)
    ax.axvline(1, ls=":", color="k", lw=0.8)
    ax.set_yticks(y, [n.replace(" (not a scribe flag)", "") for n in names], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("relative risk of cancer (95% CI)")
    save(fig, "fig_symptoms")


def fig_architecture():
    """Block diagram of the whole system (no data needed)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.set_xlim(0, 120)
    ax.set_ylim(-4, 66)
    ax.axis("off")
    FS = 6.0

    def box(x, y, w, h, text, fc, fs=FS):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc=fc, ec="#555555", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                linespacing=1.25)

    def arrow(x1, y1, x2, y2, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.7, color="#444444", linestyle=ls))

    inputs = [("Clinical photo", 52), ("Dermoscopy\n(optional or requested)", 42),
              ("Patient details", 32), ("Symptoms text", 6)]
    for t, y in inputs:
        box(1, y, 16, 6.5, t, "#eef2ff")
    spec = [("S1 photo head (evidential)", 55), ("S2 dermoscopy head", 49.5),
            ("S3 details head", 44), ("S4 old Model A", 38.5),
            ("S5 similar-case vote", 33), ("S6 embedding classifier", 27.5)]
    for t, y in spec:
        box(23, y, 22, 4.3, t, "#e2efda")
    box(23, 17, 22, 6.5, "Quality agent\nblur, light, grain, covered", "#fff2cc")
    box(23, 6, 22, 6.5, "Scribe\nwarning signs raise urgency", "#fff2cc")
    arrow(17.6, 55, 22.6, 57)
    arrow(17.6, 45, 22.6, 51.5)
    arrow(17.6, 35, 22.6, 46)
    arrow(17.6, 9, 22.6, 9)
    ax.text(26, 1.5, "(the model has no text input)", fontsize=5.2, style="italic")
    box(51, 30, 18, 29, "Devil's Advocate\n\nquality Q\nlearned reliability R\n"
        "uncertainty U\ntrack record\nagreement, margin\nconsistency C\nmissingness M\n"
        "OOD distance", "#fde2e2")
    for _, y in spec:
        arrow(45.6, y + 2.1, 50.6, 44.5)
    arrow(45.6, 20, 50.6, 34)
    box(51, 8, 18, 17, "Fusion rule chosen\non validation\n\nlearned gate (v2)\n"
        "iterative EM (v1)\n+ 5 other rules", "#fde2e2")
    arrow(60, 29.6, 60, 25.4)
    box(75, 30, 18, 29, "Gatekeeper\n\nDS conflict: defer\nconformal set\n"
        "Learn-then-Test\nquality, OOD\nmissing details\ncost matrix\nacquisition value",
        "#dcfce7")
    arrow(69.6, 16, 74.6, 36)
    outs = (("HIGH answer", 53, "#c6e0b4"), ("LOW answer\n(doctor confirms)", 45, "#fce4d6"),
            ("ACQUIRE\ndermoscopy, details", 37, "#ffe699"), ("DEFER", 29, "#f4b6b6"))
    for t, y, c in outs:
        box(99, y, 19, 6.2, t, c)
        arrow(93.6, 44.5, 98.6, y + 3.1)
    box(75, 6, 43, 16, "Explainer  ·  Librarian (similar cases)  ·  Report + verifier\n\n"
        "Care platform: patient app, doctor queue, messages,\n"
        "follow-ups, data deletion, audit log, REST API", "#f3e8ff")
    arrow(108.5, 28.6, 108.5, 22.4)
    ax.plot([45.6, 72, 72], [7.5, 7.5, 9], color="#444444", lw=0.7)
    arrow(72, 9, 74.6, 10)
    # the "ask for more evidence" loop, routed above everything
    ax.plot([118.6, 119.5, 119.5, 20, 20, 18.6], [40, 40, 63.5, 63.5, 46.5, 46.5],
            color="#1f4e79", lw=0.8, ls="--")
    arrow(18.8, 46.5, 17.6, 46.5, ls="--")
    ax.text(60, 64.3, "ask the patient for more evidence, then assess again", fontsize=6,
            style="italic", color="#1f4e79", ha="center")
    save(fig, "fig_architecture")


def main():
    print("figures ->", OUT)
    for f in (fig_architecture, fig_fusion, fig_risk_coverage, fig_backend, fig_stress, fig_weights,
              fig_calibration, fig_legacy, fig_outcomes_cost, fig_confusion, fig_shift,
              fig_gate, fig_symptoms):
        try:
            f()
        except Exception as e:  # one broken figure must not stop the others
            print(f"   {f.__name__} FAILED: {e}")


if __name__ == "__main__":
    main()
