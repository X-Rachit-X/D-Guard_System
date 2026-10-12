"""
scripts/make_figures_extra.py - Additional figures for the report, slides and paper.

Data figures (class distribution, training curves, per-class F1, measured trust table,
evidential example, skin tone) are drawn from saved outputs; concept figures (decision
flow, case states, cross-fitting, database schema) are drawn from the code's logic.

    python scripts/make_figures_extra.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import C_B3, C_DG, C_GREY, J, L, OUT, save  # noqa: E402
from src.constants import MILK_CLASS_NAMES  # noqa: E402

K = len(MILK_CLASS_NAMES)


def box(ax, x, y, w, h, text, fc, fs=6.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                fc=fc, ec="#555555", lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, label=None, ls="-", color="#444444"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.7, color=color, linestyle=ls))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.8, label, fontsize=5.6, ha="center",
                style="italic", color="#333333")


# ------------------------------------------------------------------ data figures
def fig_class_distribution():
    sp = {s: L(f"outputs/specialists/{s}.npz") for s in ("train", "val", "test")}
    if any(v is None for v in sp.values()):
        return
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    x = np.arange(K)
    bottom = np.zeros(K)
    for s, c in (("train", "#9dc3e6"), ("val", "#f4b183"), ("test", C_DG)):
        n = np.bincount(sp[s]["y"], minlength=K)
        ax.bar(x, n, bottom=bottom, color=c, label=s)
        bottom += n
    for i, t in enumerate(bottom):
        ax.text(i, t + 25, int(t), ha="center", fontsize=6.5)
    ax.set_xticks(x, MILK_CLASS_NAMES, rotation=30, fontsize=7)
    ax.set_ylabel("lesions")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("MILK10k: 5,240 lesions, lesion-level split 70 / 15 / 15", fontsize=8)
    save(fig, "fig_class_distribution")


def fig_training_curves():
    h = J("runs/milk10k_resnet50_seed42/history.json")
    if not h:
        return
    ep = np.arange(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    ax.plot(ep, h["train_loss"], color=C_GREY, label="training loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax2 = ax.twinx()
    ax2.plot(ep, h["val_macro_f1"], color=C_DG, label="validation macro-F1")
    ax2.plot(ep, h["val_acc"], color=C_B3, label="validation accuracy")
    ax2.set_ylabel("validation score")
    best = int(np.argmax(h["val_macro_f1"]))
    ax2.axvline(best + 1, ls=":", color="k", lw=0.8)
    ax2.text(best + 1.3, 0.05 + min(h["val_macro_f1"]), f"best epoch {best + 1}", fontsize=6.5)
    ax.axvline(3.5, ls="--", color="#999999", lw=0.6)
    ax.text(3.7, max(h["train_loss"]) * 0.95, "backbone\nunfrozen", fontsize=6)
    lines = ax.get_lines()[:1] + ax2.get_lines()[:2]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=6.5, frameon=False, loc="center right")
    save(fig, "fig_training_curves")


def fig_per_class():
    from sklearn.metrics import f1_score
    ds, tm = L("outputs/milk_native/test.npz"), L("outputs/milk_team/test.npz")
    if ds is None or tm is None:
        return
    y = tm["y"]
    f_ds = f1_score(y, ds["full_prob"].argmax(1), labels=list(range(K)), average=None, zero_division=0)
    f_tm = f1_score(y, tm["full_prob"].argmax(1), labels=list(range(K)), average=None, zero_division=0)
    n = np.bincount(y, minlength=K)
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    x = np.arange(K)
    ax.bar(x - 0.2, f_ds, 0.4, color=C_GREY, label="evidential network (Dempster-Shafer)")
    ax.bar(x + 0.2, f_tm, 0.4, color=C_DG, label="D-GUARD team (learned gate)")
    ax.set_xticks(x, [f"{c}\n(n={k})" for c, k in zip(MILK_CLASS_NAMES, n)], fontsize=6.5)
    ax.set_ylabel("F1 (test, with dermoscopy)")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "fig_per_class_f1")


def fig_trust_table():
    import joblib
    p = ROOT / "outputs/milk_team/team_bundle.joblib"
    if not p.exists():
        return
    b = joblib.load(p)
    W = b.ctx_B.prior.W_prior
    names = {"clinical": "photo", "dermoscopy": "dermoscopy", "metadata": "details",
             "pad_ufes_A": "old Model A", "knn_cases": "similar cases", "embed_clf": "embedding clf."}
    cols = [names.get(n, n) for n in b.ctx_B.names]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    im = ax.imshow(W, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=35, fontsize=7, ha="right")
    ax.set_yticks(range(K), MILK_CLASS_NAMES, fontsize=7)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center", fontsize=5.8,
                    color="white" if W[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, label="P(specialist right | true class)")
    ax.set_title("Measured trust table (validation)", fontsize=8)
    save(fig, "fig_trust_table")


def fig_evidential_example():
    d = L("outputs/milk_native/test.npz")
    if d is None:
        return
    u = d["mod_image_u"].ravel()
    b = d["mod_image_belief"]
    conf_i = int(np.argmin(u))
    ign_i = int(np.argmax(u))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3), sharey=True)
    for ax, i, title in ((axes[0], conf_i, "photo head, most evidence"),
                         (axes[1], ign_i, "photo head, least evidence")):
        vals = list(b[i]) + [u[i]]
        cols = [C_DG] * K + [C_B3]
        ax.bar(range(K + 1), vals, color=cols)
        ax.set_xticks(range(K + 1), MILK_CLASS_NAMES + ["u"], rotation=60, fontsize=6)
        ax.set_title(f"{title} (u = {u[i]:.2f})", fontsize=7.5)
    axes[0].set_ylabel("belief / ignorance")
    fig.text(0.5, -0.08, "Real test lesions: belief per class (blue) plus ignorance u (red) always sum to 1. "
             "Little evidence means high u, not a random guess.", ha="center", fontsize=6.5)
    save(fig, "fig_evidential_example")


def fig_skin_tone():
    a = J("outputs/analysis/analysis.json")
    if not a:
        return
    st = a["skin_tone"]
    tones = sorted(st)
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    x = np.arange(len(tones))
    ax.bar(x - 0.2, [st[t]["high_cov"] * 100 for t in tones], 0.4, color=C_GREY, label="HIGH share")
    ax.bar(x + 0.2, [(st[t]["high_miss"] or 0) * 100 for t in tones], 0.4, color=C_B3,
           label="missed cancers among HIGH")
    ax.set_xticks(x, [f"tone {t}\n(n={st[t]['n']})" for t in tones], fontsize=6.5)
    ax.set_ylabel("%")
    ax.legend(fontsize=6.5, frameon=False)
    save(fig, "fig_skin_tone")


# ------------------------------------------------------------------ concept figures
def fig_decision_flow():
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    box(ax, 38, 54, 24, 6, "Fused distribution p,\nignorance u, conflict C", "#eef2ff")
    box(ax, 38, 43, 24, 6, "Conflict above\nvalidation p90?", "#fde2e2")
    box(ax, 72, 43, 18, 6, "DEFER\nto a clinician", "#f4b6b6", bold=True)
    box(ax, 38, 32, 24, 6, "Details missing and\nno dermoscopy?", "#fde2e2")
    box(ax, 72, 32, 18, 6, "ACQUIRE details", "#ffe699", bold=True)
    box(ax, 26, 17, 48, 11, "Any gate fires?\nconformal set still contains 'malignant'\n"
        "1-u below the Learn-then-Test threshold  ·  severe photo issue\n"
        "out-of-distribution  ·  answering costs more than a review", "#fde2e2", fs=5.8)
    box(ax, 4, 18, 18, 9, "HIGH answer\n(certified tier)", "#c6e0b4", bold=True)
    box(ax, 38, 4, 24, 7, "Dermoscopy available and\ncost(ask) < cost(review)?", "#fde2e2")
    box(ax, 72, 4, 18, 7, "ACQUIRE\ndermoscopy", "#ffe699", bold=True)
    box(ax, 4, 4, 18, 7, "LOW answer\n(doctor confirms)", "#fce4d6", bold=True)
    arrow(ax, 50, 54, 50, 49.2)
    arrow(ax, 62, 46, 71.6, 46, "yes")
    arrow(ax, 50, 43, 50, 38.2, "no")
    arrow(ax, 62, 35, 71.6, 35, "yes")
    arrow(ax, 50, 32, 50, 27.2, "no")
    arrow(ax, 26, 22.5, 22.4, 22.5, "no")
    arrow(ax, 50, 17, 50, 11.2, "yes")
    arrow(ax, 62, 7.5, 71.6, 7.5, "yes")
    arrow(ax, 38, 7.5, 22.4, 7.5, "no")
    ax.plot([90.4, 96, 96, 62.4], [7.5, 7.5, 57, 57], color=C_DG, lw=0.8, ls="--")
    arrow(ax, 63, 57, 62.4, 57, ls="--", color=C_DG)
    ax.text(97, 32, "reassess with\nthe new evidence", fontsize=6, rotation=90, va="center",
            color=C_DG, style="italic")
    save(fig, "fig_decision_flow")


def fig_case_states():
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")
    st = {"OPEN": (2, 18, "#eef2ff"), "AWAITING_PATIENT": (24, 32, "#ffe699"),
          "ASSESSED": (24, 4, "#c6e0b4"), "AWAITING_DOCTOR": (52, 18, "#fce4d6"),
          "FOLLOW_UP": (78, 32, "#dcfce7"), "CLOSED": (78, 4, "#d9d9d9")}
    for k, (x, y, c) in st.items():
        box(ax, x, y, 18, 7, k.replace("_", " "), c, fs=6.5, bold=True)
    arrow(ax, 20, 23, 23.6, 34, "ACQUIRE")
    arrow(ax, 20, 20, 23.6, 8, "HIGH")
    arrow(ax, 20, 21.5, 51.6, 21.5)
    ax.text(44, 19.2, "LOW / DEFER", fontsize=5.6, style="italic", ha="center", color="#333333")
    arrow(ax, 30, 31.6, 30, 11.4)
    ax.text(29, 25.5, "evidence sent,\nreassessed", fontsize=5.6, style="italic", ha="right",
            color="#333333")
    arrow(ax, 42, 8, 51.6, 19, "follow-up shows change")
    arrow(ax, 70, 23, 77.6, 34, "doctor: re-check")
    arrow(ax, 70, 20, 77.6, 8, "doctor: verdict")
    arrow(ax, 87, 31.6, 87, 11.4, "re-check done")
    save(fig, "fig_case_states")


def fig_cross_fitting():
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    for k in range(5):
        y = 32 - k * 6.5
        ax.text(1, y + 2, f"fold {k + 1}", fontsize=6.5, va="center")
        for j in range(5):
            c = "#f4b183" if j == k else "#9dc3e6"
            ax.add_patch(FancyBboxPatch((9 + j * 8, y), 7.4, 4.2, boxstyle="square,pad=0", fc=c,
                                        ec="white"))
    ax.text(29, 38, "786 validation lesions", ha="center", fontsize=7)
    box(ax, 55, 26, 18, 9, "fit every fusion rule\non the blue 4/5", "#9dc3e6", fs=6.3)
    box(ax, 55, 13, 18, 9, "predict the\norange 1/5\n(out-of-fold)", "#f4b183", fs=6.3)
    box(ax, 79, 26, 20, 9, "choose the rule:\nbest out-of-fold\nmacro-F1", "#e2efda", fs=6.2)
    box(ax, 79, 13, 20, 9, "calibrate conformal sets,\nLearn-then-Test on\nout-of-fold probabilities", "#e2efda", fs=6)
    box(ax, 55, 1, 44, 8, "test set (787 lesions): rules refitted on all of\nvalidation; scored once",
        "#d9d9d9", fs=6.2)
    arrow(ax, 50, 30, 54.6, 30)
    arrow(ax, 64, 25.6, 64, 22.4)
    arrow(ax, 73.4, 17.5, 78.6, 30)
    arrow(ax, 73.4, 17.5, 78.6, 17.5)
    save(fig, "fig_cross_fitting")


def fig_db_schema():
    tables = [("patients", ["id", "name", "age", "sex", "skin_tone"]),
              ("cases", ["id", "patient_id", "site", "status", "urgency"]),
              ("evidence", ["id", "case_id", "kind", "path", "payload"]),
              ("assessments", ["id", "case_id", "stage", "action", "tentative", "p_malignant", "result", "policy"]),
              ("reviews", ["id", "case_id", "doctor", "verdict", "diagnosis"]),
              ("messages", ["id", "case_id", "sender", "kind", "body"]),
              ("followups", ["id", "case_id", "due_date", "status"]),
              ("audit", ["id", "case_id", "actor", "event", "detail"])]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    pos = {"patients": (2, 30), "cases": (22, 30), "evidence": (44, 33), "assessments": (64, 25),
           "reviews": (84, 33), "messages": (44, 5), "followups": (64, 5), "audit": (84, 5)}
    for name, cols in tables:
        x, y = pos[name]
        h = 3 + 2.4 * len(cols)
        ax.add_patch(FancyBboxPatch((x, y), 15, h, boxstyle="round,pad=0.2,rounding_size=0.6",
                                    fc="#f8fbff", ec="#555555", lw=0.7))
        ax.add_patch(FancyBboxPatch((x, y + h - 3), 15, 3, boxstyle="square,pad=0", fc="#1f4e79", ec="#1f4e79"))
        ax.text(x + 7.5, y + h - 1.5, name, color="white", fontsize=6.5, ha="center", va="center",
                weight="bold")
        for i, c in enumerate(cols):
            ax.text(x + 1, y + h - 4.6 - 2.4 * i, c, fontsize=5.6, va="center",
                    weight="bold" if c == "id" else "normal")
    arrow(ax, 17.4, 40, 21.6, 40, "1 : n")
    heights = {n: 3 + 2.4 * len(c) for n, c in tables}
    for t in ("evidence", "assessments", "reviews"):
        x, y = pos[t]
        arrow(ax, 37.4, 36, x - 0.4, y + heights[t] / 2)
    for t in ("messages", "followups"):
        x, y = pos[t]
        arrow(ax, 37.4, 32, x + 7.5, y + heights[t] + 0.4)
    save(fig, "fig_db_schema")


def main():
    for f in (fig_class_distribution, fig_training_curves, fig_per_class, fig_trust_table,
              fig_evidential_example, fig_skin_tone, fig_decision_flow, fig_case_states,
              fig_cross_fitting, fig_db_schema):
        try:
            f()
        except Exception as e:
            print(f"   {f.__name__} FAILED: {e}")


if __name__ == "__main__":
    main()
