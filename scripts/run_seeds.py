"""
scripts/run_seeds.py - Repeat the WHOLE pipeline for extra training seeds and
report mean +- sd.

Each Kaggle seed run (notebooks/kaggle_train_dguard.py, run_seeds((7, 2026)))
trains a new network AND draws a new lesion-level split (the split uses the
same seed), so the spread below covers both training randomness and which
lesions land in test - repeated random splits, not just re-initialisation.

Put each downloaded run folder under runs/, e.g.
    runs/milk10k_resnet50_seed7/{checkpoints/best.pt, manifest.json, metrics.json}
    runs/milk10k_resnet50_seed2026/...
(from Kaggle: unzip dguard_runs.zip; folder dguard_milk_s7 -> rename or pass as is)

    python scripts/run_seeds.py runs/milk10k_resnet50_seed7 runs/milk10k_resnet50_seed2026

Per seed (outputs/seeds/<run name>/):
    milk_native_infer -> build_specialists -> run_team (fusion picked on val)
    -> run_eade_milk (cost 5) -> analyze_milk
Then outputs/seeds/summary.json: every headline number per seed + mean, sd,
including the main seed-42 results already in outputs/.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

HEADLINE = [
    ("answered", lambda a, t, r: a["ci"]["answered"]["point"]),
    ("high_coverage", lambda a, t, r: a["ci"]["high_cov"]["point"]),
    ("high_accuracy", lambda a, t, r: a["ci"]["high_acc"]["point"]),
    ("high_missed_cancer", lambda a, t, r: a["ci"]["high_miss"]["point"]),
    ("b3_missed_cancer", lambda a, t, r: a["ci"]["b3_miss"]["point"]),
    ("b4_missed_cancer", lambda a, t, r: a["ci"]["b4_miss"]["point"]),
    ("team_macro_f1_B", lambda a, t, r: a["ci"]["f1_team"]["point"]),
    ("ds_macro_f1_B", lambda a, t, r: a["ci"]["f1_ds"]["point"]),
    ("selected_fusion_B", lambda a, t, r: t["selected"]["B"]),
    ("selected_fusion_A", lambda a, t, r: t["selected"]["A"]),
    ("defer_rate", lambda a, t, r: r["tiers"]["defer_rate"]),
    ("acquire_rate", lambda a, t, r: r["dguard"]["acquire_rate"]),
]


def sh(*args):
    print("  $", " ".join(str(a) for a in args), flush=True)
    r = subprocess.run([PY, *map(str, args)], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"step failed: {args[0]}")


def run_one(run: Path) -> Path:
    out = ROOT / "outputs" / "seeds" / run.name
    nat, spec, team, eade = out / "milk_native", out / "specialists", out / "milk_team", out / "eade"
    print(f"\n=== {run.name} ===")
    if not (nat / "test.npz").exists():
        sh("scripts/milk_native_infer.py", "--run", run, "--out", nat)
    if not (spec / "test.npz").exists():
        sh("scripts/build_specialists.py", "--run", run, "--out", spec)
    if not (team / "test.npz").exists():
        sh("scripts/run_team.py", "--native", nat, "--spec", spec, "--out", team, "--no-subsets")
    if not (eade / "report.json").exists():
        sh("scripts/run_eade_milk.py", "--native", team, "--missed-cost", 5, "--out", eade)
    sh("scripts/analyze_milk.py", "--eade", eade.relative_to(ROOT), "--team",
       team.relative_to(ROOT), "--native", nat.relative_to(ROOT), "--eade-em", "none",
       "--out", (out / "analysis").relative_to(ROOT))
    return out


def collect(name, analysis, team_report, eade_report):
    a = json.loads(Path(analysis).read_text())
    t = json.loads(Path(team_report).read_text())
    r = json.loads(Path(eade_report).read_text())
    return {"run": name, **{k: f(a, t, r) for k, f in HEADLINE}}


def main() -> int:
    runs = [Path(p) for p in sys.argv[1:]]
    rows = [collect("seed42 (main)", ROOT / "outputs/analysis/analysis.json",
                    ROOT / "outputs/milk_team/team_report.json",
                    ROOT / "outputs/eade_team/cost_5/report.json")]
    for run in runs:
        if not (run / "checkpoints" / "best.pt").exists():
            print(f"skip {run}: no checkpoints/best.pt")
            continue
        out = run_one(run)
        rows.append(collect(run.name, out / "analysis/analysis.json",
                            out / "milk_team/team_report.json", out / "eade/report.json"))

    import numpy as np
    summary = {"runs": rows, "mean": {}, "sd": {}}
    print(f"\n{'metric':22} " + " ".join(f"{r['run'][:16]:>16}" for r in rows) + "   mean +- sd")
    for k, _ in HEADLINE:
        vals = [r[k] for r in rows]
        if all(isinstance(v, (int, float)) for v in vals):
            summary["mean"][k], summary["sd"][k] = float(np.mean(vals)), float(np.std(vals, ddof=1)) \
                if len(vals) > 1 else 0.0
            print(f"{k:22} " + " ".join(f"{v:16.3f}" for v in vals)
                  + f"   {summary['mean'][k]:.3f} +- {summary['sd'][k]:.3f}")
        else:
            print(f"{k:22} " + " ".join(f"{str(v)[:16]:>16}" for v in vals))
    dest = ROOT / "outputs" / "seeds" / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
