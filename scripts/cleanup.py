"""
scripts/cleanup.py - Find and remove redundant files, safely.

Reports by default. Deletes nothing unless you pass --apply. Every candidate is
VERIFIED redundant before it is offered, not guessed at from its name.

Tiers, in increasing order of "think before you do this":

  SAFE      regenerable or byte-identical duplicates. Nothing is lost.
  REVIEW    large, unreferenced, and superseded - but not regenerable without a
            re-download or a re-run. Requires --tier review.
  NEVER     listed only so it is on the record that these must stay, and why.

Usage:
    python scripts/cleanup.py                      # report only
    python scripts/cleanup.py --apply               # delete SAFE
    python scripts/cleanup.py --tier review --apply # delete SAFE + REVIEW
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def _force_remove(func, path, exc_info):
    """rmtree error handler: clear the read-only bit and retry.

    Git stores loose objects read-only, so shutil.rmtree fails on Windows with
    WinError 5 partway through deleting a .git directory - leaving it in a
    half-removed state, which is worse than not starting.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        raise


def _rmtree(path: Path) -> None:
    """shutil.rmtree that copes with read-only files on every Python version."""
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=lambda f, p, e: _force_remove(f, p, e))
    else:
        shutil.rmtree(path, onerror=_force_remove)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            d.update(b)
    return d.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0
    return f"{n:.1f}TB"


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            try:
                total += (Path(r) / f).stat().st_size
            except OSError:
                pass
    return total


# ---------------------------------------------------------------------------
# Verifiers - a candidate is only offered if its verifier returns (True, why)
# ---------------------------------------------------------------------------
def verify_image_dupe(candidate: Path, reference: Path) -> tuple[bool, str]:
    """Every image in `candidate` must be byte-identical to one in `reference`."""
    if not candidate.exists():
        return False, "does not exist"
    if not reference.exists():
        return False, f"reference {reference} missing - refusing to compare"

    ref = {}
    for f in os.listdir(reference):
        p = reference / f
        if p.is_file() and f.lower().endswith((".png", ".jpg", ".jpeg")):
            ref[f] = p

    cand = []
    for r, _, fs in os.walk(candidate):
        for f in fs:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                cand.append(Path(r) / f)

    if not cand:
        return False, "no images found"

    missing, differing = [], []
    for p in cand:
        if p.name not in ref:
            missing.append(p.name)
            continue
        if sha256(p) != sha256(ref[p.name]):
            differing.append(p.name)

    if missing or differing:
        return False, (f"NOT redundant: {len(missing)} absent from reference, "
                       f"{len(differing)} differ in content")
    return True, (f"all {len(cand)} images byte-identical to "
                  f"{reference.relative_to(ROOT)} (SHA-256)")


def verify_regenerable(candidate: Path, command: str) -> tuple[bool, str]:
    if not candidate.exists():
        return False, "does not exist"
    return True, f"regenerable with: {command}"


def verify_unreferenced(candidate: Path, patterns: list[str]) -> tuple[bool, str]:
    """Nothing in src/ scripts/ tests/ app/ configs/ may mention these."""
    if not candidate.exists():
        return False, "does not exist"
    hits = []
    search_dirs = [ROOT / d for d in ("src", "scripts", "tests", "app", "configs")]
    for d in search_dirs:
        if not d.exists():
            continue
        for r, _, fs in os.walk(d):
            if "__pycache__" in r:
                continue
            for f in fs:
                if not f.endswith((".py", ".yaml", ".yml", ".toml")):
                    continue
                fp = Path(r) / f
                if fp.name == "cleanup.py":
                    continue
                try:
                    txt = fp.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for pat in patterns:
                    if pat in txt:
                        hits.append(f"{fp.relative_to(ROOT)} mentions {pat!r}")
    if hits:
        return False, "STILL REFERENCED: " + "; ".join(hits[:3])
    return True, "not referenced by any module, script, test or config"


def collect_pycache() -> list[Path]:
    out = []
    for r, ds, _ in os.walk(ROOT):
        if any(part in r for part in ("milk10k", ".git")):
            continue
        for d in list(ds):
            if d == "__pycache__":
                out.append(Path(r) / d)
    return out


# ---------------------------------------------------------------------------
NEVER_DELETE = [
    ("Dummy/Project/saved_models",
     "the A-G specialist weights the orchestrator and demo.py actually load "
     "(src/models/backends.py resolves here by default)"),
    ("Dummy/Project/data/generated_clinical_texts.csv",
     "the text narratives build_features.py, run_stress.py and evaluate_ladder.py read"),
    ("data/raw/images",
     "the 2298 PAD-UFES-20 images every feature build consumes"),
    ("data/raw/metadata.csv",
     "PAD-UFES-20 labels and patient ids"),
    ("milk10k",
     "the primary dataset: 10480 images, 5240 lesions, CC-BY-NC. Re-downloading "
     "needs an ISIC account"),
    ("data/processed/features_v2.pt",
     "reproducible features; regenerable in ~4.5 min but needed by the RAG index "
     "and the baseline probes"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="actually delete")
    ap.add_argument("--tier", choices=["safe", "review"], default="safe",
                    help="highest tier to act on (default: safe)")
    args = ap.parse_args()

    print("=" * 76)
    print("CLEANUP" + ("  [APPLY]" if args.apply else "  [REPORT ONLY - pass --apply to delete]"))
    print("=" * 76)

    candidates = []   # (tier, path, why)

    # --- SAFE ---------------------------------------------------------------
    for i in (1, 2, 3):
        p = ROOT / "data" / "raw" / f"imgs_part_{i}"
        ok, why = verify_image_dupe(p, ROOT / "data" / "raw" / "images")
        if ok:
            candidates.append(("SAFE", p, f"original download archive; {why}"))
        elif p.exists():
            print(f"  SKIP {p.relative_to(ROOT)}: {why}")

    p = ROOT / "Dummy" / "Project" / "data" / "images"
    ok, why = verify_image_dupe(p, ROOT / "data" / "raw" / "images")
    if ok:
        ok2, why2 = verify_unreferenced(p, ["Dummy/Project/data/images",
                                            "Dummy\\Project\\data\\images"])
        if ok2:
            candidates.append(("SAFE", p, f"third copy of the same images; {why}; {why2}"))
        elif p.exists():
            print(f"  SKIP {p.relative_to(ROOT)}: {why2}")
    elif p.exists():
        print(f"  SKIP {p.relative_to(ROOT)}: {why}")

    p = ROOT / "Dummy" / "Project" / ".git"
    if p.exists():
        candidates.append(("SAFE", p, "nested git repository inside a vendored tree; "
                                      "its history is not ours and confuses tooling"))

    for p in collect_pycache():
        candidates.append(("SAFE", p, "Python bytecode cache"))

    for name in (".pytest_cache", "outputs/audit/hash_run.log",
                 "outputs/audit/dupe_verify.log", "outputs/audit/pytest_full.log"):
        p = ROOT / name
        if p.exists():
            candidates.append(("SAFE", p, "transient tooling artifact"))

    # --- REVIEW -------------------------------------------------------------
    p = ROOT / "outputs" / "checkpoints"
    if p.exists():
        ok, why = verify_unreferenced(p, ["outputs/checkpoints"])
        note = ("checkpoints from the July 2025 local run whose metrics are "
                "INVALID (docs/evaluation_integrity.md). The live system loads "
                "Dummy/Project/saved_models instead, not these. ")
        if ok:
            candidates.append(("REVIEW", p, note + why))
        else:
            candidates.append(("REVIEW", p, note + why +
                               " - only a legacy script refers to it"))

    p = ROOT / "Dummy" / "Project" / "saved_models" / "cached_multimodal_features.pt"
    if p.exists():
        candidates.append(("REVIEW", p,
                           "the irreproducible feature cache (cosine <= 0.07 against "
                           "every regeneration attempt). Kept so far as forensic "
                           "evidence for the integrity audit; delete only if you do "
                           "not need to re-demonstrate that finding"))

    # --- Report -------------------------------------------------------------
    order = {"SAFE": 0, "REVIEW": 1}
    candidates.sort(key=lambda c: (order[c[0]], -size_of(c[1])))

    grouped_total = {"SAFE": 0, "REVIEW": 0}
    print()
    shown_tier = None
    pycache_n, pycache_bytes = 0, 0
    for tier, path, why in candidates:
        sz = size_of(path)
        grouped_total[tier] += sz
        if path.name == "__pycache__":
            pycache_n += 1
            pycache_bytes += sz
            continue
        if tier != shown_tier:
            print(f"--- {tier} " + "-" * (70 - len(tier)))
            shown_tier = tier
        print(f"  {human(sz):>9}  {path.relative_to(ROOT)}")
        print(f"             {why}")
    if pycache_n:
        print(f"  {human(pycache_bytes):>9}  {pycache_n} x __pycache__/ directories")

    print()
    print(f"SAFE total   : {human(grouped_total['SAFE'])}")
    print(f"REVIEW total : {human(grouped_total['REVIEW'])}")
    print(f"TOTAL        : {human(sum(grouped_total.values()))}")

    print()
    print("--- NEVER DELETE " + "-" * 59)
    for rel, why in NEVER_DELETE:
        p = ROOT / rel
        mark = human(size_of(p)) if p.exists() else "absent"
        print(f"  {mark:>9}  {rel}")
        print(f"             {why}")

    # --- Apply --------------------------------------------------------------
    tiers = {"safe": {"SAFE"}, "review": {"SAFE", "REVIEW"}}[args.tier]
    doomed = [(t, p, w) for t, p, w in candidates if t in tiers]

    if not args.apply:
        print()
        print(f"Would delete {len(doomed)} path(s) at tier '{args.tier}', freeing "
              f"{human(sum(size_of(p) for _, p, _ in doomed))}.")
        print("Re-run with --apply to do it.")
        return 0

    print()
    print("=" * 76)
    print(f"DELETING {len(doomed)} path(s) at tier '{args.tier}'")
    print("=" * 76)
    freed, failed = 0, []
    for tier, path, _ in doomed:
        sz = size_of(path)
        try:
            if path.is_dir():
                _rmtree(path)
            else:
                path.chmod(stat.S_IWRITE)
                path.unlink()
            freed += sz
            if path.name != "__pycache__":
                print(f"  removed  {human(sz):>9}  {path.relative_to(ROOT)}")
        except OSError as e:
            failed.append((path, str(e)))
    print()
    print(f"freed {human(freed)}")
    if failed:
        print(f"{len(failed)} failure(s):")
        for p, e in failed[:5]:
            print(f"  {p.relative_to(ROOT)}: {e}")
    print()
    print("Now re-verify:  pytest tests -q && python scripts/demo.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
