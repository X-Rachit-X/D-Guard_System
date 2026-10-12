"""Extract key functions from the repository into the report's code appendix."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PICKS = [
    ("The learned evidence gate (Devil's Advocate v2): features and fitting",
     "src/fusion/gating.py", ["LearnedGate.fit", "LearnedGate.fuse"]),
    ("All fusion rules behind one interface", "src/fusion/methods.py", ["fuse_method"]),
    ("Cross-fitting and rule selection", "scripts/run_team.py", ["cv_folds", "run_methods"]),
    ("Occlusion check of the quality agent", "src/agents/quality.py", ["_centre", "flat_fraction"]),
    ("Quality and missing-details gates in the agent pipeline", "src/agents/pipeline.py",
     ["has_details", "DGuardPipeline.assess"]),
    ("Data deletion", "src/care/service.py", ["CareService._remove_uploads",
                                              "CareService.delete_patient"]),
]


def get_source(path, dotted):
    src = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()
    parts = dotted.split(".")
    nodes = tree.body
    node = None
    for part in parts:
        node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
        nodes = node.body
    start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
    return "\n".join(lines[start:node.end_lineno])


out = ["# *Appendix I: Selected Source Code", "",
       "This appendix reproduces the most important functions of the system exactly as they "
       "appear in the repository. The complete source code (about 14,600 lines of Python), "
       "the tests and the experiment scripts are in the project repository.", ""]
for i, (title, path, names) in enumerate(PICKS, 1):
    out += [f"**A.{i}  {title}** (file: `{path}`)", ""]
    for n in names:
        out += ["```", *get_source(path, n).splitlines(), "```", ""]
(ROOT / "reports/source/12_appendix_code.md").write_text("\n".join(out), encoding="utf-8")
print("written", sum(len(get_source(p, n).splitlines()) for _, p, ns in PICKS for n in ns), "lines")
