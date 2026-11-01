"""Overestimation comparison: the job worker, one job per process (OVERESTIMATION_RULE.md). User-run in the recorded
WSL environment; agents never run this file and never open its checkpoints.

It is the native screen's worker (`../native_screen/screen_worker.py`: Stage 1's `warmup` and `branch`, the dose
screen's `bca_branch` at kappa 63, the seven declared bindings including the authenticated dataset loader) with one
job kind added, `oe_evaluate` (oe_eval.py). Integrity: this worker verifies its own job's record (this folder as the
package folder) before any package import, refuses any importable file added to this folder or the native screen's,
and hides --job from the native screen worker's own check (whose package folder is the native screen's).
"""
from pathlib import Path
import argparse
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CAND = HERE.parent
NS = CAND / "native_screen"
V3 = CAND.parent / "knee_edge_ood_v3"
F2 = CAND / "formal_g2"
for _p in (str(V3), str(F2), str(NS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import integrity                                             # noqa: E402  (v3 integrity.py, standard library only)
integrity.verify(sys.argv, __file__)

if "--job" in sys.argv:
    _job = json.loads(Path(sys.argv[sys.argv.index("--job") + 1]).read_text(encoding="utf-8"))
    _added = [p for p in integrity.changed_files({**_job["integrity"], "package_dir": str(NS)}) if p.startswith("added: ")]
    if _added:
        raise SystemExit("importable files added to the native screen folder: " + ", ".join(_added))

_argv, sys.argv = sys.argv, [a for a in sys.argv if a != "--job"]
try:
    import screen_worker as SW                               # noqa: E402  applies the seven bindings; no --job seen
finally:
    sys.argv = _argv
import oe_eval                                               # noqa: E402

SW.KINDS["oe_evaluate"] = oe_eval.oe_evaluate
run_job = SW.run_job

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True)
    result = run_job(parser.parse_args().job)
    sys.exit(0 if result.get("status") == "completed" else 1)
