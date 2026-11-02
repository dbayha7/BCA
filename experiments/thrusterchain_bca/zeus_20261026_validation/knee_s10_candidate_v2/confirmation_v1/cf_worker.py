"""Confirmation: the job worker, one job per process (CONFIRMATION_RULE.md). User-run in the recorded WSL
environment; agents never run this file and never open its checkpoints.

It is the native screen's worker (Stage 1's `warmup` and `branch`, the dose screen's `bca_branch`, the seven
declared bindings) with three in-process changes of this package: (1) research training seeds are the confirmation
rows (`cf_seeds.json`), never the screen's; (2) the BCA dose is the job's `kappa` (21, 63 or 189), applied to the
runtime before the branch and checked by the branch and the policy restore; (3) the job kind `cf_evaluate`, the
exploratory package's `measure` with the confirmation's visited-state and probe-row seeds. Integrity: this worker
verifies its own job's record (this folder as the package folder) before any package import, refuses importable
files added to this folder, the exploratory package's or the native screen's, and hides --job from the native
screen worker's own check.

Change 4 (amendment A2): the recorded device kind is the one allowed kind this machine has (`device_kinds.py`: Zeus's
laptop GPU or Titan's desktop GPU), bound right after the worker environment is set, before any training bind or
policy restore reads it; every report keeps the runtime's device kinds."""
from pathlib import Path
import argparse
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CAND = HERE.parent
NS, OE = CAND / "native_screen", CAND / "overestimation_v1"
V3 = CAND.parent / "knee_edge_ood_v3"
F2 = CAND / "formal_g2"
for _p in (str(V3), str(F2), str(NS), str(OE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import integrity                                             # noqa: E402  (v3 integrity.py, standard library only)
integrity.verify(sys.argv, __file__)

if "--job" in sys.argv:
    _job = json.loads(Path(sys.argv[sys.argv.index("--job") + 1]).read_text(encoding="utf-8"))
    for _folder in (NS, OE):
        _added = [p for p in integrity.changed_files({**_job["integrity"], "package_dir": str(_folder)}) if p.startswith("added: ")]
        if _added:
            raise SystemExit(f"importable files added to {_folder.name}: " + ", ".join(_added))

_argv, sys.argv = sys.argv, [a for a in sys.argv if a != "--job"]
try:
    import screen_worker as SW                               # noqa: E402  applies the seven bindings; no --job seen
finally:
    sys.argv = _argv
import numpy as np                                           # noqa: E402
import oe_eval as OE_EVAL                                    # noqa: E402
import cf_seeds as CS                                        # noqa: E402
import device_kinds as DK                                    # noqa: E402
from core import require                                     # noqa: E402

DOSES = (21.0, 63.0, 189.0)
SB, W1, E = SW.SB, SW.W1, SW.E


def bind_device_kind():
    """Change 4 (amendment A2): the recorded device kind is the one allowed kind this machine has."""
    import jax
    kinds = sorted({d.device_kind for d in jax.devices()})
    require(len(kinds) == 1 and kinds[0] in DK.ALLOWED_DEVICE_KINDS,
            f"Device kinds {kinds} are not exactly one allowed kind {sorted(DK.ALLOWED_DEVICE_KINDS)}")
    SW.step03.K.RECORDED_DEVICE_KIND = kinds[0]
    return kinds[0]


_set_worker_environment = SW.step03.set_worker_environment


def set_worker_environment():
    """The recorded worker environment (unchanged), then change 4's device-kind binding."""
    _set_worker_environment()
    bind_device_kind()


SW.step03.set_worker_environment = set_worker_environment


def protocol_view(job):
    """Change 1. Research jobs: the confirmation's three seed rows; smoke jobs as Stage 1."""
    if job.get("provenance", "research") == "research":
        require(job.get("training_seeds") is None, "Research jobs use the confirmation's sealed training seeds")
        return {"seeds": CS.training_rows()}
    return SB.protocol_view(job)


W1.protocol_view = protocol_view


def apply_kappa(job):
    """Change 2. The BCA dose of this process is the job's kappa; the branch and the restore check it."""
    kappa = float(job.get("kappa", 63.0))
    require(kappa in DOSES, f"kappa {kappa} is not one of the doses {DOSES}")
    SW.KAPPA = kappa

    def rerun1():
        runtime, evaluate_worker, gradient_audit, rerun_inputs = SW.step03.rerun1()
        return SB.patch_runtime(runtime, kappa), evaluate_worker, gradient_audit, rerun_inputs
    SW.rerun1 = rerun1
    return kappa


def cf_evaluate(job, out, deadline):
    """Change 3. The exploratory measurement with the confirmation's seeds, at the job's world and arm."""
    world = job.get("world")
    require(job.get("arm") in ("native", "bca") and world in OE_EVAL.WORLDS, "cf_evaluate: arm native|bca, world stall3|nostall")
    runtime, evaluate_worker, _, _ = SW.rerun1()
    budget = W1.Budget(job["transitions"], job["timeout_seconds"])
    seeds = CS.load()["numpy"]
    dataset, plan = W1.load_world(job, world)
    dirs = {"warmup": job["runs"]["warmup"], "branch": job["runs"]["branch"]}
    restore = W1.policies if job["arm"] == "native" else SW.policies_bca
    _, pols, assertions, adapter = restore(job, runtime, evaluate_worker, dataset, plan, OE_EVAL.DATA_ARM, dirs, (OE_EVAL.CHECKPOINT,))
    policy = pols[OE_EVAL.CHECKPOINT]
    deadline()
    reads = OE_EVAL.measure(policy, dataset.arrays, E.bank_states(W1.E2_BANK), seeds, budget, deadline, world)
    policy.assert_unchanged()
    return {"kind": "cf_evaluate", "package": "knee_s10_candidate_v2/confirmation_v1", "arm": job["arm"], "kappa": SW.KAPPA,
            "seed_index": job["seed_index"], "checkpoint": OE_EVAL.CHECKPOINT, "world": world, "assertions": assertions,
            "dataset_sha256": dataset.digest, "seeds": dict(seeds), "training_seeds_sha256": SW.sha_file(CS.SEEDS_PATH), **reads}


SW.KINDS["cf_evaluate"] = cf_evaluate


def run_job(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    apply_kappa(job)
    return SW.run_job(job_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True)
    result = run_job(parser.parse_args().job)
    sys.exit(0 if result.get("status") == "completed" else 1)
