"""Train a GAPPED TD3+BC host with the repo's own train.py on the action-gap copy of the data (design: the approved
action-gap design; experiments/truth/README.md, 'Action gap').

    $PY -m experiments.truth.gap_train --task hopper --gap W/gap/gap.json --gap-sha256 HEX --seed S \
        --output runs/td3-hopper-gap15-sS [--data-dir D] [--device cuda] [--prepare-only]

No existing file changes. The wrapper:
  1. verifies the gap manifest (its sha256 equals --gap-sha256 and the frozen sidecar written by `gap freeze`), the
     gapped file's sha256 (in --data-dir, default: where `gap define` wrote it) and the pinned sha256 of
     runtime/td3_bc.py, train.py and experiments/truth/gap.py; any difference refuses;
  2. patches train.resolve: the native resolve(configs/td3_bc.yaml, 'host', S, out, task) runs unchanged, then the
     row's cache pin becomes the gapped copy, run_id gets '-gap15-<8 hex of the gap sha>' and row['action_gap']
     records the manifest path and sha256, c*, rho, q, the gapped file's sha256, the projection rule and this code's
     sha256 (gap.gapped_row). train.py writes the row into resolved.json before training; typed() ignores the key;
  3. patches runtime.td3_bc._reserve with the projected reservation (gap.projection): gapped training = native
     training minus the gap, likewise withheld and bank, matched by raw row; prepare and train.py's own
     validate_prepared both run it;
  4. calls train.main([...]) unchanged: the same GPU lock, hyperparameters (configs/td3_bc.yaml host block),
     checkpoints at 10k/50k/1M, evaluation banks (evaluation_seed_k of the seed), events journal, source snapshot
     and result.json. Then it writes OUT/action_gap.json.
Full-data controls are plain train.py runs (README). If LD_LIBRARY_PATH lacks MuJoCo, train.py re-executes the
process with the same arguments, which re-runs this wrapper; exporting it beforehand avoids the re-exec.

There is no BCA path: --method bca refuses with the wording of R5-signal's --allow-bca-band refusal, and no BCA flag
is exposed. --synthetic-test is for tests on synthetic files only (it refuses real data).
"""

import argparse
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.truth import gap as GAP  # noqa: E402
from experiments.truth.state_restore import file_sha256, require, task_spec  # noqa: E402

SCHEMA = "bca-truth-gap-train-v1"


class BandFitDisabled(RuntimeError):
    """The same refusal as acceptance.BandFitDisabled (R5-signal's --allow-bca-band)."""


def refuse_bca():
    raise BandFitDisabled("gap_train trains TD3+BC hosts only: a BCA arm or band fit is off unless --allow-bca-band "
                          "is given (and approved), and gap_train exposes no such flag")


def verify(manifest_path, gap_sha256, task, data_dir=None):
    """(manifest, gapped file path) after every pin check."""
    m = GAP.read_manifest(manifest_path, gap_sha256, require_frozen=True)
    require(m.data["defined"], "the gap manifest defines no gap (G0 failed)")
    require(m.data["task"] == task_spec(task).name, f"the gap manifest is for {m.data['task']}, not {task}")
    current = GAP.code_pins()
    changed = sorted(k for k, v in m.data["code_sha256"].items() if current.get(k) != v)
    require(not changed, f"pinned files changed since the gap was defined: {changed}; refusing (the projected "
            "reservation patches runtime/td3_bc.py's private _reserve)")
    freeze = json.loads((m.dir / "freeze.json").read_text(encoding="utf8"))
    require(freeze["gap_sha256"] == m.sha256, "freeze.json belongs to another gap.json")
    return m, m.gapped_path(data_dir), freeze


@contextlib.contextmanager
def patched(manifest):
    """train.resolve -> gapped row; runtime.td3_bc._reserve -> projected reservation."""
    import train as T

    native_resolve = T.resolve

    def resolve(path, method, seed, output_dir, dataset=None):
        if method != "host":
            refuse_bca()
        return GAP.gapped_row(native_resolve(path, method, seed, output_dir, dataset), manifest)

    with mock.patch.object(T, "resolve", resolve), GAP.projection(manifest):
        yield


def run(task, gap, gap_sha256, seed, output, *, data_dir=None, device="cuda", prepare_only=False, lock=None,
        method="host", synthetic_test=False, allow_reexec=False):
    if method != "host":
        refuse_bca()
    import train as T

    mujoco_bin = str(Path(os.environ.get("MUJOCO_PY_MUJOCO_PATH", str(Path.home() / ".mujoco/mujoco210"))) / "bin")
    require(allow_reexec or mujoco_bin in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep),
            f"export LD_LIBRARY_PATH with {mujoco_bin} first: train.py would re-execute this process")

    m, gapped, freeze = verify(gap, gap_sha256, task, data_dir)
    require(not freeze["synthetic_g0_override"] or m.data["synthetic"], "G0 override on real data")
    if synthetic_test:
        require(m.data["synthetic"], "--synthetic-test refuses real data")
    else:
        require(not m.data["synthetic"], "a synthetic gap needs --synthetic-test (tests only)")
    output = Path(output).expanduser().resolve()
    argv = ["--algorithm", "td3_bc", "--dataset", task_spec(task).name, "--method", "host", "--seed", str(int(seed)),
            "--output", str(output), "--data-dir", str(gapped.parent), "--device", device]
    if prepare_only:
        argv.append("--prepare-only")
    if lock is not None:
        argv += ["--lock", str(lock)]
    with patched(m):
        T.main(argv)
    resolved = json.loads((output / "resolved.json").read_text(encoding="utf8"))
    preparation = json.loads((output / "preparation.json").read_text(encoding="utf8"))
    reservation = preparation["metadata"]["reservation"]
    record = dict(schema=SCHEMA, run_dir=str(output), run_id=resolved["run_id"], seed=int(seed), task=m.data["task"],
                  gap_manifest=str(m.path.resolve()), gap_sha256=m.sha256,
                  definition_sha256=m.data["definition_sha256"], centre=m.data["gap"]["centre"],
                  rho=m.data["gap"]["rho"], gapped_file=str(gapped), gapped_sha256=m.data["gapped"]["sha256"],
                  projected_counts=reservation["counts"], prepare_only=bool(prepare_only),
                  gap_train_sha256=file_sha256(Path(__file__)), synthetic=m.data["synthetic"],
                  utc=datetime.now(timezone.utc).isoformat(),
                  note=GAP.SYNTHETIC_NOTE if m.data["synthetic"] else "gapped TD3+BC host (action gap)")
    path = output / "action_gap.json"
    with path.open("x", encoding="utf8") as handle:
        handle.write(GAP.strict_json(record))
    return record


def attach_random_checkpoint(run_dir, step=0):
    """TEST ONLY: give a --prepare-only gapped run a randomly initialised checkpoint and result.json, as
    pools.write_synthetic_run does for full-data fixtures. Refuses non-synthetic data."""
    import jax.numpy as jnp
    from flax import serialization

    from algorithms import td3_bc_bca as P
    from experiments.truth import pools as PL
    from experiments.wbcp import freeze_scores as FS
    from runtime.provenance import write

    run_dir = Path(run_dir)
    row = json.loads((run_dir / "resolved.json").read_text(encoding="utf8"))
    record = json.loads((run_dir / "action_gap.json").read_text(encoding="utf8"))
    require(record["synthetic"] and row["action_gap"]["synthetic"], "attach_random_checkpoint refuses real data")
    context, m = GAP.host_projection(row)
    with context:
        (_, args, spec, _), prepared = FS.prepare(row, Path(record["gapped_file"]).parent)
    rng, state, _ = P.initialize(args, spec.config(), prepared.training.obs.shape[1],
                                 prepared.training.action.shape[1], prepared.max_action)
    payload = serialization.to_bytes({"state": state, "training_rng": rng, "step": jnp.int32(step)})
    (run_dir / f"checkpoint_{step}.msgpack").write_bytes(payload)
    write(run_dir / "result.json", dict(completed=False, synthetic=True, checkpoints=[
        dict(step=step, path=f"checkpoint_{step}.msgpack", sha256=PL.sha256_hex(payload))]))
    (run_dir / "SYNTHETIC_HOST.txt").write_text(PL.SYNTHETIC_HOST_NOTE + "\n", encoding="utf8")
    return dict(run_dir=str(run_dir), checkpoint_sha256=PL.sha256_hex(payload), step=step)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a gapped TD3+BC host with train.py (action gap).")
    parser.add_argument("--task", required=True, help="hopper | walker2d")
    parser.add_argument("--gap", required=True, type=Path, help="W/gap/gap.json")
    parser.add_argument("--gap-sha256", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, help="directory of the gapped copy (default: where define wrote it)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--lock", help="train.py's GPU lock path (default: train.py's)")
    parser.add_argument("--method", default="host", help="host only; bca refuses")
    parser.add_argument("--synthetic-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.method != "host":
        refuse_bca()
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    record = run(args.task, args.gap, args.gap_sha256, args.seed, args.output, data_dir=args.data_dir,
                 device=args.device, prepare_only=args.prepare_only, lock=args.lock, method=args.method,
                 synthetic_test=args.synthetic_test, allow_reexec=True)
    print(GAP.strict_json({k: record[k] for k in ("run_id", "gap_sha256", "projected_counts", "prepare_only")}))


if __name__ == "__main__":
    main()
