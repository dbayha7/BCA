"""Information-only check (b) forecast for gate seed 2 (V7_SPEC.md C7 step 2b; P1a, P1b, P1c and P2). NOT A v7 RESULT:
it runs on the cloud's Xeon (mode 'emulated': an AVX2-only processor emulated; mode 'default': the Xeon as it is, a
control), outside the v7 wrapper, before prereg_v7.json exists. The laptop's self-test replaces it.

It calls the functions of V7/code/v7_probes.py directly -- run_p1 (P1a, then P1b, then P1c, then the record with the
forecast_of gaps F_ref and F_fresh) and run_p2 -- not its command line, whose environment check
(v7_verify.check_environment) refuses XLA_FLAGS, OPENBLAS_* and NPY_* and whose set-up needs a draft prereg_v7.json.
What the command line's set-up does is done here in the same order, minus those two: one processor, the constants
substituted by v7_constants.apply, then the pinned room harness. The probe record's 'inputs' block is written here,
with the same fingerprints where they exist (constants, v7_probes.py, the gate's pre-registration, v6's code manifest)
and None for the draft's re-make block, which does not exist yet.

The reference files and records that v7_probes writes go under --r7, a scratchpad folder (never runs/). This script
then copies the record (with its sidecar) into --outdir and writes there a run file with the environment, the
processor, the constants facts and the software-check note.

Usage (from /home/user/bca, one processor):
  python forecast_seed2.py p1 --world stall|no_stall --mode emulated|default --r7 DIR --outdir DIR
  python forecast_seed2.py p2 --mode emulated|default --r7 DIR --outdir DIR
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xla_emul_common as X  # noqa: E402

SEED = 2
T0 = time.perf_counter()
ap = argparse.ArgumentParser()
ap.add_argument("what", choices=("p1", "p2"))
ap.add_argument("--world", choices=("stall", "no_stall"))
ap.add_argument("--mode", choices=X.MODES, required=True)
ap.add_argument("--r7", required=True)
ap.add_argument("--outdir", required=True)
a = ap.parse_args()
if a.what == "p1" and a.world is None:
    raise SystemExit("p1 needs --world")
R7, OUTDIR = Path(a.r7), Path(a.outdir)
if X.K.inside(R7, X.ROOT / "runs"):
    raise SystemExit("refused: --r7 must not be under runs/")
pid = f"p1_seed{SEED}_{a.world}" if a.what == "p1" else f"p2_seed{SEED}"
RUN = OUTDIR / f"run_{pid}.json"
run = dict(kind="stage2-v7-evidence-forecast-run", information_only=X.INFO, probe=pid, mode=a.mode, seed=SEED,
           world=a.world if a.what == "p1" else "stall", argv=sys.argv[1:],
           script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           common_sha256=hashlib.sha256((HERE / "xla_emul_common.py").read_bytes()).hexdigest(),
           r7=str(R7), calls=("v7_probes.run_p1 (p1a_call, p1b_call, p1c_call, forecast_of)" if a.what == "p1"
                              else "v7_probes.run_p2 (p2_call)"))
problems = X.check_mode(a.mode)
if problems:
    X.refuse(RUN, run, "; ".join(problems))
run["environment"] = X.env_facts(a.mode)
if not run["environment"]["mode_confirmed"]:
    X.refuse(RUN, run, f"the {a.mode} mode is not confirmed by numpy's dispatch and OpenBLAS's kernels")
try:
    run["constants"] = X.substitute_constants()
except X.K.V7Refusal as error:
    X.refuse(RUN, run, f"constants: {error}")
run["gate_software_check"] = X.gate_software_check(a.mode)
H = X.harness()
import v7_probes as PR  # noqa: E402  (V7/code is on sys.path through xla_emul_common)

K = X.K
inputs = dict(constants_sha256=run["constants"]["sha256"], probe_script_sha256=K.file_sha256(X.V7 / "code" / "v7_probes.py"),
              gate_prereg_sha256=K.file_sha256(X.ROOT / "runs" / "shift" / "preregistration.json"),
              v6_code_manifest_sha256=K.V6["code_manifest_sha256"], remake_block_sha256=None,
              prereg_v7_sha256_at_run=None, prereg_v7_was_final=False,
              note=("information only: run by evidence/xla/forecast_seed2.py on the cloud's Xeon, mode " + a.mode +
                    ", outside the v7 wrapper; no prereg_v7.json existed"))
if inputs["gate_prereg_sha256"] != K.GATE_PREREG_SHA256:
    X.refuse(RUN, run, "the gate's pre-registration is not 68d9e877...")


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


t0 = time.perf_counter()
try:
    rec = PR.run_p1(H, SEED, a.world, R7, inputs, log=log) if a.what == "p1" else PR.run_p2(H, SEED, R7, inputs, log=log)
except K.V7Refusal as error:
    X.refuse(RUN, run, f"v7_probes: {error}")
run["seconds"] = dict(probe=time.perf_counter() - t0, process=time.perf_counter() - T0)
src = PR.record_path(R7, pid)
OUTDIR.mkdir(parents=True, exist_ok=True)
for p in (src, src.with_name(src.name + ".sha256")):
    dst = OUTDIR / p.name
    if dst.exists():
        X.refuse(RUN, run, f"{dst} exists")
    shutil.copyfile(p, dst)
run["record"] = dict(file=src.name, sha256=K.file_sha256(OUTDIR / src.name), result_line=PR.result_line(rec))
if a.what == "p1":
    run["forecast"] = rec["forecast"]
else:
    run["actor"] = rec["actor"]
X.write(RUN, run)
print(f"{pid} {a.mode}: {run['record']['result_line']}; {run['seconds']['probe']:.0f} s")
sys.stdout.flush()
os._exit(0)
