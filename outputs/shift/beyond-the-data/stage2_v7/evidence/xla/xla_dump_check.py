"""Information-only check that the emulation changes the machine code of the real training program (NOT A v7 RESULT).

The emulated runs in this folder gave the same bits as the cloud's processor. That is informative only if capping XLA
at AVX2 (XLA_FLAGS=--xla_cpu_max_isa=AVX2) actually changes the instructions XLA generates for the training program;
if the cloud's own code used no AVX-512 instruction at all, the cap would change nothing. This script compiles and runs
one 4,000-step chunk of the gate's check-2 actor-only program (seed 2, stalling world, the gapped, full and random
critics side by side, as in xla_repro.py and P1a; 4,000 steps is one chunk, actor_only_chunk in
experiments/shift/learners.py:139, so the compiled program is the one the full 20,000-step run uses), asks XLA to
write its compiled object files to --dump (a scratchpad folder), and counts, with objdump, the instructions that use
512-bit registers (zmm, AVX-512 only) and 256-bit registers (ymm, AVX or AVX2).

Set-up as in xla_repro.py (one processor, the mode's environment checked and confirmed, the constants substituted by
V7/code/v7_constants.apply), except: the dump flag is appended to XLA_FLAGS inside the process before JAX starts its
processor backend, so in BOTH modes the gate's software check runs with the XLA_FLAGS entry excused
(xla_emul_common.gate_software_check is applied as for the emulated mode); everything else it checks is still checked.

Usage (from /home/user/bca, one processor): python xla_dump_check.py --mode default|emulated --dump DIR --out OUT.json
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xla_emul_common as X  # noqa: E402

STEPS = 4000
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=X.MODES, required=True)
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
OUT, DUMP = Path(a.out), Path(a.dump)
out = dict(kind="stage2-v7-evidence-xla-dump-check", information_only=X.INFO, mode=a.mode, seed=2, steps=STEPS,
           argv=sys.argv[1:], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           common_sha256=hashlib.sha256((HERE / "xla_emul_common.py").read_bytes()).hexdigest())
problems = X.check_mode(a.mode)
if X.K.inside(DUMP, X.ROOT):
    problems.append("--dump must be outside /home/user/bca")
if DUMP.exists() and any(DUMP.iterdir()):
    problems.append(f"{DUMP} is not empty")
if problems:
    X.refuse(OUT, out, "; ".join(problems))
out["environment"] = X.env_facts(a.mode)
if not out["environment"]["mode_confirmed"]:
    X.refuse(OUT, out, f"the {a.mode} mode is not confirmed by numpy's dispatch and OpenBLAS's kernels")
DUMP.mkdir(parents=True, exist_ok=True)
os.environ["XLA_FLAGS"] = " ".join(x for x in (os.environ.get("XLA_FLAGS"), f"--xla_dump_to={DUMP}") if x)
out["xla_flags_in_process"] = os.environ["XLA_FLAGS"]
try:
    out["constants"] = X.substitute_constants()
except X.K.V7Refusal as error:
    X.refuse(OUT, out, f"constants: {error}")
out["gate_software_check"] = X.gate_software_check("emulated")
H = X.harness()
L = H.L
import jax  # noqa: E402

t0 = time.perf_counter()
ctx = H.SeedContext(2, worlds=("stall",))
pg = ctx.prepared("stall")
named = [("gapped", ctx.critic("stall", "gap")[0]), ("full", ctx.critic("stall", "none")[0]),
         ("random", ctx.critic("stall", "random")[0])]
clone, _ = ctx.clone("stall")
ao = L.ActorOnly(pg, clone, ctx.learn, ctx.rep, alpha=H.ALPHA, steps=STEPS, chunk=STEPS)
stacked, info = ao.run([L.Arm(n, critic=i) for i, (n, _) in enumerate(named)], [c for _, c in named],
                       weighted=False, log=lambda *a: None)
jax.block_until_ready(stacked)
out["arms_after_one_chunk_sha256"] = {n: L.tree_sha256(ao.arm_params(stacked, i)) for i, (n, _) in enumerate(named)}
out["seconds"] = time.perf_counter() - t0

ZMM, YMM = re.compile(r"%zmm\d+"), re.compile(r"%ymm\d+")
modules = {}
for obj in sorted(DUMP.glob("*.o")):
    text = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(obj)], capture_output=True, text=True).stdout
    lines = [ln for ln in text.splitlines() if "\t" in ln]
    z = sum(1 for ln in lines if ZMM.search(ln))
    y = sum(1 for ln in lines if YMM.search(ln))
    mod = obj.name.split(".")[0] + "." + obj.name.split(".")[1]
    m = modules.setdefault(mod, dict(object_files=0, instructions=0, zmm_instructions=0, ymm_instructions=0))
    m["object_files"] += 1
    m["instructions"] += len(lines)
    m["zmm_instructions"] += z
    m["ymm_instructions"] += y
prefer = sorted({m for f in DUMP.glob("*ir-with-opt.ll")
                 for m in re.findall(r'"prefer-vector-width"="(\d+)"', f.read_text(errors="replace"))})
big = max(modules.items(), key=lambda kv: kv[1]["instructions"]) if modules else (None, None)
out["dump"] = dict(folder=str(DUMP), object_files=sum(m["object_files"] for m in modules.values()),
                   instructions=sum(m["instructions"] for m in modules.values()),
                   zmm_instructions=sum(m["zmm_instructions"] for m in modules.values()),
                   ymm_instructions=sum(m["ymm_instructions"] for m in modules.values()),
                   prefer_vector_width=prefer, largest_module=dict(name=big[0], counts=big[1]), modules=modules)
X.write(OUT, out)
d = out["dump"]
print(f"{a.mode}: {d['object_files']} object files, {d['instructions']} instructions, {d['zmm_instructions']} use 512-bit "
      f"registers, {d['ymm_instructions']} use 256-bit registers; {out['seconds']:.0f} s")
sys.stdout.flush()
os._exit(0)
