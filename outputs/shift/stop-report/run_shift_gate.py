"""Run the ThrusterChain harm gate in its pre-registered order, up to the stop before any BCA run.

Each command is one experiments.shift.acceptance stage process, pinned to one CPU (4 slots). Per-seed stages run in
parallel over the slots; pooled stages run alone. The stop rule and the alpha ladder are decided by the gate's own
code (variant "auto", alpha "auto"): this runner only asks it what is due next and stops when a stage refuses.
Nothing BCA-related is ever called. Log: runs/shift_runner/progress.txt.
"""
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path("/home/user/bca")
PY = "/tmp/claude-0/bcavenv/bin/python"
LOG = REPO / "runs/shift_runner"  # outside runs/shift: the pre-registration refuses any non-"pre" file there
LOG.mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
for k in ("JAX_ENABLE_X64", "XLA_FLAGS", "JAX_DEFAULT_MATMUL_PRECISION"):
    ENV.pop(k, None)
SLOTS = ["0", "1", "2", "3"]
MOD = ["-m", "experiments.shift.acceptance"]


def note(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG / "progress.txt", "a") as fh:
        fh.write(line + "\n")


def run_phase(name, cmds):
    """Run cmds (lists of stage args) over the slots; return {cmd string: (exit code, last log line)}."""
    note(f"phase {name}: {len(cmds)} commands")
    q, results = queue.Queue(), {}
    for c in cmds:
        q.put(c)

    def worker(cpu):
        while True:
            try:
                c = q.get_nowait()
            except queue.Empty:
                return
            tag = "_".join(a.lstrip("-").replace("/", "-") for a in c)
            t0 = time.time()
            with open(LOG / f"{tag}.log", "w") as fh:
                rc = subprocess.call(["nice", "-n", "5", "taskset", "-c", cpu, PY, "-u", *MOD, *c], cwd=REPO, env=ENV,
                                     stdout=fh, stderr=subprocess.STDOUT)
            tail = (LOG / f"{tag}.log").read_text().strip().splitlines()[-1:] or [""]
            results[" ".join(c)] = (rc, tail[0])
            note(f"  [cpu {cpu}] rc={rc} {time.time() - t0:7.0f}s {' '.join(c)} | {tail[0][:200]}")

    threads = [threading.Thread(target=worker, args=(s,)) for s in SLOTS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bad = [k for k, (rc, _) in results.items() if rc != 0]
    note(f"phase {name} done: {len(results) - len(bad)} ok, {len(bad)} failed" + (f": {bad}" if bad else ""))
    return results


def single(name, args):
    rc, tail = run_phase(name, [args])[" ".join(args)]
    return rc == 0, tail


def gate_state():
    """Ask the gate's own code where the harm checks stand (no training; reads finished outputs only)."""
    code = ("import json, experiments.shift.acceptance as A\n"
            "p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))\n"
            "v = A.active_variant(p)\n"
            "print(json.dumps(dict(A.harm_state(p, v), active=v)))\n")
    out = subprocess.run([PY, "-c", code], cwd=REPO, env=ENV, capture_output=True, text=True)
    if out.returncode != 0:
        note(f"gate_state failed: {out.stderr.strip().splitlines()[-1:]}")
        return None
    import json
    return json.loads(out.stdout.strip().splitlines()[-1])


def harm_checks(variant, seeds):
    """H1 then the alpha ladder of H2, for one variant. Returns the harm state afterwards."""
    pooled = next((REPO / "runs/shift").glob(f"*/harm_critic/{variant}/pooled.json"), None)
    if pooled is not None:
        note(f"[{variant}] H1 already pooled ({pooled}), not rerun")
    else:
        run_phase(f"[{variant}] H1 critics", [["harm_critic", "--variant", variant, "--seeds", str(s)] for s in seeds])
        single(f"[{variant}] H1 pooled", ["harm_critic", "--variant", variant, "--pooled"])
    while True:
        st = gate_state()
        note(f"[{variant}] harm state: {st}")
        if st is None:
            raise SystemExit("STOPPED: could not read the gate's harm state; nothing further is run")
        if st.get("failed") or st.get("alpha") is not None or st.get("next_alpha") is None:
            return st
        a = f"{st['next_alpha']:g}"
        run_phase(f"[{variant}] H2 at alpha {a}",
                  [["harm_return", "--variant", variant, "--alpha", a, "--seeds", str(s)] for s in seeds])
        ok, _ = single(f"[{variant}] H2 pooled at alpha {a}", ["harm_return", "--variant", variant, "--alpha", a,
                                                                  "--pooled"])
        if not ok:
            st = gate_state()
            if st is None:
                raise SystemExit("STOPPED: could not read the gate's harm state; nothing further is run")
            return st


def room_and_power(variant, seeds):
    run_phase(f"[{variant}] H3 room", [["room", "--variant", variant, "--seeds", str(s)] for s in seeds])
    ok, _ = single(f"[{variant}] H3 pooled", ["room", "--variant", variant, "--pooled"])
    if not ok:
        return
    run_phase(f"[{variant}] H4 power", [["power", "--variant", variant, "--seeds", str(s)] for s in seeds])
    single(f"[{variant}] H4 pooled", ["power", "--variant", variant, "--pooled"])


def main():
    pre = REPO / "runs/shift/pre"
    steps = [("build checks", ["build_checks"], pre / "build_checks.json"),
             ("stage0 base", ["stage0", "--variant", "base"], pre / "stage0_base.json"),
             ("stage0 retune", ["stage0", "--variant", "retune"], pre / "stage0_retune.json")]
    import json

    def passed(path):
        try:
            return bool(json.loads(path.read_text()).get("passed"))
        except (OSError, ValueError):
            return None

    for name, args, done in steps:
        if passed(done):
            note(f"{name}: already passed ({done.name}), not rerun")
            continue
        ok, tail = single(name, args)
        verdict = passed(done)
        note(f"{name}: exit {'0' if ok else 'non-zero'}, recorded passed = {verdict}")
        if (not ok or not verdict) and name != "stage0 retune":
            note(f"STOPPED: {name} did not pass: {tail}")
            note("SHIFT RUNNER FINISHED")
            return 1
    if os.environ.get("SHIFT_DATA_DONE") == "1":
        note("data: already generated by the first attempt (both commands exited 0), not rerun")
    else:
        run_phase("data", [["data", "--variant", "base"], ["data", "--variant", "retune", "--roles", "gate", "calibration"]])
    if (REPO / "runs/shift/preregistration.json").exists():
        ok, tail = True, "exists"
        note("pre-registration: already written, not rewritten")
    else:
        ok, tail = single("pre-registration", ["prereg"])
    if not ok:
        note(f"STOPPED: pre-registration refused: {tail}")
        note("SHIFT RUNNER FINISHED")
        return 1
    for variant, seeds in (("base", [0, 1, 2, 3, 4]), ("retune", [5, 6, 7, 8, 9])):
        if variant == "retune":
            st0 = gate_state()
            if st0 is None or st0.get("active") != "retune":
                note(f"STOPPED: the gate does not allow the retune (state {st0}); nothing further is run")
                break
        st = harm_checks(variant, seeds)
        if st and st.get("alpha") is not None:
            note(f"[{variant}] harm certified at alpha {st['alpha']:g}")
            room_and_power(variant, seeds)
            break
        note(f"[{variant}] harm checks failed: {st and st.get('reason')}" +
             (" -> stop rule: the pre-registered retune" if variant == "base" else " -> stop: no critic-caused harm"))
    st = gate_state()
    gate_seeds = [5, 6, 7, 8, 9] if st and st.get("variant") == "retune" else [0, 1, 2, 3, 4]
    run_phase("report cards", [["report", "--seeds", str(s)] for s in gate_seeds])
    single("stop report", ["report"])
    note("SHIFT RUNNER FINISHED (stopped before any BCA run; the user decides)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
