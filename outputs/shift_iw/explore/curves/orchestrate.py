"""Run every learning-curve job with a fixed number of slots (EXPLORATORY; see curves.py).

Pinned jobs (version 5's runs: they were trained one per CPU, so they reproduce only when pinned to one CPU with one
thread): taskset -c <cpu>, OMP_NUM_THREADS=1. Unpinned jobs (the channel arms, trained unpinned): OMP_NUM_THREADS=2.
A job whose output exists is skipped by curves.py itself. Logs: /root/logs_curves/<job>.log; progress:
/root/logs_curves/orchestrate.log.

Usage: python orchestrate.py <pinned slots> <unpinned slots>
"""
import os
import subprocess
import sys
import time
from pathlib import Path

QUALITIES = ("random", "medium-replay", "medium", "medium-expert", "expert")
SEEDS = (0, 1, 2, 3, 4)
PY = "/opt/bca-shift-venv/bin/python"
CURVES = "/mnt/c/Users/David/dev/bca-shift-work/curves/curves.py"
LOGS = Path("/root/logs_curves")
CPUS = list(range(2, 24))


def jobs():
    pinned, unpinned = [], []
    for run in ("capped_unweighted", "capped_exact_strong"):          # longest first
        pinned += [(f"{run}_{q}_{s}", ["cotrain", q, str(s), run]) for q in QUALITIES for s in SEEDS]
    for run in ("host_gap", "host_full"):
        pinned += [(f"{run}_{q}_{s}", ["cotrain", q, str(s), run]) for q in QUALITIES for s in SEEDS]
    pinned += [(f"actor_v5_{q}_{s}", ["actor", q, str(s), "band", "oracle"]) for q in QUALITIES for s in SEEDS]
    unpinned += [(f"actor_new_{q}_{s}", ["actor", q, str(s), "cmd", "f1", "est"]) for q in QUALITIES for s in SEEDS]
    return pinned, unpinned


def main(n_pinned, n_unpinned):
    LOGS.mkdir(exist_ok=True)
    pinned, unpinned = jobs()
    free_cpus = CPUS[:n_pinned]
    running = {}       # name -> (Popen, cpu or None)
    log = open(LOGS / "orchestrate.log", "a", buffering=1)
    log.write(time.strftime("%H:%M:%S ") + f"start: {len(pinned)} pinned, {len(unpinned)} unpinned\n")
    base = dict(os.environ, PYTHONPATH="/root/bca-shift", PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu")
    while pinned or unpinned or running:
        for name, (p, cpu) in list(running.items()):
            if p.poll() is not None:
                log.write(time.strftime("%H:%M:%S ") + f"done {name} rc {p.returncode}\n")
                if cpu is not None:
                    free_cpus.append(cpu)
                del running[name]
        while pinned and free_cpus:
            name, args = pinned.pop(0)
            cpu = free_cpus.pop(0)
            env = dict(base, OMP_NUM_THREADS="1")
            cmd = ["nice", "-n", "10", "taskset", "-c", str(cpu), PY, CURVES] + args
            running[name] = (subprocess.Popen(cmd, env=env, cwd="/root/bca-shift", stdout=open(LOGS / f"{name}.log", "w"),
                                              stderr=subprocess.STDOUT), cpu)
            log.write(time.strftime("%H:%M:%S ") + f"start {name} on cpu {cpu}\n")
        while unpinned and sum(1 for _, c in running.values() if c is None) < n_unpinned:
            name, args = unpinned.pop(0)
            env = dict(base, OMP_NUM_THREADS="2")
            cmd = ["nice", "-n", "10", PY, CURVES] + args
            running[name] = (subprocess.Popen(cmd, env=env, cwd="/root/bca-shift", stdout=open(LOGS / f"{name}.log", "w"),
                                              stderr=subprocess.STDOUT), None)
            log.write(time.strftime("%H:%M:%S ") + f"start {name} (unpinned)\n")
        time.sleep(10)
    log.write(time.strftime("%H:%M:%S ") + "all done\n")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
