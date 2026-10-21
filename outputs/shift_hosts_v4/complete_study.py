"""Check, personally approve, and run the tested version-4 ThrusterChain supplement.

Run --check from WSL for a read-only readiness check. The human user runs
complete_study.ps1 in PowerShell: it freezes the seven tested files, calls the
existing interactive approval program, commits that approval, and launches a
worker that trains the remaining grid and exports the complete results.

This helper is outside the experiment's code manifest. It never changes the
experiment code, creates an approval itself, or relaxes a stop rule.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True

REPO = Path("/root/bca-hosts")
TRAIN = "/opt/bca-venv/bin/python"
SCORE = "/opt/bca-shift-venv/bin/python"
BRANCH = "claude/shift-hosts"
TRANSFER = "refs/heads/codex/shift-hosts-v4-transfer"
EXPECTED = "d3538fd0b39698361e3318666e28d8316c37df9175769d5a2cb54225e5e2f71a"
FILES = tuple("experiments/shift_hosts/" + name for name in (
    "INTERFACES.md", "PRESPEC.md", "prespec.json", "stages.py",
    "supplement.py", "tests/test_claims.py", "tests/test_stages.py",
))


def call(args, *, capture=False, env=None):
    return subprocess.run(
        [str(a) for a in args], cwd=REPO, env=env, check=True,
        text=True, stdout=subprocess.PIPE if capture else None,
    ).stdout


def git(*args, capture=False):
    return call(["git", *args], capture=capture)


def ready():
    from experiments.shift_hosts import gate, stages, supplement

    actual, _, _ = gate.fingerprint()
    if actual != EXPECTED:
        raise RuntimeError("The package changed since this launcher was checked; refusing to freeze or run it.")
    if gate.bytecode_files():
        raise RuntimeError("Compiled bytecode exists beside the frozen code; the experiment gate refuses it.")
    s0 = gate.require_s0("completion readiness")
    frozen = stages.frozen_config()
    supplement.check_sources(frozen["supplement"])
    changes = supplement.code_unchanged(gate.code_manifest())
    if changes:
        raise RuntimeError(f"Training/scoring files differ from version 3: {changes}")
    if git("branch", "--show-current", capture=True).strip() != BRANCH:
        raise RuntimeError(f"Expected the existing {BRANCH} branch; refusing to switch branches.")
    run_ids = supplement.run_ids(frozen["plan"], frozen["seeds"], frozen["learners"])
    report = dict(
        fingerprint=actual, stage0_passed=s0["passed"], suite=s0["suite_run"],
        stage1_runs=20, stage2_runs={q: len(ids) for q, ids in run_ids.items()},
        total_runs=20 + sum(len(ids) for ids in run_ids.values()),
        source_files_verified=True, training_and_scoring_unchanged=True,
        approved=gate.approval_path(actual).is_file(),
        root=str(supplement.ROOT),
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def transfer():
    # Transfer to a non-checked-out ref in Windows; its working files stay intact.
    git("push", "origin", f"HEAD:{TRANSFER}")


def freeze():
    from experiments.shift_hosts import approve_hosts

    if git("diff", "--cached", "--name-only", capture=True).strip():
        raise RuntimeError("The WSL index already has staged changes; refusing to include or unstage them.")
    dirty = set(git("diff", "--name-only", capture=True).splitlines())
    dirty.update(git("ls-files", "--others", "--exclude-standard", capture=True).splitlines())
    if dirty - set(FILES):
        raise RuntimeError(f"Other WSL work is present; refusing to commit it: {sorted(dirty - set(FILES))}")
    if dirty:
        ready()  # Recheck immediately before staging, including the exact fingerprint.
        git("add", "--", *FILES)
        message = (
            f"Unifloral hosts: freeze pre-specification version 4 (fingerprint {EXPECTED[:16]})\n\n"
            "Train and score every scope excluded by version 3's S7 check.\n"
            "Stage-0 suite 20261020T183357Z passed on these exact files.\n"
            "Version 3's judged results and all training/scoring code stay unchanged.\n\n"
            "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>\n"
            "Claude-Session: https://claude.ai/code/session_01BAXKUJJESi4SM9NLR3X4Rp\n"
        )
        git("-c", "user.name=Claude", "-c", "user.email=noreply@anthropic.com",
            "commit", "--only", "-m", message, "--", *FILES)
    # The approval program independently requires every manifest file to be committed.
    info = approve_hosts.prepare()
    if info["fingerprint"] != EXPECTED:
        raise RuntimeError("The committed fingerprint differs from the tested one.")
    transfer()


def commit_approval():
    from experiments.shift_hosts import approve_hosts, gate

    # Only the existing interactive program can have created this file.
    path = gate.approval_path(EXPECTED)
    files = [path.relative_to(REPO).as_posix(), gate.sidecar(path).relative_to(REPO).as_posix()]
    if not path.is_file() or not gate.sidecar(path).is_file():
        raise RuntimeError("Your terminal approval has not been written; refusing to commit or run.")
    if any(not gate.committed_in_head(REPO / f) for f in files):
        if git("diff", "--cached", "--name-only", capture=True).strip():
            raise RuntimeError("Another process staged changes; the approval is written but not committed.")
        git("add", "--", *files)
        git("-c", f"user.name={approve_hosts.COMMIT_NAME}", "-c", f"user.email={gate.APPROVAL_EMAIL}",
            "commit", "--only", "-m", f"Unifloral hosts: approve the pre-specification {EXPECTED[:16]}",
            "--", *files)
    gate.require_run_approval("completion launcher")
    transfer()


def launch():
    from experiments.shift_hosts import gate, stages, supplement

    ready()
    gate.require_run_approval("completion launcher")
    args = stages.parser().parse_args([
        "run", "--stage0-card-hours", "0", "--stage0-processor-hours", "0",
    ])
    stages.preflight(args)
    driver = supplement.ROOT / "driver"
    if stages.lock_holder(driver) is not None:
        raise RuntimeError("The supplement already has a running driver; leaving it alone.")
    if (driver / "state.json").exists():
        raise RuntimeError("The supplement has already started. Inspect its state before resuming; nothing was retried.")
    driver.mkdir(parents=True, exist_ok=True)
    log = driver / "completion_launcher.log"
    with log.open("ab") as handle:
        child = subprocess.Popen(
            ["nohup", TRAIN, "-B", str(Path(__file__).resolve()), "--worker"],
            cwd=REPO, env=dict(os.environ), stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
    print(f"Started completion worker {child.pid}. Log: {log}", flush=True)
    print("Keep WSL running. The worker exports results after all runs finish; a stop rule leaves a stop report.")


def worker():
    from experiments.shift_hosts import gate, supplement

    ready()
    gate.require_run_approval("completion worker")
    root = supplement.ROOT
    result = subprocess.run(
        [TRAIN, "-B", "-m", "experiments.shift_hosts.stages", "run",
         "--stage0-card-hours", "0", "--stage0-processor-hours", "0"], cwd=REPO,
    )
    completion = dict(driver_exit=result.returncode, exported=False, utc=datetime.now(timezone.utc).isoformat())
    if result.returncode == 0:
        status = json.loads((root / "driver" / "status.json").read_text())
        if status.get("state") != "finished":
            raise RuntimeError("Driver exited zero without a finished status; refusing to claim completion.")
        call([SCORE, "-B", "-m", "experiments.shift_hosts.supplement", "export"])
        completion.update(exported=True, results=str(REPO / "outputs/shift_hosts/results.json"))
    path = root / "driver" / "completion.json"
    path.write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion, indent=2), flush=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="read-only readiness check (the default)")
    mode.add_argument("--freeze", action="store_true", help="human terminal: commit and transfer the tested files")
    mode.add_argument("--approve", action="store_true", help="human terminal: run the existing approval prompt")
    mode.add_argument("--commit-approval", action="store_true", help="human terminal: commit and transfer your approval")
    mode.add_argument("--launch", action="store_true", help="launch after the approval commit")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not REPO.is_dir():
        parser.error("Run this helper inside WSL as root; /root/bca-hosts must exist.")
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", GIT_CONFIG_GLOBAL="/root/.bca-gitconfig")
    # A WSL login shell adds /usr/lib/wsl/lib to LD_LIBRARY_PATH. The driver's
    # launch_env sets the approved MuJoCo-only value for each training child.
    for key in ("LD_LIBRARY_PATH", "JAX_PLATFORMS", "XLA_FLAGS", "SHIFT_HOSTS_DEVICE"):
        os.environ.pop(key, None)
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    try:
        if args.freeze or args.approve or args.commit_approval:
            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                raise RuntimeError("Run complete_study.ps1 yourself at a terminal; these steps require your input.")
            ready()
            if args.freeze:
                freeze()
            elif args.approve:
                # The user personally types the fingerprint into the existing program.
                call([TRAIN, "-B", "-m", "experiments.shift_hosts.approve_hosts"])
            else:
                commit_approval()
        elif args.launch:
            launch()
        elif args.worker:
            return worker()
        else:
            ready()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
