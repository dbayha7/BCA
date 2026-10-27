"""The stage driver: stage 1 and its checks, choice 19's sham-alone runs, stage 2a (medium-expert) and stage 2b (medium)
one whole seed at a time, a card queue beside processor scoring lanes, the stop rules, a status file the main session
can poll, and a report on any stop (plan sections 5.1, 6.1, 7.3 and 7.4). It refuses to start without the user's
terminal approval (gate.require_run_approval), and so does every run and lane it starts.

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. It runs no
learner itself: card runs are runner.py processes in the training environment (/opt/bca-venv); scoring, S7 and the
coverage hand-offs run in the scoring environment (/opt/bca-shift-venv); band readouts and S7's training-side commands
run in the training environment on the processor. Every child is pinned with taskset, with PYTHONDONTWRITEBYTECODE=1 and
OMP_NUM_THREADS=1; card runs also get the Dockerfile's variables and XLA_PYTHON_CLIENT_PREALLOCATE=false.

What it runs, in this order (plan 7.3-7.4; the task's stage-driver list):
  stage 1     medium-expert, data seed 0, 15,000 outer steps (one refresh), per learner: alone, standard and IW one at a
              time, then IW with every weight 1 (S2) and a repeat of IW (S8) two at once. Then S7 both halves on every
              stage-1 actor (scoring lanes), and the checks: S2, S8, S10 (a readout instead, with choice 19, for a
              learner whose test (a2) on the card matched only to rounding: TD3+BC and IQL by the stage-0 results,
              CHOICE19),
              S4, S7 (every stage-1 actor's s7.json must say it passed), S9 and S6 (speeds, the projection of stages 1-2
              against the caps, each learner alone at most twice its random-array reference, and the two-at-once rule
              budget.TWO_AT_ONCE_RULE). No return is read.
  stage 2a    medium-expert: choice 19's sham-alone runs (seed 0, 100,000 steps, the learners above), then seeds 0-4,
              each seed all 12 setups (per learner alone, standard, IW), then the 20 room references (each learner alone
              on the full log, seeds 0-4) last. S6b before every one of those units, against the quality's cap.
  stage 2b    medium, the same, only if every stop rule passed so far (approval item 2; its own cap).
  lanes       beside the card queue, on up to 8 processor lanes: every finished actor scored on the 7 banks at 50% and
              100% (evaluate.score_actor; seed 0's final actors keep the viewer's episodes, and the BCA ones get the
              viewer's band arrays, band_readout viewer); every BCA run's coverage at its last refresh (the scoring
              side's actor and visited transitions, then band_readout coverage); S7 both halves on every seed-0 final
              actor (and every stage-1 actor). After each quality's last lane, every seed-0 S7 result must say it
              passed.
Stop rules (plan 6.1): S5 on a learner alone or a room reference is that learner's Claim 0 outcome, recorded, and the
driver goes on; S5 on a BCA setup halts until it is traced (rerun with --accept-s5 RUN_ID if it is the setup's outcome,
not a package defect); S1 (a refused run: the approval no longer matches the code; a run record of another fingerprint;
the approval re-verified before every unit), S3, S11, S2, S4, S6, S6b, S8, S9 and S10 (where it is not a readout)
halt. S7 (version 3, PRESPEC.md section 20, post hoc): a failed float32 check stops a scope within its quality instead
of the driver. An alone, standard or weights-of-1 actor stops its learner, an IW actor (or IW's repeat) stops only that
learner's IW setup (S7_SCOPE); state.json "s7_stopped" records it, that scope's queued stage-2 runs and lanes are
dropped and its later units leave it out, and everything else goes on. Stage 1 still runs all 20 runs and every check.
An s7.json that does not verify still halts. S5 also covers a non-finite return: a scoring lane whose scores hold one exits 4 (EXIT_NONFINITE), and the driver
records it as that learner's Claim 0 outcome (alone, room) or halts (a BCA setup, unless accepted). A halt launches
nothing new (not even the next step of a lane already running), lets running children finish, writes driver/STOP.json
and driver/STOP.txt, and exits 3. An unexpected error or SIGTERM also writes the stop report, after waiting up to an hour
for the running children (it then names any still running), and exits 1 (error) or 3 (interrupted). Nothing is retried
silently: a resumed driver skips finished runs and lanes (their records and outputs verify; a lane's own verdict, S7's
s7.json and the scores' finite returns, is read again, so a failed check never passes on resume), refuses to start over a
stop report unless told to (--resume-after-stop), and a run that died without a record is restarted only when named
(--replace-incomplete RUN_ID ...), and only once its process is gone.

Version 4 (PRESPEC.md section 21, post hoc: the S7 supplement; version 3's text above stays as history). S7 records a
failed float32 check in state.json "s7_stopped" (with its scope, as version 3 did, and version 4) and every lane's
verdict in "s7_results", and drops nothing: every queued run and lane goes on, the dropped counts stay 0, and no S7
failure halts (an s7.json that does not verify, or an exit status that disagrees with it either way, 3 on a passing
file or 0 on a failing one, still does). On real data the driver trains only the frozen supplement plan (prespec.json
"supplement", supplement.py: every stage-2 run of every scope version 3's S7 stopped) under its own root
runs/shift_hosts/supplement_v4, with its own driver files and ledger; stage 1 always runs whole. S7's lanes run on the
final actor of every planned alone, standard and IW run at seeds 0-4 and on its 50,000-step actor too
(lanes/s7-step50000); only seed 0's final actors and stage 1's get check keys (S7/<quality>/seed0/<learner>/<setup>,
S7/<learner>/<setup>). Every other stop rule halts as approved.

One driver per root: driver/driver.lock is held (flock) for the driver's life; a second driver on the same root is
refused before it changes anything (a pending STOP_REQUEST included). A resumed driver also reuses stage 1's S6 decision
while stage 1's records are unchanged (it is never recomputed against totals that already hold stage-2 work), projects
S6b only for the runs of a unit that have no record yet, and skips S6b for a unit with nothing left to run.

Modes. driver/state.json and driver/ledger.json record their mode ("dry-run" or "real") and the driver's effective
configuration (learners, qualities, seeds, device, choice 19, schedule, version 4's plan and S7 lanes); a driver of the
other mode, or a resume with another configuration, is refused. A dry run never uses the real roots (runs/shift_hosts,
and version 4's runs/shift_hosts/supplement_v4 or anything inside it). On real data the driver reads choice 19, the
learners, the qualities, the data seeds, the S9 signature and version 4's supplement plan and S7 lanes from
prespec.json, and refuses options that would change them (--choice19, --learners, --qualities, --seeds, --card-device,
--root, --supplement-plan, --s7-all-seeds, --s7-half) or the limits (--caps, --inject, more than 2 card cores or 8
lanes); before it starts it also requires stop rule S0 (gate.require_s0: the stage-0 summary was made on exactly this
code), stage 0's compute entry (in the supplement's own ledger), version 3's files as the frozen block names them
(supplement.check_sources) and every file but supplement.CHANGEABLE byte-identical to version 3's approved code
(supplement.code_unchanged).

Files (under --root, default runs/shift_hosts/supplement_v4, version 4's frozen root; version 3's was runs/shift_hosts):
driver/status.json (polled by the main session), driver/driver.log,
driver/ledger.json (budget.py; with ledger.json.prev), driver/state.json (decisions and check results; with
state.json.prev), driver/driver.lock, driver/logs/<job>.log, driver/STOP.*; run folders as runner.py writes them, each
with lanes/ (scores, S7, coverage, viewer). driver/STOP_REQUEST asks a running driver to halt gracefully.

Start it detached (WSL, as root, from /root/bca-hosts), after the approval:
  PYTHONDONTWRITEBYTECODE=1 /opt/bca-venv/bin/python -m experiments.shift_hosts.stages start \\
      --stage0-card-hours <h> --stage0-processor-hours <h>
which runs `... stages run ...` under nohup (driver/nohup.out); `... stages status` prints the status file.
Dry run (made-up arrays, processor, tiny step counts; no approval needed, no return produced):
  ... stages run --dry-run [--root runs/shift_hosts/stage0/runner/dryrun] [--seeds 0 1] [--inject RUN_ID=unit0:15]
      [--supplement-plan PLAN.json] [--s7-all-seeds] [--s7-half]
(a dry run without --supplement-plan trains the full grid; the plan file is {quality: {learner: [setups]}}, as
supplement.check_plan reads it).
In a dry run the lanes skip everything that would roll a learned policy out in the simulator (scoring, the visited
transitions, S7's closed loop): S7's open loop runs on made-up states and coverage on the made-up strong-law rows only.
"""

import argparse
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from experiments.shift_hosts import budget as B
from experiments.shift_hosts import gate, guard
from experiments.shift_hosts import handoff as H
from experiments.shift_hosts import runner as RN
from experiments.shift_hosts import supplement as S
from experiments.shift_hosts import unifloral_import as UI

sys.dont_write_bytecode = True

TRAINING_PYTHON = "/opt/bca-venv/bin/python"
SCORING_PYTHON = "/opt/bca-shift-venv/bin/python"
STATUS_SCHEMA = "shift-hosts-driver-status-v1"
STATE_SCHEMA = "shift-hosts-driver-state-v1"
STOP_SCHEMA = "shift-hosts-driver-stop-v1"
LEARNERS = RN.LEARNERS
QUALITIES = RN.QUALITIES
SEEDS = (0, 1, 2, 3, 4)
STAGE1_SINGLE = ("alone", "standard", "iw")
STAGE1_PAIR = ("iw_ones", "iw_repeat")
STAGE2_SETUPS = ("alone", "standard", "iw")
# Version 3's S7 rule (PRESPEC.md section 20): what a failed float32 check stops within its quality. The weights-of-1
# actor equals standard's in every bit (S2) and IW's repeat equals IW's (S8), so each stops what its twin would.
S7_SCOPE = dict(alone="learner", standard="learner", iw_ones="learner", iw="iw", iw_repeat="iw")
S7_INJECT = "s7fail"   # dry run only: --inject RUN_ID=s7fail makes that run's S7 check fail (the tests of the rule)
# Choice 19 (plan 4.4): learners whose test (a2) matched only to rounding on the card in stage 0 (TD3+BC: the logged
# losses by 1-2 rounding units, state bit for bit; IQL: every leaf within 3.8e-7). CQL (with its pre-refresh program)
# and ReBRAC were bit for bit. On real data prespec.json's choices.19.applies is used and must equal this; --choice19
# changes it in dry runs only.
CHOICE19 = dict(td3_bc=True, cql=False, iql=True, rebrac=False)
# The dry run's S6 references: twice the environment report's one-core speeds (TD3+BC 7.7 ms, CQL 107, IQL 5.2,
# ReBRAC 70.6; measured on an idle core with long chunks), because the dry run's processes share hyperthreaded cores
# and pay a fixed cost per 10-step chunk (stage-0 measurements: IQL alone 9.0 ms, CQL 155 ms per outer step). The real
# run uses the plan's card references (budget.REFERENCE_MS).
REFERENCE_MS_PROCESSOR = dict(td3_bc=2 * 7.7, cql=2 * 107.0, iql=2 * 5.2, rebrac=2 * 70.6)
POLL = 0.5             # seconds between reaping children (it sets how finely wall times are measured)
STATUS_EVERY = 5.0     # seconds between status-file writes
HEARTBEAT_EVERY = 30.0  # seconds between ledger heartbeats (open entries of running children; the driver's own time)
CHILD_WAIT = 3600.0    # seconds a failing driver waits for its running children before it writes its stop report
DRY_SCHEDULE = dict(num_updates=40, eval_interval=10, warmup=20, every=10, stage1_steps=30)
EXIT_NONFINITE = 4     # a scoring lane whose scores hold a non-finite return (S5; the driver decides what it means)
LOCK_NAME = "driver.lock"
MAX_CARD_CORES, MAX_LANE_CORES = 2, 8   # plan 7.4: one or two card runs at once; up to 8 processor lanes
_LOCKS = []            # file descriptors of held driver locks (kept open for the process's life)


def utc_now():
    return RN.utc_now()


# ------------------------------------------------------------------------------------------------ checks on records

def read_record(run_dir):
    return H.read_json(Path(run_dir) / "record.json")


def check_s4(records, key_setups=("alone", "standard", "iw")):
    """S4 (plan 6.1): every setup of a learner (room included) starts from the same initial parameters, and its alone,
    standard and IW setups carry the same Unifloral key out of the chunks ending at 2,500, 50,000 and 100,000 (with the
    same training data and code, equal keys mean equal batches). A stopped run is compared up to where it stopped."""
    init = {s: r["hashes"]["initial"]["agent_state"]["all"] for s, r in records.items()}
    same_init = len(set(init.values())) == 1
    keys = {}
    for s, r in records.items():
        if s not in key_setups:
            continue
        stop = (r.get("stop") or {}).get("step") if r.get("outcome") == "stopped" else None
        # a stopped run counts only before its first bad step: a BCA step that is not valid keeps the old carry,
        # key included, so the keys it carries out afterwards are not its batch stream
        keys[s] = {t: v for t, v in r["hashes"]["key_carry"].items() if stop is None or int(t) < int(stop)}
    steps = sorted({int(t) for k in keys.values() for t in k}, key=int)
    diff = []
    for t in steps:
        vals = {s: k.get(str(t)) for s, k in keys.items() if k.get(str(t)) is not None}
        if len(set(vals.values())) > 1:
            diff.append(t)
    passed = same_init and not diff and len(keys) >= 2
    return dict(passed=passed, initial_equal=same_init, setups=sorted(records), key_steps=steps, key_differs_at=diff)


S2_REFUSAL_NOTE = ("a refresh could not certify (R = +inf): the weights-of-1 run carries IW's refusal rule (most "
                   "caution: dose 1.5, IQL weight 1 on positive advantages) where standard BCA gives dose 1 (IQL weight "
                   "0), so S2 differs by design after such a refresh, not because the weights path is wrong (PRESPEC "
                   "section 17); the driver still stops")


def check_s2(standard, iw_ones):
    """S2 (plan 6.1): IW with every weight 1 equals standard BCA in every bit after at least one refresh: the final key,
    learner state and BCA state (all but the carried refusal rule, the one intended difference), every per-step output,
    and at each refresh R(1) (the IW table's one grid point) equals the stored standard threshold in float32. When a
    refresh could not certify, the report says that S2 then differs by design (S2_REFUSAL_NOTE)."""
    fs, fi = standard["hashes"]["final"], iw_ones["hashes"]["final"]
    same = dict(key=fs["key"] == fi["key"], agent_state=fs["agent_state"] == fi["agent_state"],
                bca_state=fs["bca_state_without_refusal_rule"] == fi["bca_state_without_refusal_rule"],
                outputs=fs["losses_sha256"] == fi["losses_sha256"])
    per = []
    for a, b in zip(standard.get("refreshes") or [], iw_ones.get("refreshes") or []):
        tab = (b.get("iw") or {}).get("table") or {}
        r1 = tab.get("thresholds", [None])[0]
        ok = (a["step"] == b["step"] and a["threshold"] == b["threshold"] and tab.get("grid") == [1.0]
              and r1 is not None and np.float32(float(r1)).tobytes() == np.float32(float(a["threshold"])).tobytes()
              and b.get("s3_reproduced") is True)
        per.append(dict(step=a["step"], standard=a["threshold"], iw_ones=b["threshold"], r_at_1=r1, passed=bool(ok)))
    passed = (all(same.values()) and per and all(p["passed"] for p in per)
              and standard["outcome"] == iw_ones["outcome"] == "complete")
    refused = [r.get("step") for r in (standard.get("refreshes") or [])
               if r.get("certified") is False or r.get("threshold") == "inf"]
    out = dict(passed=bool(passed), equal=same, refreshes=per, uncertified_refreshes=refused)
    if refused:
        out["note"] = S2_REFUSAL_NOTE
    return out


def check_s8(first, repeat):
    """S8 (plan 6.1): two identical runs (the IW setup, seed 0, 15,000 steps) differ in no bit: final key, learner
    state, BCA state, every per-step output, and every refresh's threshold."""
    f1, f2 = first["hashes"]["final"], repeat["hashes"]["final"]
    same = {k: f1[k] == f2[k] for k in ("key", "agent_state", "bca_state", "losses_sha256")}
    pairs = zip(first.get("refreshes") or [], repeat.get("refreshes") or [])
    thr = [a["threshold"] == b["threshold"] for a, b in pairs]
    passed = all(same.values()) and all(thr) and first["outcome"] == repeat["outcome"] == "complete"
    return dict(passed=bool(passed), equal=same, refresh_thresholds_equal=thr)


def snapshot_difference(path_a, path_b, prefixes=("agent/", "rng/")):
    """Leaf-by-leaf comparison of two snapshots' learner state and key (numpy only): leaves compared, leaves that
    differ in any bit, and the largest relative difference max|a - b| / max|b| over the differing leaves."""
    a, _ = H.read(path_a)
    b, _ = H.read(path_b)
    names = sorted(n for n in a if n.startswith(prefixes))
    differ, worst = [], 0.0
    for n in names:
        x, y = a[n], b.get(n)
        if y is None or x.dtype != y.dtype or x.shape != y.shape:
            differ.append(n)
            worst = math.inf
            continue
        if x.tobytes() != y.tobytes():
            differ.append(n)
            if x.dtype.kind == "f":
                d = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                s = float(np.max(np.abs(y.astype(np.float64))))
                worst = max(worst, d / s if s > 0 else math.inf)
            else:
                worst = math.inf
    return dict(leaves=len(names), differing=len(differ), max_relative=worst, first_differing=differ[:5])


def check_s10(alone, bca, readout_only):
    """S10 (plan 6.1): at outer step 10,000, just before the first refresh, the BCA setup's learner state (Unifloral's
    parameters, optimizer states) and key carry equal the learner alone's in every bit. With choice 19 (test (a2)
    matched only to rounding on the card) this is a readout: reported with the snapshots' leaf differences, not a
    stop."""
    a, b = alone["hashes"].get("params_10000"), bca["hashes"].get("params_10000")
    if a is None or b is None:
        return dict(passed=readout_only, readout=readout_only, note="no hash at outer step 10,000 (a run stopped)")
    equal = a["agent_state"] == b["agent_state"] and a["key"] == b["key"]
    out = dict(equal=bool(equal), readout=bool(readout_only), step=a["step"],
               parts_equal={k: a["parts"][k] == b["parts"].get(k) for k in a.get("parts", {})})
    if not equal:
        sa = (alone.get("snapshots") or {}).get(str(a["step"]))
        sb = (bca.get("snapshots") or {}).get(str(b["step"]))
        if sa and sb:
            out["difference"] = snapshot_difference(sa["path"], sb["path"])
    out["passed"] = bool(equal or readout_only)
    return out


# S9 (plan 6.1): the environment signature and its check live in runner.py, which also checks a real run before its
# first chunk; the driver checks every card run's record again against the same frozen signature.
env_signature = RN.env_signature
check_s9 = RN.check_s9


def s7_result(run_dir, rec):
    """A finished run's S7 verdict from its lane's s7.json (lanes/s7-step<final>/s7.json): passed only when the file
    verifies and says s7_passed true. A run that did not complete has no S7 lane (its record says why)."""
    if rec.get("outcome") != "complete":
        return dict(passed=True, note="no S7 lane: the run did not complete (its record says why)")
    path = Path(run_dir) / "lanes" / f"s7-step{rec['timings']['outer_steps']}" / "s7.json"
    try:
        body = H.read_json(path)
    except (H.HandoffError, OSError, ValueError) as error:
        return dict(passed=False, file=str(path), note=f"no verified s7.json ({error})")
    return dict(passed=body.get("s7_passed") is True, file=str(path), s7_passed=body.get("s7_passed"))


def _lane_step(path):
    """The actor step of an S7 lane's file (its folder lanes/s7-step<T>), or None for another path."""
    name = Path(str(path)).parent.name
    return int(name[len("s7-step"):]) if name.startswith("s7-step") and name[len("s7-step"):].isdigit() else None


# ------------------------------------------------------------------------------------------------ lanes (scoring side)

def lane_permission(made_up, purpose):
    """Made-up lanes (the dry run) use guard.MADE_UP, which evaluate accepts only for actor files marked made-up (and,
    for anything that rolls an actor out, never trained); real lanes need the user's verified approval."""
    return guard.MADE_UP if made_up else gate.require_run_approval(purpose)


def lane_main(argv):
    """python -m experiments.shift_hosts.stages lane <kind> ...: one scoring-environment step of a lane job."""
    p = argparse.ArgumentParser(prog="stages lane")
    sub = p.add_subparsers(dest="kind", required=True)
    s = sub.add_parser("score")
    s.add_argument("--actor", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--keep-viewer", action="store_true")
    t = sub.add_parser("transitions")
    t.add_argument("--actor", required=True)
    t.add_argument("--prepared", required=True)
    t.add_argument("--out-actor", required=True)
    t.add_argument("--out-visited", required=True)
    o = sub.add_parser("s7-states")
    o.add_argument("--actor", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--made-up-states", help="dry run: a made-up prepared file whose strong-law states are used")
    c = sub.add_parser("s7-check")
    c.add_argument("--states", required=True)
    c.add_argument("--commands", required=True)
    c.add_argument("--actor", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--no-closed-loop", action="store_true", help="dry run: no rollout of a learned policy")
    c.add_argument("--inject-failure", action="store_true", help="dry run only: write a failed check (S7's tests)")
    for q in (s, t, o, c):
        q.add_argument("--made-up", action="store_true")
    a = p.parse_args(argv)
    permission = lane_permission(a.made_up, f"scoring lane {a.kind} of {getattr(a, 'actor', '')}")   # first
    if getattr(a, "inject_failure", False) and not a.made_up:
        raise gate.ApprovalError("an S7 failure is injected in dry runs only (--made-up)")
    from experiments.shift_hosts import evaluate as E

    if a.kind == "score":
        result = E.score_actor(a.actor, permission, keep_viewer=a.keep_viewer, out=a.out)
        bad = sorted(b for (readout, b), sc in result.items() if readout == "scored" and not sc.summary.get("finite"))
        if bad:                                     # stop rule S5's non-finite return: the driver decides (module doc)
            print(f"NONFINITE: a non-finite return on bank(s) {bad} (stop rule S5); the scores are written", flush=True)
            return EXIT_NONFINITE
        return 0
    if a.kind == "transitions":
        from experiments.shift_hosts import actors_f32 as A32
        from experiments.shift_hosts import policies as P

        arr, _ = H.read(a.prepared, names=["test_strong/state", "test_strong/episode", "test_strong/t"])
        actor = A32.read_actor_file(a.actor)
        eps = None
        if actor.learner == "cql":      # the strong bank's declared noise at the test step's own episode and step
            ep, t = arr["test_strong/episode"].astype(np.int64), arr["test_strong/t"].astype(np.int64)
            eps = P.cql_noise("strong")[ep, t]
        E.actor_transitions(a.actor, arr["test_strong/state"], a.out_actor, permission, eps=eps)
        E.visited_transitions(a.actor, a.out_visited, permission)
        return 0
    if a.kind == "s7-states":
        if a.made_up_states:
            if not a.made_up:
                raise gate.ApprovalError("made-up states go with --made-up")
            arr, man = H.read(a.made_up_states, names=["test_strong/state"])
            if not (man.get("meta") or {}).get("made_up"):
                raise gate.ApprovalError(f"{a.made_up_states} is not a made-up file")
            states = arr["test_strong/state"]
            noise = np.random.default_rng([B_MASTER, 1004, 99]).standard_normal((len(states), 4)).astype(np.float32)
            E.open_loop_states(a.actor, a.out, permission, banks=(), extra_states=states, extra_eps=noise)
        else:
            E.open_loop_states(a.actor, a.out, permission)
        return 0
    res = E.open_loop_compare(a.states, a.commands, a.actor)
    res["closed_loop"] = None if a.no_closed_loop else E.closed_loop_check(a.actor, permission)
    passed = bool(res["passed"]) and (res["closed_loop"] is None or bool(res["closed_loop"]["passed"]))
    if a.inject_failure:
        passed, res["injected_failure"] = False, True
    res["s7_passed"] = passed
    H.write_json(a.out, res)
    return 0 if passed else 3


B_MASTER = 20261016


# ------------------------------------------------------------------------------------------------ jobs

@dataclass
class CardJob:
    label: str              # the run id
    quality: str            # the plan's quality name (budget, caps)
    rep: int
    learner: str
    setup: str
    stage: str
    steps: int
    run_dir: Path
    argv: list
    unit: str               # stage1 / sham / seed<k> / room
    result: dict = None


@dataclass
class LaneJob:
    label: str
    kind: str               # score / coverage / s7 / viewer
    quality: str
    commands: list          # [(argv, env, python name)], run one after another on one core
    outputs: list           # files whose sidecars verify when the job is done (skip on resume)
    result: dict = None
    step_index: int = 0
    run: str = None         # the run id whose actor the lane reads
    setup: str = None       # that run's setup (S5's verdict on a non-finite return depends on it)
    learner: str = None     # that run's learner (version 3's S7 rule stops a learner or its IW setup)


@dataclass
class Config:
    root: Path
    dry: bool
    learners: tuple
    qualities: tuple
    seeds: tuple
    card_device: str
    card_cores: list
    lane_cores: list
    choice19: dict
    stage1_steps: int
    steps: int
    schedule_args: list
    refreshes_per_run: int
    stage1_refreshes: int
    accept_s5: tuple = ()
    inject: dict = field(default_factory=dict)
    replace_incomplete: tuple = ()          # run ids the operator named (--replace-incomplete RUN_ID ...)
    reference_ms: dict = field(default_factory=lambda: dict(B.REFERENCE_MS))
    s9_signature: dict = None               # real mode: prespec.json's environment.s9_signature
    plan: dict = None                       # version 4: {quality: {learner: [setups]}} stage 2 trains (None: the grid)
    s7_all_seeds: bool = False              # version 4 (D3): S7 on every seed's final actor, not only seed 0's
    s7_half: bool = False                   # version 4 (D3): and on the 50% actor (the 50,000-step one on real data)
    supplement: dict = None                 # version 4, real mode: prespec.json's frozen "supplement" block

    @property
    def mode(self):
        return "dry-run" if self.dry else "real"


def config_record(cfg):
    """The configuration a driver's state records (Modes, module docstring); a resume must match it (version 4: its
    plan and S7 lanes included)."""
    plan = None if cfg.plan is None else {q: {l: list(s) for l, s in by.items()} for q, by in cfg.plan.items()}
    return dict(mode=cfg.mode, learners=list(cfg.learners), qualities=list(cfg.qualities), seeds=list(cfg.seeds),
                card_device=cfg.card_device, choice19=dict(cfg.choice19), stage1_steps=int(cfg.stage1_steps),
                steps=int(cfg.steps), schedule_args=list(cfg.schedule_args),
                refreshes_per_run=int(cfg.refreshes_per_run), stage1_refreshes=int(cfg.stage1_refreshes), plan=plan,
                s7_all_seeds=bool(cfg.s7_all_seeds), s7_half=bool(cfg.s7_half))


# ------------------------------------------------------------------------------------------------ the driver

class Halt(Exception):
    pass


class Interrupted(Exception):
    """SIGTERM reached the driver."""


def _on_sigterm(signum, frame):
    raise Interrupted(f"signal {signum}")


class Driver:
    def __init__(self, cfg, approval):
        self.cfg, self.approval = cfg, approval
        self.mode = cfg.mode
        self.dir = cfg.root / "driver"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        try:
            self.ledger = B.Ledger(self.dir / "ledger.json", mode=self.mode)
        except ValueError as error:
            raise gate.ApprovalError(f"refused: {error}") from error
        self.session = B.session_id()
        self.notes = [self.ledger.note] if self.ledger.note else []
        state_path = self.dir / "state.json"
        if state_path.exists() or B.kept_path(state_path).exists():
            self.state, note = B.read_json_kept(state_path)
            if note:
                self.notes.append(note)
            have = self.state.get("mode")
            if (have is None and self.mode == "real") or (have is not None and have != self.mode):
                raise gate.ApprovalError(f"refused: {state_path} is a {have or 'mode-less'} driver's state; a {self.mode} "
                                         "driver never uses it")
            self.state["mode"] = self.mode
        else:
            self.state = dict(schema=STATE_SCHEMA, mode=self.mode, decisions={}, checks={}, claim0_failures=[],
                              card_walls={}, s9_reference=None, lanes_done=[], accepted_s5=[], s7_stopped=[],
                              s7_results=[], config=None)
        self.state.setdefault("accepted_s5", [])
        self.state.setdefault("s7_stopped", [])
        self.state.setdefault("s7_results", [])     # version 4: every S7 lane's verdict (_record_s7)
        if cfg.supplement is not None:              # version 4: the frozen block it runs (never version 3's entries)
            self.state.setdefault("supplement", S.provenance(cfg.supplement, cfg.plan))
        want = config_record(cfg)
        if self.state.get("config") is None:
            self.state["config"] = want
        elif self.state["config"] != want:
            raise gate.ApprovalError(f"refused: this root's driver started with another configuration "
                                     f"({json.dumps(self.state['config'], sort_keys=True)}); a resume must use the same "
                                     f"one, not {json.dumps(want, sort_keys=True)}")
        self.card_queue, self.lane_queue = deque(), deque()
        self.cards, self.lanes = {}, {}          # pid -> (job, popen, core, start)
        self.open = {}                           # pid -> (open card entry or None, open processor entry)
        self.free_card, self.free_lane = list(cfg.card_cores), list(cfg.lane_cores)
        self.cap = 1
        self.halt = None
        self.events = deque(maxlen=30)
        self.phase = "starting"
        self.quality = "medium-expert"           # the quality the driver's own processor time is counted against
        # version 4 drops nothing: cards_dropped and lanes_dropped stay 0 (kept so the status file's schema stays)
        self.counts = dict(cards_done=0, cards_skipped=0, lanes_done=0, lanes_skipped=0, cards_dropped=0,
                           lanes_dropped=0)
        self.started_utc = utc_now()
        self._last_status = -math.inf
        self._last_beat = time.monotonic()
        self._cpu_mark = B.own_processor_seconds()
        orphaned = self.ledger.orphan_open_entries(self.session)
        if orphaned:
            self.notes.append(f"{orphaned} ledger entries of children running when an earlier driver ended stay "
                              "counted to their last heartbeat")
        self.save_state()                                # the mode and the configuration are on disk from the start
        self.save_ledger()

    # -------------------------------------------------------------------------------------------- bookkeeping

    def log(self, msg):
        line = f"[{utc_now()}] {msg}"
        self.events.append(line)
        with open(self.dir / "driver.log", "a", encoding="utf8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def save_state(self):
        B.write_json_kept(self.dir / "state.json", self.state)

    def save_ledger(self):
        """The ledger, with the driver's own processor time since the last save added (budget.py, processor-hour)."""
        now = B.own_processor_seconds()
        delta, self._cpu_mark = max(0.0, now - self._cpu_mark), now
        self.ledger.add_to(self.quality, f"driver/{self.session}", delta, "driver")
        self.ledger.save()

    def heartbeat(self, now=None):
        """Move the open ledger entries of the running children forward (their card interval's end, their processor
        time so far), so a child that outlives this driver is still counted (budget.py, crash safety)."""
        now = time.monotonic() if now is None else now
        for pid, (ce, pe) in self.open.items():
            if ce is not None:
                ce["end"] = float(now)
            secs = B.child_processor_seconds(pid)
            if secs is not None:
                pe["seconds"] = float(secs)
        self._last_beat = now
        self.save_ledger()

    def status(self, final=None):
        now = time.monotonic()
        body = dict(
            schema=STATUS_SCHEMA, mode=self.mode,
            state=final or ("stopping" if self.halt else "running"), phase=self.phase, pid=os.getpid(),
            started_utc=self.started_utc, updated_utc=utc_now(), concurrency=self.cap,
            card=dict(running=[dict(label=j.label, core=c, seconds=round(now - t0, 1)) for j, _, c, t0 in
                               self.cards.values()], queued=len(self.card_queue)),
            lanes=dict(running=[dict(label=j.label, kind=j.kind, core=c, seconds=round(now - t0, 1)) for j, _, c, t0 in
                                self.lanes.values()], queued=len(self.lane_queue)),
            counts=self.counts, budget=B.describe(self.ledger), decisions=self.state.get("decisions"),
            checks={k: v.get("passed") for k, v in self.state.get("checks", {}).items()},
            claim0_failures=self.state.get("claim0_failures"), accepted_s5=self.state.get("accepted_s5"),
            s7_stopped=self.state.get("s7_stopped"), s7_results=len(self.state.get("s7_results") or []),
            supplement=self.state.get("supplement"), stop=self.halt, notes=self.notes, last_events=list(self.events))
        H.write_json(self.dir / "status.json", body)

    def stop(self, rule, reason, **detail):
        if self.halt is None:
            self.halt = dict(rule=rule, reason=reason, at_utc=utc_now(), phase=self.phase, **H.plain(detail))
            self.log(f"STOP ({rule}): {reason}")

    # -------------------------------------------------------------------------------------------- S7 and the plan

    def in_plan(self, quality, learner, setup):
        """Version 4: whether this quality's stage 2 trains this learner's setup (the frozen supplement plan,
        prespec.json "supplement"; a dry run without --supplement-plan trains the full grid)."""
        plan = self.cfg.plan
        return plan is None or setup in (plan.get(quality) or {}).get(learner, ())

    def s7_failed(self, quality, learner, setup, run, file, label):
        """Version 4's S7 rule (module docstring; PRESPEC.md section 21): the float32 check of `run`'s actor failed (its
        verified s7.json says so). It is recorded once in state.json s7_stopped with its scope (S7_SCOPE: the learner,
        or only its IW setup) and version 4, nothing is dropped, and the driver goes on. No judged result changes:
        version 3's judgment is carried (of a scope it stopped, and of a learner it judged, whose comparator or stage-1
        actor this may be), and the supplement's numbers are shown, never judged. A setup without a scope still halts.
        S7's limit is unchanged; nothing is retried."""
        scope = S7_SCOPE.get(setup)
        if scope is None or learner not in LEARNERS:     # no scope for it: the old rule, a halt
            return self.stop("S7", f"{label}: the float32 check failed (see its s7.json)", lane=label)
        entry = dict(quality=quality, learner=learner, scope=scope, setup=setup, run=run)
        if not any(all(e.get(k) == v for k, v in entry.items()) for e in self.state["s7_stopped"]):
            self.state["s7_stopped"].append(dict(entry, stage=str(run).split("/")[0], file=str(file),
                                                 step=_lane_step(file), version=4, at_utc=utc_now()))
            self.save_state()
            what = f"learner {learner}" if scope == "learner" else f"{learner}'s IW setup"
            self.log(f"S7 (version 4): {label}: the float32 check failed (see its s7.json); recorded with its scope "
                     f"({what} on {quality}); nothing is dropped; no judged result changes (version 3's are carried), "
                     f"and the supplement's numbers are shown, never judged")
        return None

    def _record_s7(self, job, body):
        """Version 4: an S7 lane's verdict, from its verified s7.json, upserted into state.json s7_results by lane label
        (a resumed driver reading it again adds nothing): the run, quality, learner, setup, seed, the actor's step, the
        verdict and the open loop's, the closed loop's largest and mean per-start points per readout and bank
        (evaluate.closed_loop_check), and the file with its SHA-256."""
        path = Path(job.outputs[0])
        sha = H.sha256_file(path)
        results = self.state.setdefault("s7_results", [])
        old = next((i for i, r in enumerate(results) if r.get("label") == job.label), None)
        if old is not None and results[old].get("sha256") == sha:
            return results[old]
        closed = body.get("closed_loop")
        seed = str(job.run or "").split("/")[2] if len(str(job.run or "").split("/")) == 4 else ""
        entry = dict(label=job.label, run=job.run, quality=job.quality, learner=job.learner, setup=job.setup,
                     seed=int(seed[4:]) if seed.startswith("seed") and seed[4:].isdigit() else None,
                     step=_lane_step(path), s7_passed=body.get("s7_passed"), open_loop_passed=body.get("passed"),
                     open_loop_max_abs={r: body[r].get("max_abs") for r in ("scored", "tanh_mean")
                                        if isinstance(body.get(r), dict)},
                     closed_loop=None if not isinstance(closed, dict) else {
                         k: {f: v.get(f) for f in ("max_points", "mean_points", "max_return", "passed")}
                         for k, v in closed.items() if "/" in k and isinstance(v, dict)},
                     file=str(path), sha256=sha, at_utc=utc_now())
        if old is None:
            results.append(entry)
        else:
            results[old] = entry
        self.save_state()
        return entry

    def verify_approval(self, where):
        """S1 before a unit (real mode): the approval must still verify from disk (gate.verify); else halt."""
        if self.cfg.dry:
            return
        try:
            gate.verify(self.approval)
        except gate.ApprovalError as error:
            self.stop("S1", f"before {where}: the approval no longer verifies ({error})")
            self.wait()

    # -------------------------------------------------------------------------------------------- environments

    def card_env(self):
        env = UI.launch_env(dict(os.environ))
        if self.cfg.card_device == "gpu":
            env.pop("JAX_PLATFORMS", None)
            env.update(SHIFT_HOSTS_DEVICE="gpu", XLA_PYTHON_CLIENT_PREALLOCATE="false")
        else:
            env.update(SHIFT_HOSTS_DEVICE="cpu", JAX_PLATFORMS="cpu")
        return env

    def training_cpu_env(self):
        env = UI.launch_env(dict(os.environ))
        env.update(SHIFT_HOSTS_DEVICE="cpu", JAX_PLATFORMS="cpu")
        return env

    def scoring_env(self):
        env = dict(os.environ)
        env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", JAX_PLATFORMS="cpu", SHIFT_HOSTS_DEVICE="cpu")
        return env

    # -------------------------------------------------------------------------------------------- processes

    def _spawn(self, label, argv, env, core):
        log_path = self.dir / "logs" / (label.replace("/", "__") + ".log")
        with open(log_path, "ab") as log:
            log.write(f"\n[{utc_now()}] core {core}: {' '.join(argv)}\n".encode())
            log.flush()
            return subprocess.Popen(["taskset", "-c", str(core)] + list(argv), cwd=str(RN.REPO), env=env,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def _reap(self):
        while True:
            try:
                pid, status, ru = os.wait4(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                return
            code = os.waitstatus_to_exitcode(status)
            cpu = ru.ru_utime + ru.ru_stime
            end = time.monotonic()
            ce, pe = self.open.pop(pid, (None, None))
            if pid in self.cards:
                job, popen, core, start = self.cards.pop(pid)
                popen.returncode = code
                self.free_card.append(core)
                self._card_done(job, code, cpu, start, end, ce, pe)
            elif pid in self.lanes:
                job, popen, core, start = self.lanes.pop(pid)
                popen.returncode = code
                self._lane_step_done(job, code, cpu, core, start, end, pe)

    def _launch_card(self, job):
        core = self.free_card.pop(0)
        popen = self._spawn(job.label, job.argv, self.card_env(), core)
        start = time.monotonic()
        self.cards[popen.pid] = (job, popen, core, start)
        self.open[popen.pid] = (
            self.ledger.add_card(job.quality, job.label, self.session, start, start, open=True),
            self.ledger.add_processor(job.quality, job.label, 0.0, "card", open=True, session=self.session))
        self.save_ledger()
        self.log(f"card {job.label} on core {core}")

    def _launch_lane_step(self, job, core):
        argv, env_name = job.commands[job.step_index]
        env = dict(training=self.training_cpu_env, scoring=self.scoring_env)[env_name]()
        popen = self._spawn(job.label, argv, env, core)
        self.lanes[popen.pid] = (job, popen, core, time.monotonic())
        self.open[popen.pid] = (None, self.ledger.add_processor(job.quality, job.label, 0.0, f"lane:{job.kind}",
                                                                open=True, session=self.session))

    def _outputs_done(self, outputs):
        for f in outputs:
            try:
                H.verify_sidecar(f)
            except (H.HandoffError, OSError):
                return False
        return bool(outputs)

    def pump(self):
        self._reap()
        if (self.dir / "STOP_REQUEST").exists():
            self.stop("operator", "a stop was requested (driver/STOP_REQUEST)")
        if self.halt is None:
            while self.card_queue and len(self.cards) < self.cap and self.free_card:
                self._launch_card(self.card_queue.popleft())
            while self.lane_queue and self.free_lane and self.halt is None:
                job = self.lane_queue.popleft()
                if self._outputs_done(job.outputs):
                    job.result = dict(status="skipped")
                    self.counts["lanes_skipped"] += 1
                    self._verdict(job)              # a lane's own verdict is read again on resume (module docstring)
                    continue
                self._launch_lane_step(job, self.free_lane.pop(0))
        now = time.monotonic()
        if now - self._last_beat >= HEARTBEAT_EVERY:
            self.heartbeat(now)
        if now - self._last_status >= STATUS_EVERY:
            self.status()
            self._last_status = time.monotonic()
        time.sleep(POLL)

    def wait(self, jobs=None, lanes=False):
        """Pump until the given card jobs are done (or, lanes=True, until no lane is queued or running); on a halt,
        until no child is running."""
        while True:
            pending = [j for j in (jobs or []) if j.result is None]
            busy = bool(self.lane_queue or self.lanes) if lanes else False
            if self.halt is not None:
                self.card_queue.clear()
                self.lane_queue.clear()
                if not self.cards and not self.lanes:
                    raise Halt()
            elif not pending and not busy:
                return
            self.pump()

    def _drain(self, timeout):
        """After an error or a signal: launch nothing, reap the running children for up to `timeout` seconds. Returns
        the labels still running."""
        self.card_queue.clear()
        self.lane_queue.clear()
        deadline = time.monotonic() + timeout
        while (self.cards or self.lanes) and time.monotonic() < deadline:
            try:
                self._reap()
                self.status()
            except Exception as error:                  # noqa: BLE001  (the stop report must still be written)
                self.log(f"while draining children: {error!r}")
                break
            time.sleep(POLL)
        left = [dict(label=j.label, pid=pid) for pid, (j, _, _, _) in list(self.cards.items()) + list(self.lanes.items())]
        try:
            self.heartbeat()
        except Exception as error:                      # noqa: BLE001
            self.log(f"final heartbeat failed: {error!r}")
        return left

    # -------------------------------------------------------------------------------------------- card runs

    def card_job(self, quality, rep, learner, setup, stage, steps, unit):
        cfg = self.cfg
        label_q = f"made-up-{quality}" if cfg.dry else quality
        run_setup = "iw" if setup == "iw_ones" else setup
        argv = [TRAINING_PYTHON, "-m", "experiments.shift_hosts.runner", "run", "--learner", learner, "--setup",
                run_setup, "--stage", stage, "--out-root", str(cfg.root), "--device", cfg.card_device,
                "--steps", str(steps)]
        if setup == "iw_ones":
            argv.append("--weights-one")
        if cfg.dry:
            argv += ["--made-up-file", str(self.made_up_file(quality, rep))] + list(cfg.schedule_args)
        else:
            argv += ["--quality", quality, "--seed", str(rep)]
        rid = RN.run_id(stage, label_q, rep, learner, setup)
        if rid in tuple(cfg.replace_incomplete or ()):
            argv.append("--replace-incomplete")         # only the runs the operator named
        if rid in cfg.inject and cfg.inject[rid] != S7_INJECT:     # an S7 failure is the lane's, not the runner's
            argv += ["--inject-fault", cfg.inject[rid]]
        return CardJob(rid, quality, rep, learner, setup, stage, steps,
                       RN.run_dir(cfg.root, stage, label_q, rep, learner, setup), argv, unit)

    def made_up_file(self, quality, rep):
        return self.cfg.root / "data" / quality / f"seed{rep}.npz"

    def run_cards(self, jobs, concurrency):
        self.cap = max(1, min(concurrency, len(self.cfg.card_cores)))
        self.card_queue.extend(jobs)
        self.wait(jobs)

    def _status_line(self, job):
        """The runner's last JSON status line in the job's log (status complete / stopped / skipped; a stop before
        training also names its rule there, with record false), or None."""
        log_path = self.dir / "logs" / (job.label.replace("/", "__") + ".log")
        try:
            for line in reversed(log_path.read_text(encoding="utf8", errors="replace").splitlines()[-40:]):
                if line.startswith("{") and '"status"' in line:
                    return json.loads(line)
        except (OSError, ValueError):
            pass
        return None

    def _has_record(self, job):
        try:
            read_record(job.run_dir)
            return True
        except Exception:                               # noqa: BLE001  (no record, or one that does not verify)
            return False

    def _card_done(self, job, code, cpu, start, end, ce=None, pe=None):
        line = self._status_line(job) or {}
        status = line.get("status")
        job.result = dict(exit=code, status=status, processor_seconds=cpu, start=start, end=end,
                          wall=end - start)
        self.counts["cards_done"] += 1
        if pe is None:
            pe = self.ledger.add_processor(job.quality, job.label, 0.0, "card")
        pe["seconds"] = float(cpu)
        pe.pop("open", None)
        if status != "skipped":
            if ce is None:
                ce = self.ledger.add_card(job.quality, job.label, self.session, start, end)
            ce["end"] = float(end)
            ce.pop("open", None)
            self.state["card_walls"][job.label] = dict(wall=end - start, session=self.session, start=start, end=end,
                                                       processor_seconds=cpu)
        else:
            if ce is not None and ce in self.ledger.body["card"]:
                self.ledger.body["card"].remove(ce)     # a skipped run's process never touched the card
            self.counts["cards_skipped"] += 1
        self.save_ledger()
        self.save_state()
        self.log(f"card {job.label} exit {code} ({status}), {end - start:.1f}s wall, {cpu:.1f} processor-s")
        if code == RN.EXIT_REFUSED:
            return self.stop("S1", f"{job.label} was refused (the approval no longer matches the code or settings, "
                                   "or its folder holds another run): see its log", run=job.label)
        if code not in (RN.EXIT_COMPLETE, RN.EXIT_STOPPED):
            return self.stop("error", f"{job.label} failed without a stop rule (exit {code}); nothing retries it: "
                                      "see its log", run=job.label)
        if code == RN.EXIT_STOPPED and line.get("record") is False:
            st = line.get("stop") or {}
            return self.stop(st.get("rule") or "error", f"{job.label} stopped before training: {st.get('reason')}",
                             run=job.label)
        try:
            rec = read_record(job.run_dir)
        except Exception as error:                      # noqa: BLE001  (any unreadable record halts)
            return self.stop("error", f"{job.label}: its record does not verify ({error})", run=job.label)
        job.result["record"] = rec
        if status == "skipped":
            self._recover(job, rec)
        if not self.cfg.dry and (rec.get("code") or {}).get("fingerprint") != self.approval.fingerprint:
            return self.stop("S1", f"{job.label}: its record names fingerprint "
                                   f"{str((rec.get('code') or {}).get('fingerprint'))[:16]}, not the approved "
                                   f"{self.approval.fingerprint[:16]}", run=job.label)
        if rec["outcome"] == "stopped":
            rule = rec["stop"]["rule"]
            if rule == "S5" and job.setup in ("alone", "room"):
                entry = dict(run=job.label, step=rec["stop"]["step"], reason=rec["stop"]["reason"])
                if entry not in self.state["claim0_failures"]:
                    self.state["claim0_failures"].append(entry)
                    self.save_state()
                self.log(f"{job.label}: S5 on Unifloral's own step is that learner's Claim 0 outcome, recorded")
            elif rule == "S5" and job.label in self.cfg.accept_s5:
                self._accept_s5(job.label)
            else:
                return self.stop(rule, f"{job.label} stopped: {rec['stop']['reason']} (outer step "
                                       f"{rec['stop']['step']})", run=job.label,
                                 record=str(job.run_dir / "record.json"))
        self._check_s9(job, rec)
        if job.stage != "stage1" and job.setup == "alone" and status != "skipped":
            self._check_s6_alone(job, rec)
        if self.halt is None:                           # version 4: every completed run gets its lanes
            for lane in self.lanes_for(job, rec):
                self.lane_queue.append(lane)

    def _accept_s5(self, run):
        self.log(f"{run}: S5 accepted by the operator as the setup's outcome (--accept-s5)")
        if run not in self.state["accepted_s5"]:
            self.state["accepted_s5"].append(run)
            self.save_state()

    def _recover(self, job, rec):
        """A run found complete (or stopped) on resume: if no closed ledger entry counts it (it finished while no driver
        watched it), add what its record says it used (budget.Ledger.recover), and its stage-1 wall time."""
        t = rec.get("timings") or {}
        got = self.ledger.recover(job.quality, job.label, t.get("wall_seconds"), t.get("processor_seconds"))
        if got is not None:
            self.log(f"{job.label}: it finished while no driver watched it; recovered {got[0]:.1f} card-s and "
                     f"{got[1]:.1f} processor-s from its record into the ledger")
            self.save_ledger()
        if job.label not in self.state["card_walls"]:
            self.state["card_walls"][job.label] = dict(wall=float(t.get("wall_seconds") or 0.0),
                                                       session=f"recovered/{job.label}", start=None, end=None,
                                                       processor_seconds=float(t.get("processor_seconds") or 0.0),
                                                       recovered=True)
            self.save_state()

    # -------------------------------------------------------------------------------------------- per-run checks

    def _s9_reference(self, rec):
        """The frozen environment S9 compares with. Real mode: always prespec.json's environment.s9_signature (the
        driver refused to start without a complete one). Dry run: the first card run's own signature (a processor run
        is not the frozen card environment), recorded as such."""
        if not self.cfg.dry:
            ref = self.cfg.s9_signature
            if self.state.get("s9_reference") is None or self.state["s9_reference"].get("signature") != ref:
                self.state["s9_reference"] = dict(signature=ref, source="prespec.json environment.s9_signature")
                self.save_state()
            return ref
        if self.state.get("s9_reference") is None:
            self.state["s9_reference"] = dict(signature=env_signature(rec),
                                              source=f"the first card run ({rec['run_id']}); a dry run's processor "
                                                     "runs are not the frozen card environment")
            self.save_state()
        return self.state["s9_reference"]["signature"]

    def _check_s9(self, job, rec):
        ref = self._s9_reference(rec)
        if not ref:
            return self.stop("S9", f"{job.label}: no frozen environment signature to compare with", run=job.label)
        res = check_s9(rec, ref)
        if not res["passed"]:
            self.state["checks"][f"S9/{job.label}"] = res
            self.save_state()
            self.stop("S9", f"{job.label}: its environment differs from the frozen record", **res)

    def _check_s6_alone(self, job, rec):
        dec = self.state["decisions"].get("s6") or {}
        ms1 = ((dec.get("alone_ms_at_concurrency") or {}).get(job.learner))
        ms = rec["timings"].get("steady_ms_per_outer_step")
        if ms1 and ms and ms > B.SLOWDOWN_FACTOR * ms1:
            self.stop("S6", f"{job.label}: {ms:.3f} ms per outer step, more than twice its stage-1 time "
                            f"({ms1:.3f} ms at this concurrency): stop and re-cost", run=job.label)

    # -------------------------------------------------------------------------------------------- lanes

    def lanes_for(self, job, rec):
        """The scoring lanes a finished run needs (module docstring; dry runs: S7's open loop and coverage only).
        Version 4: S7 on the final actor of every seed's alone, standard and IW run with s7_all_seeds (seed 0's only
        without), and on its 50% actor too with s7_half (lanes/s7-step<50%>, the 50,000-step actor on real data)."""
        cfg, out = self.cfg, []
        if rec["outcome"] != "complete":
            return out
        snaps = rec.get("snapshots") or {}
        final = str(rec["timings"]["outer_steps"])
        lanes_dir = job.run_dir / "lanes"
        actor = lambda t: snaps[str(t)]["actor"]["path"]
        made = ["--made-up"] if cfg.dry else []
        bca = job.setup in ("standard", "iw")
        seed0 = job.rep == 0
        if job.stage == "stage1" or ((seed0 or cfg.s7_all_seeds) and job.setup in STAGE2_SETUPS):
            lane = self._s7_lane(job, actor(final), lanes_dir / f"s7-step{final}", made)
            lane.run, lane.setup, lane.learner = job.label, job.setup, job.learner
            out.append(lane)
            half = str(cfg.steps // 2)
            if job.stage != "stage1" and cfg.s7_half and half in snaps and half != final:
                lane = self._s7_lane(job, actor(half), lanes_dir / f"s7-step{half}", made,
                                     label=f"{job.label}/s7-{half}")
                lane.run, lane.setup, lane.learner = job.label, job.setup, job.learner
                out.append(lane)
        if job.stage == "stage1":
            return out
        if not cfg.dry:
            half = str(cfg.steps // 2)
            for t in (half, final):
                if t not in snaps:
                    continue
                keep = seed0 and t == final and job.setup in STAGE2_SETUPS
                score = lanes_dir / f"scores-step{t}.npz"
                cmds = [([SCORING_PYTHON, "-m", "experiments.shift_hosts.stages", "lane", "score", "--actor",
                          actor(t), "--out", str(score)] + (["--keep-viewer"] if keep else []), "scoring")]
                outputs = [str(score)]
                if keep and bca:
                    view = lanes_dir / f"viewer-step{t}.npz"
                    cmds.append(([TRAINING_PYTHON, "-m", "experiments.shift_hosts.band_readout", "viewer", "--run-dir",
                                  str(job.run_dir), "--step", t, "--scores", str(score), "--out", str(view)],
                                 "training"))
                    outputs.append(str(view))
                out.append(LaneJob(f"{job.label}/score-{t}", "score" if len(cmds) == 1 else "viewer", job.quality,
                                   cmds, outputs, run=job.label, setup=job.setup, learner=job.learner))
        if bca:
            last = max(r["step"] for r in rec["refreshes"]) if rec.get("refreshes") else None
            if last is not None:
                cov = lanes_dir / f"coverage-step{last}"
                cmds = []
                extra = []
                if not cfg.dry:
                    ta, tv = cov / "actor_transitions.npz", cov / "visited_transitions.npz"
                    cmds.append(([SCORING_PYTHON, "-m", "experiments.shift_hosts.stages", "lane", "transitions",
                                  "--actor", actor(last), "--prepared", rec["data"]["path"], "--out-actor", str(ta),
                                  "--out-visited", str(tv)], "scoring"))
                    extra = ["--actor-transitions", str(ta), "--visited-transitions", str(tv)]
                else:
                    extra = ["--made-up-file", rec["data"]["path"]]
                cmds.append(([TRAINING_PYTHON, "-m", "experiments.shift_hosts.band_readout", "coverage", "--run-dir",
                              str(job.run_dir), "--step", str(last), "--out", str(cov)] + extra, "training"))
                out.append(LaneJob(f"{job.label}/coverage-{last}", "coverage", job.quality, cmds,
                                   [str(cov / "coverage.json")], run=job.label, setup=job.setup, learner=job.learner))
        return out

    def _s7_lane(self, job, actor, d, made, label=None):
        states, commands, result = d / "states.npz", d / "commands.npz", d / "s7.json"
        st = [SCORING_PYTHON, "-m", "experiments.shift_hosts.stages", "lane", "s7-states", "--actor", actor,
              "--out", str(states)] + made
        if self.cfg.dry:
            st += ["--made-up-states", str(self.made_up_file(job.quality, job.rep))]
        chk = [SCORING_PYTHON, "-m", "experiments.shift_hosts.stages", "lane", "s7-check", "--states", str(states),
               "--commands", str(commands), "--actor", actor, "--out", str(result)] + made
        if self.cfg.dry:
            chk.append("--no-closed-loop")
            if self.cfg.inject.get(job.label) == S7_INJECT:
                chk.append("--inject-failure")
        cmds = [(st, "scoring"),
                ([TRAINING_PYTHON, "-m", "experiments.shift_hosts.runner", "s7-commands", "--states", str(states),
                  "--actor", actor, "--out", str(commands)], "training"),
                (chk, "scoring")]
        return LaneJob(label or f"{job.label}/s7", "s7", job.quality, cmds, [str(result)])

    @staticmethod
    def _is_score_step(argv):
        return len(argv) > 4 and argv[3] == "lane" and argv[4] == "score"

    def _verdict(self, job):
        """A lane's own verdict from its outputs, read when it finishes and again when a resumed driver finds it done:
        S7 (version 4: every verified s7.json is recorded in s7_results; s7_passed false is recorded through s7_failed
        and drops nothing; a file that does not verify, or says neither, halts) and S5's non-finite return (the scores'
        "scored" summaries must all be finite: for an alone or room run that is its learner's Claim 0 outcome,
        recorded; for a BCA setup it halts unless the run was accepted with --accept-s5)."""
        if job.kind == "s7":
            try:
                body = H.read_json(job.outputs[0])
            except (H.HandoffError, OSError, ValueError) as error:
                return self.stop("S7", f"{job.label}: its s7.json does not verify ({error})", lane=job.label)
            self._record_s7(job, body)
            if body.get("s7_passed") is False:
                return self.s7_failed(job.quality, job.learner, job.setup, job.run, job.outputs[0], job.label)
            if body.get("s7_passed") is not True:
                return self.stop("S7", f"{job.label}: its s7.json says neither passed nor failed", lane=job.label)
            return None
        if job.kind not in ("score", "viewer"):
            return None
        try:
            summ = (H.read_manifest(job.outputs[0]).get("meta") or {}).get("summaries", {}).get("scored") or {}
        except (H.HandoffError, OSError, ValueError) as error:
            return self.stop("lane", f"{job.label}: its scores file does not verify ({error})", lane=job.label)
        bad = sorted(b for b, s in summ.items() if not s.get("finite"))
        if not bad:
            return None
        reason = f"a non-finite return on bank(s) {bad} ({Path(job.outputs[0]).name})"
        if job.setup in ("alone", "room"):
            entry = dict(run=job.run, step=None, reason=f"S5: {reason}", source="scores")
            if entry not in self.state["claim0_failures"]:
                self.state["claim0_failures"].append(entry)
                self.save_state()
            self.log(f"{job.run}: {reason}; S5 on Unifloral's own step is that learner's Claim 0 outcome, recorded")
            return None
        if job.run in self.cfg.accept_s5:
            self._accept_s5(job.run)
            return None
        return self.stop("S5", f"{job.run}: {reason}; S5 on a BCA setup halts until it is traced (rerun with "
                               "--accept-s5 RUN_ID --resume-after-stop if it is the setup's outcome)",
                         run=job.run, lane=job.label)

    def _lane_step_done(self, job, code, cpu, core, start, end, pe=None):
        if pe is None:
            pe = self.ledger.add_processor(job.quality, job.label, 0.0, f"lane:{job.kind}")
        pe["seconds"] = float(cpu)
        pe.pop("open", None)
        self.save_ledger()
        argv = job.commands[job.step_index][0]
        score_step = self._is_score_step(argv)
        if code != 0 and not (score_step and code == EXIT_NONFINITE):
            self.free_lane.append(core)
            job.result = dict(status="failed", exit=code, step=job.step_index)
            if job.kind == "s7" and code == 3 and job.step_index == len(job.commands) - 1:
                job.result = dict(status="complete", s7_passed=False)   # the check ran; its verdict is a failure
                self.counts["lanes_done"] += 1
                self.log(f"lane {job.label} done: the float32 check failed")
                self._verdict(job)                      # its s7.json: recorded (version 4), or a halt
                if self.halt is None and not any(r.get("label") == job.label and r.get("s7_passed") is False
                                                 for r in self.state["s7_results"]):
                    self.stop("S7", f"{job.label}: the check exited 3 but its s7.json does not say it failed",
                              lane=job.label)       # the exit status and this lane's own file must agree
                return None
            if job.kind == "s7" and code == 3:
                return self.stop("S7", f"{job.label}: the float32 check failed (see its s7.json)", lane=job.label)
            return self.stop("lane", f"{job.label} failed at its step {job.step_index + 1} (exit {code}); nothing "
                                     "retries it: see its log", lane=job.label)
        if score_step:
            self._verdict(job)                          # S5's non-finite return (exit 4) or a clean score
        if self.halt is not None:                       # a halt launches nothing new, a lane's next step included
            self.free_lane.append(core)
            job.result = dict(status="halted", step=job.step_index)
            return
        job.step_index += 1                             # version 4: a lane of a scope S7 failed goes on too
        if job.step_index < len(job.commands):
            self._launch_lane_step(job, core)
            return
        self.free_lane.append(core)
        job.result = dict(status="complete")
        self.counts["lanes_done"] += 1
        self.log(f"lane {job.label} done")
        if job.kind == "s7":                            # version 4: a passing check is recorded in s7_results when it
            self._verdict(job)                          # ends too (an s7.json that does not verify halts)
            if self.halt is None and any(r.get("label") == job.label and r.get("s7_passed") is False
                                         for r in self.state["s7_results"]):
                self.stop("S7", f"{job.label}: the check exited 0 but its s7.json says it failed",
                          lane=job.label)       # the exit status and this lane's own file must agree, both ways

    def committed_lane_hours(self):
        pending = list(self.lane_queue) + [j for j, _, _, _ in self.lanes.values()]
        return sum(B.lane_cost(self.ledger, j.kind) for j in pending) / 3600.0

    # -------------------------------------------------------------------------------------------- stage 1

    def stage1(self):
        cfg, q = self.cfg, "medium-expert"
        self.quality = q
        self.phase = "stage 1"
        self.verify_approval("stage 1")
        jobs = {}
        for learner in cfg.learners:
            single = [self.card_job(q, 0, learner, s, "stage1", cfg.stage1_steps, "stage1") for s in STAGE1_SINGLE]
            self.run_cards(single, 1)
            pair = [self.card_job(q, 0, learner, s, "stage1", cfg.stage1_steps, "stage1") for s in STAGE1_PAIR]
            self.run_cards(pair, 2)
            for j in single + pair:
                jobs[(learner, j.setup)] = j
        self.phase = "stage 1 lanes (S7)"
        self.wait(lanes=True)
        self.stage1_checks(jobs)

    def stage1_checks(self, jobs):
        cfg = self.cfg
        recs = {k: read_record(j.run_dir) for k, j in jobs.items()}
        checks = {}
        for l in cfg.learners:
            r = {s: recs[(l, s)] for s in STAGE1_SINGLE + STAGE1_PAIR}
            checks[f"S2/{l}"] = check_s2(r["standard"], r["iw_ones"])
            checks[f"S8/{l}"] = check_s8(r["iw"], r["iw_repeat"])
            checks[f"S4/{l}"] = check_s4(r, key_setups=STAGE1_SINGLE + STAGE1_PAIR)
            for s in ("standard", "iw", "iw_ones", "iw_repeat"):
                checks[f"S10/{l}/{s}"] = check_s10(r["alone"], r[s], cfg.choice19.get(l, False))
            for s in STAGE1_SINGLE + STAGE1_PAIR:
                res = checks[f"S7/{l}/{s}"] = s7_result(jobs[(l, s)].run_dir, r[s])
                if res.get("s7_passed") is False:       # version 4: recorded (again, once), nothing dropped, no halt
                    res["scoped_stop"] = S7_SCOPE[s]
                    self.s7_failed(jobs[(l, s)].quality, l, s, jobs[(l, s)].label, res["file"],
                                   f"{jobs[(l, s)].label}/s7")
        s8_ok = all(checks[f"S8/{l}"]["passed"] for l in cfg.learners)
        records = {j.label: H.sha256_file(j.run_dir / "record.json") for j in jobs.values()}
        saved = self.state["decisions"].get("s6")
        if saved and saved.get("stage1_records") == records:
            s6 = saved                  # a resume: stage 1's decision stands; never recomputed against later totals
            self.log(f"S6: stage 1's decision of {saved.get('decided_utc')} is reused (stage 1's records unchanged)")
        else:
            s6 = self.s6(jobs, recs, s8_ok)
            s6.update(stage1_records=records, decided_utc=utc_now())
        checks["S6"] = s6
        self.state["checks"].update(checks)
        self.state["decisions"]["s6"] = s6
        self.state["decisions"]["concurrency"] = s6["concurrency"]
        self.save_state()
        failed = [k for k, v in checks.items() if not v.get("passed") and not v.get("scoped_stop")]
        for k in failed:
            note = checks[k].get("note")
            self.stop(k.split("/")[0], f"stage 1 check {k} failed" + (f" ({note})" if note else ""), check=k,
                      detail=checks[k])
            break
        if self.halt is not None:
            self.wait()
        readout = [l for l in cfg.learners if cfg.choice19.get(l)]
        s10 = f"readout for {', '.join(readout)}" if readout else "bit for bit"
        scoped = sorted({(e["learner"], e["scope"]) for e in self.state["s7_stopped"] if e["stage"] == "stage1"})
        s7 = ("S7" if not scoped else "S7 failed (recorded, nothing dropped; version 4) on medium-expert: "
              + ", ".join(l if sc == "learner" else f"{l}'s IW setup" for l, sc in scoped))
        self.log(f"stage 1 passed: S2, S4, {s7}, S8, S10 ({s10}), S6, S9; concurrency {s6['concurrency']}")

    def _wall(self, job):
        w = self.state["card_walls"].get(job.label)
        return w["wall"] if w else job.result["wall"]

    def _cpu(self, job):
        w = self.state["card_walls"].get(job.label)
        return w.get("processor_seconds") if w else job.result.get("processor_seconds")

    def s6(self, jobs, recs, s8_ok):
        """S6 at the end of stage 1 (module docstring; budget.py)."""
        cfg = self.cfg
        speeds = {}
        for l in cfg.learners:
            for kind in STAGE1_SINGLE:
                speeds[(l, kind)] = B.run_speed(recs[(l, kind)], self._wall(jobs[(l, kind)]),
                                                self._cpu(jobs[(l, kind)]))
        single = {l: self._wall(jobs[(l, "iw")]) for l in cfg.learners}
        pair = {}
        for l in cfg.learners:
            ws = [self.state["card_walls"].get(jobs[(l, s)].label) for s in STAGE1_PAIR]
            if all(w and w.get("start") is not None and w["session"] == ws[0]["session"] for w in ws):
                pair[l] = max(w["end"] for w in ws) - min(w["start"] for w in ws)
            else:
                pair[l] = sum(self._wall(jobs[(l, s)]) for s in STAGE1_PAIR)   # not measured together: no gain
        conc, two = B.two_at_once(single, pair, cfg.stage1_steps, s8_ok)
        slow = [recs[(l, s)]["timings"]["steady_ms_per_outer_step"] / recs[(l, "iw")]["timings"][
            "steady_ms_per_outer_step"] for l in cfg.learners for s in STAGE1_PAIR]
        slowdown = float(np.mean(slow)) if conc == 2 else 1.0
        ms_alone = {l: speeds[(l, "alone")]["ms_per_step"] for l in cfg.learners}
        # each learner alone ran by itself in stage 1, as the reference was measured: no slowdown on either side
        alone_ok, alone_detail = B.s6_alone_check(ms_alone, cfg.reference_ms)
        gain = two["gain"] if conc == 2 else 1.0
        proj = {}
        for qi, q in enumerate(cfg.qualities):
            units = self.plan_units(q)
            card = proc = 0.0
            for _, us in units:
                c, p = B.project_unit(us, speeds, concurrency=conc, gain=gain, ledger=self.ledger)
                card += c
                proc += p
            tot = self.ledger.totals(q)
            total = dict(card_hours=tot["card_hours"] + card, processor_hours=tot["processor_hours"] + proc)
            proj[q] = dict(projected=dict(card_hours=card, processor_hours=proc), so_far=tot, total=total,
                           cap=B.CAPS[q])
            proj[q]["passed"] = (proj[q]["total"]["card_hours"] <= B.CAPS[q]["card_hours"]
                                 and proj[q]["total"]["processor_hours"] <= B.CAPS[q]["processor_hours"])
        passed = alone_ok and all(p["passed"] for p in proj.values())
        return dict(passed=bool(passed), concurrency=conc, two_at_once=two, slowdown=slowdown,
                    speeds={f"{l}/{k}": v for (l, k), v in speeds.items()}, alone=alone_detail,
                    alone_ms_at_concurrency={l: ms_alone[l] * slowdown for l in cfg.learners}, projection=proj)

    # -------------------------------------------------------------------------------------------- stage 2

    def plan_units(self, quality):
        """[(unit name, [run dicts])] of a quality's stage 2 in the plan's order, for projections; each unit's runs are
        in unit_jobs' order. Version 4: only the runs of the supplement plan (in_plan; the full grid without one); the
        unit names stay. S7's lanes: seed 0's final actors, every seed's with s7_all_seeds, and two per run (the 50%
        actor too) with s7_half."""
        cfg = self.cfg
        lanes_bca = dict(score=2, coverage=1)
        lanes_alone = dict(score=2)
        units = []
        live = lambda l, s: self.in_plan(quality, l, s)
        if any(cfg.choice19.get(l) for l in cfg.learners):
            units.append(("sham", [dict(learner=l, setup="sham_alone", steps=cfg.steps, refreshes=0,
                                        lanes=dict(score=2)) for l in cfg.learners
                                   if cfg.choice19.get(l) and live(l, "sham_alone")]))
        for k in cfg.seeds:
            us = []
            for l in cfg.learners:
                for s in STAGE2_SETUPS:
                    if not live(l, s):
                        continue
                    lanes = dict(lanes_bca if s != "alone" else lanes_alone)
                    if k == 0 or cfg.s7_all_seeds:
                        lanes["s7"] = 2 if cfg.s7_half else 1
                    if k == 0 and s != "alone":
                        lanes["viewer"] = 1
                    us.append(dict(learner=l, setup=s, steps=cfg.steps,
                                   refreshes=0 if s == "alone" else cfg.refreshes_per_run, lanes=lanes))
            units.append((f"seed{k}", us))
        units.append(("room", [dict(learner=l, setup="room", steps=cfg.steps, refreshes=0, lanes=dict(score=2))
                               for k in cfg.seeds for l in cfg.learners if live(l, "room")]))
        if cfg.dry:
            for _, us in units:
                for u in us:
                    u["lanes"] = {k: v for k, v in u["lanes"].items() if k in ("s7", "coverage")}
        return units

    def unit_jobs(self, quality, name):
        """The card jobs of a unit, in plan_units' order: version 4, only the runs of the supplement plan (in_plan; the
        full grid without one)."""
        cfg = self.cfg
        live = lambda l, s: self.in_plan(quality, l, s)
        if name == "sham":
            return [self.card_job(quality, 0, l, "sham_alone", "stage2", cfg.steps, name) for l in cfg.learners
                    if cfg.choice19.get(l) and live(l, "sham_alone")]
        if name == "room":
            return [self.card_job(quality, k, l, "room", "stage2", cfg.steps, name) for k in cfg.seeds
                    for l in cfg.learners if live(l, "room")]
        k = int(name[len("seed"):])
        return [self.card_job(quality, k, l, s, "stage2", cfg.steps, name) for l in cfg.learners for s in STAGE2_SETUPS
                if live(l, s)]

    def stage2(self, quality):
        cfg = self.cfg
        self.quality = quality
        dec = self.state["decisions"]
        s6 = dec["s6"]
        speeds = {tuple(k.split("/")): v for k, v in s6["speeds"].items()}
        conc = int(dec["concurrency"])
        gain = s6["two_at_once"]["gain"] if conc == 2 else 1.0
        for name in [n for n, _ in self.plan_units(quality)]:
            self.phase = f"stage 2 {quality} {name}"
            self.verify_approval(f"{quality} {name}")
            units = dict(self.plan_units(quality))[name]
            jobs = self.unit_jobs(quality, name)
            if len(jobs) != len(units):
                raise AssertionError(f"{quality} {name}: {len(jobs)} jobs against {len(units)} planned runs")
            todo = [u for u, j in zip(units, jobs) if not self._has_record(j)]
            if not jobs:                            # version 4: the supplement plan has no run in this unit
                self.state["checks"].setdefault(f"S6b/{quality}/{name}", dict(   # an earlier check stays as it was
                    quality=quality, passed=True, skipped=True, runs_with_records=0,
                    note="nothing to run: this unit has no supplement run (version 4)"))
                self.log(f"S6b before {quality} {name}: skipped, this unit has no supplement run (version 4)")
                self.save_state()
            elif not todo:                          # a resume: every run of the unit already has its record
                key = f"S6b/{quality}/{name}"
                if key not in self.state["checks"]:     # else the check made before the unit ran stays as it was
                    self.state["checks"][key] = dict(
                        quality=quality, passed=True, skipped=True, runs_with_records=len(jobs),
                        note="every run of this unit has a record (a resumed driver): nothing new to project")
                self.log(f"S6b before {quality} {name}: skipped, every run of the unit has a record")
                self.save_state()
            else:
                projected = B.project_unit(todo, speeds, concurrency=conc, gain=gain, ledger=self.ledger)
                ok, detail = B.s6b_check(self.ledger, quality, projected, self.committed_lane_hours())
                detail.update(runs_projected=len(todo), runs_with_records=len(jobs) - len(todo))
                self.state["checks"][f"S6b/{quality}/{name}"] = detail
                self.save_state()
                if not ok:
                    self.stop("S6b", f"before {quality} {name}: the running totals plus its projected cost pass the "
                                     "approved cap; stopping between whole units", detail=detail)
                    self.wait()
            self.run_cards(jobs, conc)
            if name.startswith("seed"):
                self.seed_checks(quality, jobs, name)
            if name == "room":
                self.room_checks(quality, jobs)

    def seed_checks(self, quality, jobs, name):
        by = {}
        for j in jobs:
            by.setdefault(j.learner, {})[j.setup] = read_record(j.run_dir)
        for l, r in by.items():
            if len(r) < 2:                              # a guard: version 4's frozen plan trains three per seed
                self.state["checks"][f"S4/{quality}/{name}/{l}"] = dict(
                    passed=True, skipped=True, setups=sorted(r),
                    note="fewer than two of the learner's runs in this seed's plan (version 4's supplement plan): "
                         "nothing to compare")
                continue
            res = check_s4(r)
            self.state["checks"][f"S4/{quality}/{name}/{l}"] = res
            if not res["passed"]:
                self.stop("S4", f"{quality} {name} {l}: the setups do not share their start or batch stream",
                          detail=res)
        self.save_state()
        if self.halt is not None:
            self.wait()

    def room_checks(self, quality, jobs):
        for j in jobs:                                  # version 4: room runs come with their alone runs in the plan
            alone = RN.run_dir(self.cfg.root, "stage2", j.run_dir.parts[-3], j.rep, j.learner, "alone")
            res = check_s4(dict(alone=read_record(alone), room=read_record(j.run_dir)), key_setups=())
            res["passed"] = res["initial_equal"]
            self.state["checks"][f"S4/{quality}/seed{j.rep}/{j.learner}/room"] = res
            if not res["passed"]:
                self.stop("S4", f"{j.label}: the room reference does not start from its learner's parameters",
                          detail=res)
        self.save_state()
        if self.halt is not None:
            self.wait()

    def s7_seed0_checks(self, quality):
        """After a quality's last lane: every planned seed-0 final actor's S7 (alone, standard, IW) is read again, so a
        failure can never be forgotten by a resume (module docstring). Version 4: a failed check is recorded once
        (s7_failed: nothing is dropped, no halt); a missing or unverified record or result halts. Only these and stage
        1's get check keys (S7/<quality>/seed0/<learner>/<setup>): the S7 results at seeds 1-4 and at the 50% actor are
        in state.json s7_results only."""
        if 0 not in self.cfg.seeds:
            return
        for j in self.unit_jobs(quality, "seed0"):
            key = f"S7/{quality}/seed0/{j.learner}/{j.setup}"
            try:
                rec = read_record(j.run_dir)
            except Exception as error:                  # noqa: BLE001
                self.stop("error", f"{j.label}: its record does not verify ({error})", run=j.label)
                break
            res = s7_result(j.run_dir, rec)
            self.state["checks"][key] = res
            if res.get("s7_passed") is False:
                res["scoped_stop"] = S7_SCOPE[j.setup]
                self.s7_failed(quality, j.learner, j.setup, j.label, res["file"], f"{j.label}/s7")
            elif not res["passed"]:
                self.stop("S7", f"{j.label}: its S7 result did not pass ({res.get('note') or res.get('file')})",
                          run=j.label)
                break
        self.save_state()
        if self.halt is not None:
            self.wait()

    # -------------------------------------------------------------------------------------------- the whole

    def prepare_inputs(self):
        cfg = self.cfg
        if cfg.dry:
            for qi, q in enumerate(cfg.qualities):
                for k in sorted(set(cfg.seeds) | {0}):
                    f = self.made_up_file(q, k)
                    if not f.exists():
                        RN.write_made_up_prepared(f, 1000 * (qi + 1) + k, rep=k, quality=f"made-up-{q}")
        else:
            for q in cfg.qualities:
                for k in sorted(set(cfg.seeds) | {0}):
                    H.verify_sidecar(RN.prepared_path(q, k))

    def run(self):
        cfg = self.cfg
        self.log(f"driver starts ({'dry run' if cfg.dry else 'real'}): learners {list(cfg.learners)}, qualities "
                 f"{list(cfg.qualities)}, seeds {list(cfg.seeds)}, card {cfg.card_device} cores {cfg.card_cores}, "
                 f"lanes {cfg.lane_cores}, choice 19 {cfg.choice19}, replace-incomplete {list(cfg.replace_incomplete)}")
        self.log(f"version 4: stage-2 plan {'the full grid' if cfg.plan is None else json.dumps(cfg.plan)}; S7 on "
                 f"{'every seed' if cfg.s7_all_seeds else 'seed 0'}'s final actors"
                 f"{' and their 50% actors' if cfg.s7_half else ''}; a failed S7 check is recorded and drops nothing")
        for note in self.notes:
            self.log(f"note: {note}")
        previous = signal.signal(signal.SIGTERM, _on_sigterm)
        try:
            self.prepare_inputs()
            self.stage1()
            for q in cfg.qualities:
                self.stage2(q)
                self.phase = f"stage 2 {q}: its last lanes (the next stage starts only if every stop rule passed)"
                self.wait(lanes=True)
                self.s7_seed0_checks(q)
            self.phase = "final lanes"
            self.wait(lanes=True)
        except Halt:
            self.save_ledger()
            self.write_stop()
            self.status("stopped")
            return 3
        except (Exception, KeyboardInterrupt) as error:     # noqa: BLE001  (the stop report must be written)
            interrupted = isinstance(error, (Interrupted, KeyboardInterrupt))
            self.stop("operator" if interrupted else "error",
                      f"the driver {'was interrupted' if interrupted else 'failed'}: {error!r}",
                      traceback=traceback.format_exc()[-4000:])
            left = self._drain(CHILD_WAIT)
            if left and self.halt is not None:
                self.halt["children_still_running"] = left
            self.write_stop()
            self.status("stopped")
            return 3 if interrupted else 1
        finally:
            signal.signal(signal.SIGTERM, previous)
        self.phase = "finished"
        self.save_ledger()
        scoped = self.state.get("s7_stopped") or []
        if scoped:
            what = sorted({f"{e['learner']}{'' if e['scope'] == 'learner' else ' IW'} on {e['quality']}" for e in scoped})
            self.log(f"finished: every unit of every stage ran and no stop rule halted; S7 failures recorded under "
                     f"version 4 (nothing dropped; the supplement's numbers are shown, never judged): "
                     f"{', '.join(what)}")
        else:
            self.log("finished: every unit of every stage ran and every stop rule passed")
        self.status("finished")
        return 0

    def write_stop(self):
        h = dict(self.halt or {}, schema=STOP_SCHEMA, mode=self.mode,
                 budget=B.describe(self.ledger), claim0_failures=self.state.get("claim0_failures"),
                 s7_stopped=self.state.get("s7_stopped"),
                 next_steps=("Nothing was retried. Read the named log or record. A stop that needs a code fix waits "
                             "for a new pre-specification and the user's terminal approval (plan 6.1). An S5 on a BCA "
                             "setup that is not a package defect is that setup's outcome: rerun with --accept-s5 "
                             "RUN_ID --resume-after-stop."))
        H.write_json(self.dir / "STOP.json", h)
        scoped = h.get("s7_stopped") or []
        text = [f"The stage driver stopped ({h.get('rule')}).", f"Why: {h.get('reason')}",
                f"When: {h.get('at_utc')}, during {h.get('phase')}.",
                f"Budget so far: {json.dumps(H.plain(h['budget']))}",
                f"S7 failures recorded so far (version 4; nothing was dropped for them): {len(scoped)}"
                + (f": {', '.join(sorted({str(e.get('run')) for e in scoped}))}" if scoped else ""), h["next_steps"]]
        if h.get("children_still_running"):
            text.insert(3, f"Still running when this was written: {json.dumps(h['children_still_running'])}")
        (self.dir / "STOP.txt").write_text("\n".join(text) + "\n", encoding="utf8")
        self.log(f"stop report: {self.dir / 'STOP.json'}")


# ------------------------------------------------------------------------------------------------ the lock

def acquire_lock(d):
    """Hold <root>/driver/driver.lock (flock, for this process's life; the descriptor is not inherited by children):
    one driver per root. Refuses (gate.ApprovalError) while another process holds it."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(d / LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        holder = os.pread(fd, 4096, 0).decode("utf8", "replace").strip()
        os.close(fd)
        raise gate.ApprovalError(f"refused: another stage driver holds {d / LOCK_NAME} ({holder or 'unknown'}); one "
                                 "driver per root") from None
    os.ftruncate(fd, 0)
    os.pwrite(fd, json.dumps(dict(pid=os.getpid(), boot_id=RN.boot_id(), started_utc=utc_now()),
                             sort_keys=True).encode("utf8"), 0)
    _LOCKS.append(fd)
    return fd


def lock_holder(d):
    """None when no process holds <root>/driver/driver.lock, else what the lock file says (checked without keeping it)."""
    p = Path(d) / LOCK_NAME
    if not p.exists():
        return None
    fd = os.open(str(p), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    except OSError:
        return os.pread(fd, 4096, 0).decode("utf8", "replace").strip() or "unknown"
    finally:
        os.close(fd)


# ------------------------------------------------------------------------------------------------ command line

def frozen_config(package=gate.PACKAGE):
    """What prespec.json fixes for the driver on real data (Modes, module docstring): the learners, the qualities, the
    data seeds, choice 19, the default cores, the complete S9 signature and, version 4, the frozen supplement block (its
    root, plan and S7 lanes, supplement.validate). Refuses (gate.ApprovalError) when any is missing or does not fit this
    code."""
    path = Path(package) / gate.PRESPEC_NAME
    try:
        pre = json.loads(path.read_text(encoding="utf8"))
        out = dict(learners=tuple(pre["setups"]["learners"]), qualities=tuple(pre["data"]["qualities"]),
                   seeds=tuple(int(s) for s in pre["seeds"]["data_seeds"]),
                   choice19={str(k): bool(v) for k, v in pre["choices"]["19"]["applies"].items()},
                   card_cores=[int(c) for c in pre["stage_order"]["card_cores_default"]],
                   lane_cores=[int(c) for c in pre["stage_order"]["lane_cores_default"]],
                   s9_signature=dict(pre["environment"]["s9_signature"]), supplement=pre["supplement"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise gate.ApprovalError(f"refused: {path} does not give the driver's frozen configuration ({error!r})") \
            from error
    problems = []
    if not set(out["learners"]) <= set(LEARNERS) or not out["learners"]:
        problems.append(f"learners {out['learners']}")
    if not set(out["qualities"]) <= set(QUALITIES) or not out["qualities"]:
        problems.append(f"qualities {out['qualities']}")
    if out["choice19"] != {l: CHOICE19.get(l) for l in out["choice19"]} or set(out["choice19"]) != set(LEARNERS):
        problems.append(f"choice 19 {out['choice19']} against this code's {CHOICE19}")
    missing = [k for k in RN.S9_KEYS if k not in out["s9_signature"]]
    if missing:
        problems.append(f"environment.s9_signature lacks {missing}")
    if problems:
        raise gate.ApprovalError(f"refused: prespec.json does not fit this code: {'; '.join(problems)}")
    if RN.NUM_UPDATES != S.FINAL_STEP or RN.NUM_UPDATES // 2 != S.HALF_STEP:
        raise gate.ApprovalError("refused: the supplement's S7 steps are not this runner's final and 50% steps")
    v = S.validate(out["supplement"], out["learners"], out["qualities"], out["seeds"], out["choice19"])
    out.update(root=S.ROOT, plan=v["plan"], s7_all_seeds=v["s7_all_seeds"], s7_half=v["s7_half"])
    return out


def check_real_args(a, frozen):
    """Real mode: refuse every option that would change a frozen choice or limit (Modes, module docstring; prespec.json
    driver_command.not_used_on_real_data). Values equal to the frozen ones are no change and pass."""
    bad = []
    if a.dry_run:
        bad.append("--dry-run")
    if a.root and Path(a.root).resolve() != S.ROOT.resolve():        # version 4: only the supplement's frozen root
        bad.append(f"--root {a.root}")
    if a.supplement_plan:
        bad.append(f"--supplement-plan {a.supplement_plan}")
    if a.s7_all_seeds:
        bad.append("--s7-all-seeds")
    if a.s7_half:
        bad.append("--s7-half")
    for flag, given, want in (("--learners", a.learners, frozen["learners"]),
                              ("--qualities", a.qualities, frozen["qualities"]),
                              ("--seeds", a.seeds, frozen["seeds"])):
        if given is not None and tuple(given) != tuple(want):
            bad.append(f"{flag} {' '.join(map(str, given))}")
    if a.card_device and a.card_device != "gpu":
        bad.append(f"--card-device {a.card_device}")
    if a.choice19:
        merged = dict(frozen["choice19"])
        for item in a.choice19:
            l, _, v = item.partition("=")
            merged[l] = v.lower() in ("1", "true", "yes", "on")
        if merged != frozen["choice19"]:
            bad.append("--choice19")
    if a.caps:
        bad.append("--caps")
    if a.inject:
        bad.append("--inject")
    card = list(a.card_cores) if a.card_cores else list(frozen["card_cores"])
    lanes = list(a.lane_cores) if a.lane_cores else list(frozen["lane_cores"])
    if len(card) > MAX_CARD_CORES or len(set(card)) != len(card):
        bad.append(f"--card-cores {card} (at most {MAX_CARD_CORES}, distinct)")
    if len(lanes) > MAX_LANE_CORES or len(set(lanes)) != len(lanes) or set(lanes) & set(card):
        bad.append(f"--lane-cores {lanes} (at most {MAX_LANE_CORES}, distinct, none a card core)")
    if bad:
        raise gate.ApprovalError(f"refused: on real data these options would change a frozen choice or limit: {bad} "
                                 "(prespec.json; a change needs a new pre-specification version and your terminal "
                                 "approval, plan 6.1)")


def config_from(a, frozen=None):
    """The driver's configuration from its command line; on real data (frozen given: frozen_config()) the learners,
    qualities, seeds, choice 19, S9 signature and version 4's plan and S7 lanes come from prespec.json
    (check_real_args refused any change first). The real root is version 4's runs/shift_hosts/supplement_v4; a dry run
    refuses it, anything inside it, and version 3's runs/shift_hosts."""
    dry = bool(a.dry_run)
    root = Path(a.root) if a.root else (RN.RUNS / "stage0" / "runner" / "dryrun" if dry else S.ROOT)
    if dry and (root.resolve() in (RN.RUNS.resolve(), S.ROOT.resolve()) or S.ROOT.resolve() in root.resolve().parents):
        raise gate.ApprovalError(f"refused: a dry run never uses the real study's roots {RN.RUNS} or {S.ROOT}")
    if dry:
        d = DRY_SCHEDULE
        sched = ["--num-updates", str(d["num_updates"]), "--eval-interval", str(d["eval_interval"]),
                 "--refresh-warmup", str(d["warmup"]), "--refresh-every", str(d["every"])]
        refreshes = len(RN.make_schedule(num_updates=d["num_updates"], eval_interval=d["eval_interval"],
                                         warmup=d["warmup"], every=d["every"]).refresh_steps)
        steps, s1 = d["num_updates"], d["stage1_steps"]
        s1_ref = len(RN.make_schedule(num_updates=d["num_updates"], eval_interval=d["eval_interval"], steps=s1,
                                      warmup=d["warmup"], every=d["every"]).refresh_steps)
        device, reference = "cpu", dict(REFERENCE_MS_PROCESSOR)
    else:
        sched, refreshes, steps, s1, s1_ref = [], 18, RN.NUM_UPDATES, RN.STAGE1_STEPS, 1
        device, reference = "gpu", dict(B.REFERENCE_MS)
    choice19 = dict(CHOICE19)
    for item in a.choice19 or []:
        l, _, v = item.partition("=")
        choice19[l] = v.lower() in ("1", "true", "yes", "on")
    inject = {}
    for item in a.inject or []:
        rid, _, fault = item.partition("=")
        inject[rid] = fault
    if inject and not dry:
        raise gate.ApprovalError("faults are injected in dry runs only")
    learners = tuple(a.learners or LEARNERS)
    qualities = tuple(a.qualities or QUALITIES)
    seeds = tuple(a.seeds if a.seeds is not None else SEEDS)
    card_cores = [int(c) for c in (a.card_cores or [0, 1])]
    lane_cores = [int(c) for c in (a.lane_cores or ([2, 3, 4, 5] if dry else list(range(2, 10))))]
    s9, plan, supplement = None, None, None
    s7_all_seeds, s7_half = bool(a.s7_all_seeds), bool(a.s7_half)
    if a.supplement_plan:
        if not dry:
            raise gate.ApprovalError("a supplement plan file is a dry run's; on real data it comes from prespec.json")
        try:
            given = json.loads(Path(a.supplement_plan).read_text(encoding="utf8"))
        except (OSError, ValueError) as error:
            raise gate.ApprovalError(f"refused: {a.supplement_plan} is not a JSON plan ({error})") from error
        plan = S.check_plan(given, learners, qualities, choice19)
    if not dry and frozen is not None:
        learners, qualities, seeds = tuple(frozen["learners"]), tuple(frozen["qualities"]), tuple(frozen["seeds"])
        choice19, s9 = dict(frozen["choice19"]), dict(frozen["s9_signature"])
        card_cores = [int(c) for c in (a.card_cores or frozen["card_cores"])]
        lane_cores = [int(c) for c in (a.lane_cores or frozen["lane_cores"])]
        plan, supplement = frozen["plan"], frozen["supplement"]
        s7_all_seeds, s7_half = bool(frozen["s7_all_seeds"]), bool(frozen["s7_half"])
    return Config(root=root, dry=dry, learners=learners, qualities=qualities, seeds=seeds,
                  card_device=a.card_device or device, card_cores=card_cores, lane_cores=lane_cores,
                  choice19=choice19, stage1_steps=s1, steps=steps, schedule_args=sched, refreshes_per_run=refreshes,
                  stage1_refreshes=s1_ref, accept_s5=tuple(a.accept_s5 or ()), inject=inject,
                  replace_incomplete=tuple(a.replace_incomplete or ()), reference_ms=reference, s9_signature=s9,
                  plan=plan, s7_all_seeds=s7_all_seeds, s7_half=s7_half, supplement=supplement)


def add_run_args(r):
    r.add_argument("--dry-run", action="store_true", help="made-up arrays, processor, tiny step counts")
    r.add_argument("--root", help="output root (default runs/shift_hosts/supplement_v4, version 4's frozen root; dry "
                                  "run: runs/shift_hosts/stage0/runner/dryrun)")
    r.add_argument("--learners", nargs="+", choices=LEARNERS)
    r.add_argument("--qualities", nargs="+", choices=QUALITIES)
    r.add_argument("--seeds", nargs="+", type=int)
    r.add_argument("--card-device", choices=("cpu", "gpu"))
    r.add_argument("--card-cores", nargs="+", type=int, help="one core per card run at once (two at most)")
    r.add_argument("--lane-cores", nargs="+", type=int, help="up to 8 processor lanes")
    r.add_argument("--choice19", nargs="+", help="dry run only: learner=yes|no (test (a2) matched only to rounding)")
    r.add_argument("--accept-s5", nargs="+", help="run ids whose S5 the operator traced to the setup itself")
    r.add_argument("--inject", nargs="+", help="dry run only: RUN_ID=unit0:<step>, RUN_ID=nan:<step> or "
                                               f"RUN_ID={S7_INJECT} (that run's S7 check fails)")
    r.add_argument("--replace-incomplete", nargs="+", metavar="RUN_ID",
                   help="the runs (by run id) that died without a record and may start again")
    r.add_argument("--resume-after-stop", action="store_true")
    r.add_argument("--stage0-card-hours", type=float)
    r.add_argument("--stage0-processor-hours", type=float)
    r.add_argument("--caps", help="dry run only: card,processor hours for both qualities (tests of S6b)")
    r.add_argument("--supplement-plan", metavar="PLAN.json",
                   help="dry run only: version 4's stage-2 plan {quality: {learner: [setups]}} (default: the full "
                        "grid; real data: prespec.json's supplement)")
    r.add_argument("--s7-all-seeds", action="store_true",
                   help="dry run only: S7 on every seed's final alone, standard and IW actor, not only seed 0's")
    r.add_argument("--s7-half", action="store_true",
                   help="dry run only: S7 also on those runs' 50%% actors (the 50,000-step actors on real data)")


def preflight(a):
    """Real mode, before anything is written (module docstring): the approval (S1), prespec.json's frozen configuration
    and the options against it, stop rule S0 on exactly this code, version 4's checks (version 3's files against the
    frozen supplement block, supplement.check_sources; every file but supplement.CHANGEABLE byte-identical to version
    3's approved code, supplement.code_unchanged), and stage 0's compute entry (in the supplement's own ledger, or both
    hours given). Returns (approval, frozen)."""
    approval = gate.require_run_approval("stage driver")
    frozen = frozen_config()
    check_real_args(a, frozen)
    gate.require_s0("stage driver")
    S.check_sources(frozen["supplement"])
    changed = S.code_unchanged(gate.code_manifest())
    if changed:
        raise gate.ApprovalError(f"refused: version 4 may change only {list(S.CHANGEABLE)} against version 3's "
                                 f"approved code; these differ: {json.dumps(changed)}")
    led = S.ROOT / "driver" / "ledger.json"            # version 4's own ledger (decision D2)
    stage0 = None
    if led.exists() or B.kept_path(led).exists():
        try:
            body, _ = B.read_json_kept(led)
        except H.HandoffError as error:
            raise gate.ApprovalError(f"refused: {led} does not verify ({error})") from error
        if body.get("mode") != "real":
            raise gate.ApprovalError(f"refused: {led} is not a real study's ledger (mode {body.get('mode')!r})")
        stage0 = body.get("stage0")
        if stage0 is not None and "dry run" in str(stage0.get("source")):
            raise gate.ApprovalError(f"refused: {led}'s stage-0 entry comes from a dry run")
    if stage0 is None and (a.stage0_card_hours is None or a.stage0_processor_hours is None):
        raise gate.ApprovalError("the first start names stage 0's compute (--stage0-card-hours, "
                                 "--stage0-processor-hours): the medium-expert cap covers stages 0-2a")
    return approval, frozen


def command_run(a):
    if not a.dry_run:
        approval, frozen = preflight(a)                 # first: nothing is written before it
    else:
        approval, frozen = None, None
    cfg = config_from(a, frozen)
    if a.caps:
        if not cfg.dry:
            raise gate.ApprovalError("caps are the approved ones; only a dry run may test others")
        c, p = (float(x) for x in a.caps.split(","))
        for q in B.CAPS:
            B.CAPS[q] = dict(B.CAPS[q], card_hours=c, processor_hours=p)
    d = cfg.root / "driver"
    acquire_lock(d)                                     # one driver per root, before anything else changes
    if (d / "STOP.json").exists():
        if not a.resume_after_stop:
            raise gate.ApprovalError(f"{d / 'STOP.json'} exists: a stop report is waiting to be read; rerun with "
                                     "--resume-after-stop once it is dealt with (nothing retries silently)")
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        for n in ("STOP.json", "STOP.json.sha256", "STOP.txt"):
            if (d / n).exists():
                shutil.move(str(d / n), str(d / f"{n}.{stamp}"))
    if (d / "STOP_REQUEST").exists():
        (d / "STOP_REQUEST").unlink()
    driver = Driver(cfg, approval)
    stage0 = driver.ledger.body.get("stage0")
    if stage0 is None:
        if cfg.dry:
            driver.ledger.set_stage0(a.stage0_card_hours or 0.0, a.stage0_processor_hours or 0.0, "dry run")
        elif a.stage0_card_hours is None or a.stage0_processor_hours is None:
            raise gate.ApprovalError("the first start names stage 0's compute (--stage0-card-hours, "
                                     "--stage0-processor-hours): the medium-expert cap covers stages 0-2a")
        else:
            driver.ledger.set_stage0(a.stage0_card_hours, a.stage0_processor_hours, "operator, at the first start")
    elif not cfg.dry and "dry run" in str(stage0.get("source")):
        raise gate.ApprovalError("refused: the ledger's stage-0 entry comes from a dry run")
    return driver.run()


def command_start(argv):
    """Start `stages run ...` detached under nohup (driver/nohup.out), after the same checks `run` makes before it
    writes anything (preflight, the configuration, no other driver on the root); prints its process id and the
    status file."""
    a = parser().parse_args(["run"] + argv)
    frozen = None
    if not a.dry_run:
        _, frozen = preflight(a)
    cfg = config_from(a, frozen)
    d = cfg.root / "driver"
    holder = lock_holder(d)
    if holder is not None:
        raise gate.ApprovalError(f"refused: another stage driver runs on {cfg.root} ({holder})")
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    with open(d / "nohup.out", "ab") as out:
        p = subprocess.Popen(["nohup", sys.executable, "-m", "experiments.shift_hosts.stages", "run"] + argv,
                             cwd=str(RN.REPO), env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(pid=p.pid, status=str(d / "status.json"), log=str(d / "driver.log"),
                          nohup=str(d / "nohup.out")), sort_keys=True))
    return 0


def parser():
    p = argparse.ArgumentParser(prog="python -m experiments.shift_hosts.stages")
    sub = p.add_subparsers(dest="command", required=True)
    add_run_args(sub.add_parser("run", help="run the stages in the foreground"))
    sub.add_parser("start", help="run the stages detached under nohup (same options as run)")
    st = sub.add_parser("status", help="print the status file")
    st.add_argument("--root")
    st.add_argument("--dry-run", action="store_true")
    rq = sub.add_parser("request-stop", help="ask a running driver to halt gracefully")
    rq.add_argument("--root")
    rq.add_argument("--dry-run", action="store_true")
    sub.add_parser("lane", help="one scoring-environment step of a lane job (the driver starts these)")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "lane":
        try:
            return lane_main(argv[1:])
        except gate.ApprovalError as error:
            print(f"REFUSED: {error}", file=sys.stderr, flush=True)
            return 2
        except Exception:
            traceback.print_exc()
            return 1
    if argv and argv[0] == "start":
        try:
            return command_start(argv[1:])
        except gate.ApprovalError as error:
            print(f"REFUSED: {error}", file=sys.stderr, flush=True)
            return 2
    a = parser().parse_args(argv)
    try:
        if a.command == "run":
            return command_run(a)
        root = Path(a.root) if a.root else (RN.RUNS / "stage0" / "runner" / "dryrun" if a.dry_run else S.ROOT)
        if a.command == "status":
            print(json.dumps(H.read_json(root / "driver" / "status.json"), indent=1, sort_keys=True))
            return 0
        (root / "driver").mkdir(parents=True, exist_ok=True)
        (root / "driver" / "STOP_REQUEST").write_text(utc_now() + "\n", encoding="utf8")
        print(f"stop requested: {root / 'driver' / 'STOP_REQUEST'}")
        return 0
    except gate.ApprovalError as error:
        print(f"REFUSED: {error}", file=sys.stderr, flush=True)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
