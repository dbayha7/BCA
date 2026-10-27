"""The stage driver's and the budget's stage-0 tests (plan sections 6.1 and 7.3-7.4; area "runner"), on made-up arrays
only: no ThrusterChain data is trained on, no band is fitted on real arrays, no learned policy's return is produced.
An experiment-local attachment (plan section 3.1) that changes none of the repository's declared methods.

  budget    The arithmetic of budget.py: card-hours as the union of card-busy intervals per session (two runs at once
            never count twice; sessions are summed), processor-hours as summed measured seconds, stage 0's entry in
            medium-expert's totals only, the speed model (start-up, steps x ms, refreshes, tables; two at once), the
            S6b check at the cap's boundary, the two-at-once rule at 1.15 and with S8 failing, S6's alone check.
  checks    The stop-rule checks on records: S2, S8, S9 pass on equal records and fail on each kind of change.
  S6b       The driver halts before a unit whose projected cost would pass the cap, launching nothing (in process).
  dry run   The whole driver as a subprocess in dry-run mode (made-up arrays, processor, tiny step counts: 40 outer
            steps in chunks of 10, refreshes after 20 and 30, stage 1 = 30 steps): stage 1 for the four learners
            (alone, standard, IW one at a time; weights-of-1 and repeat two at once), its checks (S2, S4, S8, S10 with
            choice 19's readouts, S6, S7's open loop on made-up states, S9), the sham-alone runs, stage 2 for both
            qualities with data seeds 0 and 1 (12 setups per seed, then the room references), S6b before every unit,
            the coverage lanes (made-up strong-law rows) and S7 on the seed-0 actors. Then the same command again:
            everything is skipped (records and lane outputs verify) and it finishes.
  stops     A simulated S5 (BCA's unit set to 0 in one BCA run of seed 1): the driver halts, writes its stop report,
            launches nothing after it (no room reference); a restart over the report is refused; a restart with
            --resume-after-stop halts again on the same recorded stop (never retried); with --accept-s5 it finishes.
            A simulated S6 (a tiny card cap): the driver halts after stage 1's projection. Version 4's supplement
            (PRESPEC.md section 21) end to end: a supplement-shaped plan (IQL's learner scope and TD3+BC's IW scope on
            medium-expert, TD3+BC's learner scope on medium) with S7 on every seed's final and 50% actors, and S7
            failures injected on stage 1's TD3+BC standard actor, seed 0's IQL alone actor and seed 1's TD3+BC IW
            actor: every failure is recorded (version 4) and nothing is dropped; stage 1 runs whole, stage 2 trains
            exactly the plan, the driver finishes, and a resume retrains nothing.
  refusal   Without the approval the driver refuses to start (real mode), writing nothing.
  review    The stage-0 review's fixes, in process on made-up folders: a failed S7 result is never skipped as done
            on resume (version 4: it is recorded; an s7.json that does not verify halts); version 4's S7 rule (a failure
            is recorded once and drops nothing, a lane of its scope goes on, the end-of-quality read, the exit-3
            cross-check), its S7 lanes (every seed, final and 50% actors; lanes/s7-step50000 on real data) and the check
            keys export.py reads, the frozen supplement plan's units (medium-expert 56 stage-2 runs, medium 42; stage 1
            whole), its root and its own ledger; a halt starts no next lane step; a non-finite scored return is a Claim
            0 outcome (alone, room) or an S5 halt (BCA, unless accepted); a stop before training names its rule (S11);
            modes, the recorded configuration (version 4's plan included) and the per-root lock; the real-data options
            that would change a frozen choice are refused (version 4's plan, S7 lanes and root included); S6b on resume
            projects only runs without a record; an unexpected error writes the stop report; the ledger's open entries,
            recovered runs, the driver's own time and the kept previous version; the gate's manifest coverage, its
            bytecode refusal and stop rule S0's summary check.

Run from /root/bca-hosts in /opt/bca-venv with the Dockerfile's variables, on the processor (cores 0-5):
  env D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DISPLAY= \\
    MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210 LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin \\
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu SHIFT_HOSTS_DEVICE=cpu \\
    taskset -c 0-5 /opt/bca-venv/bin/python -m experiments.shift_hosts.tests.test_stages
Result: runs/shift_hosts/stage0/runner/test_stages_cpu.json (+ .sha256); dry runs under
runs/shift_hosts/stage0/runner/test_stages/.
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.shift_hosts import budget as B
from experiments.shift_hosts import gate
from experiments.shift_hosts import handoff as H
from experiments.shift_hosts import runner as RN
from experiments.shift_hosts import stages as ST
from experiments.shift_hosts import supplement as S
from experiments.shift_hosts import tests as T
from experiments.shift_hosts import unifloral_import as UI

AREA = "runner"
ROOT = T.STAGE0 / AREA / "test_stages"
TRAINING_PYTHON = "/opt/bca-venv/bin/python"
READOUTS = {}


def driver(args, timeout=5400):
    """Run the stage driver as a subprocess (training environment); returns (exit code, output)."""
    env = UI.launch_env(dict(os.environ))
    env.update(JAX_PLATFORMS="cpu", SHIFT_HOSTS_DEVICE="cpu")
    out = subprocess.run([TRAINING_PYTHON, "-m", "experiments.shift_hosts.stages"] + args, cwd=str(RN.REPO), env=env,
                         capture_output=True, text=True, timeout=timeout)
    return out.returncode, out.stdout + out.stderr


def state_of(root):
    return H.read_json(root / "driver" / "state.json")


def status_of(root):
    return H.read_json(root / "driver" / "status.json")


def fresh(name):
    d = ROOT / name
    if d.exists():
        shutil.rmtree(d)
    return d


# ------------------------------------------------------------------------------------------------ budget

class BudgetTest(unittest.TestCase):
    def test_union_and_ledger(self):
        """Card-hours are the union of busy intervals per session; processor-hours are summed; stage 0 counts once."""
        self.assertEqual(B.union_seconds([(0, 10), (5, 15), (20, 25)]), 20.0)
        self.assertEqual(B.union_seconds([(0, 10), (2, 3)]), 10.0)
        self.assertEqual(B.union_seconds([(0, 5), (5, 10)]), 10.0)
        self.assertEqual(B.union_seconds([]), 0.0)
        with self.assertRaises(ValueError):
            B.union_seconds([(5, 1)])
        d = fresh("ledger")
        led = B.Ledger(d / "ledger.json")
        led.set_stage0(0.5, 2.0, "test")
        led.set_stage0(9.0, 9.0, "ignored: stage 0 is entered once")
        led.add_card("medium-expert", "a", "s1", 0, 3600)
        led.add_card("medium-expert", "b", "s1", 1800, 5400)       # overlaps a: 1.5 h of union in session s1
        led.add_card("medium-expert", "c", "s2", 0, 1800)          # another session: its clock is not comparable
        led.add_card("medium", "d", "s2", 0, 3600)
        led.add_processor("medium-expert", "a", 3600, "card")
        led.add_processor("medium-expert", "x", 1800, "lane:score")
        led.add_processor("medium-expert", "y", 3600, "lane:score")
        led.add_processor("medium", "d", 7200, "card")
        led.save()
        again = B.Ledger(d / "ledger.json")
        self.assertAlmostEqual(again.card_hours("medium-expert"), 1.5 + 0.5 + 0.5, places=12)
        self.assertAlmostEqual(again.card_hours("medium"), 1.0, places=12)
        self.assertAlmostEqual(again.processor_hours("medium-expert"), 1.0 + 0.5 + 1.0 + 2.0, places=12)
        self.assertAlmostEqual(again.processor_hours("medium"), 2.0, places=12)
        self.assertEqual(again.lane_seconds("score"), 2700.0)
        self.assertIsNone(again.lane_seconds("s7"))
        self.assertEqual(B.lane_cost(again, "s7"), B.DEFAULT_LANE_SECONDS["s7"])
        self.assertEqual(B.CAPS["medium-expert"]["card_hours"], 14.0)
        self.assertEqual((B.CAPS["medium"]["card_hours"], B.CAPS["medium"]["processor_hours"]), (12.0, 16.0))

    def test_speed_model_and_rules(self):
        """The run model, the unit projection, S6b at the cap, the two-at-once rule and S6's alone check."""
        speed = dict(ms_per_step=1.0, startup_seconds=60.0, refresh_seconds=2.0, table_seconds=10.0, busy=0.8)
        self.assertEqual(B.project_run(speed, refreshes=18, tables=18, readout_tables=2), (396.0, 396.0 * 0.8))
        c2, p2 = B.project_run(speed, refreshes=18, tables=18, readout_tables=2, concurrency=2, gain=1.6)
        self.assertAlmostEqual(c2, 396.0 / 1.6, places=9)
        self.assertAlmostEqual(p2, 2 * 396.0 / 1.6 * 0.8, places=9)
        speeds = {("td3_bc", "alone"): dict(speed, refresh_seconds=0.0, table_seconds=0.0),
                  ("td3_bc", "standard"): speed, ("td3_bc", "iw"): speed}
        units = [dict(learner="td3_bc", setup="alone", lanes=dict(score=2)),
                 dict(learner="td3_bc", setup="standard", lanes=dict(score=2, coverage=1)),
                 dict(learner="td3_bc", setup="iw", lanes=dict(coverage=1)),
                 dict(learner="td3_bc", setup="room"), dict(learner="td3_bc", setup="sham_alone", refreshes=0)]
        card, proc = B.project_unit(units, speeds)
        walls = [160.0, 196.0, 396.0, 160.0, 160.0]
        self.assertAlmostEqual(card, sum(walls) / 3600, places=12)
        lanes = 4 * B.DEFAULT_LANE_SECONDS["score"] + 2 * B.DEFAULT_LANE_SECONDS["coverage"]
        self.assertAlmostEqual(proc, (0.8 * sum(walls) + lanes) / 3600, places=12)
        d = fresh("ledger_s6b")
        led = B.Ledger(d / "ledger.json")
        led.set_stage0(10.0, 15.0, "test")
        ok, detail = B.s6b_check(led, "medium-expert", (4.0, 6.0))
        self.assertTrue(ok)
        self.assertEqual(detail["after"], dict(card_hours=14.0, processor_hours=21.0))
        self.assertFalse(B.s6b_check(led, "medium-expert", (4.0 + 1e-9, 6.0))[0])
        self.assertFalse(B.s6b_check(led, "medium-expert", (4.0, 5.0), committed_processor_hours=1.5)[0])
        self.assertTrue(B.s6b_check(led, "medium", (12.0, 16.0))[0])            # medium has its own cap
        c, det = B.two_at_once(dict(a=100.0, b=100.0), dict(a=170.0, b=175.0), 15000, True)
        self.assertEqual(c, 2)
        self.assertAlmostEqual(det["gain"], 400.0 / 345.0, places=12)
        self.assertEqual(B.two_at_once(dict(a=100.0, b=100.0), dict(a=170.0, b=175.0), 15000, False)[0], 1)
        self.assertEqual(B.two_at_once(dict(a=100.0, b=100.0), dict(a=180.0, b=180.0), 15000, True)[0], 1)
        self.assertEqual(B.two_at_once(dict(a=115.0), dict(a=200.0), 15000, True)[0], 2)     # gain exactly 1.15
        ok, det = B.s6_alone_check(dict(td3_bc=1.2, cql=4.3), dict(td3_bc=0.6, cql=2.1))
        self.assertFalse(ok)
        self.assertEqual((det["td3_bc"]["passed"], det["cql"]["passed"]), (True, False))
        # both sides one run at a time: the limit is twice the reference, never scaled by a two-at-once slowdown
        self.assertEqual(det["cql"]["limit_ms"], 2 * 2.1)
        with self.assertRaises(TypeError):
            B.s6_alone_check(dict(cql=4.3), dict(cql=2.1), slowdown=1.1)
        rec = dict(run_id="x", timings=dict(steady_ms_per_outer_step=2.0, outer_steps=15000, refresh_seconds=[3.0],
                                            table_seconds=[7.0], readout_seconds=[], wall_seconds=50.0,
                                            processor_seconds=40.0))
        sp = B.run_speed(rec, process_wall=60.0, process_cpu=48.0)
        self.assertEqual((sp["startup_seconds"], sp["busy"], sp["refresh_seconds"], sp["table_seconds"]),
                         (60.0 - 30.0 - 3.0 - 7.0, 0.8, 3.0, 7.0))
        READOUTS["budget"] = dict(two_at_once_rule=B.TWO_AT_ONCE_RULE)


# ------------------------------------------------------------------------------------------------ checks on records

def _rec(**over):
    final = dict(key="k", agent_state="a", bca_state="b", bca_state_without_refusal_rule="bw", losses_sha256="l")
    base = dict(run_id="r", outcome="complete", hashes=dict(final=final),
                refreshes=[dict(step=20, threshold=0.5, s3_reproduced=True,
                                iw=dict(table=dict(grid=[1.0], thresholds=[0.5])))])
    base.update(over)
    return base


class CheckTest(unittest.TestCase):
    def test_s2_s8_s9(self):
        """S2, S8 and S9 pass on equal records and fail on each kind of change."""
        std = _rec(refreshes=[dict(step=20, threshold=0.5)])
        self.assertTrue(ST.check_s2(std, _rec(hashes=dict(final=dict(_rec()["hashes"]["final"], bca_state="other"))))
                        ["passed"])                                   # the refusal rule itself may differ
        self.assertFalse(ST.check_s2(std, _rec(hashes=dict(final=dict(_rec()["hashes"]["final"],
                                                                       bca_state_without_refusal_rule="x"))))["passed"])
        self.assertFalse(ST.check_s2(std, _rec(hashes=dict(final=dict(_rec()["hashes"]["final"], losses_sha256="x"))))
                         ["passed"])
        wrong = _rec(refreshes=[dict(step=20, threshold=0.5, s3_reproduced=True,
                                     iw=dict(table=dict(grid=[1.0], thresholds=[0.5000001])))])
        self.assertFalse(ST.check_s2(std, wrong)["passed"])
        self.assertFalse(ST.check_s2(std, _rec(refreshes=[]))["passed"])
        self.assertTrue(ST.check_s8(_rec(), _rec())["passed"])
        self.assertFalse(ST.check_s8(_rec(), _rec(hashes=dict(final=dict(_rec()["hashes"]["final"], key="x"))))
                         ["passed"])
        self.assertFalse(ST.check_s8(_rec(), _rec(outcome="stopped"))["passed"])
        rec = dict(environment=dict(import_record=dict(python="3.10.12", packages=dict(jax="0.6.2"), env=dict(A="1"),
                                                       xla_flags=dict(in_effect="--x"))))
        sig = ST.env_signature(rec)                       # every key S9 compares
        self.assertEqual(sorted(sig), sorted(RN.S9_KEYS))
        self.assertTrue(ST.check_s9(rec, sig)["passed"])
        self.assertFalse(ST.check_s9(rec, dict(sig, packages=dict(jax="0.6.1")))["passed"])
        partial = dict(python="3.10.12", packages=dict(jax="0.6.2"), env=dict(A="1"))
        res = ST.check_s9(rec, partial)                   # a partial frozen signature cannot skip a field
        self.assertFalse(res["passed"])
        self.assertIn("xla_flags", res["differs"])
        # S2's report when a refresh could not certify (R = +inf): it still fails, and says why it differs by design
        inf = _rec(refreshes=[dict(step=20, threshold="inf", certified=False)])
        res = ST.check_s2(inf, _rec(hashes=dict(final=dict(_rec()["hashes"]["final"], losses_sha256="x"))))
        self.assertFalse(res["passed"])
        self.assertEqual(res["uncertified_refreshes"], [20])
        self.assertIn("differs by design", res["note"])
        self.assertNotIn("note", ST.check_s2(std, _rec()))


class S6bWiringTest(unittest.TestCase):
    def test_driver_halts_before_a_unit_over_the_cap(self):
        """The driver checks S6b before a unit and halts there, launching nothing."""
        d = fresh("s6b")
        args = ST.parser().parse_args(["run", "--dry-run", "--root", str(d), "--learners", "td3_bc", "--qualities",
                                       "medium-expert", "--seeds", "0"])
        cfg = ST.config_from(args)
        drv = ST.Driver(cfg, None)
        drv.ledger.set_stage0(14.0, 0.0, "test: the card cap is already used up")
        speed = dict(ms_per_step=1.0, startup_seconds=1.0, refresh_seconds=0.0, table_seconds=0.0, busy=1.0, steps=30,
                     wall=1.0)
        drv.state["decisions"] = dict(concurrency=1, s6=dict(speeds={f"td3_bc/{k}": speed for k in ST.STAGE1_SINGLE},
                                                             two_at_once=dict(gain=1.0)))
        with self.assertRaises(ST.Halt):
            drv.stage2("medium-expert")
        self.assertEqual(drv.halt["rule"], "S6b")
        self.assertEqual(drv.counts["cards_done"], 0)
        self.assertFalse((d / "stage2").exists())
        self.assertIn("S6b/medium-expert/sham", drv.state["checks"])
        READOUTS["s6b"] = drv.halt["detail"]["after"]


# ------------------------------------------------------------------------------------------------ the review's rules

def dry_cfg(root, *more):
    return ST.config_from(ST.parser().parse_args(["run", "--dry-run", "--root", str(root), "--learners", "td3_bc",
                                                  "--qualities", "medium-expert", "--seeds", "0"] + list(more)))


def fake_scores(path, finite):
    """A scores file in evaluate.write_scores' layout as far as the driver reads it (meta summaries), made-up."""
    H.write(path, dict(x=np.zeros(1, np.float32)),
            meta=dict(summaries=dict(scored={b: dict(finite=bool(finite or b != "strong"))
                                             for b in ("none", "strong")})))
    return path


def plan_file(path, plan):
    """A dry run's --supplement-plan file: version 4's stage-2 plan {quality: {learner: [setups]}}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan), encoding="utf8")
    return str(path)


def made_up_record(steps, refresh_steps=(), snapshot_steps=None):
    """Just enough of a complete run's record for lanes_for: its outer steps, refreshes, data path and snapshot actors
    (/actors/<step>, never opened)."""
    snaps = snapshot_steps or (steps // 2, steps)
    return dict(outcome="complete", timings=dict(outer_steps=steps), refreshes=[dict(step=t) for t in refresh_steps],
                data=dict(path="made-up/prepared.npz"),
                snapshots={str(t): dict(actor=dict(path=f"/actors/{t}")) for t in snaps})


def s7_lanes(lanes):
    """(label, its folder under lanes/, the actor it checks) of each S7 lane."""
    out = []
    for lane in lanes:
        if lane.kind == "s7":
            check = lane.commands[-1][0]
            out.append((lane.label, Path(lane.outputs[0]).parent.name, check[check.index("--actor") + 1]))
    return out


class DriverRulesTest(unittest.TestCase):
    """The stop-rule and bookkeeping fixes, in process on made-up folders (nothing trains, nothing is scored)."""

    def test_failed_s7_never_passes_on_resume(self):
        """A lane whose outputs verify is skipped on resume only after its own verdict: an s7.json that says
        s7_passed false is never skipped as done (version 4: it is recorded in s7_stopped and s7_results, nothing is
        dropped; a lane with no scope halts as before); a passing one is skipped; one that does not verify halts."""
        d = fresh("rules_s7")
        drv = ST.Driver(dry_cfg(d), None)
        bad, good = d / "run" / "lanes" / "s7-step1" / "s7.json", d / "good" / "s7.json"
        H.write_json(bad, dict(passed=False, s7_passed=False))
        H.write_json(good, dict(passed=True, s7_passed=True))
        never = [["/bin/false"], "scoring"]
        ok_job = ST.LaneJob("x/s7-good", "s7", "medium-expert", [never], [str(good)])
        drv.lane_queue.append(ok_job)
        drv.pump()
        self.assertEqual((ok_job.result, drv.halt, drv.lanes), (dict(status="skipped"), None, {}))
        bad_job = ST.LaneJob("x/s7-bad", "s7", "medium-expert", [never], [str(bad)], run="x", setup="iw",
                             learner="td3_bc")
        drv.lane_queue.append(bad_job)
        drv.pump()
        self.assertIsNone(drv.halt)
        self.assertEqual([(e["learner"], e["scope"], e["version"]) for e in drv.state["s7_stopped"]],
                         [("td3_bc", "iw", 4)])
        self.assertEqual([(r["label"], r["s7_passed"]) for r in drv.state["s7_results"]],
                         [("x/s7-good", True), ("x/s7-bad", False)])
        self.assertEqual(drv.lanes, {})                       # nothing launched
        unscoped = ST.LaneJob("x/s7-bad", "s7", "medium-expert", [never], [str(bad)])
        drv.lane_queue.append(unscoped)
        drv.pump()
        self.assertEqual(drv.halt["rule"], "S7")
        res = ST.s7_result(d / "run", dict(outcome="complete", timings=dict(outer_steps=1)))
        self.assertEqual((res["passed"], res["s7_passed"]), (False, False))   # stage 1's and seed 0's S7 checks
        self.assertFalse(ST.s7_result(d / "none", dict(outcome="complete", timings=dict(outer_steps=1)))["passed"])
        self.assertTrue(ST.s7_result(d / "none", dict(outcome="stopped", timings=dict(outer_steps=1)))["passed"])
        broken = d / "broken" / "s7.json"
        H.write_json(broken, dict(passed=False, s7_passed=False))
        broken.write_text(broken.read_text() + " ")          # its sidecar no longer matches
        drv2 = ST.Driver(dry_cfg(fresh("rules_s7b")), None)
        drv2._verdict(ST.LaneJob("y/s7", "s7", "medium-expert", [never], [str(broken)], run="y", setup="standard",
                                 learner="td3_bc"))
        self.assertEqual(drv2.halt["rule"], "S7")
        self.assertEqual(drv2.state["s7_stopped"], [])

    def test_s7_failure_drops_nothing(self):
        """Version 4's S7 rule (PRESPEC.md section 21): a failed check (here IQL's IW actor at the 50% step, and
        TD3+BC's standard actor whose check exits 3 at its last step) is recorded once in s7_stopped, with its scope,
        actor step and version 4, and in s7_results, and drops nothing: no halt, the queued card runs and lanes stay
        queued (stage 2's and stage 1's), the dropped counts stay 0, no S7 check key is written for it, and the units
        list exactly the supplement plan's runs (supplement.run_ids; the other quality's plan untouched). Exit 3 at an
        earlier step still halts."""
        d = fresh("rules_s7_v4")
        q = "medium-expert"
        plan = {q: {"td3_bc": ["alone", "standard", "iw"], "iql": ["sham_alone", "alone", "standard", "iw", "room"]},
                "medium": {"td3_bc": ["alone", "standard", "iw"]}}
        drv = ST.Driver(dry_cfg(d, "--learners", "td3_bc", "iql", "--qualities", q, "medium", "--seeds", "0", "1",
                                "--supplement-plan", plan_file(d / "plan.json", plan), "--s7-all-seeds", "--s7-half"),
                        None)
        self.assertEqual(ST.S7_SCOPE, dict(alone="learner", standard="learner", iw_ones="learner", iw="iw",
                                           iw_repeat="iw"))
        seed1 = drv.unit_jobs(q, "seed1")
        stage1 = drv.card_job(q, 0, "td3_bc", "iw", "stage1", 30, "stage1")
        drv.card_queue.extend(seed1 + [stage1])
        s7_lane1 = ST.LaneJob(f"{stage1.label}/s7", "s7", q, [], [], run=stage1.label, setup="iw", learner="td3_bc")
        score = ST.LaneJob("s/score", "score", q, [], [], run=seed1[0].label, setup="alone", learner="td3_bc")
        drv.lane_queue.extend([s7_lane1, score])
        cards = [j.label for j in drv.card_queue]

        def failing(name, step):
            p = d / name / "lanes" / f"s7-step{step}" / "s7.json"
            H.write_json(p, dict(passed=True, s7_passed=False))
            return str(p)

        iw0 = drv.card_job(q, 0, "iql", "iw", "stage2", 40, "seed0")
        iw_lane = ST.LaneJob(f"{iw0.label}/s7-20", "s7", q, [], [failing("iql_iw", 20)], run=iw0.label, setup="iw",
                             learner="iql")
        drv._verdict(iw_lane)
        self.assertIsNone(drv.halt)
        self.assertEqual([j.label for j in drv.card_queue], cards)              # IQL's seed-1 IW run stays queued
        self.assertTrue(all(j.result is None for j in seed1))
        self.assertEqual(list(drv.lane_queue), [s7_lane1, score])
        std = drv.card_job(q, 0, "td3_bc", "standard", "stage2", 40, "seed0")
        lane = ST.LaneJob(f"{std.label}/s7", "s7", q, [(["a"], "scoring"), (["b"], "training"), (["c"], "scoring")],
                          [failing("td3_bc_std", 40)], run=std.label, setup="standard", learner="td3_bc")
        lane.step_index = 2                                   # the check itself, which exits 3 on a failure
        core = drv.free_lane.pop(0)
        drv._lane_step_done(lane, 3, 0.0, core, 0.0, 1.0)
        self.assertIsNone(drv.halt)
        self.assertEqual(lane.result, dict(status="complete", s7_passed=False))
        self.assertIn(core, drv.free_lane)
        self.assertEqual([j.label for j in drv.card_queue], cards)              # TD3+BC's runs stay queued too
        self.assertEqual(list(drv.lane_queue), [s7_lane1, score])
        self.assertEqual((drv.counts["cards_dropped"], drv.counts["lanes_dropped"]), (0, 0))
        for job in (iw_lane, lane):                                              # read again: recorded once
            drv._verdict(job)
        got = [(e["quality"], e["learner"], e["scope"], e["setup"], e["stage"], e["step"], e["version"])
               for e in drv.state["s7_stopped"]]
        self.assertEqual(got, [(q, "iql", "iw", "iw", "stage2", 20, 4), (q, "td3_bc", "learner", "standard", "stage2",
                                                                        40, 4)])
        self.assertEqual([(r["label"], r["seed"], r["step"], r["s7_passed"]) for r in drv.state["s7_results"]],
                         [(iw_lane.label, 0, 20, False), (lane.label, 0, 40, False)])
        self.assertEqual([k for k in drv.state["checks"] if k.startswith("S7")], [])
        drv.status()
        st = status_of(d)
        self.assertEqual((st["s7_results"], st["counts"]["cards_dropped"], st["counts"]["lanes_dropped"]), (2, 0, 0))
        want = S.run_ids(plan, drv.cfg.seeds, drv.cfg.learners, made_up=True)
        for qq in (q, "medium"):                                                 # exactly the plan's runs
            units = drv.plan_units(qq)
            self.assertEqual([j.label for name, _ in units for j in drv.unit_jobs(qq, name)], want[qq], qq)
            self.assertEqual(sum(len(us) for _, us in units), len(want[qq]), qq)
        units = dict(drv.plan_units(q))
        self.assertEqual([(u["learner"], u["setup"]) for u in units["seed0"]],
                         [(l, s) for l in ("td3_bc", "iql") for s in ST.STAGE2_SETUPS])
        self.assertEqual([u["learner"] for u in units["sham"]], ["iql"])
        self.assertEqual([u["learner"] for u in units["room"]], ["iql", "iql"])
        other = dict(drv.plan_units("medium"))                                   # the other quality: its own plan
        self.assertEqual([(u["learner"], u["setup"]) for u in other["seed1"]],
                         [("td3_bc", s) for s in ST.STAGE2_SETUPS])
        self.assertEqual((other["sham"], other["room"]), ([], []))
        early = ST.LaneJob("z/s7", "s7", q, [(["a"], "scoring"), (["b"], "training")], [str(d / "none.json")],
                           run="z", setup="alone", learner="iql")
        drv._lane_step_done(early, 3, 0.0, drv.free_lane.pop(0), 0.0, 1.0)         # exit 3 before the check: halt
        self.assertEqual(drv.halt["rule"], "S7")

    def test_s7_scope_branches(self):
        """Version 4's S7 rule, the remaining branches: the end-of-quality read of seed 0's final actors records a
        failure once (its check key names the scope) and halts on a planned run whose s7.json does not verify, that
        has a record but no s7.json, or that has no record; a running lane of a scope whose check failed starts its
        next step; a check that exits 3 while its s7.json says it passed halts; a unit with no supplement run is
        skipped with version 4's note, keeping an earlier S6b record; S4 is not computed for a learner with fewer than
        two runs in a seed."""
        q = "medium-expert"
        d = fresh("rules_s7_branches")
        drv = ST.Driver(dry_cfg(d), None)
        alone, std, iw = drv.unit_jobs(q, "seed0")
        for j in (alone, std, iw):
            H.write_json(j.run_dir / "record.json", dict(run_id=j.label, outcome="complete", timings=dict(outer_steps=40)))
            H.write_json(j.run_dir / "lanes" / "s7-step40" / "s7.json", dict(passed=True, s7_passed=j is not std))
        drv.s7_seed0_checks(q)
        drv.s7_seed0_checks(q)                                                  # read again (a resume)
        self.assertIsNone(drv.halt)
        c = drv.state["checks"]
        self.assertEqual({s: c[f"S7/{q}/seed0/td3_bc/{s}"]["passed"] for s in ST.STAGE2_SETUPS},
                         dict(alone=True, standard=False, iw=True))
        self.assertEqual(c[f"S7/{q}/seed0/td3_bc/standard"]["scoped_stop"], "learner")
        self.assertEqual([(e["run"], e["step"], e["version"]) for e in drv.state["s7_stopped"]], [(std.label, 40, 4)])
        broken = iw.run_dir / "lanes" / "s7-step40" / "s7.json"
        broken.write_text(broken.read_text() + " ")                             # it no longer verifies
        with self.assertRaises(ST.Halt):
            drv.s7_seed0_checks(q)
        self.assertEqual(drv.halt["rule"], "S7")
        for name, missing, rule in (("rules_s7_branches_nos7", "s7.json", "S7"),
                                    ("rules_s7_branches_norec", "record.json", "error")):
            drv1 = ST.Driver(dry_cfg(fresh(name)), None)
            for j in drv1.unit_jobs(q, "seed0"):
                if not (j.setup == "standard" and missing == "record.json"):
                    H.write_json(j.run_dir / "record.json", dict(run_id=j.label, outcome="complete",
                                                                 timings=dict(outer_steps=40)))
                if j.setup != "standard":
                    H.write_json(j.run_dir / "lanes" / "s7-step40" / "s7.json", dict(passed=True, s7_passed=True))
            with self.assertRaises(ST.Halt, msg=missing):
                drv1.s7_seed0_checks(q)
            self.assertEqual(drv1.halt["rule"], rule, missing)
            self.assertIn(f"{q}/seed0/td3_bc-standard", drv1.halt["reason"])

        drv2 = ST.Driver(dry_cfg(fresh("rules_s7_branches2")), None)
        drv2.s7_failed(q, "td3_bc", "iw", iw.label, d / "x", "x/s7")              # td3_bc's IW setup failed S7
        launched = []
        drv2._launch_lane_step = lambda job, core: launched.append((job.label, job.step_index, core))
        nxt = ST.LaneJob("r/score-40", "viewer", q, [(["a"], "scoring"), (["b"], "training")], [str(d / "v")],
                         run=iw.label, setup="iw", learner="td3_bc")
        core = drv2.free_lane.pop(0)
        drv2._lane_step_done(nxt, 0, 0.0, core, 0.0, 1.0)
        self.assertEqual(launched, [("r/score-40", 1, core)])                    # its next step starts
        self.assertIsNone(nxt.result)
        self.assertEqual(drv2.counts["lanes_dropped"], 0)
        good = d / "good" / "s7.json"
        H.write_json(good, dict(passed=True, s7_passed=True))
        odd = ST.LaneJob("s/s7", "s7", q, [(["a"], "scoring"), (["b"], "training"), (["c"], "scoring")], [str(good)],
                         run=alone.label, setup="alone", learner="td3_bc")
        odd.step_index = 2
        drv2._lane_step_done(odd, 3, 0.0, drv2.free_lane.pop(0), 0.0, 1.0)
        self.assertEqual(drv2.halt["rule"], "S7")
        self.assertIn("does not say it failed", drv2.halt["reason"])

        d3 = fresh("rules_s7_branches3")
        drv3 = ST.Driver(dry_cfg(d3, "--qualities", q, "medium", "--supplement-plan",
                                 plan_file(d3 / "plan.json", {"medium": {"td3_bc": ["alone", "standard", "iw"]}})),
                         None)
        drv3.state["decisions"] = dict(concurrency=1, s6=dict(speeds={}, two_at_once=dict(gain=1.0)))
        drv3.run_cards = lambda jobs, conc: None
        kept = dict(quality=q, passed=True, note="made before the plan was read")
        drv3.state["checks"][f"S6b/{q}/sham"] = kept
        drv3.stage2(q)                                                            # medium-expert: no supplement run
        self.assertIsNone(drv3.halt)
        self.assertEqual(drv3.state["checks"][f"S6b/{q}/sham"], kept)
        for name in ("seed0", "room"):
            self.assertIn("no supplement run (version 4)", drv3.state["checks"][f"S6b/{q}/{name}"]["note"])
        drv3.seed_checks(q, [alone], "seed0")                                     # one run of the learner this seed
        res = drv3.state["checks"][f"S4/{q}/seed0/td3_bc"]
        self.assertTrue(res["skipped"])
        self.assertIn("version 4", res["note"])
        self.assertIsNone(drv3.halt)

    def test_s7_stopped_while_running(self):
        """Version 4, in process: stage-2 runs whose scope's S7 check failed while they trained (TD3+BC's IW setup, then
        the learner) still get their lanes when they finish; the room checks check every room job (version 3 skipped
        one dropped with its learner; version 4 drops none)."""
        q = "medium-expert"
        d = fresh("rules_s7_running")
        drv = ST.Driver(dry_cfg(d, "--seeds", "0", "1"), None)
        _, std, iw = drv.unit_jobs(q, "seed0")
        drv.s7_failed(q, "td3_bc", "iw", iw.label, d / "x", "x/s7")                         # td3_bc's IW setup
        drv.s7_failed(q, "td3_bc", "standard", std.label, d / "y", "y/s7")                  # and the learner
        drv._check_s9 = lambda job, rec: None                                                 # no frozen signature here
        drv.lanes_for = lambda job, rec: [f"{job.label}/lane"]
        for j in (std, iw):
            H.write_json(j.run_dir / "record.json", dict(run_id=j.label, outcome="complete", timings=dict(outer_steps=40)))
            drv._card_done(j, ST.RN.EXIT_COMPLETE, 1.0, 0.0, 2.0)
        self.assertIsNone(drv.halt)
        self.assertEqual(list(drv.lane_queue), [f"{std.label}/lane", f"{iw.label}/lane"])
        self.assertFalse(any("no lanes" in e for e in drv.events))
        self.assertEqual((drv.counts["cards_dropped"], drv.counts["lanes_dropped"]), (0, 0))
        room0, room1 = drv.unit_jobs(q, "room")
        init = dict(initial=dict(agent_state=dict(all="same")), key_carry={})
        for j in (room0, room1):
            alone_dir = ST.RN.run_dir(drv.cfg.root, "stage2", j.run_dir.parts[-3], j.rep, "td3_bc", "alone")
            for path in (alone_dir, j.run_dir):
                H.write_json(path / "record.json", dict(outcome="complete", hashes=init))
        room1.result = dict(status="dropped")                 # a version-3 mark is no reason to skip it any more
        drv.room_checks(q, [room0, room1])
        self.assertIsNone(drv.halt)
        self.assertEqual(sorted(k for k in drv.state["checks"] if k.endswith("/room")),
                         [f"S4/{q}/seed0/td3_bc/room", f"S4/{q}/seed1/td3_bc/room"])

    def test_s7_lanes_seeds_and_steps(self):
        """Version 4 (decision D3): with --s7-all-seeds and --s7-half every seed's alone, standard and IW run gets two
        S7 lanes, on its final actor (lanes/s7-step<final>, label <run>/s7) and on its 50% actor (lanes/s7-step<50%>,
        label <run>/s7-<50%>), and the projections count both; sham-alone runs and room references get none, and stage
        1 only its final actor's; without the options only seed 0's final actor is checked. On real data (prespec.json's
        frozen block) the lanes are lanes/s7-step100000 and lanes/s7-step50000 under the supplement's root, with S7's
        closed loop. Every lane's verdict goes to state.json s7_results when it ends (a pass, exit 0, as well as a
        failure, exit 3; a finished lane whose s7.json does not verify halts, and so does exit 0 with an s7.json that
        says it failed); check keys are written only for seed 0's final actors (and stage 1's), in the two shapes
        export._s7_check_key reads."""
        from experiments.shift_hosts import export as E

        q = "medium-expert"
        d = fresh("rules_s7_lanes")
        plan = {q: {"iql": ["sham_alone", "alone", "standard", "iw", "room"]}}
        drv = ST.Driver(dry_cfg(d, "--learners", "iql", "--seeds", "0", "1", "2", "3", "4", "--supplement-plan",
                                plan_file(d / "plan.json", plan), "--s7-all-seeds", "--s7-half"), None)
        rec = made_up_record(40, refresh_steps=(20, 30), snapshot_steps=(20, 30, 40))
        checked = 0
        for name, units in drv.plan_units(q):
            for u, job in zip(units, drv.unit_jobs(q, name)):
                got = s7_lanes(drv.lanes_for(job, rec))
                if job.setup in ST.STAGE2_SETUPS:
                    self.assertEqual(got, [(f"{job.label}/s7", "s7-step40", "/actors/40"),
                                           (f"{job.label}/s7-20", "s7-step20", "/actors/20")], job.label)
                    self.assertEqual(u["lanes"]["s7"], 2, job.label)
                    checked += 1
                else:
                    self.assertEqual(got, [], job.label)
                    self.assertNotIn("s7", u["lanes"], job.label)
        self.assertEqual(checked, 3 * 5)
        for lane in drv.lanes_for(drv.card_job(q, 3, "iql", "iw", "stage2", 40, "seed3"), rec):
            if lane.kind == "s7":
                self.assertEqual((lane.run, lane.setup, lane.learner),
                                 (f"stage2/made-up-{q}/seed3/iql-iw", "iw", "iql"))
                self.assertIn("--no-closed-loop", lane.commands[-1][0])         # a dry run rolls no policy out
        one = drv.card_job(q, 0, "iql", "iw", "stage1", 30, "stage1")
        self.assertEqual(s7_lanes(drv.lanes_for(one, made_up_record(30, snapshot_steps=(20, 30)))),
                         [(f"{one.label}/s7", "s7-step30", "/actors/30")])           # stage 1: the final actor only
        plain = ST.Driver(dry_cfg(fresh("rules_s7_lanes_plain"), "--seeds", "0", "1"), None)
        for k, want in ((0, ["s7-step40"]), (1, [])):
            job = plain.card_job(q, k, "td3_bc", "iw", "stage2", 40, f"seed{k}")
            self.assertEqual([x[1] for x in s7_lanes(plain.lanes_for(job, rec))], want, k)
            self.assertEqual([u["lanes"].get("s7") for u in dict(plain.plan_units(q))[f"seed{k}"]],
                             [1, 1, 1] if k == 0 else [None, None, None])

        frozen = ST.frozen_config()                                              # real data: prespec.json's block
        cfg = ST.config_from(ST.parser().parse_args(["run"]), frozen)
        self.assertEqual((cfg.steps, cfg.root, cfg.s7_all_seeds, cfg.s7_half), (100_000, S.ROOT, True, True))
        stub = types.SimpleNamespace(cfg=cfg)
        stub._s7_lane = types.MethodType(ST.Driver._s7_lane, stub)
        real = made_up_record(cfg.steps, refresh_steps=(95_000,), snapshot_steps=(50_000, 95_000, 100_000))
        for k in (0, 3):
            job = ST.Driver.card_job(stub, q, k, "iql", "iw", "stage2", cfg.steps, f"seed{k}")
            self.assertEqual(job.run_dir, S.ROOT / "stage2" / q / f"seed{k}" / "iql-iw")
            lanes = ST.Driver.lanes_for(stub, job, real)
            self.assertEqual(s7_lanes(lanes), [(f"{job.label}/s7", "s7-step100000", "/actors/100000"),
                                               (f"{job.label}/s7-50000", "s7-step50000", "/actors/50000")])
            for lane in lanes:
                if lane.kind == "s7":
                    self.assertEqual(Path(lane.outputs[0]).parents[1], S.ROOT / "stage2" / q / f"seed{k}" / "iql-iw" /
                                     "lanes")
                    self.assertNotIn("--no-closed-loop", lane.commands[-1][0])    # S7's closed loop runs
        room = ST.Driver.card_job(stub, q, 2, "iql", "room", "stage2", cfg.steps, "room")
        self.assertEqual(s7_lanes(ST.Driver.lanes_for(stub, room, real)), [])

        iw3 = drv.card_job(q, 3, "iql", "iw", "stage2", 40, "seed3")             # verdicts at seed 3: s7_results only
        lanes3 = [lane for lane in drv.lanes_for(iw3, rec) if lane.kind == "s7"]
        for lane, ok in zip(lanes3, (True, False)):                               # the final actor passes, 50% fails
            H.write_json(lane.outputs[0], dict(passed=True, s7_passed=ok, closed_loop=None))
            lane.step_index = len(lane.commands) - 1                              # the check exits 0, or 3 on a failure
            drv._lane_step_done(lane, 0 if ok else 3, 0.0, drv.free_lane.pop(0), 0.0, 1.0)
            self.assertEqual(lane.result, dict(status="complete") if ok else dict(status="complete", s7_passed=False))
        self.assertIsNone(drv.halt)
        self.assertEqual([(r["label"], r["seed"], r["step"], r["s7_passed"]) for r in drv.state["s7_results"]],
                         [(f"{iw3.label}/s7", 3, 40, True), (f"{iw3.label}/s7-20", 3, 20, False)])
        self.assertEqual([(e["run"], e["scope"], e["step"], e["version"]) for e in drv.state["s7_stopped"]],
                         [(iw3.label, "iw", 20, 4)])
        self.assertEqual([k for k in drv.state["checks"] if k.startswith("S7")], [])
        for j in drv.unit_jobs(q, "seed0"):
            H.write_json(j.run_dir / "record.json", dict(run_id=j.label, outcome="complete", timings=dict(outer_steps=40)))
            H.write_json(j.run_dir / "lanes" / "s7-step40" / "s7.json", dict(passed=True, s7_passed=True))
        drv.s7_seed0_checks(q)
        self.assertIsNone(drv.halt)
        keys = sorted(k for k in drv.state["checks"] if k.startswith("S7"))
        self.assertEqual(keys, sorted(f"S7/{q}/seed0/iql/{s}" for s in ST.STAGE2_SETUPS))
        for k in keys:
            self.assertEqual(E._s7_check_key(k), (q, "iql", k.rsplit("/", 1)[1]))
        self.assertIsNone(E._s7_check_key(f"S7/{q}/seed3/iql/iw"))   # a seed-3 key: export could not place it
        odd = ST.Driver(dry_cfg(fresh("rules_s7_lanes_odd"), "--seeds", "0", "1", "--s7-all-seeds"), None)
        lane = [x for x in odd.lanes_for(odd.card_job(q, 1, "td3_bc", "alone", "stage2", 40, "seed1"), rec)
                if x.kind == "s7"][0]
        H.write_json(lane.outputs[0], dict(passed=True, s7_passed=True))
        Path(lane.outputs[0]).write_text(Path(lane.outputs[0]).read_text() + " ")    # it no longer verifies
        lane.step_index = len(lane.commands) - 1
        odd._lane_step_done(lane, 0, 0.0, odd.free_lane.pop(0), 0.0, 1.0)            # exit 0, but no verified file
        self.assertEqual(odd.halt["rule"], "S7")
        self.assertIn("does not verify", odd.halt["reason"])
        odd0 = ST.Driver(dry_cfg(fresh("rules_s7_lanes_odd0"), "--seeds", "0", "1", "--s7-all-seeds"), None)
        lane = [x for x in odd0.lanes_for(odd0.card_job(q, 1, "td3_bc", "standard", "stage2", 40, "seed1"), rec)
                if x.kind == "s7"][0]
        H.write_json(lane.outputs[0], dict(passed=True, s7_passed=False))
        lane.step_index = len(lane.commands) - 1
        odd0._lane_step_done(lane, 0, 0.0, odd0.free_lane.pop(0), 0.0, 1.0)          # exit 0, but its file failed
        self.assertEqual(odd0.halt["rule"], "S7")
        self.assertIn("exited 0 but its s7.json says it failed", odd0.halt["reason"])
        self.assertEqual([(r["label"], r["s7_passed"]) for r in odd0.state["s7_results"]], [(lane.label, False)])
        READOUTS["s7_lanes"] = dict(real=s7_lanes(ST.Driver.lanes_for(stub, job, real)))

    def test_supplement_plan_units(self):
        """Version 4's frozen plan (prespec.json "supplement", supplement.validate through frozen_config): stage 2
        lists exactly the plan's runs, in supplement.run_ids' order (medium-expert 56: ReBRAC's learner scope 20,
        IQL's 21, TD3+BC's IW scope 15; medium 42: TD3+BC's and IQL's learner scopes, 21 each), with two S7 lanes
        projected for every alone, standard and IW run; stage 1 still runs all 20 runs (every learner, five setups)."""
        frozen = ST.frozen_config()
        block = S.prespec_block()
        self.assertEqual(frozen["plan"], block["plan"])
        self.assertEqual(frozen["supplement"], block)
        self.assertEqual({q: len(ids) for q, ids in S.run_ids(frozen["plan"], frozen["seeds"]).items()},
                         {"medium-expert": 56, "medium": 42})
        d = fresh("rules_plan_units")
        cfg = ST.config_from(ST.parser().parse_args(["run", "--dry-run", "--root", str(d), "--supplement-plan",
                                                     plan_file(d / "plan.json", frozen["plan"]), "--s7-all-seeds",
                                                     "--s7-half"]))
        self.assertEqual((cfg.learners, cfg.qualities, cfg.seeds), (RN.LEARNERS, RN.QUALITIES, ST.SEEDS))
        drv = ST.Driver(cfg, None)
        self.assertEqual((drv.state["config"]["plan"], drv.state["config"]["s7_all_seeds"],
                          drv.state["config"]["s7_half"]), (frozen["plan"], True, True))
        made = S.run_ids(frozen["plan"], cfg.seeds, cfg.learners, made_up=True)
        per = {}
        for q, want in (("medium-expert", 56), ("medium", 42)):
            units = drv.plan_units(q)
            labels = [j.label for name, _ in units for j in drv.unit_jobs(q, name)]
            self.assertEqual(labels, made[q], q)
            self.assertEqual(len(labels), want, q)
            self.assertEqual(sum(len(us) for _, us in units), want, q)
            for _, us in units:
                for u in us:
                    self.assertEqual(u["lanes"].get("s7"), 2 if u["setup"] in ST.STAGE2_SETUPS else None, u)
            for rid in labels:
                key = (q, rid.split("/")[3].split("-")[0])
                per[key] = per.get(key, 0) + 1
        self.assertEqual(per, {("medium-expert", "rebrac"): 20, ("medium-expert", "iql"): 21,
                               ("medium-expert", "td3_bc"): 15, ("medium", "td3_bc"): 21, ("medium", "iql"): 21})
        launched = []
        drv.run_cards = lambda jobs, conc: launched.extend(jobs)                   # nothing is launched
        drv.wait = lambda *a, **k: None
        drv.stage1_checks = lambda jobs: None
        drv.stage1()
        self.assertEqual(len(launched), 20)
        self.assertEqual(sorted((j.learner, j.setup) for j in launched),
                         sorted((l, s) for l in RN.LEARNERS for s in ST.STAGE1_SINGLE + ST.STAGE1_PAIR))
        self.assertEqual({(j.stage, j.rep) for j in launched}, {("stage1", 0)})
        READOUTS["supplement_plan"] = dict(runs={q: len(v) for q, v in made.items()}, per_scope={
            f"{q}/{l}": n for (q, l), n in per.items()}, stage1=len(launched))

    def test_supplement_root_and_ledger(self):
        """Version 4 writes only under runs/shift_hosts/supplement_v4 with its own driver files and ledger: the real
        configuration's root is it (run folders and the runner's --out-root under it); a dry run refuses it, anything
        inside it and version 3's runs/shift_hosts; a driver's ledger is <root>/driver/ledger.json; preflight reads the
        supplement's own ledger, never version 3's runs/shift_hosts/driver/ledger.json (so the first start needs both
        stage-0 hours although version 3's ledger holds its stage-0 entry), checks version 3's files against the frozen
        block, and refuses a code change outside supplement.CHANGEABLE; status and request-stop default to the
        supplement's root (a scratch folder stands in for it: nothing is written under runs/shift_hosts/supplement_v4
        or version 3's folders)."""
        self.assertEqual(S.ROOT, RN.RUNS / "supplement_v4")
        self.assertEqual(S.ROOT_REL, "runs/shift_hosts/supplement_v4")
        for bad in (RN.RUNS, S.ROOT, S.ROOT / "driver", S.ROOT / "stage2" / "x"):
            with self.assertRaises(gate.ApprovalError, msg=str(bad)):
                ST.config_from(ST.parser().parse_args(["run", "--dry-run", "--root", str(bad)]))
        frozen = ST.frozen_config()
        cfg = ST.config_from(ST.parser().parse_args(["run"]), frozen)
        self.assertEqual(cfg.root, S.ROOT)
        stub = types.SimpleNamespace(cfg=cfg)
        for stage, steps in (("stage1", cfg.stage1_steps), ("stage2", cfg.steps)):
            job = ST.Driver.card_job(stub, "medium", 0, "iql", "alone", stage, steps, stage)
            self.assertEqual(job.run_dir, S.ROOT / stage / "medium" / "seed0" / "iql-alone")
            self.assertEqual(job.argv[job.argv.index("--out-root") + 1], str(S.ROOT))
        d = fresh("rules_root")
        drv = ST.Driver(dry_cfg(d), None)
        self.assertEqual((drv.dir, drv.ledger.path), (d / "driver", d / "driver" / "ledger.json"))

        tmp = fresh("rules_preflight") / "supplement_v4"
        read, sources = [], []
        kept = B.read_json_kept

        def spy(path, *a, **k):
            read.append(Path(path))
            return kept(path, *a, **k)

        first = ST.parser().parse_args(["run"])
        hours = ST.parser().parse_args(["run", "--stage0-card-hours", "0", "--stage0-processor-hours", "0"])
        with mock.patch.object(gate, "require_run_approval", lambda purpose: "approval"), \
                mock.patch.object(gate, "require_s0", lambda purpose: None), \
                mock.patch.object(S, "check_sources", lambda block: sources.append(block)), \
                mock.patch.object(S, "ROOT", tmp), mock.patch.object(B, "read_json_kept", spy):
            with self.assertRaisesRegex(gate.ApprovalError, "first start names stage 0's compute"):
                ST.preflight(first)                     # version 3's ledger, with its stage-0 entry, is not read
            approval, fz = ST.preflight(hours)          # the real code manifest: supplement.code_unchanged is []
            self.assertEqual((approval, fz["root"], fz["plan"]), ("approval", tmp, frozen["plan"]))
            self.assertEqual(sources, [frozen["supplement"]] * 2)
            led = B.Ledger(tmp / "driver" / "ledger.json", mode="real")
            led.set_stage0(0.0, 0.0, "test: the supplement's own ledger, entered at its first start")
            led.save()
            ST.preflight(first)                         # now the supplement's own ledger holds stage 0
            H.write_json(tmp / "driver" / "ledger.json", dict(led.body, mode="dry-run"))
            with self.assertRaisesRegex(gate.ApprovalError, "not a real study's ledger"):
                ST.preflight(first)
            with mock.patch.object(S, "code_unchanged", lambda manifest: [dict(path="experiments/shift_hosts/"
                                                                                    "export.py", change="changed")]):
                with self.assertRaisesRegex(gate.ApprovalError, "may change only"):
                    ST.preflight(hours)
        self.assertTrue(read)
        self.assertTrue(all(tmp in p.parents for p in read), read)
        self.assertFalse([p for p in read if RN.RUNS / "driver" in (p, *p.parents)])
        with mock.patch.object(S, "ROOT", tmp):        # status and request-stop default to the supplement's root
            H.write_json(tmp / "driver" / "status.json", dict(schema=ST.STATUS_SCHEMA, state="made-up"))
            self.assertEqual(ST.main(["status"]), 0)
            self.assertEqual(ST.main(["request-stop"]), 0)
            self.assertTrue((tmp / "driver" / "STOP_REQUEST").exists())

    def test_halt_launches_no_next_lane_step(self):
        """After a halt, a lane whose step finished does not start its next step (the module docstring's promise)."""
        d = fresh("rules_halt")
        drv = ST.Driver(dry_cfg(d), None)
        job = ST.LaneJob("x/s7", "s7", "medium-expert", [(["a"], "scoring"), (["b"], "training")], [str(d / "o")])
        core = drv.free_lane.pop(0)
        drv.halt = dict(rule="S5", reason="test")
        drv._lane_step_done(job, 0, 0.0, core, 0.0, 1.0)
        self.assertEqual(job.result, dict(status="halted", step=0))
        self.assertEqual(drv.lanes, {})
        self.assertIn(core, drv.free_lane)

    def test_nonfinite_return_is_s5(self):
        """A score lane's non-finite return (exit 4, or found on resume): the learner's Claim 0 outcome for alone and
        room runs (recorded, no halt), a halt for a BCA setup unless the run was accepted with --accept-s5."""
        d = fresh("rules_s5")
        path = fake_scores(d / "scores.npz", finite=False)
        drv = ST.Driver(dry_cfg(d), None)
        alone = ST.LaneJob("r1/score-40", "score", "medium-expert", [], [str(path)], run="r1", setup="alone")
        drv._verdict(alone)
        self.assertIsNone(drv.halt)
        self.assertEqual(drv.state["claim0_failures"][0]["run"], "r1")
        bca = ST.LaneJob("r2/score-40", "score", "medium-expert", [], [str(path)], run="r2", setup="standard")
        drv._verdict(bca)
        self.assertEqual(drv.halt["rule"], "S5")
        acc = ST.Driver(dry_cfg(fresh("rules_s5b"), "--accept-s5", "r2"), None)
        acc._verdict(bca)
        self.assertIsNone(acc.halt)
        self.assertEqual(acc.state["accepted_s5"], ["r2"])
        clean = fake_scores(d / "clean.npz", finite=True)
        acc._verdict(ST.LaneJob("r3/score-40", "score", "medium-expert", [], [str(clean)], run="r3", setup="iw"))
        self.assertIsNone(acc.halt)
        self.assertEqual(ST.EXIT_NONFINITE, 4)

    def test_stop_before_training_names_its_rule(self):
        """A runner stop before execute (S11 here) writes no record; the driver takes the rule from its status line."""
        d = fresh("rules_s11")
        drv = ST.Driver(dry_cfg(d), None)
        job = drv.card_job("medium-expert", 0, "td3_bc", "alone", "stage2", 40, "seed0")
        log = drv.dir / "logs" / (job.label.replace("/", "__") + ".log")
        log.write_text(json.dumps(dict(status="stopped", record=False, run_id=None, outcome="stopped",
                                       stop=dict(rule="S11", step=0, reason="rows differ"))) + "\n", encoding="utf8")
        drv._card_done(job, RN.EXIT_STOPPED, 0.5, 0.0, 1.0)
        self.assertEqual(drv.halt["rule"], "S11")
        self.assertIn("rows differ", drv.halt["reason"])

    def test_modes_configuration_and_lock(self):
        """A dry run never uses the real roots (version 3's and version 4's); a driver of the other mode, or a resume
        with another configuration (version 4's plan and S7 lanes included), is refused; one driver per root (the
        lock), and a refused second start leaves a pending STOP_REQUEST alone."""
        for root in (RN.RUNS, S.ROOT, S.ROOT / "stage1"):
            with self.assertRaises(gate.ApprovalError, msg=str(root)):
                ST.config_from(ST.parser().parse_args(["run", "--dry-run", "--root", str(root)]))
        d = fresh("rules_modes")
        ST.Driver(dry_cfg(d), None)
        with self.assertRaises(gate.ApprovalError):                       # another configuration
            ST.Driver(dry_cfg(d, "--seeds", "0", "1"), None)
        plan = plan_file(ROOT / "rules_modes.plan.json", {"medium-expert": {"td3_bc": ["alone", "standard", "iw"]}})
        for more in (["--supplement-plan", plan], ["--s7-all-seeds"], ["--s7-half"]):
            with self.assertRaises(gate.ApprovalError, msg=more):         # version 4: another plan or S7 lanes
                ST.Driver(dry_cfg(d, *more), None)
        with self.assertRaises(gate.ApprovalError):                       # a plan naming a learner not run
            dry_cfg(d, "--supplement-plan", plan_file(ROOT / "rules_modes.bad.json",
                                                      {"medium-expert": {"iql": ["alone", "standard", "iw"]}}))
        real = ST.config_from(ST.parser().parse_args(["run", "--root", str(d)]))
        with self.assertRaises(gate.ApprovalError):                       # a real driver on a dry run's files
            ST.Driver(real, None)
        with self.assertRaises(ValueError):
            B.Ledger(d / "driver" / "ledger.json", mode="real")
        ST.Driver(dry_cfg(d), None)                                         # the same configuration resumes
        lock_root = fresh("rules_lock")
        (lock_root / "driver").mkdir(parents=True)
        (lock_root / "driver" / "STOP_REQUEST").write_text("pending\n")
        fd = ST.acquire_lock(lock_root / "driver")
        try:
            self.assertIsNotNone(ST.lock_holder(lock_root / "driver"))
            code, out = driver(["run", "--dry-run", "--root", str(lock_root), "--learners", "td3_bc", "--qualities",
                                "medium-expert", "--seeds", "0"], timeout=600)
            self.assertEqual(code, 2, out[-2000:])
            self.assertIn("another stage driver holds", out)
            self.assertTrue((lock_root / "driver" / "STOP_REQUEST").exists())
            self.assertFalse((lock_root / "driver" / "state.json").exists())
        finally:
            ST._LOCKS.remove(fd)
            os.close(fd)
        self.assertIsNone(ST.lock_holder(lock_root / "driver"))

    def test_real_options_are_refused(self):
        """On real data the options that would change a frozen choice or limit are refused (prespec.json's values pass,
        as no change): version 4's --supplement-plan, --s7-all-seeds and --s7-half, and any --root but the supplement's
        (version 3's runs/shift_hosts included). The real configuration takes the plan and the S7 lanes from
        prespec.json's frozen block (frozen_config: supplement.validate), which refuses a block whose plan is not what
        its scopes give, S7 steps other than the runner's final and 50% steps, or no block at all."""
        frozen = dict(learners=RN.LEARNERS, qualities=RN.QUALITIES, seeds=(0, 1, 2, 3, 4), choice19=dict(ST.CHOICE19),
                      card_cores=[0, 1], lane_cores=list(range(2, 10)), s9_signature={k: None for k in RN.S9_KEYS},
                      root=S.ROOT, plan={"medium": {"cql": ["alone", "standard", "iw"]}}, s7_all_seeds=True,
                      s7_half=True, supplement=None)
        ok = ST.parser().parse_args(["run", "--seeds", "0", "1", "2", "3", "4", "--learners"] + list(RN.LEARNERS))
        ST.check_real_args(ok, frozen)
        ST.check_real_args(ST.parser().parse_args(["run", "--root", str(S.ROOT)]), frozen)
        for more in (["--choice19", "td3_bc=no"], ["--learners", "cql"], ["--qualities", "medium"],
                     ["--seeds", "0"], ["--card-device", "cpu"], ["--root", str(ROOT)], ["--caps", "1,1"],
                     ["--card-cores", "0", "1", "2"], ["--lane-cores"] + [str(c) for c in range(2, 11)],
                     ["--lane-cores", "1", "2"], ["--root", str(RN.RUNS)], ["--supplement-plan", "plan.json"],
                     ["--s7-all-seeds"], ["--s7-half"]):
            with self.assertRaises(gate.ApprovalError, msg=more):
                ST.check_real_args(ST.parser().parse_args(["run"] + more), frozen)
        cfg = ST.config_from(ST.parser().parse_args(["run"]), dict(frozen, seeds=(0, 1)))
        self.assertEqual((cfg.seeds, cfg.choice19, cfg.root), ((0, 1), ST.CHOICE19, S.ROOT))
        self.assertEqual((cfg.plan, cfg.s7_all_seeds, cfg.s7_half), (frozen["plan"], True, True))
        with self.assertRaises(gate.ApprovalError):                       # a plan file is a dry run's only
            ST.config_from(ST.parser().parse_args(["run", "--supplement-plan", "plan.json"]), frozen)
        real = ST.frozen_config()
        block = S.prespec_block()
        self.assertEqual((real["root"], real["plan"], real["s7_all_seeds"], real["s7_half"], real["supplement"]),
                         (S.ROOT, block["plan"], True, True, block))
        pre = json.loads((gate.PACKAGE / gate.PRESPEC_NAME).read_text(encoding="utf8"))
        short = json.loads(json.dumps(block))
        short["plan"]["medium"]["iql"].remove("room")                     # not what its scopes give
        extra = json.loads(json.dumps(block))
        extra["plan"]["medium"]["cql"] = ["alone", "standard", "iw"]      # a learner no scope names
        quarter = json.loads(json.dumps(block))
        quarter["s7"]["steps"] = [25_000, 100_000]
        for name, body in (("plan without IQL's room references", dict(pre, supplement=short)),
                           ("plan with CQL", dict(pre, supplement=extra)),
                           ("S7 at 25,000 steps", dict(pre, supplement=quarter)),
                           ("no supplement block", {k: v for k, v in pre.items() if k != "supplement"})):
            pkg = fresh("rules_frozen") / "package"
            pkg.mkdir(parents=True)
            (pkg / gate.PRESPEC_NAME).write_text(json.dumps(body, indent=1) + "\n", encoding="utf8")
            with self.assertRaises(gate.ApprovalError, msg=name):
                ST.frozen_config(package=pkg)
        (pkg / gate.PRESPEC_NAME).write_text(json.dumps(pre, indent=1) + "\n", encoding="utf8")
        self.assertEqual(ST.frozen_config(package=pkg)["plan"], block["plan"])     # the unchanged copy passes

    def test_s6_and_s6b_on_resume(self):
        """A resume reuses stage 1's S6 decision while stage 1's records are unchanged, and projects S6b only for the
        runs of a unit without a record (none left: the check is skipped, never failed on totals that already hold
        the unit's own work)."""
        speed = dict(ms_per_step=1.0, startup_seconds=100.0, refresh_seconds=0.0, table_seconds=0.0, busy=1.0,
                     steps=30, wall=1.0)              # one dry run: 100 s + 40 steps x 1 ms = 0.0278 card-hours

        def fresh_driver(d):
            drv = ST.Driver(dry_cfg(d), None)
            drv.state["decisions"] = dict(concurrency=1, s6=dict(
                speeds={f"td3_bc/{k}": speed for k in ST.STAGE1_SINGLE}, two_at_once=dict(gain=1.0)))
            drv.run_cards = lambda jobs, conc: None    # the records stand for the runs: nothing is launched
            drv.seed_checks = drv.room_checks = lambda *a: None
            return drv

        d = fresh("rules_s6b")
        drv = fresh_driver(d)
        drv.ledger.set_stage0(13.99, 0.0, "test: the card cap all but used, by the units' own runs (a resume)")
        for name in ("sham", "seed0", "room"):          # every run of every unit already has its record
            for j in drv.unit_jobs("medium-expert", name):
                H.write_json(j.run_dir / "record.json", dict(run_id=j.label, outcome="complete"))
        drv.stage2("medium-expert")
        self.assertIsNone(drv.halt)                      # it used to project every unit again on top of the totals
        for name in ("sham", "seed0", "room"):
            self.assertTrue(drv.state["checks"][f"S6b/medium-expert/{name}"]["skipped"])
        d2 = fresh("rules_s6b_part")
        drv2 = fresh_driver(d2)
        drv2.ledger.set_stage0(13.95, 0.0, "test: one more run fits under the card cap (13.978), three do not (14.033)")
        for name in ("sham", "seed0", "room"):
            jobs = drv2.unit_jobs("medium-expert", name)
            for j in jobs[1:] if name == "seed0" else jobs:   # seed 0's first run (td3_bc alone) has no record yet
                H.write_json(j.run_dir / "record.json", dict(run_id=j.label, outcome="complete"))
        drv2.stage2("medium-expert")
        self.assertIsNone(drv2.halt)
        det = drv2.state["checks"]["S6b/medium-expert/seed0"]
        self.assertEqual((det["runs_projected"], det["runs_with_records"]), (1, 2))
        self.assertAlmostEqual(det["projected"]["card_hours"], 100.04 / 3600, places=9)

    def test_unexpected_error_writes_the_stop_report(self):
        """An exception inside the driver: it launches nothing more, writes STOP.json (rule "error") and exits 1."""
        d = fresh("rules_error")
        drv = ST.Driver(dry_cfg(d), None)

        def boom():
            raise RuntimeError("made-up failure")

        drv.prepare_inputs = boom
        self.assertEqual(drv.run(), 1)
        stop = H.read_json(d / "driver" / "STOP.json")
        self.assertEqual(stop["rule"], "error")
        self.assertIn("made-up failure", stop["reason"])
        self.assertIn("S7 failures recorded so far (version 4; nothing was dropped for them): 0",
                      (d / "driver" / "STOP.txt").read_text(encoding="utf8"))
        self.assertEqual(H.read_json(d / "driver" / "status.json")["state"], "stopped")


class LedgerTest(unittest.TestCase):
    def test_crash_safety(self):
        """Open entries count to their last heartbeat and are marked orphaned by the next session; a run that finished
        unwatched is recovered from its record once, less what open entries already count; a ledger whose sidecar
        does not match falls back to its kept previous version; the driver's own processor time is counted."""
        d = fresh("ledger_crash")
        led = B.Ledger(d / "ledger.json", mode="dry-run")
        ce = led.add_card("medium-expert", "r", "s1", 100.0, 100.0, open=True)
        pe = led.add_processor("medium-expert", "r", 0.0, "card", open=True, session="s1")
        ce["end"], pe["seconds"] = 160.0, 50.0                  # a heartbeat
        led.save()
        again = B.Ledger(d / "ledger.json", mode="dry-run")
        self.assertEqual(again.orphan_open_entries("s2"), 2)
        self.assertAlmostEqual(again.card_hours("medium-expert"), 60.0 / 3600, places=12)
        self.assertEqual(again.recover("medium-expert", "r", 100.0, 80.0), (40.0, 30.0))
        self.assertAlmostEqual(again.card_hours("medium-expert"), 100.0 / 3600, places=12)
        self.assertAlmostEqual(again.processor_hours("medium-expert"), 80.0 / 3600, places=12)
        again.add_card("medium-expert", "q", "s2", 0.0, 10.0)
        self.assertIsNone(again.recover("medium-expert", "q", 99.0, 99.0))   # a closed interval counts it already
        again.add_to("medium-expert", "driver/s2", 3.0, "driver")
        again.add_to("medium-expert", "driver/s2", 4.0, "driver")
        self.assertEqual([e["seconds"] for e in again.body["processor"] if e["kind"] == "driver"], [7.0])
        again.save()
        again.save()
        (d / "ledger.json.sha256").write_text("0" * 64 + "  ledger.json\n")   # a crash between file and sidecar
        kept = B.Ledger(d / "ledger.json", mode="dry-run")
        self.assertIn("previous version", kept.note)
        self.assertEqual(kept.body["mode"], "dry-run")
        with self.assertRaises(ValueError):
            B.Ledger(d / "ledger.json", mode="real")


class GateRulesTest(unittest.TestCase):
    def test_manifest_coverage_bytecode_and_s0(self):
        """The code manifest hashes every package file (any name or suffix) but prespec.json and the root approval
        files; bytecode beside the code is found (require_run_approval refuses it); stop rule S0 passes only for a
        verified, passing stage-0 summary made on exactly the code on disk."""
        tmp = fresh("gate_rules")
        pkg = tmp / "experiments" / "shift_hosts"
        (pkg / "tests").mkdir(parents=True)
        for name in ("module.py", "approval_helpers.py", "notes.cfg", "prespec.json", "tests/approval_0123456789abcdef.json",
                     "approval_0123456789abcdef.json", "approval_0123456789abcdef.json.sha256"):
            (pkg / name).write_text("x\n")
        rel = sorted(p.relative_to(pkg).as_posix() for p in gate.package_files(pkg))
        self.assertEqual(rel, ["approval_helpers.py", "module.py", "notes.cfg", "tests/approval_0123456789abcdef.json"])
        self.assertEqual(gate.bytecode_files(pkg, tmp), [])
        (pkg / "__pycache__").mkdir()
        (pkg / "__pycache__" / "module.cpython-310.pyc").write_bytes(b"\0")
        try:
            self.assertTrue(gate.bytecode_files(pkg, tmp))
            self.assertNotIn("experiments/shift_hosts/__pycache__/module.cpython-310.pyc",
                             gate.code_manifest(pkg, tmp))
            with self.assertRaisesRegex(gate.ApprovalError, "compiled bytecode"):
                gate.require_run_approval("test", package=pkg, repo=tmp)
        finally:                                  # the suite's hygiene check looks for any __pycache__ under the repo
            shutil.rmtree(pkg / "__pycache__")
        self.assertEqual(gate.bytecode_files(), [])            # the real package and its external code: clean
        summary = tmp / "SUMMARY.json"
        now = gate.manifest_sha256(gate.code_manifest())
        for body, want in ((dict(schema=gate.S0_SCHEMA, status=0, s0=dict(passed=True), made_up_arrays_only=True,
                                 manifest=dict(code_manifest_sha256=now)), True),
                           (dict(schema=gate.S0_SCHEMA, status=0, s0=dict(passed=True), made_up_arrays_only=True,
                                 manifest=dict(code_manifest_sha256="0" * 64)), False),
                           (dict(schema=gate.S0_SCHEMA, status=1, s0=dict(passed=False), made_up_arrays_only=True,
                                 manifest=dict(code_manifest_sha256=now)), False)):
            H.write_json(summary, body)
            self.assertEqual(gate.s0_status(summary=summary)["passed"], want, body)
        H.write_json(summary, dict(schema=gate.S0_SCHEMA, status=0, s0=dict(passed=True), made_up_arrays_only=True,
                                   manifest=dict(code_manifest_sha256=now)))
        summary.write_text(summary.read_text() + " ")
        self.assertIn("sidecar", gate.s0_status(summary=summary)["reason"])
        self.assertEqual(gate.manifest_sha256(gate.code_manifest()), RN.canonical_sha256(gate.code_manifest()))
        READOUTS["s0_now"] = gate.s0_status()


# ------------------------------------------------------------------------------------------------ the dry run

class DryRunTest(unittest.TestCase):
    """The whole driver in dry-run mode, then again (resumed)."""

    root = ROOT / "dryrun"

    def test_1_full_dry_run(self):
        """Every stage and check on made-up arrays, tiny step counts, processor."""
        if self.root.exists():
            shutil.rmtree(self.root)
        code, out = driver(["run", "--dry-run", "--root", str(self.root), "--seeds", "0", "1"])
        self.assertEqual(code, 0, out[-4000:])
        st, state = status_of(self.root), state_of(self.root)
        self.assertEqual((st["state"], st["stop"]), ("finished", None))
        failed = [k for k, v in state["checks"].items() if not v.get("passed")]
        self.assertEqual(failed, [])
        names = set(state["checks"])
        for l in RN.LEARNERS:
            for rule in ("S2", "S8", "S4"):
                self.assertIn(f"{rule}/{l}", names)
            for s in ("standard", "iw", "iw_ones", "iw_repeat"):
                self.assertIn(f"S10/{l}/{s}", names)
                self.assertEqual(state["checks"][f"S10/{l}/{s}"]["readout"], ST.CHOICE19[l])
        for q in RN.QUALITIES:
            for unit in ("sham", "seed0", "seed1", "room"):
                self.assertIn(f"S6b/{q}/{unit}", names)
            for l in RN.LEARNERS:
                for s in ST.STAGE2_SETUPS:                    # every seed-0 S7 result read after the quality's lanes
                    self.assertTrue(state["checks"][f"S7/{q}/seed0/{l}/{s}"]["passed"])
        for l in RN.LEARNERS:
            for s in ST.STAGE1_SINGLE + ST.STAGE1_PAIR:       # every stage-1 S7 result read by stage 1's checks
                self.assertTrue(state["checks"][f"S7/{l}/{s}"]["passed"])
        self.assertEqual((state["mode"], state["config"]["seeds"]), ("dry-run", [0, 1]))
        self.assertEqual(state["claim0_failures"], [])
        recs = list(self.root.glob("stage*/*/seed*/*/record.json"))
        sham = sum(ST.CHOICE19.values())
        self.assertEqual(len(recs), 20 + 2 * (sham + 2 * 12 + 2 * 4))
        for p in recs:
            self.assertEqual(H.read_json(p)["outcome"], "complete", str(p))
        s7 = list(self.root.glob("stage*/*/seed*/*/lanes/s7-step*/s7.json"))
        self.assertEqual(len(s7), 20 + 2 * 12)
        for p in s7:
            self.assertTrue(H.read_json(p)["s7_passed"], str(p))
        # version 4 without a plan: the full grid, S7 on seed 0's final actors only, every verdict in s7_results
        self.assertEqual((state["config"]["plan"], state["config"]["s7_all_seeds"], state["config"]["s7_half"]),
                         (None, False, False))
        self.assertEqual(state["s7_stopped"], [])
        self.assertEqual(sorted(r["file"] for r in state["s7_results"]), sorted(str(p) for p in s7))
        self.assertTrue(all(r["s7_passed"] is True for r in state["s7_results"]))
        self.assertEqual((st["counts"]["cards_dropped"], st["counts"]["lanes_dropped"]), (0, 0))
        cov = list(self.root.glob("stage2/*/seed*/*/lanes/coverage-step*/coverage.json"))
        self.assertEqual(len(cov), 2 * 2 * 8)
        dec = state["decisions"]
        led = B.Ledger(self.root / "driver" / "ledger.json")
        two = dec["concurrency"] == 2
        overlap = 0
        if two:
            iv = sorted((e["start"], e["end"]) for e in led.body["card"] if e["label"].startswith("stage2/"))
            overlap = sum(1 for a, b in zip(iv, iv[1:]) if b[0] < a[1])
            self.assertGreater(overlap, 0)
        READOUTS["dry_run"] = dict(concurrency=dec["concurrency"], gain=dec["s6"]["two_at_once"]["gain"],
                                   s6_passed=dec["s6"]["passed"], alone_ms=dec["s6"]["alone"], records=len(recs),
                                   s7=len(s7), coverage=len(cov), overlapping_stage2_pairs=overlap,
                                   budget=B.describe(led), s9_reference=state["s9_reference"]["source"],
                                   counts=st["counts"])

    def test_2_resume_skips_everything(self):
        """The same command again: every run and lane is skipped, and it finishes."""
        code, out = driver(["run", "--dry-run", "--root", str(self.root), "--seeds", "0", "1"])
        self.assertEqual(code, 0, out[-4000:])
        st = status_of(self.root)
        self.assertEqual(st["state"], "finished")
        self.assertEqual(st["counts"]["cards_done"], st["counts"]["cards_skipped"])
        self.assertEqual(st["counts"]["lanes_done"], 0)
        log = (self.root / "driver" / "driver.log").read_text(encoding="utf8")
        self.assertIn("S6: stage 1's decision of", log)           # reused, not recomputed against later totals
        self.assertIn("is reused (stage 1's records unchanged)", log)
        self.assertIn("skipped, every run of the unit has a record", log)
        led = B.Ledger(self.root / "driver" / "ledger.json")
        self.assertTrue(any(e["kind"] == "driver" and e["seconds"] > 0 for e in led.body["processor"]))
        self.assertFalse(any(e.get("open") is True for e in led.body["card"] + led.body["processor"]))
        READOUTS["resume"] = st["counts"]


class StopTest(unittest.TestCase):
    def test_simulated_s5_halts_and_is_never_retried(self):
        """A BCA run stopped by S5 halts the driver; restarts never retry it; accepting it as the outcome finishes."""
        root = fresh("stop_s5")
        target = "stage2/made-up-medium-expert/seed1/td3_bc-standard"
        base = ["run", "--dry-run", "--root", str(root), "--learners", "td3_bc", "--qualities", "medium-expert",
                "--seeds", "0", "1"]
        code, out = driver(base + ["--inject", f"{target}=unit0:15"])
        self.assertEqual(code, 3, out[-4000:])
        stop = H.read_json(root / "driver" / "STOP.json")
        self.assertEqual(stop["rule"], "S5")
        self.assertIn(target, stop["reason"])
        self.assertTrue((root / "driver" / "STOP.txt").exists())
        rec = H.read_json(RN.run_dir(root, "stage2", "made-up-medium-expert", 1, "td3_bc", "standard") / "record.json")
        self.assertEqual((rec["outcome"], rec["stop"]["rule"], rec["stop"]["step"]), ("stopped", "S5", 11))
        self.assertFalse(list(root.glob("stage2/*/seed*/td3_bc-room")))     # nothing after the stop
        code, out = driver(base)
        self.assertEqual(code, 2, out[-2000:])                              # a stop report is waiting
        code, out = driver(base + ["--resume-after-stop"])
        self.assertEqual(code, 3, out[-2000:])                              # the recorded stop again, not retried
        again = H.read_json(RN.run_dir(root, "stage2", "made-up-medium-expert", 1, "td3_bc", "standard") /
                            "record.json")
        self.assertEqual(again, rec)
        code, out = driver(base + ["--resume-after-stop", "--accept-s5", target])
        self.assertEqual(code, 0, out[-4000:])
        self.assertEqual(status_of(root)["state"], "finished")
        self.assertEqual(len(list(root.glob("stage2/*/seed*/td3_bc-room/record.json"))), 2)
        READOUTS["stop_s5"] = dict(rule=stop["rule"], reason=stop["reason"][:200])

    def test_supplement_dry_run(self):
        """Version 4 end to end (PRESPEC.md section 21): a dry run of a supplement-shaped plan (supplement.plan_of:
        IQL's learner scope and TD3+BC's IW scope on medium-expert, TD3+BC's learner scope on medium) with S7 on every
        seed's final and 50% actors, and S7 failures injected on stage 1's TD3+BC standard actor, seed 0's IQL alone
        actor and seed 1's TD3+BC IW actor (both its lanes). The driver finishes without a stop report; stage 1 runs
        whole (10 runs); stage 2 trains exactly the plan's runs (supplement.run_ids) and drops nothing; each failure is
        recorded once in s7_stopped with version 4, every lane's verdict in s7_results (seeds 0 and 1, steps 20 and
        40); check keys exist only for stage 1 and seed 0, in export's two shapes; the log says version 4. The same
        command again retrains and records nothing new. test_claims builds the supplement's export over this root."""
        from experiments.shift_hosts import export as E

        root = fresh("supplement")
        q = "medium-expert"
        scopes = [dict(quality=q, learner="iql", scope="learner"), dict(quality=q, learner="td3_bc", scope="iw"),
                  dict(quality="medium", learner="td3_bc", scope="learner")]
        plan = S.plan_of(S.scope_map(scopes), ST.CHOICE19)
        s1 = "stage1/made-up-medium-expert/seed0/td3_bc-standard"
        s2 = "stage2/made-up-medium-expert/seed0/iql-alone"
        s3 = "stage2/made-up-medium-expert/seed1/td3_bc-iw"
        args = (["run", "--dry-run", "--root", str(root), "--learners", "td3_bc", "iql", "--seeds", "0", "1",
                 "--supplement-plan", plan_file(ROOT / "supplement.plan.json", plan), "--s7-all-seeds", "--s7-half",
                 "--inject"] + [f"{r}={ST.S7_INJECT}" for r in (s1, s2, s3)])
        code, out = driver(args)
        self.assertEqual(code, 0, out[-4000:])
        self.assertFalse((root / "driver" / "STOP.json").exists())
        st, state = status_of(root), state_of(root)
        self.assertEqual((st["state"], st["stop"]), ("finished", None))
        self.assertEqual((st["counts"]["cards_dropped"], st["counts"]["lanes_dropped"]), (0, 0))
        self.assertEqual(state["config"]["plan"], plan)
        self.assertEqual(len(list(root.glob("stage1/*/seed0/*/record.json"))), 10)          # stage 1 whole
        want = S.run_ids(plan, (0, 1), ("td3_bc", "iql"), made_up=True)
        planned = sorted(r for ids in want.values() for r in ids)
        trained = sorted(p.parent.relative_to(root).as_posix() for p in root.glob("stage2/*/seed*/*/record.json"))
        self.assertEqual(trained, planned)
        self.assertEqual(len(trained), 15 + 9)
        for p in root.glob("stage*/*/seed*/*/record.json"):
            self.assertEqual(H.read_json(p)["outcome"], "complete", str(p))
        stopped = sorted((e["learner"], e["scope"], e["setup"], e["stage"], e["run"], e["version"])
                         for e in state["s7_stopped"])
        self.assertEqual(stopped, sorted([("td3_bc", "learner", "standard", "stage1", s1, 4),
                                          ("iql", "learner", "alone", "stage2", s2, 4),
                                          ("td3_bc", "iw", "iw", "stage2", s3, 4)]))
        for rid in planned:                             # S7 on the final and 50% actors of every alone, standard, IW
            setup = rid.split("/")[3].partition("-")[2]
            got = sorted(p.parent.name for p in (root / rid / "lanes").glob("s7-step*/s7.json"))
            self.assertEqual(got, ["s7-step20", "s7-step40"] if setup in ST.STAGE2_SETUPS else [], rid)
        results = {r["label"]: r for r in state["s7_results"]}
        self.assertEqual(len(results), len(state["s7_results"]))                             # one entry per lane
        self.assertEqual(len(results), 10 + 2 * 2 * (6 + 3))
        self.assertEqual({(r["seed"], r["step"]) for r in state["s7_results"] if r["run"].startswith("stage2/")},
                         {(0, 20), (0, 40), (1, 20), (1, 40)})
        self.assertEqual(sorted(k for k, r in results.items() if r["s7_passed"] is not True),
                         sorted([f"{s1}/s7", f"{s2}/s7", f"{s2}/s7-20", f"{s3}/s7", f"{s3}/s7-20"]))
        for r in state["s7_results"]:
            self.assertEqual(r["sha256"], H.sha256_file(r["file"]), r["label"])
            self.assertEqual(H.read_json(r["file"])["s7_passed"], r["s7_passed"], r["label"])
        checks = state["checks"]
        keys = [k for k in checks if k.startswith("S7/")]
        self.assertEqual([k for k in keys if re.match(r"^S7/[^/]+/seed[1-9]/", k)], [])
        self.assertTrue(all(E._s7_check_key(k) is not None for k in keys), keys)
        self.assertEqual(sorted(k for k in keys if "/seed0/" in k),
                         sorted(f"S7/{qq}/seed0/{l}/{s}" for qq, by in plan.items() for l, setups in by.items()
                                for s in setups if s in ST.STAGE2_SETUPS))
        self.assertEqual(checks["S7/td3_bc/standard"]["scoped_stop"], "learner")
        self.assertEqual(checks[f"S7/{q}/seed0/iql/alone"]["scoped_stop"], "learner")
        self.assertEqual([k for k, v in checks.items() if not v.get("passed") and not v.get("scoped_stop")], [])
        log = (root / "driver" / "driver.log").read_text(encoding="utf8")
        self.assertIn(f"S7 (version 4): {s2}/s7", log)
        self.assertIn("finished: every unit of every stage ran and no stop rule halted; S7 failures recorded under "
                      "version 4 (nothing dropped; the supplement's numbers are shown, never judged): iql on "
                      "medium-expert, td3_bc IW on medium-expert, td3_bc on medium-expert", log)
        self.assertIsNone(re.search(r"card \S+ dropped", log))
        before = state
        code, out = driver(args)                        # resumed: nothing retrains, nothing new is recorded
        self.assertEqual(code, 0, out[-4000:])
        again, st2 = state_of(root), status_of(root)
        self.assertEqual((st2["state"], st2["counts"]["cards_done"]), ("finished", st2["counts"]["cards_skipped"]))
        self.assertEqual(st2["counts"]["cards_done"], 10 + len(planned))
        self.assertEqual(st2["counts"]["lanes_done"], 0)
        self.assertEqual((again["s7_stopped"], again["s7_results"]), (before["s7_stopped"], before["s7_results"]))
        READOUTS["supplement"] = dict(s7_stopped=stopped, s7_results=len(results), counts=st["counts"],
                                      resume=st2["counts"])

    def test_simulated_s6_halts_after_stage1(self):
        """A card cap below stage 1's own use: S6's projection stops the driver before stage 2."""
        root = fresh("stop_s6")
        code, out = driver(["run", "--dry-run", "--root", str(root), "--learners", "td3_bc", "--qualities",
                            "medium-expert", "--seeds", "0", "--caps", "0.0001,100"])
        self.assertEqual(code, 3, out[-4000:])
        self.assertEqual(H.read_json(root / "driver" / "STOP.json")["rule"], "S6")
        self.assertFalse((root / "stage2").exists())
        self.assertEqual(len(list(root.glob("stage1/*/seed0/*/record.json"))), 5)

    def test_refuses_without_approval(self):
        """Real mode: no approval, no start, nothing written."""
        root = fresh("refuse")
        code, out = driver(["run", "--root", str(root)], timeout=300)
        self.assertEqual(code, 2)
        self.assertIn("REFUSED", out)
        self.assertFalse(root.exists())
        with self.assertRaises(gate.ApprovalError):
            ST.config_from(ST.parser().parse_args(["run", "--inject", "x=unit0:1"]))


def extra():
    return dict(readouts=READOUTS)


if __name__ == "__main__":
    sys.exit(T.run(sys.modules[__name__], AREA, extra=extra))
