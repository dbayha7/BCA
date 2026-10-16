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
            A simulated S6 (a tiny card cap): the driver halts after stage 1's projection.
  refusal   Without the approval the driver refuses to start (real mode), writing nothing.
  review    The stage-0 review's fixes, in process on made-up folders: a failed S7 result halts a resumed driver
            instead of being skipped as done; a halt starts no next lane step; a non-finite scored return is a Claim 0
            outcome (alone, room) or an S5 halt (BCA, unless accepted); a stop before training names its rule (S11);
            modes, the recorded configuration and the per-root lock; the real-data options that would change a frozen
            choice are refused; S6b on resume projects only runs without a record; an unexpected error writes the stop
            report; the ledger's open entries, recovered runs, the driver's own time and the kept previous version;
            the gate's manifest coverage, its bytecode refusal and stop rule S0's summary check.

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
import shutil
import subprocess
import sys
import unittest

import numpy as np

from experiments.shift_hosts import budget as B
from experiments.shift_hosts import gate
from experiments.shift_hosts import handoff as H
from experiments.shift_hosts import runner as RN
from experiments.shift_hosts import stages as ST
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


class DriverRulesTest(unittest.TestCase):
    """The stop-rule and bookkeeping fixes, in process on made-up folders (nothing trains, nothing is scored)."""

    def test_failed_s7_never_passes_on_resume(self):
        """A lane whose outputs verify is skipped on resume only after its own verdict: an s7.json that says
        s7_passed false halts with S7 (it used to be skipped as done); a passing one is skipped."""
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
        bad_job = ST.LaneJob("x/s7-bad", "s7", "medium-expert", [never], [str(bad)])
        drv.lane_queue.append(bad_job)
        drv.pump()
        self.assertEqual(drv.halt["rule"], "S7")
        self.assertEqual(drv.lanes, {})                       # nothing launched
        res = ST.s7_result(d / "run", dict(outcome="complete", timings=dict(outer_steps=1)))
        self.assertEqual((res["passed"], res["s7_passed"]), (False, False))   # stage 1's and seed 0's S7 checks
        self.assertFalse(ST.s7_result(d / "none", dict(outcome="complete", timings=dict(outer_steps=1)))["passed"])
        self.assertTrue(ST.s7_result(d / "none", dict(outcome="stopped", timings=dict(outer_steps=1)))["passed"])

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
        """A dry run never uses the real root; a driver of the other mode, or a resume with another configuration, is
        refused; one driver per root (the lock), and a refused second start leaves a pending STOP_REQUEST alone."""
        with self.assertRaises(gate.ApprovalError):
            ST.config_from(ST.parser().parse_args(["run", "--dry-run", "--root", str(RN.RUNS)]))
        d = fresh("rules_modes")
        ST.Driver(dry_cfg(d), None)
        with self.assertRaises(gate.ApprovalError):                       # another configuration
            ST.Driver(dry_cfg(d, "--seeds", "0", "1"), None)
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
        as no change)."""
        frozen = dict(learners=RN.LEARNERS, qualities=RN.QUALITIES, seeds=(0, 1, 2, 3, 4), choice19=dict(ST.CHOICE19),
                      card_cores=[0, 1], lane_cores=list(range(2, 10)), s9_signature={k: None for k in RN.S9_KEYS})
        ok = ST.parser().parse_args(["run", "--seeds", "0", "1", "2", "3", "4", "--learners"] + list(RN.LEARNERS))
        ST.check_real_args(ok, frozen)
        for more in (["--choice19", "td3_bc=no"], ["--learners", "cql"], ["--qualities", "medium"],
                     ["--seeds", "0"], ["--card-device", "cpu"], ["--root", str(ROOT)], ["--caps", "1,1"],
                     ["--card-cores", "0", "1", "2"], ["--lane-cores"] + [str(c) for c in range(2, 11)],
                     ["--lane-cores", "1", "2"]):
            with self.assertRaises(gate.ApprovalError, msg=more):
                ST.check_real_args(ST.parser().parse_args(["run"] + more), frozen)
        cfg = ST.config_from(ST.parser().parse_args(["run"]), dict(frozen, seeds=(0, 1)))
        self.assertEqual((cfg.seeds, cfg.choice19, cfg.root), ((0, 1), ST.CHOICE19, RN.RUNS))

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
