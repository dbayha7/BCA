"""Unit tests of version 9's runner: the phase table (the main setting only), the refusals, the memory rule against
version 7's, the ledger's interruption classes, the training-file check, the safe stop, the budget guard, the
plumbing, and the end-to-end stub run (with a seed where BCA refuses). Synthetic data and stubs only: nothing is
trained (every learner is replaced by a stand-in that takes no gradient step), no recorded seed 5-49 is generated,
loaded or read, and nothing is written into the repository (temporary folders only).

Run (plain unittest), from /tmp:
  PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=/home/user/bca \\
    /tmp/claude-0/bcavenv/bin/python /home/user/bca/outputs/shift/beyond-the-data/near80_v9/code/test_runner.py
It takes a few minutes (the end-to-end stub run is most of it).
"""
import sys

sys.dont_write_bytecode = True

import ast  # noqa: E402
import fcntl  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import math  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
import unittest.mock  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/user/bca")
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import near80_constants as K  # noqa: E402
import ledger as LG  # noqa: E402
import phases as PH  # noqa: E402
import phase_main as PM  # noqa: E402
import near80_runner as R  # noqa: E402

PY = sys.executable
TEST_ENV = {k: v for k, v in K.PHASE_ENV.items() if k != "PYTHONPYCACHEPREFIX"}   # as stub phases (Cfg.phase_env)
STUB_PATH = os.environ.get("NEAR80_STUB_PYTHONPATH")
APPROVAL = dict(prereg_sha256="a" * 64, prefix="a" * 16, fixture=True)
BIG_MEM = "MemTotal:       67108864 kB\nMemAvailable:   62914560 kB\n"


def tmpdir(test):
    d = Path(tempfile.mkdtemp(prefix="near80_test_runner_"))
    test.addCleanup(shutil.rmtree, d, True)
    if LG.inside(d, ROOT):
        raise RuntimeError("temporary folders must lie outside the repository")
    return d


def partners_present():
    """A's approval chain and C's looks are needed by the end-to-end stub run only."""
    have_pooled = (K.CODE / "pooled.py").is_file() or bool(STUB_PATH)
    return (K.CODE / "approval_check.py").is_file() and have_pooled


def phase_env():
    env = dict(TEST_ENV)
    if STUB_PATH:
        env["PYTHONPATH"] = STUB_PATH + ":" + env["PYTHONPATH"]
    return env


# --------------------------------------------------------------------------------------------- the table

class TestTable(unittest.TestCase):
    def test_kinds_settings_and_counts(self):
        keys = PH.plan()
        ids = [PH.phase_id(k) for k in keys]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({k.kind for k in keys}, set(K.PHASE_KINDS) - {"look_exploratory"})
        count = {}
        for k in keys:
            count[k.kind] = count.get(k.kind, 0) + 1
        n = len(K.CONFIRMATORY)
        self.assertEqual(count, dict(data_main=2 * n, critic=2 * n, critic_full=n, critic_second=2 * n, switch=2 * n,
                                     clone=2 * n, reference_actors=2 * n, policy=2 * n, lock=1, evaluate=2 * n,
                                     look_confirmatory=1))
        for k in keys:
            if k.kind in PH.LEARNER_KINDS or k.kind in ("data_main", "evaluate"):
                self.assertEqual(k.setting, "main")
            if k.kind == "critic_full":
                self.assertEqual((k.world, k.seed in K.CONFIRMATORY), ("stall", True))
            if k.seed is not None:
                self.assertIn(k.seed, K.CONFIRMATORY)

    def test_order_of_the_table(self):
        kinds = [k.kind for k in PH.plan()]
        firsts = [kinds.index(x) for x in K.PHASE_KINDS if x in kinds]
        self.assertEqual(firsts, sorted(firsts))

    def test_ids_round_trip(self):
        for k in PH.plan() + PH.plan(spec="smoke"):
            self.assertEqual(PH.parse_id(PH.phase_id(k)), k)
        self.assertEqual(PH.phase_id(PH.make_key("policy", "main", "stall", 3)), "policy.main.stall.s3")
        self.assertEqual(PH.phase_id(PH.make_key("lock", group="exploratory")), "lock.x.x.exploratory")

    def test_needs_are_in_the_plan_and_acyclic(self):
        s = K.CONFIRMATORY[0]
        for spec in ("full", "smoke", f"seeds:0,{s}"):
            keys = PH.plan(spec=spec)
            inside = set(keys)
            for k in keys:
                for d in PH.needs(k, keys):
                    self.assertIn(d, inside, (spec, k, d))
            pr = PH.priorities(keys)              # raises on a cycle
            self.assertTrue(all(v > 0 for v in pr.values()))

    def test_specific_needs(self):
        s = K.CONFIRMATORY[1]
        k = PH.parse_id(f"policy.main.no_stall.s{s}")
        n = set(PH.phase_id(x) for x in PH.needs(k))
        self.assertEqual(n, {f"switch.main.stall.s{s}", f"switch.main.no_stall.s{s}", f"critic.main.no_stall.s{s}",
                             f"clone.main.no_stall.s{s}", f"reference_actors.main.no_stall.s{s}",
                             f"policy.main.stall.s{s}"})
        k = PH.parse_id(f"reference_actors.main.stall.s{s}")
        self.assertIn(f"critic_full.x.stall.s{s}", [PH.phase_id(x) for x in PH.needs(k)])
        k = PH.parse_id(f"reference_actors.main.no_stall.s{s}")
        self.assertNotIn("critic_full", " ".join(PH.phase_id(x) for x in PH.needs(k)))
        self.assertEqual(PH.needs(PH.parse_id(f"data_main.main.stall.s{s}")), [])
        self.assertEqual([PH.phase_id(x) for x in PH.needs(PH.parse_id(f"switch.main.stall.s{s}"))],
                         [f"critic_second.main.stall.s{s}"])

    def test_est_minutes(self):
        s = K.CONFIRMATORY[0]
        self.assertEqual(PH.est_minutes(PH.parse_id("policy.main.stall.s0")), 44)
        self.assertEqual(PH.est_minutes(PH.parse_id(f"policy.main.no_stall.s{s}")), 11)
        self.assertEqual(PH.est_minutes(PH.parse_id(f"critic.main.stall.s{s}")), 6)
        self.assertEqual(PH.est_minutes(PH.parse_id("data_main.main.stall.s0")), 1)

    def test_outputs_include_sidecars_and_stop(self):
        k = PH.parse_id("policy.main.stall.s0")
        out = PH.outputs(k)
        for rel in ("policy/main/stall/seed0/lock.json", "policy/main/stall/seed0/lock.json.sha256",
                    "policy/main/stall/seed0/actors.msgpack", "policy/main/stall/seed0/stop.json",
                    "stops/policy.main.stall.s0.json"):
            self.assertIn(rel, out)


class TestRefusals(unittest.TestCase):
    def test_reserved_and_outside_seeds_for_every_kind_and_mode(self):
        for kind in K.PHASE_KINDS:
            for seed in list(K.RESERVED) + list(K.SPENT) + [15, 34, 50, 99, 100]:
                for mode in ("official", "scratch", "stub"):
                    with self.assertRaises(PH.Refused):
                        PH.check_key(PH.parse_id(f"{kind}.main.stall.s{seed}"), mode,
                                     {"official": "full", "scratch": "smoke"}[mode] if mode != "stub"
                                     else f"seeds:0,{K.CONFIRMATORY[0]}")
        for seed in K.RESERVED + K.SPENT:
            with self.assertRaises(PH.Refused):
                PH.parse_id(f"critic.main.stall.s{seed}")

    def test_phase_main_refuses_reserved_seeds_in_every_mode(self):
        d = tmpdir(self)
        fx = d / "fx"
        fx.mkdir()
        (fx / K.TEST_MARKER).write_text(K.TEST_MARKER_TEXT)
        for seed in K.RESERVED + K.SPENT:
            for kind in ("data_main", "critic", "policy", "evaluate"):
                pid = f"{kind}.main.stall.s{seed}"
                self.assertEqual(PM.main(["--phase-id", pid, "--out", str(K.OUT), "--mode", "official"]), 97)
                self.assertEqual(PM.main(["--phase-id", pid, "--out", str(d / "scratch"), "--mode", "scratch"]), 97)
                self.assertEqual(PM.main(["--phase-id", pid, "--out", str(fx / "out"), "--mode", "stub",
                                          "--fixture", str(fx), "--plan", f"seeds:0,{seed}"]), 97)
        self.assertFalse((d / "scratch").exists())
        self.assertFalse((fx / "out").exists())

    def test_scratch_never_reaches_fresh_seeds(self):
        """Scratch mode is the plumbing on seed 0 (section 8.2): a phase of a fresh seed, or of the confirmatory
        group, is refused in scratch mode for every kind and plan, before any approval or data step."""
        d = tmpdir(self)
        s0, s1 = K.CONFIRMATORY[0], K.CONFIRMATORY[-1]
        for pid in (f"data_main.main.stall.s{s0}", f"policy.main.stall.s{s1}", f"critic_full.x.stall.s{s0}"):
            self.assertEqual(PM.main(["--phase-id", pid, "--out", str(d / "scratch"), "--mode", "scratch"]), 97)
            self.assertEqual(PM.main(["--phase-id", pid, "--out", str(d / "scratch"), "--mode", "scratch", "--plan",
                                      "smoke"]), 97)
        self.assertFalse((d / "scratch").exists())
        for kind in K.PHASE_KINDS:
            for seed in K.CONFIRMATORY:
                for setting in ("main", "x"):
                    for spec in ("full", "smoke", f"seeds:0,{seed}"):
                        with self.assertRaises(PH.Refused) as cm:
                            PH.check_key(PH.make_key(kind, setting, "stall", seed), "scratch", spec)
                        self.assertTrue(str(cm.exception).startswith("refused"))
        for kind in ("lock", "look_confirmatory"):
            with self.assertRaises(PH.Refused):
                PH.check_key(PH.make_key(kind, group="confirmatory"), "scratch", "smoke")
        with self.assertRaises(PH.Refused):                       # scratch is the smoke plan only
            PH.check_key(PH.parse_id("data_main.main.stall.s0"), "scratch", "full")
        PH.check_key(PH.parse_id("data_main.main.stall.s0"), "scratch", "smoke")
        for pid in (f"data_main.main.stall.s{s0}", f"policy.main.stall.s{s1}"):
            ok, why = PH.admit(PH.parse_id(pid), APPROVAL, None, "scratch")
            self.assertFalse(ok)
            self.assertEqual(why, PH.SCRATCH_FRESH)
        with self.assertRaises(PH.Refused):
            R.Cfg(mode="scratch", out=d / "o", plan="full")
        self.assertEqual(R.Cfg(mode="scratch", out=d / "o").plan, "smoke")

    def test_combinations_outside_the_table(self):
        s0 = K.CONFIRMATORY[0]
        for pid in ("critic.main.stall.s0", f"v6_band.main.stall.s{s0}", "critic_full.x.stall.s0",
                    "data_band.main.stall.s0", f"readouts.main.stall.s{s0}", "policy.band.stall.s0",
                    "critic." + "near80" + f".stall.s{s0}", f"policy.random_rows.stall.s{s0}", "lock.x.x.x",
                    "nonsense.x.x.x",
                    "critic.main.moon.s0", "look_exploratory.x.x.exploratory"):
            with self.assertRaises(PH.Refused):
                PH.check_key(PH.parse_id(pid))
        PH.check_key(PH.parse_id("critic_full.x.stall.s0"), "scratch", "smoke")
        with self.assertRaises(PH.Refused):
            PH.check_key(PH.parse_id("critic_full.x.stall.s0"), "official", "smoke")

    def test_admission(self):
        s0 = K.CONFIRMATORY[0]
        main = PH.parse_id(f"data_main.main.stall.s{s0}")
        crit = PH.parse_id(f"critic.main.stall.s{s0}")
        self.assertFalse(PH.admit(main, None)[0])                                  # no approval
        self.assertFalse(PH.admit(PH.parse_id("critic.main.stall.s0"), None, None, "scratch")[0])
        self.assertEqual(PH.admit(main, APPROVAL), (True, None))                   # no look is needed
        self.assertEqual(PH.admit(crit, APPROVAL), (True, None))
        ok, why = PH.admit(PH.parse_id("critic.main.stall.s0"), APPROVAL, None, "official")
        self.assertFalse(ok)                                                        # seed 0 is the plumbing's
        self.assertEqual(PH.admit(PH.parse_id("critic.main.stall.s0"), APPROVAL, None, "scratch"), (True, None))
        self.assertEqual(PH.admit(main, APPROVAL, None, "stub"), (True, None))
        self.assertFalse(PH.is_skip("anything"))

    def test_skip_rules_from_files(self):
        out = tmpdir(self)
        keys = PH.plan()
        s1, s2, s3 = K.CONFIRMATORY[1], K.CONFIRMATORY[2], K.CONFIRMATORY[3]
        self.assertIsNone(PH.rule_skip(PH.parse_id(f"critic.main.stall.s{s1}"), out, keys))
        # a stopped need skips its dependents, through chains
        stop = PH.parse_id(f"critic_second.main.stall.s{s2}")
        LG.write_json_once(out / PH.stop_rel(stop), dict(kind="near80v9-phase-stop", reason="x"))
        self.assertEqual(PH.done(stop, out), "stopped")
        self.assertIn("was stopped", PH.rule_skip(PH.parse_id(f"switch.main.stall.s{s2}"), out, keys))
        self.assertIn("was skipped", PH.rule_skip(PH.parse_id(f"policy.main.stall.s{s2}"), out, keys))
        self.assertIn("was skipped", PH.rule_skip(PH.parse_id(f"evaluate.main.no_stall.s{s2}"), out, keys))
        self.assertEqual(PH.needs_state(PH.parse_id(f"evaluate.main.no_stall.s{s2}"), out, keys)[0], "skip")
        self.assertIsNone(PH.rule_skip(PH.parse_id("lock.x.x.confirmatory"), out, keys))
        # a seed whose data failed the gate's checks: every phase of that seed
        LG.write_json_once(out / K.LAYOUT["data_main"].format(world="stall", seed=s3),
                           dict(kind="near80v9-data-main", seed_failed_reason="the gate's checks failed"))
        for pid in (f"critic.main.no_stall.s{s3}", f"critic_full.x.stall.s{s3}", f"evaluate.main.stall.s{s3}"):
            self.assertIn("failed the gate's checks", PH.rule_skip(PH.parse_id(pid), out, keys))
        self.assertIsNone(PH.rule_skip(PH.parse_id(f"critic.main.stall.s{s1}"), out, keys))

    def test_out_root_rules(self):
        d = tmpdir(self)
        with self.assertRaises(PM.Refusal):
            PM.check_out_root(ROOT / "runs" / "shift_explore" / "near80_x", "scratch")
        with self.assertRaises(PM.Refusal):
            PM.check_out_root(ROOT / "runs" / "shift_explore" / "near80_v9", "scratch")
        with self.assertRaises(PM.Refusal):
            PM.check_out_root(ROOT / "runs" / "shift_explore" / "restructure" / "step2", "official")
        with self.assertRaises(PM.Refusal):
            PM.check_out_root(d / "elsewhere", "official")
        self.assertEqual(PM.check_out_root(K.OUT, "official"), Path(os.path.realpath(K.OUT)))
        with self.assertRaises(PM.Refusal):
            PM.check_out_root(d / "out", "stub", fixture=d)            # no marker
        (d / K.TEST_MARKER).write_text(K.TEST_MARKER_TEXT)
        PM.check_out_root(d / "out", "stub", fixture=d)
        with self.assertRaises(PM.Refusal):
            PM.check_out_root(Path(tempfile.gettempdir()) / "x", "stub", fixture=d)   # outside the fixture
        with self.assertRaises(PH.Refused):
            R.Cfg(mode="scratch", out=ROOT / "runs" / "x")
        with self.assertRaises(PH.Refused):
            R.Cfg(mode="official", out=d)

    def test_pinned_list_equals_step2s(self):
        sys.path.insert(0, str(K.S2_DIR))
        try:
            spec = importlib.util.spec_from_file_location("near80_test_step2lib", str(K.S2_DIR / "step2lib.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            sys.path.remove(str(K.S2_DIR))
        self.assertEqual(tuple(str(p) for p in mod.PINNED), tuple(str(ROOT / r) for r in PM.PINNED_REL))

    def test_environment_and_processors(self):
        PM.check_environment("official", environ=dict(K.PHASE_ENV), affinity={3})
        with self.assertRaises(PM.Refusal):
            PM.check_environment("official", environ=dict(K.PHASE_ENV, EXTRA="1"), affinity={3})
        with self.assertRaises(PM.Refusal):
            PM.check_environment("official", environ=dict(K.PHASE_ENV, OMP_NUM_THREADS="4"), affinity={3})
        PM.check_environment("scratch", environ=dict(K.PHASE_ENV, EXTRA="1"), affinity={0})
        for mode in ("official", "scratch", "stub"):
            with self.assertRaises(PM.Refusal):
                PM.check_environment(mode, environ=dict(K.PHASE_ENV), affinity={0, 1})

    def test_write_once(self):
        out = tmpdir(self)
        k = PH.parse_id("critic.main.stall.s1")
        PM.check_write_once(k, out)
        (out / "learners/main/stall/seed1").mkdir(parents=True)
        (out / "learners/main/stall/seed1/critic.msgpack").write_bytes(b"x")
        with self.assertRaises(PM.Refusal):
            PM.check_write_once(k, out)
        LG.write_json_once(out / PH.record_rel(k), dict(kind="near80v9-critic"))
        with self.assertRaises(PM.Refusal):
            PM.check_write_once(k, out)
        with self.assertRaises(LG.RecordError):
            LG.write_json_once(out / PH.record_rel(k), dict(kind="near80v9-critic"))

    def test_sizes_only_outside_official(self):
        a = PM.parser().parse_args(["--phase-id", "x", "--out", "y", "--steps", "40"])
        with self.assertRaises(PM.Refusal):
            PM.check_sizes(a, "official")
        PM.check_sizes(a, "scratch")
        a = PM.parser().parse_args(["--phase-id", "x", "--out", "y", "--steps", "40", "--chunk", "30"])
        with self.assertRaises(PM.Refusal):
            PM.check_sizes(a, "scratch")

    def test_lanes(self):
        self.assertIsNotNone(R.lane_cpus(0)[1])
        self.assertIsNotNone(R.lane_cpus(17)[1])
        cpus, why = R.lane_cpus(2, given=[5, 7, 9])
        self.assertEqual((cpus, why), ([5, 7], None))
        self.assertIsNotNone(R.lane_cpus(3, given=[5, 7])[1])
        out = []
        self.assertEqual(R.main(["run", "--lanes", "0"], out=out.append), 97)
        self.assertEqual(R.main(["plumbing", "--lanes", "17"], out=out.append), 97)
        self.assertEqual(R.main(["stub", "--fixture", "/nonexistent_near80", "--lanes", "17"], out=out.append), 97)
        self.assertTrue(all(line.startswith("refused") for line in out))

    def test_stub_refuses_repository_fixture_and_reserved_seeds(self):
        out = []
        self.assertEqual(R.main(["stub", "--fixture", str(ROOT / "runs" / "x")], out=out.append), 97)
        d = tmpdir(self)
        self.assertEqual(R.main(["stub", "--fixture", str(d), "--seeds", "0,7"], out=out.append), 97)
        self.assertFalse(any(d.iterdir()))

    def test_second_runner_and_version8(self):
        out = tmpdir(self)
        cfg = R.Cfg(mode="scratch", out=out / "o", poll=0.05, approval_fn=lambda: APPROVAL, r8=out / "r8")
        (out / "o").mkdir()
        with open(out / "o" / "runner.lock", "a") as h:
            fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lines = []
            self.assertEqual(R.run_locked(cfg, [], APPROVAL, lines.append, signals=False), 97)
            self.assertIn("another", lines[-1])
        (out / "r8" / "locks").mkdir(parents=True)
        self.assertFalse(R.version8_running(out / "r8"))
        with open(out / "r8" / "runner.lock", "a") as h:
            fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertTrue(R.version8_running(out / "r8"))
            lines = []
            self.assertEqual(R.cmd_run(cfg, lines.append), 97)
            self.assertEqual(lines[-1], R.V8_MESSAGE)
        with open(out / "r8" / "locks" / "policy.main.stall.s11.lock", "a") as h:
            fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertTrue(R.version8_running(out / "r8"))
            self.assertEqual(R.held_phase_locks(out / "r8"), ["policy.main.stall.s11"])
        self.assertFalse(R.version8_running(out / "r8"))

    def test_version8_must_have_finished(self):
        out = tmpdir(self)
        r8, res = out / "r8", out / "v8" / "results-x" / "RESULTS.json"
        cfg = R.Cfg(mode="scratch", out=out / "o", approval_fn=lambda: APPROVAL, r8=r8, v8_results=res)
        ok, why = R.version8_finished(cfg)
        self.assertFalse(ok)
        self.assertIn(f"{R.RUN_N80} version8-ended", why)
        r8.mkdir()
        (r8 / "status.json").write_text(json.dumps(dict(phases={"a": dict(state="done"),
                                                               "b": dict(state="unfinished")})))
        self.assertFalse(R.version8_finished(cfg)[0])
        (r8 / "status.json").write_text(json.dumps(dict(phases={"a": dict(state="done"), "b": dict(state="skipped"),
                                                               "c": dict(state="stopped")})))
        self.assertTrue(R.version8_finished(cfg)[0])
        (r8 / "status.json").unlink()
        LG.write_json_once(res, dict(kind="near80-results", finished=False))
        self.assertFalse(R.version8_finished(cfg)[0])
        res.unlink()
        (res.parent / "RESULTS.json.sha256").unlink()
        LG.write_json_once(res, dict(kind="near80-results", finished=True))
        self.assertTrue(R.version8_finished(cfg)[0])
        res.unlink()
        (res.parent / "RESULTS.json.sha256").unlink()
        lines = []
        self.assertEqual(R.cmd_version8_ended(cfg, lines.append, isatty=False), 97)
        self.assertEqual(R.cmd_version8_ended(cfg, lines.append, isatty=True, argv=["version8-ended"]), 0)
        self.assertTrue(R.version8_finished(cfg)[0])
        self.assertEqual(R.cmd_version8_ended(cfg, lines.append, isatty=True, argv=["version8-ended"]), 97)

    def test_version8s_gathered_results_are_in_place(self):
        ok, why = R.version8_finished(R.Cfg(mode="scratch", out=tmpdir(self) / "o"))
        self.assertTrue(ok, why)

    def test_plumbing_record_required(self):
        out = tmpdir(self)
        with self.assertRaises(PH.Refused):
            R.require_plumbing(out, APPROVAL)
        p = out / K.LAYOUT["plumbing"]
        LG.write_json_once(p, dict(kind="near80v9-plumbing", passed=False, prereg_sha256=APPROVAL["prereg_sha256"]))
        with self.assertRaises(PH.Refused):
            R.require_plumbing(out, APPROVAL)
        out2 = tmpdir(self)
        p2 = out2 / K.LAYOUT["plumbing"]
        LG.write_json_once(p2, dict(kind="near80v9-plumbing", passed=True, prereg_sha256="c" * 64))
        with self.assertRaises(PH.Refused):
            R.require_plumbing(out2, APPROVAL)
        out3 = tmpdir(self)
        p3 = out3 / K.LAYOUT["plumbing"]
        LG.write_json_once(p3, dict(kind="near80v9-plumbing", passed=True, prereg_sha256=APPROVAL["prereg_sha256"]))
        R.require_plumbing(out3, APPROVAL)
        p3.write_text(p3.read_text().replace("true", "false"))
        with self.assertRaises(PH.Refused):
            R.require_plumbing(out3, APPROVAL)
        out4 = tmpdir(self)
        LG.write_json_once(out4 / K.LAYOUT["plumbing"], dict(kind="near80v9-plumbing", passed=True,
                                                              prereg_sha256=APPROVAL["prereg_sha256"]))
        LG.write_json_once(out4 / K.LAYOUT["reproduction_differs"], dict(kind="near80v9-reproduction-differs"))
        with self.assertRaises(PH.Refused) as cm:
            R.require_plumbing(out4, APPROVAL)
        self.assertIn("reproduction_differs.json exists", str(cm.exception))

    def test_git_identity(self):
        import approval_check as AC
        d = tmpdir(self)
        env = dict(os.environ, GIT_CONFIG_GLOBAL=str(d / "global"), GIT_CONFIG_NOSYSTEM="1", HOME=str(d))
        subprocess.run(["git", "init", "-q", str(d / "repo")], check=True, env=env)
        old = dict(os.environ)
        os.environ.update(GIT_CONFIG_GLOBAL=str(d / "global"), GIT_CONFIG_NOSYSTEM="1", HOME=str(d))
        try:
            R.refuse_identity(d / "repo")                                    # unset: passes
            for email in (K.APPROVER_EMAIL, "noreply@anthropic.com"):
                subprocess.run(["git", "-C", str(d / "repo"), "config", "user.email", email], check=True, env=env)
                with self.assertRaises(PH.Refused):
                    R.refuse_identity(d / "repo")
            subprocess.run(["git", "-C", str(d / "repo"), "config", "user.email", "someone@example.org"], check=True,
                           env=env)
            R.refuse_identity(d / "repo")
            self.assertTrue(hasattr(AC, "refuse_git_identity"))
        finally:
            os.environ.clear()
            os.environ.update(old)


# --------------------------------------------------------------------------------------------- memory and budget

def load_runner_v7():
    lib = K.V7_DIR / "laptop" / "lib"
    for p in (str(lib), str(K.V7_DIR / "code")):
        if p not in sys.path:
            sys.path.append(p)
    spec = importlib.util.spec_from_file_location("near80_test_runner_v7", str(lib / "runner_v7.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestMemoryRule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v7 = load_runner_v7()

    def test_constants_equal_version7s(self):
        self.assertEqual((K.UNMEASURED_GIB, K.HEADROOM_GIB, K.MEASURED_FACTOR, K.POLICY_CAP_UNMEASURED, K.MAX_LANES),
                         (self.v7.UNMEASURED_GIB, self.v7.HEADROOM_GIB, self.v7.MEASURED_FACTOR,
                          self.v7.POLICY_TRAIN_CAP, self.v7.MAX_LANES))

    def test_reservation_and_admit_equal_version7s(self):
        G = K.GIB
        peaks = [None, 1 * G // 4, 2 * G, 5 * G]
        for peak in peaks:
            ours = {} if peak is None else {"critic": peak, "policy": peak}
            theirs = {} if peak is None else {"deploy": peak, "policy_train": peak}
            self.assertEqual(R.reservation("critic", ours), self.v7.reservation("deploy", theirs))
            self.assertEqual(R.reservation("policy", ours), self.v7.reservation("policy_train", theirs))
        cases = []
        for total in (16 * G, 32 * G, 128 * G):
            for avail in (6 * G, 15 * G, 30 * G, 120 * G):
                for own in (1 * G, 7 * G, 14 * G):
                    for n_run in (0, 1, 2, 3):
                        cases.append((total, avail, own, n_run))
        for total, avail, own, n_run in cases:
            for measured in (False, True):
                ours_p = {"policy": 3 * G} if measured else {}
                theirs_p = {"policy_train": 3 * G} if measured else {}
                a = R.admit("policy", own, [("policy", 7 * G)] * n_run, total, avail, ours_p)
                b = self.v7.admit("policy_train", own, [("policy_train", 7 * G)] * n_run, total, avail, theirs_p)
                self.assertEqual(a[0], b[0], (total, avail, own, n_run, measured))
                a = R.admit("critic", own, [("evaluate", 2 * G)] * n_run, total, avail, ours_p)
                b = self.v7.admit("deploy", own, [("deploy", 2 * G)] * n_run, total, avail, theirs_p)
                self.assertEqual(a[0], b[0])

    def test_policy_cap_until_measured(self):
        G = K.GIB
        ok, why = R.admit("policy", 7 * G, [("policy", 7 * G)] * 2, 512 * G, 500 * G, {})
        self.assertFalse(ok)
        self.assertIn("at most 2 policy", why)
        self.assertTrue(R.admit("policy", 7 * G, [("policy", 7 * G)] * 2, 512 * G, 500 * G, {"policy": G})[0])

    def test_pick_cpus_equals_version7s(self):
        d = tmpdir(self)
        for cpu, sib in ((0, "0,4"), (1, "1,5"), (2, "2,6"), (3, "3,7"), (4, "0,4"), (5, "1,5"), (6, "2,6"),
                         (7, "3,7")):
            (d / f"cpu{cpu}" / "topology").mkdir(parents=True)
            (d / f"cpu{cpu}" / "topology" / "thread_siblings_list").write_text(sib + "\n")
        for n in (1, 2, 4, 6):
            self.assertEqual(R.pick_cpus(n, d, set(range(8))), self.v7.pick_cpus(n, d, set(range(8))))

    def test_meminfo(self):
        d = tmpdir(self)
        (d / "m").write_text(BIG_MEM)
        self.assertEqual(R.meminfo(d / "m"), (67108864 * 1024, 62914560 * 1024))


class TestBudget(unittest.TestCase):
    def test_pause_and_cap(self):
        f = K.FORECAST_CPU_HOURS["confirmatory_main"]
        self.assertAlmostEqual(f, K.CPU_HOURS_PER_FRESH_SEED * len(K.CONFIRMATORY))
        paused, over = R.budget({"confirmatory_main": 2 * f + 1, "plumbing": 0.1})
        self.assertIn("confirmatory_main", paused)
        self.assertIn(R.PAUSE_TEXT, paused["confirmatory_main"])
        self.assertIn(f"{R.RUN_N80} budget-continue confirmatory_main", paused["confirmatory_main"])
        paused, over = R.budget({"confirmatory_main": 2 * f - 0.1})
        self.assertEqual((paused, over), ({}, None))
        paused, over = R.budget({"confirmatory_main": 2 * f - 0.1, "plumbing": 9.0})
        self.assertIsNotNone(over)
        self.assertIn("budget-continue <stage>", over)
        # the plumbing and the seed-0 smoke have no phase of the official plan: they never pause
        paused, over = R.budget({"plumbing": 4.0, "exploratory": 4.0})
        self.assertEqual((paused, over), ({}, None))
        # each typed continuation raises the stage's pause point and the cap by one forecast of the stage
        self.assertEqual(R.budget({"confirmatory_main": 2 * f + 1}, continues={"confirmatory_main": 1})[0], {})
        self.assertIn("confirmatory_main", R.budget({"confirmatory_main": 3 * f + 1},
                                                    continues={"confirmatory_main": 1})[0])
        hours = {"confirmatory_main": 2 * f - 0.1, "plumbing": 9.0}
        self.assertIsNone(R.budget(hours, continues={"confirmatory_main": 1})[1])

    def test_budget_continue_is_typed_and_recorded(self):
        out = tmpdir(self)
        cfg = R.Cfg(mode="scratch", out=out / "o", approval_fn=lambda: APPROVAL)
        lines = []
        st = "confirmatory_main"
        self.assertEqual(R.cmd_budget_continue(cfg, st, lines.append, isatty=False), 97)
        self.assertEqual(R.cmd_budget_continue(cfg, "plumbing", lines.append, isatty=True), 97)
        self.assertEqual(R.cmd_budget_continue(cfg, "exploratory", lines.append, isatty=True), 97)
        self.assertEqual(R.cmd_budget_continue(cfg, st, lines.append, isatty=True, argv=["x"]), 0)
        self.assertEqual(R.cmd_budget_continue(cfg, st, lines.append, isatty=True, argv=["x"]), 0)
        self.assertEqual(R.budget_continues(out / "o"), {st: 2})
        rec = LG.read_json_verified(out / "o" / K.LAYOUT["budget_continue"].format(stage=st, n=2))
        self.assertEqual((rec["kind"], rec["stage"], rec["n"]), ("near80v9-budget-continue", st, 2))
        self.assertIn(f"{R.RUN_N80} run --lanes 10", lines[-1])

    def test_cpu_hours_from_the_ledger(self):
        out = tmpdir(self)
        s0 = K.CONFIRMATORY[0]
        for pid, kind, setting, group, sec in (("critic.main.stall.s0", "critic", "main", "exploratory", 7200),
                                              (f"critic.main.stall.s{s0}", "critic", "main", "confirmatory", 3600),
                                              (f"critic_full.x.stall.s{s0}", "critic_full", "x", "confirmatory",
                                               1800)):
            n = LG.allocate_attempt(out, pid, mode="official")
            LG.write_usage(out, pid, n, sec, 0)
            LG.finish_attempt(out, dict(phase_kind=kind, setting=setting, world="stall", seed=0, group=group), pid,
                              n, done=True, stopped=False, exit_code=0)
        h = LG.cpu_hours(out, plumbing_seconds=360)
        self.assertAlmostEqual(h["exploratory"], 2.0)
        self.assertAlmostEqual(h["confirmatory_main"], 1.5)
        self.assertAlmostEqual(h["plumbing"], 0.1)


# --------------------------------------------------------------------------------------------- the ledger

class TestLedger(unittest.TestCase):
    def test_strict_json_and_sidecar(self):
        d = tmpdir(self)
        import numpy as np
        sha = LG.write_json_once(d / "a.json", dict(x=float("nan"), y=np.float32(1.5), z=np.arange(3), p=d))
        self.assertEqual((d / "a.json.sha256").read_text(), f"{sha}  a.json\n")
        self.assertEqual(LG.read_json_verified(d / "a.json"), dict(p=str(d), x=None, y=1.5, z=[0, 1, 2]))
        (d / "a.json").write_text((d / "a.json").read_text() + " ")
        self.assertEqual(LG.read_record(d / "a.json")[0], "partial")
        with self.assertRaises(LG.RecordError):
            LG.read_json_verified(d / "a.json")
        self.assertEqual(LG.read_record(d / "b.json"), ("missing", None))

    def test_classification(self):
        c = LG.classify_attempt
        self.assertEqual(c(True, 1, "a", "b")[0], "done")
        self.assertEqual(c(False, 2, "a", "a", stopped=True)[0], "stopped")
        self.assertEqual(c(False, 0, "a", "b")[0], "machine-wide")
        self.assertEqual(c(False, -9, "a", "a", runner_signals=[15])[0], "machine-wide")
        self.assertEqual(c(False, None, "a", "a")[0], "machine-wide")
        self.assertEqual(c(False, -15, "a", "a")[0], "machine-wide")
        self.assertEqual(c(False, 130, "a", "a")[0], "machine-wide")
        self.assertEqual(c(False, -9, "a", "a")[0], "phase-specific")
        self.assertEqual(c(False, 139, "a", "a")[0], "phase-specific")
        self.assertEqual(c(False, 97, "a", "a", log_tail="MemoryError")[0], "refused")
        self.assertEqual(c(False, 1, "a", "a", log_tail="x\nRESOURCE_EXHAUSTED: out of memory")[0], "phase-specific")
        self.assertEqual(c(False, 1, "a", "a")[0], "failed")
        self.assertEqual(c(False, 0, "a", "a")[0], "failed")

    def test_attempt_log_segments(self):
        d = tmpdir(self)
        log = d / "x.log"
        log.write_text(LG.start_line("p", 1, 0) + "\nMemoryError\n" + LG.start_line("p", 2, 0) + "\nall well\n")
        self.assertIn("MemoryError", LG.attempt_log_text(log, "p", 1))
        self.assertNotIn("MemoryError", LG.attempt_log_text(log, "p", 2))
        self.assertEqual(LG.attempt_log_text(log, "p", 3), "")

    def test_move_aside_never_deletes(self):
        d = tmpdir(self)
        k = PH.parse_id("clone.main.stall.s0")
        rels = PH.outputs(k)
        (d / "learners/main/stall/seed0").mkdir(parents=True)
        (d / "learners/main/stall/seed0/clone.msgpack").write_bytes(b"abc")
        moved = LG.move_interrupted(PH.phase_id(k), d, "machine-wide: test", rels, stamp="20261015T000000Z")
        self.assertEqual(moved, ["learners/main/stall/seed0/clone.msgpack"])
        self.assertFalse((d / "learners/main/stall/seed0/clone.msgpack").exists())
        dest = d / "interrupted" / PH.phase_id(k) / "20261015T000000Z" / "learners/main/stall/seed0/clone.msgpack"
        self.assertEqual(dest.read_bytes(), b"abc")
        rec = LG.read_json_verified(d / "interrupted" / PH.phase_id(k) / "20261015T000000Z.json")
        self.assertEqual(rec["files"], {"learners/main/stall/seed0/clone.msgpack": LG.sha256_bytes(b"abc")})
        self.assertEqual(LG.move_interrupted(PH.phase_id(k), d, "x", rels), [])

    def test_peaks(self):
        out = tmpdir(self)
        for i, (kb, cls_done) in enumerate(((1000, True), (3000, True), (9000, False))):
            pid = f"critic.main.stall.s{i}"
            n = LG.allocate_attempt(out, pid)
            LG.write_usage(out, pid, n, 1.0, 0 if cls_done else 1)
            u = LG.ledger_dir(out, pid) / f"{n}.usage.json"
            body = LG.read_json_verified(u)
            u.unlink()
            (u.parent / (u.name + ".sha256")).unlink()
            LG.write_json_once(u, dict(body, vmhwm_kb=kb))
            LG.finish_attempt(out, dict(phase_kind="critic"), pid, n, done=cls_done, stopped=False,
                              exit_code=0 if cls_done else 1)
        self.assertEqual(LG.measured_peaks(out), {"critic": 3000 * 1024})
        self.assertEqual(LG.measured_peaks(out, {"policy": 10}), {"critic": 3000 * 1024, "policy": 10 * 1024})


# --------------------------------------------------------------------------------------------- the runner with a fake phase

FAKE_PHASE = r'''
import sys
sys.dont_write_bytecode = True
import argparse, fcntl, json, os, signal, time
from pathlib import Path
sys.path.insert(0, CODE)
import ledger as LG
import phases as PH
ap = argparse.ArgumentParser()
for a in ("--phase-id", "--out", "--mode", "--plan", "--fixture", "--attempt", "--steps", "--chunk", "--updates",
          "--starts"):
    ap.add_argument(a)
a = ap.parse_args()
out = Path(a.out)
key = PH.parse_id(a.phase_id)
n = int(a.attempt)
lock = open(out / "locks" / f"{a.phase_id}.lock", "a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
plan = json.loads((out / "behavior.json").read_text())
seq = plan.get(a.phase_id, ["done"])
what = seq[min(n - 1, len(seq) - 1)]
t0 = time.time()
code = 0
(out / "started").mkdir(exist_ok=True)
(out / "started" / f"{a.phase_id}.{n}").write_text(what)
if os.sched_getaffinity(0).__len__() != 1:
    code = 97
elif what.startswith("sleep"):
    time.sleep(float(what[5:]))
    LG.write_json_once(out / PH.record_rel(key), dict(kind="fake", n=n))
elif what == "done":
    LG.write_json_once(out / PH.record_rel(key), dict(kind="fake", n=n))
elif what == "stop":
    LG.write_json_once(out / PH.stop_rel(key), dict(kind="near80v9-phase-stop", reason="a fake stop"))
    code = 2
elif what in ("kill9", "kill9partial"):
    if what == "kill9partial":
        p = out / PH.outputs(key)[0]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("partial")
    os.kill(os.getpid(), signal.SIGKILL)
elif what == "term":
    os.kill(os.getpid(), signal.SIGTERM)
elif what == "refuse":
    code = 97
elif what == "fail":
    code = 1
LG.write_usage(out, a.phase_id, n, time.time() - t0, code)
sys.exit(code)
'''


class FakeRunner:
    def __init__(self, test, keys, behavior, lanes=2, poll=0.05):
        self.d = tmpdir(test)
        self.out = self.d / "out"
        (self.out / "locks").mkdir(parents=True)
        (self.out / "behavior.json").write_text(json.dumps(behavior))
        (self.d / "meminfo").write_text(BIG_MEM)
        self.fake = self.d / "fake_phase.py"
        self.fake.write_text(f"CODE = {str(HERE)!r}\n" + FAKE_PHASE)
        self.cfg = R.Cfg(mode="scratch", out=self.out, plan="smoke", lanes=lanes, cpus=sorted(os.sched_getaffinity(0)),
                         meminfo=self.d / "meminfo", poll=poll, python=PY, phase_main=self.fake,
                         approval_fn=lambda: APPROVAL, r8=self.d / "r8")
        self.keys = keys
        self.lines = []

    def run(self, until_waiting=True):
        return R.run_locked(self.cfg, self.keys, APPROVAL, self.lines.append, until_waiting=until_waiting,
                            signals=False)

    def started(self):
        d = self.out / "started"
        return sorted(p.name for p in d.iterdir()) if d.is_dir() else []

    def records(self, pid):
        return LG.attempt_records(self.out, pid)


class TestRunnerWithFakePhases(unittest.TestCase):
    def keys(self, *pids):
        return [PH.parse_id(p) for p in pids]

    def test_done_and_dependencies(self):
        keys = self.keys("data_main.main.stall.s0", "data_main.main.no_stall.s0", "critic_full.x.stall.s0")
        fr = FakeRunner(self, keys, {})
        self.assertEqual(fr.run(), 0)
        for k in keys:
            self.assertEqual(PH.done(k, fr.out), "done")
        status = LG.read_json_verified(fr.out / "status.json")
        self.assertEqual(status["counts"], {"done": 3})
        rec = fr.records("critic_full.x.stall.s0")[1]
        self.assertEqual(LG.classification(rec), "done")
        self.assertEqual((rec["phase_kind"], rec["setting"], rec["seed"], rec["group"]),
                         ("critic_full", "x", 0, "exploratory"))
        lines = []
        R.cmd_status(fr.out, lines.append)
        self.assertTrue(any(line.startswith("counts: done 3") for line in lines))

    def test_machine_wide_reruns_and_files_move_aside(self):
        k = "critic.main.stall.s0"
        keys = self.keys(k)
        fr = FakeRunner(self, keys, {k: ["term", "kill9partial", "done"]})
        self.assertEqual(fr.run(), 0)
        recs = fr.records(k)
        self.assertEqual([LG.classification(recs[n]) for n in sorted(recs)], ["machine-wide", "phase-specific", "done"])
        moved = list((fr.out / "interrupted" / k).rglob("critic.json"))
        self.assertEqual(len(moved), 1)
        self.assertEqual(moved[0].read_text(), "partial")
        self.assertEqual(LG.read_json_verified(fr.out / PH.record_rel(keys[0]))["n"], 3)

    def test_phase_specific_three_then_alone_then_unfinished_then_retry(self):
        k = "clone.main.stall.s0"
        other = "clone.main.no_stall.s0"
        keys = self.keys(k, other)
        fr = FakeRunner(self, keys, {k: ["kill9", "kill9", "kill9", "kill9", "done"], other: ["sleep0.3"]})
        code = fr.run()
        self.assertEqual(code, R.EXIT_RETRY)
        recs = fr.records(k)
        self.assertEqual([LG.classification(recs[n]) for n in sorted(recs)], ["phase-specific"] * 4)
        self.assertEqual(R.need(fr.out, k)[0], "unfinished")
        self.assertTrue(LG.start_record(fr.out, k, 4)["alone"])
        self.assertFalse(LG.start_record(fr.out, k, 3)["alone"])
        self.assertTrue(any(line.find("UNFINISHED") >= 0 for line in fr.lines))
        self.assertTrue(any(line.find(f"type: {R.RUN_N80} retry {k}") >= 0 for line in fr.lines))
        out = []
        self.assertEqual(R.cmd_retry(fr.cfg, k, out.append, isatty=False), 97)
        self.assertEqual(R.cmd_retry(fr.cfg, k, out.append, isatty=True, argv=["retry", k]), 0)
        self.assertEqual(R.need(fr.out, k)[:2], ("alone", 1))
        self.assertEqual(R.cmd_retry(fr.cfg, k, out.append, isatty=True, argv=["retry", k]), 97)
        self.assertEqual(fr.run(), 0)
        self.assertEqual(LG.start_record(fr.out, k, 5)["retry_request"], 1)
        self.assertEqual(PH.done(keys[0], fr.out), "done")

    def test_stop_is_never_rerun_and_skips_dependents(self):
        keys = self.keys("critic_second.main.stall.s0", "switch.main.stall.s0", "critic.main.stall.s0")
        fr = FakeRunner(self, keys, {"critic_second.main.stall.s0": ["stop", "done"]})
        self.assertEqual(fr.run(), 0)
        self.assertEqual(PH.done(keys[0], fr.out), "stopped")
        status = LG.read_json_verified(fr.out / "status.json")
        self.assertEqual(status["phases"]["switch.main.stall.s0"]["state"], "skipped")
        self.assertEqual(status["phases"]["critic.main.stall.s0"]["state"], "done")
        self.assertEqual(fr.run(), 0)
        self.assertEqual([s for s in fr.started() if s.startswith("critic_second.main")],
                         ["critic_second.main.stall.s0.1"])

    def test_refusal_and_failure_halt(self):
        keys = self.keys("critic.main.stall.s0", "critic.main.stall.s1")
        fr = FakeRunner(self, keys, {"critic.main.stall.s0": ["refuse"], "critic.main.stall.s1": ["sleep0.5"]},
                        lanes=1)
        self.assertEqual(fr.run(), 97)
        self.assertEqual(fr.started(), ["critic.main.stall.s0.1"])
        fr2 = FakeRunner(self, keys, {"critic.main.stall.s0": ["fail"], "critic.main.stall.s1": ["sleep0.5"]},
                         lanes=1)
        self.assertEqual(fr2.run(), 1)
        self.assertTrue(any("FAILED" in line for line in fr2.lines))

    @unittest.mock.patch.object(K, "BUDGET_STAGES", ("exploratory", "confirmatory_main"))
    @unittest.mock.patch.dict(K.FORECAST_CPU_HOURS, {"exploratory": 0.0001})
    def test_budget_pause(self):
        """The guard's mechanics, on a scratch run whose seed-0 stage is made a budget stage for the test (in version 9
        only the confirmatory stage pauses)."""
        keys = self.keys("critic.main.stall.s0", "critic.main.no_stall.s0")
        fr = FakeRunner(self, keys, {})
        fr.cfg.forecast = dict(K.FORECAST_CPU_HOURS, exploratory=0.0001)
        pid = "critic_full.x.stall.s0"
        n = LG.allocate_attempt(fr.out, pid, mode="scratch")
        LG.write_usage(fr.out, pid, n, 10.0, 0)
        LG.finish_attempt(fr.out, dict(phase_kind="critic_full", group="exploratory"), pid, n, done=True,
                          stopped=False, exit_code=0)
        code = fr.run(until_waiting=False)
        self.assertEqual(code, R.EXIT_PAUSED)
        self.assertEqual(fr.started(), [])
        self.assertTrue(any("paused:" in line and R.PAUSE_TEXT in line for line in fr.lines))
        self.assertTrue(any(line.find("PAUSED (budget)") >= 0 for line in fr.lines))
        code = fr.run(until_waiting=False)                     # a second start pauses again: nothing changed
        self.assertEqual(code, R.EXIT_PAUSED)
        # the typed continuations lift the pause (each allows one more forecast of the stage)
        for _ in range(int(10.0 / 3600 / 0.0001) + 1):
            self.assertEqual(R.cmd_budget_continue(fr.cfg, "exploratory", [].append, isatty=True, argv=["t"]), 0)
        self.assertEqual(fr.run(), 0)
        self.assertEqual(len(fr.started()), 2)

    def test_over_the_cap_notes_every_waiting_phase(self):
        keys = self.keys("critic.main.stall.s0")
        fr = FakeRunner(self, keys, {})
        fr.cfg.cap = 0.0001
        pid = "critic_full.x.stall.s0"
        n = LG.allocate_attempt(fr.out, pid, mode="scratch")
        LG.write_usage(fr.out, pid, n, 10.0, 0)
        LG.finish_attempt(fr.out, dict(phase_kind="critic_full", group="exploratory"), pid, n, done=True,
                          stopped=False, exit_code=0)
        self.assertEqual(fr.run(until_waiting=False), R.EXIT_PAUSED)
        status = LG.read_json_verified(fr.out / "status.json")
        self.assertIn("beyond the cap", status["phases"]["critic.main.stall.s0"]["note"])
        self.assertTrue(any(line.find("PAUSED (budget): paused: the run has used") >= 0 for line in fr.lines))

    def test_a_phase_that_can_never_fit_halts(self):
        keys = self.keys("critic.main.stall.s0")
        fr = FakeRunner(self, keys, {})
        (fr.d / "meminfo").write_text("MemTotal:       12582912 kB\nMemAvailable:   12000000 kB\n")
        self.assertEqual(fr.run(until_waiting=False), 1)
        self.assertEqual(fr.started(), [])
        self.assertTrue(any("can never start" in line for line in fr.lines))

    def test_version8_running_holds_new_phases(self):
        import threading
        keys = self.keys("critic.main.stall.s0")
        fr = FakeRunner(self, keys, {})
        (fr.d / "r8").mkdir()
        box = {}
        with open(fr.d / "r8" / "runner.lock", "a") as h:
            fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
            th = threading.Thread(target=lambda: box.update(code=fr.run()))
            th.start()
            time.sleep(1.0)
            self.assertEqual(fr.started(), [])
            self.assertTrue(any("waiting: version 8 is running" in line for line in fr.lines))
        th.join(60)
        self.assertEqual(box.get("code"), 0)
        self.assertEqual(len(fr.started()), 1)

    def test_safe_stop_one_ctrl_c(self):
        d = tmpdir(self)
        out = d / "out"
        (out / "locks").mkdir(parents=True)
        pids = ["critic.main.stall.s0", "critic.main.stall.s1", "critic.main.stall.s2"]
        (out / "behavior.json").write_text(json.dumps({p: ["sleep1.5"] for p in pids}))
        (d / "meminfo").write_text(BIG_MEM)
        fake = d / "fake_phase.py"
        fake.write_text(f"CODE = {str(HERE)!r}\n" + FAKE_PHASE)
        driver = d / "driver.py"
        driver.write_text(
            "import sys\nsys.dont_write_bytecode = True\n"
            f"sys.path.insert(0, {str(HERE)!r})\n"
            "import near80_runner as R, phases as PH\n"
            f"cfg = R.Cfg(mode='scratch', out={str(out)!r}, lanes=1, cpus=[{sorted(os.sched_getaffinity(0))[0]}], "
            f"meminfo={str(d / 'meminfo')!r}, poll=0.05, python={PY!r}, phase_main={str(fake)!r}, "
            f"approval_fn=lambda: dict(prereg_sha256='a'*64), r8={str(d / 'r8')!r})\n"
            f"keys = [PH.parse_id(p) for p in {pids!r}]\n"
            "sys.exit(R.run_locked(cfg, keys, dict(prereg_sha256='a'*64)))\n")
        p = subprocess.Popen([PY, str(driver)], cwd="/tmp", env=dict(TEST_ENV), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 60
        while time.time() < deadline and not (out / "started").is_dir():
            time.sleep(0.05)
        p.send_signal(signal.SIGINT)
        text, _ = p.communicate(timeout=60)
        self.assertEqual(p.returncode, 3, text)
        self.assertEqual(len(list((out / "started").iterdir())), 1)
        self.assertIn("stop requested", text)
        self.assertIn("stopped safely", text)
        started = next((out / "started").iterdir()).name.rsplit(".", 1)[0]
        self.assertEqual(PH.done(PH.parse_id(started), out), "done")

    def test_second_ctrl_c_is_machine_wide(self):
        d = tmpdir(self)
        out = d / "out"
        (out / "locks").mkdir(parents=True)
        pid = "critic.main.stall.s0"
        (out / "behavior.json").write_text(json.dumps({pid: ["sleep30", "done"]}))
        (d / "meminfo").write_text(BIG_MEM)
        fake = d / "fake_phase.py"
        fake.write_text(f"CODE = {str(HERE)!r}\n" + FAKE_PHASE)
        driver = d / "driver.py"
        driver.write_text(
            "import sys\nsys.dont_write_bytecode = True\n"
            f"sys.path.insert(0, {str(HERE)!r})\n"
            "import near80_runner as R, phases as PH\n"
            f"cfg = R.Cfg(mode='scratch', out={str(out)!r}, lanes=1, cpus=[{sorted(os.sched_getaffinity(0))[0]}], "
            f"meminfo={str(d / 'meminfo')!r}, poll=0.05, python={PY!r}, phase_main={str(fake)!r}, "
            f"approval_fn=lambda: dict(prereg_sha256='a'*64), r8={str(d / 'r8')!r}, stop_grace=5)\n"
            f"keys = [PH.parse_id({pid!r})]\n"
            "sys.exit(R.run_locked(cfg, keys, dict(prereg_sha256='a'*64)))\n")
        p = subprocess.Popen([PY, str(driver)], cwd="/tmp", env=dict(TEST_ENV), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 60
        while time.time() < deadline and not (out / "started").is_dir():
            time.sleep(0.05)
        time.sleep(0.3)
        p.send_signal(signal.SIGINT)
        time.sleep(0.3)
        p.send_signal(signal.SIGINT)
        text, _ = p.communicate(timeout=60)
        self.assertEqual(p.returncode, 3, text)
        rec = LG.attempt_record(out, pid, 1)
        self.assertEqual(LG.classification(rec), "machine-wide", rec)
        self.assertEqual(rec["runner_signals"], [15])


# --------------------------------------------------------------------------------------------- plumbing and results

def fake_tests(ok=True):
    def f(env, out):
        return {n: dict(exit_code=0 if ok else 1, seconds=1.0, tail="") for n in K.TEST_FILES}
    return f


def fake_smoke(ok=True, code=0, calls=None):
    def f(cfg, scratch, approval, out, signals=True):
        if calls is not None:
            calls.append("smoke")
        return dict(smoke={"data_main.main.stall.s0": ok}, peaks={"data_main": 1000, "policy": 2000},
                    fresh={"stall": True, "no_stall": True}, codes=[code], cpu_seconds=5.0)
    return f


def fake_repro(differs=(), finished=True, calls=None):
    def f(cfg, folder, out):
        if calls is not None:
            calls.append("repro")
        if not finished:
            return {}, {}
        items = {w: {n: dict(passed=(w, n) not in differs, plain=f"the {n}") for n in ("critic", "obca_k1")}
                 for w in K.WORLDS}
        return items, dict(seconds=100.0, vmhwm_kb=3 * 1024 * 1024)
    return f


class TestPlumbing(unittest.TestCase):
    def cfg(self):
        out = tmpdir(self)
        return R.Cfg(mode="scratch", out=out / "o", approval_fn=lambda: APPROVAL, r8=out / "r8"), out

    def plumb(self, cfg, out, n, **kw):
        lines = []
        code = R.cmd_plumbing(cfg, lines.append, scratch=out / f"scratch{n}", signals=False, **kw)
        return code, lines

    def test_a_failure_other_than_a_difference_can_be_retried(self):
        cfg, out = self.cfg()
        calls = []
        code, lines = self.plumb(cfg, out, 1, tests_fn=fake_tests(False), smoke_fn=fake_smoke(calls=calls),
                                 repro_fn=fake_repro(calls=calls))
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])                                  # nothing after a failed unit test
        self.assertFalse((cfg.out / K.LAYOUT["plumbing"]).exists())
        failed = sorted((cfg.out / "plumbing").glob("failed_*.json"))
        self.assertEqual(len(failed), 1)
        self.assertIn("a unit test failed", LG.read_json_verified(failed[0])["cause"])
        self.assertTrue(any(f"{R.RUN_N80} plumbing --lanes 10" in line for line in lines))
        code, _ = self.plumb(cfg, out, 2, tests_fn=fake_tests(), smoke_fn=fake_smoke(False, calls=calls),
                             repro_fn=fake_repro(calls=calls))
        self.assertEqual((code, calls), (1, ["smoke"]))              # the reproduction waits for a passed smoke
        code, _ = self.plumb(cfg, out, 3, tests_fn=fake_tests(), smoke_fn=fake_smoke(),
                             repro_fn=fake_repro(finished=False))
        self.assertEqual(code, 1)
        self.assertEqual(len(list((cfg.out / "plumbing").glob("failed_*.json"))), 3)
        code, lines = self.plumb(cfg, out, 4, tests_fn=fake_tests(), smoke_fn=fake_smoke(), repro_fn=fake_repro())
        self.assertEqual(code, 0)
        rec = R.require_plumbing(cfg.out, APPROVAL)
        self.assertEqual(rec["peaks"]["policy"], 3 * 1024 * 1024)    # the reproduction's full-length peak
        self.assertEqual(rec["peaks"]["data_main"], int(math.ceil(2 * K.GIB / 1.25 / 1024)))   # the 2 GiB floor
        self.assertAlmostEqual(rec["cpu_seconds"], len(K.TEST_FILES) * 1.0 + 5.0 + 100.0)
        code, lines = self.plumb(cfg, out, 5, tests_fn=fake_tests(), smoke_fn=fake_smoke(), repro_fn=fake_repro())
        self.assertEqual(code, 97)                                   # passed: nothing to do
        self.assertIn("the plumbing has passed", lines[-1])

    def test_a_reproduction_difference_is_final(self):
        cfg, out = self.cfg()
        code, lines = self.plumb(cfg, out, 1, tests_fn=fake_tests(), smoke_fn=fake_smoke(),
                                 repro_fn=fake_repro(differs=[("no_stall", "obca_k1")]))
        self.assertEqual(code, 1)
        self.assertTrue((cfg.out / K.LAYOUT["reproduction_differs"]).is_file())
        self.assertFalse((cfg.out / K.LAYOUT["plumbing"]).exists())
        self.assertTrue(any("reproduction differs: the obca_k1 (no-stall world)" in line for line in lines))
        code, lines = self.plumb(cfg, out, 2, tests_fn=fake_tests(), smoke_fn=fake_smoke(), repro_fn=fake_repro())
        self.assertEqual(code, 97)
        self.assertIn("reproduction_differs.json exists", lines[-1])
        with self.assertRaises(PH.Refused):
            R.require_plumbing(cfg.out, APPROVAL)
        rep = K.load_local("pooled").halt_report(cfg.out, mode="scratch")
        self.assertEqual(rep["stop"], "stop.reproduction")
        self.assertIn("did not reproduce the obca_k1 on seed 0", rep["sentences"][0]["text"])

    def test_an_interrupted_smoke_is_recorded_and_stops(self):
        cfg, out = self.cfg()
        calls = []
        code, lines = self.plumb(cfg, out, 1, tests_fn=fake_tests(), smoke_fn=fake_smoke(code=R.EXIT_STOPPED),
                                 repro_fn=fake_repro(calls=calls))
        self.assertEqual((code, calls), (R.EXIT_STOPPED, []))
        failed = sorted((cfg.out / "plumbing").glob("failed_*.json"))
        self.assertEqual(LG.read_json_verified(failed[0])["cause"], "interrupted (Ctrl-C)")

    def test_plumbing_peaks_and_official_measurements(self):
        peaks = R.plumbing_peaks({"evaluate": 10, "data_main": 5 * 1024 * 1024, "critic": 7}, 4 * 1024 * 1024)
        floor = int(math.ceil(2 * K.GIB / 1.25 / 1024))
        self.assertEqual(peaks["evaluate"], floor)
        self.assertEqual(peaks["data_main"], 5 * 1024 * 1024)
        for kind in K.REPRO_PEAK_KINDS:
            self.assertEqual(peaks[kind], 4 * 1024 * 1024)
        self.assertGreaterEqual(R.reservation("evaluate", {"evaluate": floor * 1024}), 2 * K.GIB)
        out = tmpdir(self)
        pid = "evaluate.main.stall.s0"
        n = LG.allocate_attempt(out, pid)
        LG.write_usage(out, pid, n, 1.0, 0)
        u = LG.ledger_dir(out, pid) / f"{n}.usage.json"
        body = LG.read_json_verified(u)
        u.unlink()
        (u.parent / (u.name + ".sha256")).unlink()
        LG.write_json_once(u, dict(body, vmhwm_kb=500))
        LG.finish_attempt(out, dict(phase_kind="evaluate"), pid, n, done=True, stopped=False, exit_code=0)
        got = LG.measured_peaks(out, peaks)
        self.assertEqual(got["evaluate"], 500 * 1024)                # an official measurement replaces the floor
        self.assertEqual(got["critic"], 4 * 1024 * 1024 * 1024)

    def test_results_copy_records_only(self):
        d = tmpdir(self)
        src, dest = d / "out", d / "n80"
        LG.write_json_once(src / "eval/main/stall/seed0/values.json", dict(kind="near80v9-values"))
        LG.write_json_once(src / K.LAYOUT["status"], dict(kind="near80v9-runner-status",
                                                            phases={"a": dict(state="done")}))
        (src / "looks").mkdir(parents=True)
        (src / "looks" / "exploratory.md").write_text("# look\n")
        (src / "logs").mkdir()
        (src / "logs" / "critic.main.stall.s0.log").write_text("log\n")
        (src / "learners/main/stall/seed0").mkdir(parents=True)
        (src / "learners/main/stall/seed0/critic.msgpack").write_bytes(b"model")
        (src / "data/main/stall/seed0").mkdir(parents=True)
        (src / "data/main/stall/seed0/train.hdf5").write_bytes(b"rows")
        (src / "locks").mkdir()
        (src / "locks" / "x.lock").write_text("")
        lines = []
        folder = R.make_results(src, dest, lines.append, stamp="20261016T000000Z")
        files = sorted(str(q.relative_to(folder)) for q in folder.rglob("*") if q.is_file())
        self.assertEqual(files, ["RESULTS.json", "RESULTS.json.sha256", "SHA256SUMS",
                                 "files/eval/main/stall/seed0/values.json",
                                 "files/eval/main/stall/seed0/values.json.sha256",
                                 "files/logs/critic.main.stall.s0.txt", "files/looks/exploratory.md",
                                 "files/status.json", "files/status.json.sha256"])
        for line in (folder / "SHA256SUMS").read_text().splitlines():
            sha, rel = line.split("  ", 1)
            self.assertEqual(LG.file_sha256(folder / rel), sha)
        self.assertTrue(LG.read_json_verified(folder / "RESULTS.json")["finished"])
        self.assertTrue(any(f"pull --rebase origin {K.BRANCH}" in line for line in lines))
        with self.assertRaises(PH.Refused):
            R.make_results(src, dest, lines.append, stamp="20261016T000000Z")

    def test_runfiles_plan(self):
        d = tmpdir(self)
        repo = d / "repo"
        out = repo / "runs" / "shift_explore" / "near80_v9"
        (out / "learners").mkdir(parents=True)
        (out / "learners" / "critic.msgpack").write_bytes(b"model")
        (out / "x.json").write_text("{}")
        (out / "train.hdf5").write_bytes(b"rows")
        (out / "locks").mkdir()
        (out / "locks" / "a.lock").write_text("")
        n = R.runfiles_plan(out, d / "plan", repo=repo)
        self.assertEqual(n, 2)
        names = (d / "plan" / "files.list").read_bytes().split(b"\0")
        self.assertEqual(sorted(x for x in names if x), [b"runs/shift_explore/near80_v9/learners/critic.msgpack",
                                                          b"runs/shift_explore/near80_v9/x.json"])
        env = (d / "plan" / "env.sh").read_text()
        self.assertIn(f"RUNFILES_BRANCH='{K.RUNFILES_BRANCH}'", env)
        self.assertIn(f"AUTHOR_EMAIL='{K.APPROVER_EMAIL}'", env)

    def test_stub_fixture_gets_its_own_memory(self):
        fx = tmpdir(self)
        R.stub_fixture_memory(fx)
        body = json.loads((fx / "runner.json").read_text())
        self.assertEqual(R.meminfo(body["meminfo"])[0], 24 * K.GIB)
        (fx / "runner.json").write_text(json.dumps(dict(poll=0.5)))
        R.stub_fixture_memory(fx)                                     # an existing runner.json is kept
        self.assertEqual(json.loads((fx / "runner.json").read_text()), dict(poll=0.5))


# --------------------------------------------------------------------------------------------- step 1 and step 2 code

class TestRestatedCode(unittest.TestCase):
    def test_overestimates_equal_step1s(self):
        """learner_phases.overestimates against the function step 1's evaluate.py defines (executed from its source,
        with step 1's library), on a synthetic critic and synthetic transitions."""
        import numpy as np
        import jax
        import jax.numpy as jnp
        import learner_phases as LP
        from experiments.shift import learners as L
        from runtime.networks import Transition
        src = (K.S1_DIR / "evaluate.py").read_text()
        tree = ast.parse(src)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "overestimates")
        ns = dict(np=np, jnp=jnp, L=L, S=LP.step1lib())
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "step1_overestimates", "exec"), ns)
        rng = np.random.default_rng(3)
        n = 64
        tr = Transition(jnp.asarray(rng.standard_normal((n, 13)), jnp.float32),
                        jnp.asarray(rng.uniform(-1, 1, (n, 4)), jnp.float32),
                        jnp.asarray(rng.standard_normal(n), jnp.float32),
                        jnp.asarray(rng.standard_normal((n, 13)), jnp.float32), jnp.zeros(n, jnp.float32))
        critic = L.CRITIC.init(jax.random.PRNGKey(5), jnp.zeros(13), jnp.zeros(4))
        fixed = L.FixedController("stall_aware", L.TC.STALL if hasattr(L, "TC") else None, np.zeros(13, np.float32),
                                  np.ones(13, np.float32))
        rows = np.arange(0, n, 3)
        noise = np.clip(rng.standard_normal((2, len(rows), 4)) * 0.2, -0.5, 0.5).astype(np.float32)
        a = LP.overestimates(critic, tr, fixed, 0.98, rows=rows)
        b = ns["overestimates"](critic, tr, fixed, 0.98, rows=rows)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.array_equal(LP.overestimates(critic, tr, fixed, 0.98, rows=rows, noise=noise),
                                       ns["overestimates"](critic, tr, fixed, 0.98, rows=rows, noise=noise)))

    def test_reference_layout(self):
        import learner_phases as LP
        self.assertEqual(LP.reference_layout("main", "stall"),
                         [("none", "setting"), ("full_none", "full"), ("pad", "setting")])
        self.assertEqual(LP.reference_layout("main", "no_stall"), [("none", "setting"), ("pad", "setting")])
        self.assertEqual(LP.saved_reference_names("main", "stall"), ("none", "full_none"))
        self.assertEqual(LP.saved_reference_names("main", "no_stall"), ("none",))

    def test_arm_row(self):
        """The arm row from the switch records of both worlds: step 2's own table when BCA certifies, the plain twins
        beside it; when the stalling world's height is not finite, the restated table marks BCA's arms refused
        (step 2's arm_table raises there) and nothing is raised (the refusal rule, section 8.4)."""
        import policy_phase as PP
        out = tmpdir(self)

        def entry(on, height, gap=0.4):
            return dict(t_lo=0.7, t_up=0.98, certificate=dict(on=on, height=height),
                        groups=dict(n_full_throttle=50, n_rest=150, mean_full_throttle=gap + 0.1, mean_rest=0.1,
                                    q90_full_throttle=gap + 0.2, q90_rest=0.1), rank_test=dict(U=1.0, p=0.01))

        def write(seed, stall, no_stall):
            for w, e in (("stall", stall), ("no_stall", no_stall)):
                LG.write_json_once(out / K.LAYOUT["switch"].format(setting="main", world=w, seed=seed),
                                   dict(kind="near80v9-switch", entry=e))

        write(0, entry(True, 2.0), entry(False, 0.05, gap=0.02))
        row = PP.arm_row("main", 0, "stall", out)
        self.assertEqual({k: row["arms"]["obca_k1"][k] for k in ("switch", "h", "k")}, dict(switch=True, h=2.0, k=1.0))
        self.assertFalse(row["arms"]["obca_k1"]["refused"])
        self.assertEqual(row["arms"]["meandiff_k1"]["switch"], True)
        self.assertAlmostEqual(row["arms"]["meandiff_k1"]["h"], 0.4)
        self.assertEqual(row["arms"]["meandiff_limit"]["caution"], "range_hinge")
        self.assertEqual((row["refused"], row["bca_refused"]), ([], False))
        ns = PP.arm_row("main", 0, "no_stall", out)
        self.assertEqual(ns["arms"]["obca_k1"]["h"], 0.0)
        self.assertEqual({k: ns["arms"]["obca_forced"][k] for k in ("switch", "h", "k")}, dict(switch=True, h=2.0,
                                                                                              k=1.0))
        self.assertFalse(ns["arms"]["meandiff_k1"]["switch"])
        # the stalling world's height is not finite while the switch is on: step 2's arm_table refuses; version 9
        # marks the arms that read BCA's height refused and goes on
        write(1, entry(True, "inf"), entry(False, 0.05))
        row = PP.arm_row("main", 1, "stall", out)
        self.assertEqual(row["refused"], ["obca_forced", "obca_k1", "obca_k2"])
        self.assertTrue(row["bca_refused"])
        self.assertIsNotNone(row["arm_table_refusal"])
        ns = PP.arm_row("main", 1, "no_stall", out)
        self.assertEqual(ns["refused"], ["obca_forced"])
        self.assertFalse(ns["bca_refused"])
        # switched off with an infinite height: no refusal (the switch says off; obca_forced still needs the height)
        write(2, entry(False, "inf"), entry(False, 0.05))
        row = PP.arm_row("main", 2, "stall", out)
        self.assertEqual(row["refused"], ["obca_forced"])
        self.assertFalse(row["bca_refused"])
        with self.assertRaises(PH.Refused):
            PP.arm_row("near80", 0, "stall", out)

    def test_ramp_programs_are_step2s(self):
        """BCA's switched ramp and the plumbing check train in step 2's own batches in both worlds (step 2's
        train_policy.py: the stalling world runs w0_check alone, then RAMP_ARMS; the no-stall world runs [w0_check] +
        RAMP_ARMS); meandiff_k1 runs alone after them; a refused arm keeps its place as a height-0 pad."""
        import policy_phase as PP
        src = (K.S2_DIR / "train_policy.py").read_text()
        self.assertIn("trained = runner.run([MC.ramp_arm(S.W0, 0.0, 0.0)], MC.log)", src)
        self.assertIn("arms = [MC.ramp_arm(S.W0, 0.0, 0.0)] + [MC.ramp_arm(n, row[\"arms\"][n][\"h\"], "
                      "row[\"arms\"][n][\"k\"])", src)
        stall, no_stall = PP.ramp_layout("main", "stall"), PP.ramp_layout("main", "no_stall")
        self.assertEqual(stall, [(K.W0,), tuple(K.STEP2_RAMP_ARMS), ("meandiff_k1",)])
        self.assertEqual(no_stall, [(K.W0,) + tuple(K.STEP2_RAMP_ARMS), ("meandiff_k1",)])
        refused = dict(refused=["obca_k1", "obca_k2", "obca_forced"])
        st = PP.ramp_layout("main", "stall", refused)
        self.assertEqual(st[1][:3], tuple(PP.refused_pad(n) for n in ("obca_k1", "obca_k2", "obca_forced")))
        self.assertEqual(st[1][3:], tuple(K.STEP2_RAMP_ARMS[3:]))
        self.assertEqual([len(c) for c in st], [len(c) for c in stall])
        row = dict(arms={K.PRIMARY: dict(h=2.5, k=1.0)})

        class MC:
            @staticmethod
            def ramp_arm(name, h, k):
                return (name, h, k)
        self.assertEqual(PP.layout_arm(MC, row, PP.refused_pad(K.PRIMARY)), (PP.refused_pad(K.PRIMARY), 0.0, 0.0))
        self.assertEqual(PP.layout_arm(MC, row, K.W0), (K.W0, 0.0, 0.0))
        self.assertEqual(PP.layout_arm(MC, row, K.PRIMARY), (K.PRIMARY, 2.5, 1.0))

    def test_reproduction_runs_the_official_policy_function(self):
        import policy_phase as PP
        import evaluate_phase as EP
        calls = []

        def fake_train_world(run, ctx, S, MC, row, stall_lock=None, stall_lock_sha=None):
            calls.append((run.key, row, stall_lock))
            if run.key.world == "stall":
                return dict(world="stall", ramp_programs=[[K.W0]]), {K.W0: "w", K.PRIMARY: "o", "range_hinge": "r"}
            raise PH.PhaseStop(PH.STOP_REASONS["height0"], dict(checks="x"))

        run = PH.Run(PH.make_key("evaluate", "main", "stall", 0), tmpdir(self), mode="scratch")
        with unittest.mock.patch.object(PP, "train_world", fake_train_world), \
                unittest.mock.patch.object(PP, "arm_row_from", lambda st, s, w, per, S: dict(setting=st, world=w)):
            acts, detail = EP.reproduce_policy(run, "ctx", None, None, {w: {} for w in K.WORLDS})
        self.assertEqual([(k.kind, k.setting, k.world, k.seed) for k, _, _ in calls],
                         [("policy", "main", "stall", 0), ("policy", "main", "no_stall", 0)])
        self.assertIsNone(calls[0][2])
        self.assertEqual(calls[1][2], dict(world="stall", ramp_programs=[[K.W0]]))   # the stall payload is the lock
        self.assertEqual(acts["stall"][K.PRIMARY], "o")
        self.assertIsNone(acts["no_stall"])
        self.assertEqual(detail["no_stall"]["stopped"], PH.STOP_REASONS["height0"])

    def test_reproduction_lets_input_stops_through(self):
        """A saved input whose fingerprint does not match, or a training file that fails its check, inside the
        reproduction is not a reproduction difference (section 8.2): it is raised again, so the reproduction process
        exits without a verdict and the plumbing records a retryable failure. A learner or check stop stays a
        difference."""
        import policy_phase as PP
        import evaluate_phase as EP
        run = PH.Run(PH.make_key("evaluate", "main", "stall", 0), tmpdir(self), mode="scratch")
        for kind in ("fingerprint", "training_file", "learner", "check"):
            def fake_train_world(run, ctx, S, MC, row, stall_lock=None, stall_lock_sha=None, kind=kind):
                raise PH.PhaseStop(PH.STOP_REASONS[kind], dict(what="x"))
            with unittest.mock.patch.object(PP, "train_world", fake_train_world), \
                    unittest.mock.patch.object(PP, "arm_row_from", lambda st, s, w, per, S: dict(setting=st)):
                if kind in EP.REPRO_RETRYABLE_STOPS:
                    with self.assertRaises(PH.PhaseStop) as cm:
                        EP.reproduce_policy(run, "ctx", None, None, {w: {} for w in K.WORLDS})
                    self.assertEqual(cm.exception.reason, PH.STOP_REASONS[kind])
                else:
                    acts, detail = EP.reproduce_policy(run, "ctx", None, None, {w: {} for w in K.WORLDS})
                    self.assertEqual(acts, {w: None for w in K.WORLDS}, kind)
                    self.assertEqual(detail["stall"]["stopped"], PH.STOP_REASONS[kind])
        # the raised stop ends the reproduction process with no verdict; the plumbing then writes the retryable
        # failed_<utc>.json, never reproduction_differs.json (TestPlumbing's unfinished-reproduction case)
        self.assertEqual(set(EP.REPRO_RETRYABLE_STOPS), {"fingerprint", "training_file"})

    def test_reproduction_items(self):
        import evaluate_phase as EP
        good = dict(removal=["a", "b"], critic="c", clone="d", none="e", critic_second="f",
                    switch=dict(q05=1.0, height=2.0, on=True), w0_check="g", obca_k1="h", range_hinge="i")
        items = EP.reproduction_items(good, dict(good))
        self.assertTrue(all(v["passed"] for v in items.values()))
        self.assertEqual(set(items), set(EP.REPRO_PLAIN))
        bad = dict(good, switch=dict(q05=1.0, height=2.5, on=True), critic="x")
        items = EP.reproduction_items(bad, good)
        self.assertEqual(sorted(k for k, v in items.items() if not v["passed"]), ["critic", "obca_k1"])
        self.assertEqual(items["obca_k1"]["plain"], "step 2's switched-ramp actor")
        self.assertFalse(EP.reproduction_items(dict(good, none=None), dict(good, none=None))["none"]["passed"])


def _fixture(test, meminfo=True):
    import approval_check as AC
    d = tmpdir(test)
    fx = d / "fx"
    AC.write_test_fixture(fx)
    if meminfo:
        (fx / "meminfo").write_text(BIG_MEM)
        (fx / "runner.json").write_text(json.dumps(dict(meminfo=str(fx / "meminfo"), poll=0.2)))
    return fx


def _phase(fx, pid, plan="seeds:0", extra=()):
    cpu = sorted(os.sched_getaffinity(0))[0]
    cmd = ["taskset", "-c", str(cpu), PY, str(HERE / "phase_main.py"), "--phase-id", pid, "--out", str(fx / "out"),
           "--mode", "stub", "--fixture", str(fx), "--plan", plan, "--steps", "40", "--chunk", "20", "--updates", "40",
           "--starts", "5"] + list(extra)
    return subprocess.run(cmd, cwd="/tmp", env=phase_env(), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          timeout=900)


@unittest.skipUnless((K.CODE / "approval_check.py").is_file(), "needs builder A's approval_check.py")
class TestTrainingFileCheck(unittest.TestCase):
    def test_tampered_training_file_stops_and_skips(self):
        import h5py
        fx = _fixture(self)
        s = K.CONFIRMATORY[0]
        plan = f"seeds:{s}"
        r = _phase(fx, f"data_main.main.stall.s{s}", plan=plan)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rec = LG.read_json_verified(fx / "out" / K.LAYOUT["data_main"].format(world="stall", seed=s))
        self.assertTrue(rec["train_file"]["training_file_check"]["passed"])
        self.assertTrue(rec["group_count"]["defined"])
        path = fx / "out" / K.LAYOUT["train_file"].format(setting="main", world="stall", seed=s)
        with h5py.File(path, "r+") as f:
            a = f["actions"][()]
            a.view("uint8")[0] ^= 1                        # one byte of the commands
            f["actions"][...] = a
        r = _phase(fx, f"critic.main.stall.s{s}", plan=plan)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        k = PH.parse_id(f"critic.main.stall.s{s}")
        stop = LG.read_json_verified(fx / "out" / PH.stop_rel(k))
        self.assertEqual(stop["reason"], PH.STOP_REASONS["training_file"])
        self.assertFalse(stop["detail"]["training_file_check"]["passed"])
        self.assertFalse((fx / "out" / PH.record_rel(k)).exists())
        self.assertEqual(PH.done(k, fx / "out"), "stopped")
        keys = PH.plan(spec=plan)
        for pid in (f"reference_actors.main.stall.s{s}", f"policy.main.stall.s{s}", f"policy.main.no_stall.s{s}",
                    f"evaluate.main.stall.s{s}"):
            self.assertEqual(PH.effective_state(PH.parse_id(pid), fx / "out", keys), "skipped", pid)
        r = _phase(fx, f"critic.main.stall.s{s}", plan=plan)           # a stopped phase is never rerun
        self.assertEqual(r.returncode, 97)
        r = _phase(fx, f"reference_actors.main.stall.s{s}", plan=plan)  # its dependent: skipped, nothing trained
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(PH.done(PH.parse_id(f"reference_actors.main.stall.s{s}"), fx / "out"), "skipped")
        self.assertFalse((fx / "out" / f"learners/main/stall/seed{s}/reference_actors.msgpack").exists())

    def test_tampered_prepared_hash_stops_switch(self):
        """BCA's switch reads the setting's prepared rows: the training-file check runs for it too, so a data record
        whose prepared hash differs from the rows the switch would read is a recorded stop."""
        fx = _fixture(self)
        s = K.CONFIRMATORY[0]
        plan = f"seeds:{s}"
        for pid in (f"data_main.main.stall.s{s}", f"critic_second.main.stall.s{s}"):
            r = _phase(fx, pid, plan=plan)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        p = fx / "out" / K.LAYOUT["data_main"].format(world="stall", seed=s)
        body = LG.read_json_verified(p)
        body["prepared_sha256"]["train"] = "0" * 64
        p.unlink()
        (p.parent / (p.name + ".sha256")).unlink()
        LG.write_json_once(p, body)
        r = _phase(fx, f"switch.main.stall.s{s}", plan=plan)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        k = PH.parse_id(f"switch.main.stall.s{s}")
        stop = LG.read_json_verified(fx / "out" / PH.stop_rel(k))
        self.assertEqual(stop["reason"], PH.STOP_REASONS["training_file"])
        self.assertFalse((fx / "out" / PH.record_rel(k)).exists())

    def test_a_changed_heldout_fingerprint_stops_switch(self):
        """The switch's held-out rows must hash to the data record's heldout_rows_sha256 (written before training)."""
        fx = _fixture(self)
        s = K.CONFIRMATORY[0]
        plan = f"seeds:{s}"
        for pid in (f"data_main.main.stall.s{s}", f"critic_second.main.stall.s{s}"):
            r = _phase(fx, pid, plan=plan)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        p = fx / "out" / K.LAYOUT["data_main"].format(world="stall", seed=s)
        body = LG.read_json_verified(p)
        self.assertIsNotNone(body["heldout_rows_sha256"])
        body["heldout_rows_sha256"] = "0" * 64
        p.unlink()
        (p.parent / (p.name + ".sha256")).unlink()
        LG.write_json_once(p, body)
        r = _phase(fx, f"switch.main.stall.s{s}", plan=plan)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        stop = LG.read_json_verified(fx / "out" / PH.stop_rel(PH.parse_id(f"switch.main.stall.s{s}")))
        self.assertEqual(stop["reason"], PH.STOP_REASONS["fingerprint"])

    def test_every_context_checks_the_prepared_rows(self):
        """settingdata.SettingContext.prepared compares the rows it serves with the main setting's data record's
        prepared_sha256 on the fresh seeds (seed 0: the gate's own loader checks)."""
        import types
        import settingdata as SD
        out = tmpdir(self)
        s0, s1 = K.CONFIRMATORY[0], K.CONFIRMATORY[1]
        good = {k: f"{k}-sha" for k in SD.PREPARED_KEYS}
        LG.write_json_once(out / K.LAYOUT["data_main"].format(world="stall", seed=s0),
                           dict(kind="near80v9-data-main", prepared_sha256=good))
        LG.write_json_once(out / K.LAYOUT["data_main"].format(world="stall", seed=s1),
                           dict(kind="near80v9-data-main", prepared_sha256=dict(good, train="other")))

        def ctx(setting, seed, prep):
            c = SD.SettingContext.__new__(SD.SettingContext)
            c.setting, c.rep, c.out_root = setting, seed, out
            c._prepared, c._prepared_checked = {"stall": prep}, set()
            return c
        prep = types.SimpleNamespace(sha256=dict(good))
        self.assertIs(ctx("main", s0, prep).prepared("stall"), prep)
        self.assertIs(ctx("main", 0, prep).prepared("stall"), prep)          # the gate's seed: its loader checks
        with self.assertRaises(PH.PhaseStop) as cm:
            ctx("main", s1, prep).prepared("stall")
        self.assertEqual(cm.exception.reason, PH.STOP_REASONS["training_file"])
        with self.assertRaises(PH.PhaseStop):
            ctx("main", s0, types.SimpleNamespace(sha256=dict(good, bank="x"))).prepared("stall")
        with self.assertRaises(PH.Refused):
            ctx("near80", s0, prep).prepared("stall")
        self.assertIn("switch", PH.TRAINING_ROW_KINDS)


FRESH_READBACK = r"""
import json, sys
sys.dont_write_bytecode = True
code, fx, seed, world = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
sys.path.insert(0, code)
from pathlib import Path
import phases as PH, settingdata as SD, stubs
run = PH.Run(PH.make_key("critic", "main", world, seed), Path(fx) / "out", mode="stub", fixture=fx)
stubs.install(run)
seen, inner = [], SD.fresh_log
def spy(s, w, spec, sizes, slope):
    seen.append([list(sizes), slope, type(slope).__name__])
    return inner(s, w, spec, sizes, slope)
SD.fresh_log = spy
rec = json.loads(sys.stdin.read()) if len(sys.argv) > 5 else None
sd = SD.FreshSeed(seed, world, rec) if rec is not None else SD.seed_data(seed, world, Path(fx) / "out")
print(json.dumps(dict(seen=seen, rep=sd.rep, world=sd.world)))
"""
RUN_SIZE_KEYS_V8 = ["chunk", "starts", "steps", "updates"]


@unittest.skipUnless((K.CODE / "approval_check.py").is_file(), "needs builder A's approval_check.py")
class TestFreshSeedRecordRoundTrip(unittest.TestCase):
    """Version 8's amendment 1. A fresh seed's data record is written through phase_main (the wrapper every runner
    phase goes through) and read back by every later phase of that seed. Version 8's wrapper wrote its run sizes over
    the record's 'sizes'; version 9's wrapper writes them under 'run_sizes' (the root cause fixed), and the reader
    still accepts both shapes. Stub mode only: synthetic logs, nothing trained."""

    def setUp(self):
        self.fx = _fixture(self)
        self.s = K.CONFIRMATORY[0]
        self.plan = f"seeds:0,{self.s}"

    def readback(self, world="stall", record=None, ok=True):
        r = subprocess.run([PY, "-c", FRESH_READBACK, str(HERE), str(self.fx), str(self.s), world] +
                           (["record"] if record is not None else []),
                           input=json.dumps(record) if record is not None else None, cwd="/tmp", env=phase_env(),
                           capture_output=True, text=True, timeout=900)
        if not ok:
            return None, r
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return json.loads(r.stdout.strip().splitlines()[-1]), r

    def test_record_written_by_the_wrapper_reads_back(self):
        s = self.s
        r = _phase(self.fx, f"data_main.main.stall.s{s}", plan=self.plan)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rec = LG.read_json_verified(self.fx / "out" / K.LAYOUT["data_main"].format(world="stall", seed=s))
        self.assertEqual(rec["source"], "fresh")
        self.assertEqual(rec["sizes"], [500, 50, 50])                   # the log's sizes, kept (version 9's fix)
        self.assertEqual(sorted(rec["run_sizes"]), RUN_SIZE_KEYS_V8)    # the run's sizes, under their own key
        got, _ = self.readback()
        self.assertEqual(got["seen"], [[[500, 50, 50], 1.5, "float"]])  # the stub's fresh_settings: the log's sizes
        for pid in (f"critic_full.x.stall.s{s}", f"critic.main.stall.s{s}", f"critic_second.main.stall.s{s}",
                    f"clone.main.stall.s{s}"):                          # the phases that failed on version 8's desktop
            r = _phase(self.fx, pid, plan=self.plan)
            self.assertEqual(r.returncode, 0, pid + "\n" + r.stdout + r.stderr)
            self.assertEqual(PH.done(PH.parse_id(pid), self.fx / "out"), "done", pid)
        crit = LG.read_json_verified(self.fx / "out" / PH.record_rel(PH.parse_id(f"critic.main.stall.s{s}")))
        self.assertTrue(crit["training_file_check"]["passed"])

    def test_three_integers_are_used_and_anything_else_is_refused(self):
        r = _phase(self.fx, f"data_main.main.no_stall.s{self.s}", plan=self.plan)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rec = LG.read_json_verified(self.fx / "out" / K.LAYOUT["data_main"].format(world="no_stall", seed=self.s))
        got, _ = self.readback("no_stall", dict(rec, sizes=[500, 50, 50]))
        self.assertEqual(got["seen"], [[[500, 50, 50], 1.5, "float"]])
        got, _ = self.readback("no_stall", dict(rec, sizes=dict.fromkeys(RUN_SIZE_KEYS_V8)))   # the desktop's records
        self.assertEqual(got["seen"], [[[500, 50, 50], 1.5, "float"]])
        for bad in (dict(steps=None), dict(rec["run_sizes"], extra=1), dict(rec["run_sizes"], steps="40"),
                    dict(rec["run_sizes"], steps=0), [500, 50], [500, 50, "50"], [500, 50, True], [0, 50, 50],
                    [-1, 50, 50], None):
            _, r2 = self.readback("no_stall", dict(rec, sizes=bad), ok=False)
            self.assertNotEqual(r2.returncode, 0, repr(bad))
            self.assertIn("refused: the data record's sizes", r2.stderr, repr(bad))

    def test_the_stub_generator_refuses_what_dataset_generate_refuses(self):
        r = subprocess.run([PY, "-c", "import sys; sys.dont_write_bytecode = True; sys.path.insert(0, sys.argv[1]);"
                                      "import dataclasses, stubs\n"
                                      "spec = stubs._recorder_spec()\n"
                                      "stubs.stub_fresh_log(10, 'stall', spec, (500, 50, 50), 1.5)\n"
                                      "other = dataclasses.replace(spec, damping=spec.damping + 0.25)\n"
                                      "ok3 = (500, 50, 50)\n"
                                      "for sp, sizes, slope in ((spec, ('chunk', 'starts', 'steps', 'updates'), 1.5),\n"
                                      "                         (spec, (500, 50), 1.5), (spec, ok3, float('nan')),\n"
                                      "                         (spec, ok3, 1.6), (other, ok3, 1.5)):\n"
                                      "    try:\n"
                                      "        stubs.stub_fresh_log(10, 'stall', sp, sizes, slope)\n"
                                      "    except TypeError:\n"
                                      "        continue\n"
                                      "    raise SystemExit('accepted ' + repr((sp, sizes, slope)))\n"
                                      "print('ok')", str(HERE)],
                           cwd="/tmp", env=phase_env(), capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.strip().splitlines()[-1], "ok")


# --------------------------------------------------------------------------------------------- the end-to-end stub run

@unittest.skipUnless(partners_present(), "needs the approval chain's approval_check.py and the looks' pooled.py")
class TestStubEndToEnd(unittest.TestCase):
    """Every phase kind of the plan on test fixtures, in stub mode: nothing is trained (the stand-ins count their
    calls; the training functions are guards that raise and count their entries). Two runs: seed 0 (the plumbing's
    group, with the plumbing look), then two fresh seeds on a second fixture: the first confirmatory seed and the
    stub's refusing seed (stubs.STUB_REFUSING_SEEDS: too few survivors, so BCA refuses there), so that the refusal
    rule runs end to end and the confirmatory look is written."""

    @classmethod
    def run_stub(cls, fx, seeds):
        env = dict(TEST_ENV)
        if STUB_PATH:
            env["NEAR80_STUB_PYTHONPATH"] = STUB_PATH
        lanes = str(min(4, len(os.sched_getaffinity(0))))
        return subprocess.run([PY, str(HERE / "near80_runner.py"), "stub", "--fixture", str(fx), "--lanes", lanes,
                               "--seeds", seeds], cwd="/tmp", env=env, capture_output=True, text=True, timeout=3600)

    @classmethod
    def fixture(cls, name):
        import approval_check as AC
        fx = cls.d / name
        AC.write_test_fixture(fx)
        (fx / "meminfo").write_text(BIG_MEM)
        (fx / "runner.json").write_text(json.dumps(dict(meminfo=str(fx / "meminfo"), poll=0.2)))
        return fx

    @classmethod
    def setUpClass(cls):
        import stubs
        cls.d = Path(tempfile.mkdtemp(prefix="near80v9_test_e2e_"))
        cls.s0, cls.sr = K.CONFIRMATORY[0], stubs.STUB_REFUSING_SEEDS[0]
        cls.fresh = f"{cls.s0},{cls.sr}"
        cls.fx0, cls.fxf = cls.fixture("fx0"), cls.fixture("fxf")
        cls.r0 = cls.run_stub(cls.fx0, "0")
        cls.rf = cls.run_stub(cls.fxf, cls.fresh)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, True)

    def outs(self):
        return (self.fx0 / "out", "0", self.r0), (self.fxf / "out", self.fresh, self.rf)

    def rec(self, out, pid):
        return LG.read_json_verified(out / PH.record_rel(PH.parse_id(pid)))

    def test_every_phase_kind_ran(self):
        kinds = set()
        for out, seeds, r in self.outs():
            self.assertEqual(r.returncode, 0, r.stdout[-4000:] + r.stderr[-4000:])
            keys = PH.plan(spec="seeds:" + seeds)
            not_done = {PH.phase_id(k): PH.done(k, out) for k in keys if PH.done(k, out) != "done"}
            self.assertEqual(not_done, {})
            kinds |= {k.kind for k in keys}
        self.assertEqual(kinds, set(K.PHASE_KINDS))
        for out, group in ((self.fx0 / "out", "exploratory"), (self.fxf / "out", "confirmatory")):
            self.assertTrue((out / K.LAYOUT["look"].format(group=group)).is_file())
            self.assertTrue((out / K.LAYOUT["look_md"].format(group=group)).is_file())

    def test_records_are_stub_and_strict(self):
        out = self.fx0 / "out"
        v = self.rec(out, "evaluate.main.stall.s0")
        self.assertEqual((v["kind"], v["mode"], v["prereg_sha256"], v["source"]), ("near80v9-values", "stub", None,
                                                                                  "trained"))
        self.assertEqual(set(v["arm_ids"]), set(K.ARMS["main"]))
        for sit in K.SITUATIONS:
            self.assertEqual(set(v["situations"][sit]), {"points", "held70_points", "capped80_points", "unseen_gain",
                                                         "overshoot_loss", "past80_share", "zone_shares",
                                                         "lower_tail_points"})
        pol = self.rec(out, "policy.main.stall.s0")
        self.assertTrue(all(c.get("passed") for c in pol["checks"].values() if isinstance(c, dict)))
        self.assertIn("stub: not run", pol["standin"]["error"])
        self.assertTrue(pol["matches"]["even_pull"]["matched"])
        outf = self.fxf / "out"
        s0 = self.s0
        main = self.rec(outf, f"data_main.main.stall.s{s0}")
        self.assertEqual(main["source"], "fresh")
        self.assertTrue(main["train_file"]["training_file_check"]["passed"])
        self.assertTrue(main["group_count"]["defined"])
        self.assertTrue(main["group_count"]["group_checks"]["passed"])
        sw = self.rec(outf, f"switch.main.stall.s{s0}")
        self.assertTrue(sw["defined"])
        self.assertEqual(sw["groups_rule"], "pooled")
        self.assertTrue(sw["group_checks"]["passed"])
        self.assertTrue(sw["group_checks"]["counts_equal_the_outcome_free_count"])
        self.assertEqual(sw["groups"]["n_full_throttle"], main["group_count"]["n_tail"])
        self.assertEqual(sw["heldout_rows_sha256"], main["heldout_rows_sha256"])
        self.assertIn("bank_only", sw["reported_only"])
        crit = self.rec(outf, f"critic.main.no_stall.s{s0}")
        self.assertTrue(crit["training_file_check"]["passed"])

    def test_the_refusal_rule_end_to_end(self):
        """On the refusing seed BCA's switch is on with an infinite height: its height arms are padded in step 2's
        program and not scored, the matched rivals are not matched, every other arm (the plain twins included) is
        trained and scored, and the look fails that seed for both claims with the refusal's own check."""
        import policy_phase as PP
        outf, sr = self.fxf / "out", self.sr
        sw = self.rec(outf, f"switch.main.stall.s{sr}")
        self.assertTrue(sw["bca_refused"])
        self.assertTrue(sw["outcome_free_count"]["below_bar"])
        lock = self.rec(outf, f"policy.main.stall.s{sr}")
        self.assertEqual(lock["refusal"]["refused_arms"], ["obca_forced", "obca_k1", "obca_k2"])
        self.assertTrue(lock["refusal"]["bca_refused"])
        none_ids = {a for a, n in lock["arm_ids"].items() if n is None}
        self.assertEqual(none_ids, set(K.BCA_HEIGHT_ARMS) | set(K.BCA_MATCHED_ARMS))
        self.assertEqual(lock["ramp_programs"][1][:3], [PP.refused_pad(a) for a in ("obca_k1", "obca_k2",
                                                                                     "obca_forced")])
        v = self.rec(outf, f"evaluate.main.stall.s{sr}")
        scored = {a for a, n in v["arm_ids"].items() if n is not None}
        self.assertTrue(set(K.PLAIN_TWINS) | {"conformal_k1", "range_hinge", "none", "fixed_gated"} <= scored)
        look = LG.read_json_verified(outf / K.LAYOUT["look"].format(group="confirmatory"))
        self.assertIn(str(sr), look["refused"])
        P = K.load_local("pooled")
        n_tail = look["refused"][str(sr)]["stall"]["n_tail"]
        self.assertEqual(look["failed_checks"]["C1"][str(sr)], P.refused_check(n_tail, "stall"))
        self.assertEqual(look["failed_checks"]["C2"][str(sr)], P.refused_check(n_tail, "stall"))
        self.assertIn("refusal.some", [r["id"] for r in look["sentences"]])
        ok = self.rec(outf, f"policy.main.stall.s{self.s0}")
        self.assertEqual(ok["refusal"]["refused_arms"], [])

    def test_nothing_trained(self):
        calls, hits = {}, {}
        for out, _, _ in self.outs():
            for p in (out / "ledger").rglob("*.usage.json"):
                u = LG.read_json_verified(p)
                for k, v in (u.get("stub_calls") or {}).items():
                    calls[k] = calls.get(k, 0) + v
                for k, v in (u.get("stub_guard_hits") or {}).items():
                    hits[k] = hits.get(k, 0) + v
        self.assertEqual(hits, {})
        for name in ("fixed_critic", "clone", "actor_only_run", "run_ramp", "run_kind", "match_controls",
                     "gate_seed", "fresh_log", "standin_matched"):
            self.assertGreater(calls.get(name, 0), 0, name)

    def test_ramp_programs_in_the_stub_run(self):
        """Every policy phase of the stub runs ran exactly the ramp programs of policy_phase.ramp_layout with its own
        arm row (step 2's batches; refused places padded), and the lock records them."""
        import policy_phase as PP
        seen = set()
        for out, seeds, _ in self.outs():
            for k in PH.plan(spec="seeds:" + seeds):
                if k.kind != "policy" or PH.done(k, out) != "done":
                    continue
                pid = PH.phase_id(k)
                usages = [LG.read_json_verified(q) for q in (out / "ledger" / pid).glob("*.usage.json")]
                done = [u for u in usages if u.get("exit_code") == 0]
                self.assertEqual(len(done), 1, pid)
                lock = LG.read_json_verified(out / PH.record_rel(k))
                row = dict(refused=lock["refusal"]["refused_arms"])
                want = [list(c) for c in PP.ramp_layout(k.setting, k.world, row)]
                self.assertEqual(done[0]["stub_ramp_batches"], want, pid)
                self.assertEqual(lock["ramp_programs"], want)
                seen.add((k.setting, k.world))
        self.assertEqual(seen, {("main", w) for w in K.WORLDS})

    def test_no_bytecode_in_the_code_folder(self):
        self.assertFalse((HERE / "__pycache__").exists())

    def test_reserved_seeds_never_appear(self):
        for out, _, _ in self.outs():
            for p in out.rglob("*"):
                for s in K.RESERVED + K.SPENT:
                    self.assertNotIn(f".s{s}.", p.name)
                    self.assertNotIn(f"seed{s}", p.name)


class TestCommands(unittest.TestCase):
    def test_selftest_lines(self):
        out = []
        cfg = R.Cfg(mode="scratch", out=tmpdir(self) / "o", approval_fn=lambda: APPROVAL)
        ok = R.selftest(cfg, "run", out.append, checks={"a": lambda: "fine", "b": lambda: None})
        self.assertTrue(ok)
        self.assertEqual(out, ["PASS a: fine", "PASS b", "READY"])
        out = []

        def bad():
            raise RuntimeError("numpy is 1.26.4, not 2.2.6")
        self.assertFalse(R.selftest(cfg, "run", out.append, checks={"versions": bad, "b": lambda: None}))
        self.assertEqual(out, ["FAIL versions: numpy is 1.26.4, not 2.2.6", "PASS b", "NOT READY (versions)"])

    def test_default_checks_by_command(self):
        cfg = R.Cfg(mode="scratch", out=tmpdir(self) / "o", approval_fn=lambda: APPROVAL)
        self.assertNotIn("step 2's saved actors", R.default_checks(cfg, "run"))
        self.assertIn("approval", R.default_checks(cfg, "plumbing"))
        self.assertIn("step 2's saved actors", R.default_checks(cfg, "plumbing"))
        self.assertIn("the gate's saved files for the reproduction check", R.default_checks(cfg, "plumbing"))
        self.assertNotIn("step 2's saved actors", R.default_checks(cfg, "selftest"))
        self.assertIn("code manifest and pins", R.default_checks(cfg, "selftest"))
        for name in ("versions", "jax", "placement", "one processor per lane", "git identity",
                     "version 8 not running", "output root writable", "gate data"):
            self.assertIn(name, R.default_checks(cfg, "run"))

    def test_plan_listing_reads_only(self):
        out = tmpdir(self)
        cfg = R.Cfg(mode="scratch", out=out / "o", plan="smoke")
        lines = []
        self.assertEqual(R.cmd_plan(cfg, lines.append), 0)
        self.assertEqual(lines[-1], f"plan: {len(PH.plan(spec='smoke'))} phases")
        self.assertFalse((out / "o").exists())
        self.assertTrue(any("data_main.main.stall.s0" in line for line in lines))

    def test_held_phase_lock_refuses(self):
        fx = tmpdir(self) / "fx"
        fx.mkdir()
        (fx / K.TEST_MARKER).write_text(K.TEST_MARKER_TEXT)
        (fx / "out" / "locks").mkdir(parents=True)
        pid = "critic.main.stall.s0"
        with open(fx / "out" / "locks" / f"{pid}.lock", "a") as h:
            fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(PM.main(["--phase-id", pid, "--out", str(fx / "out"), "--mode", "stub", "--fixture",
                                      str(fx), "--plan", "seeds:0"]), 97)
        self.assertFalse((fx / "out" / "ledger").exists())

    @unittest.skipUnless((K.CODE / "pooled.py").is_file(), "the looks (pooled.py) are not present")
    def test_report_refuses_a_damaged_record(self):
        out = tmpdir(self) / "o"
        rel = out / K.LAYOUT["reproduction_differs"]
        rel.parent.mkdir(parents=True)
        rel.write_text('{"kind": "near80v9-reproduction-differs"}')
        rel.with_name(rel.name + ".sha256").write_text("0" * 64 + f"  {rel.name}\n")
        lines = []
        self.assertEqual(R.cmd_report(out, lines.append), R.EXIT_REFUSED)
        self.assertTrue(lines and lines[-1].startswith("refused"), lines)


class TestNoBytecode(unittest.TestCase):
    def test_no_pycache(self):
        self.assertFalse((HERE / "__pycache__").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
