"""Unit tests of the near-80 version 9 runner: the phase table (the main setting, the smoke and fresh groups), the
refusals, the memory rule against version 7's, the ledger's interruption classes, the training-file check, the safe
stop, the budget guard, step 2's saved actors, the restated step-1 and step-2 code (the arm table with the per-arm
refusal, the lone calls), and the end-to-end stub run on the plumbing's seed. Synthetic data and stubs only: nothing
is trained (every learner is replaced by a stand-in that takes no gradient step), no recorded seed is generated,
loaded or read, and nothing is written into the repository (temporary folders only). DRAFT, not frozen.

Run (plain unittest), from the work folder, with NEAR80_HOST and NEAR80_ROOT set for a host other than the desktop:
  PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=<repo> \\
    <python> <repo>/outputs/shift/beyond-the-data/near80/code/test_runner.py
It takes a few minutes (the end-to-end stub run is most of it). POSIX only (fcntl, taskset).
"""
import sys

sys.dont_write_bytecode = True

import ast  # noqa: E402
import fcntl  # noqa: E402
import importlib.util  # noqa: E402
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# The owner's rule (20 October 2026): no test names or runs a reserved fresh seed (40-59). This file's fresh-group
# tests (the plan, the fake phases, the per-seed runners, the fresh-seed record) use the synthetic labels
# K.TEST_LABELS through NEAR80_STUB_FRESH_LABELS=1, set here before the constants load and passed to every phase
# process the tests start (TEST_ENV; the runner forwards it in stub mode). Every stub run uses seed 0.
os.environ["NEAR80_STUB_FRESH_LABELS"] = "1"
import near80_constants as K  # noqa: E402

assert K.FRESH == K.TEST_LABELS and K.STUB_FRESH_ENV == "NEAR80_STUB_FRESH_LABELS"
ROOT = Path(K.ROOT)
DESKTOP_ROOT = Path(K.DESKTOP_ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ledger as LG  # noqa: E402
import phases as PH  # noqa: E402
import phase_main as PM  # noqa: E402
import near80_runner as R  # noqa: E402

PY = sys.executable
TEST_ENV = {k: v for k, v in K.PHASE_ENV.items() if k != "PYTHONPYCACHEPREFIX"}   # as stub phases (Cfg.phase_env)
for _name in ("NEAR80_HOST", "NEAR80_ROOT", "NEAR80_PYTHON", "BCA_SHIFT_DEPS", K.STUB_FRESH_ENV):
    if os.environ.get(_name):
        TEST_ENV[_name] = os.environ[_name]
STUB_PATH = os.environ.get("NEAR80_STUB_PYTHONPATH")
APPROVAL = dict(prereg_sha256="a" * 64, prefix="a" * 16, fixture=True)
BIG_MEM = "MemTotal:       67108864 kB\nMemAvailable:   62914560 kB\n"
F0, F1, F2, F3, F4 = K.FRESH[:5]           # the synthetic labels (K.TEST_LABELS), never 40-59
WORK = "/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir()


def tmpdir(test):
    d = Path(tempfile.mkdtemp(prefix="near80_test_runner_"))
    test.addCleanup(shutil.rmtree, d, True)
    if LG.inside(d, ROOT):
        raise RuntimeError("temporary folders must lie outside the repository")
    return d


def partners_present():
    """The approval chain and the looks are needed by the end-to-end stub run only."""
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
        self.assertEqual(len(keys), len(set(keys)))
        count = {}
        for k in keys:
            count[k.kind] = count.get(k.kind, 0) + 1
        n = len(K.FRESH)
        self.assertEqual(count, dict(data_main=2 * n, critic=2 * n, critic_full=n, critic_second=2 * n, switch=2 * n,
                                     clone=2 * n, reference_actors=2 * n, policy=2 * n, lock=1, evaluate=2 * n,
                                     look=1))
        for k in keys:
            if k.kind in PH.LEARNER_KINDS or k.kind in ("evaluate", "data_main"):
                self.assertEqual(k.setting, "main")
            if k.kind == "critic_full":
                self.assertEqual((k.setting, k.world), ("x", "stall"))
            if k.seed is not None:
                self.assertIn(k.seed, K.FRESH)
                self.assertNotIn(k.seed, K.USED_SEEDS)
                self.assertEqual(k.group, "fresh")
            else:
                self.assertEqual(k.group, "fresh")
        smoke = PH.plan(spec="smoke")
        self.assertEqual({k.seed for k in smoke if k.seed is not None}, {0})
        self.assertEqual({k.group for k in smoke}, {"smoke"})
        self.assertEqual({k.kind for k in smoke}, set(K.PHASE_KINDS))

    def test_order_of_the_table(self):
        kinds = [k.kind for k in PH.plan()]
        firsts = [kinds.index(x) for x in K.PHASE_KINDS]
        self.assertEqual(firsts, sorted(firsts))

    def test_ids_round_trip(self):
        for k in PH.plan() + PH.plan(spec="smoke"):
            self.assertEqual(PH.parse_id(PH.phase_id(k)), k)
        self.assertEqual(PH.phase_id(PH.make_key("policy", "main", "stall", F0)), f"policy.main.stall.s{F0}")
        self.assertEqual(PH.phase_id(PH.make_key("lock", group="fresh")), "lock.x.x.fresh")
        self.assertEqual(PH.phase_id(PH.make_key("look", group="smoke")), "look.x.x.smoke")

    def test_needs_are_in_the_plan_and_acyclic(self):
        for spec in ("full", "smoke", f"seeds:0,{F0}", f"seeds:{F0},{F1}"):
            keys = PH.plan(spec=spec)
            inside = set(keys)
            for k in keys:
                for d in PH.needs(k, keys):
                    self.assertIn(d, inside, (spec, k, d))
            pr = PH.priorities(keys)              # raises on a cycle
            self.assertTrue(all(v > 0 for v in pr.values()))

    def test_specific_needs(self):
        k = PH.parse_id(f"policy.main.no_stall.s{F2}")
        n = set(PH.phase_id(x) for x in PH.needs(k))
        self.assertEqual(n, {f"switch.main.stall.s{F2}", f"switch.main.no_stall.s{F2}", f"critic.main.no_stall.s{F2}",
                             f"clone.main.no_stall.s{F2}", f"reference_actors.main.no_stall.s{F2}",
                             f"policy.main.stall.s{F2}"})
        k = PH.parse_id(f"reference_actors.main.stall.s{F1}")
        self.assertIn(f"critic_full.x.stall.s{F1}", [PH.phase_id(x) for x in PH.needs(k)])
        k = PH.parse_id(f"reference_actors.main.no_stall.s{F1}")
        self.assertNotIn("critic_full", " ".join(PH.phase_id(x) for x in PH.needs(k)))
        self.assertEqual(PH.needs(PH.parse_id(f"data_main.main.stall.s{F0}")), [])
        self.assertEqual([PH.phase_id(x) for x in PH.needs(PH.parse_id(f"switch.main.stall.s{F0}"))],
                         [f"critic_second.main.stall.s{F0}"])
        ev = PH.parse_id(f"evaluate.main.stall.s{F0}")
        self.assertEqual([PH.phase_id(x) for x in PH.needs(ev)], ["lock.x.x.fresh", f"policy.main.stall.s{F0}"])
        look = PH.parse_id("look.x.x.fresh")
        kinds = {x.kind for x in PH.needs(look, PH.plan())}
        self.assertEqual(kinds, {"evaluate", "data_main", "switch"})

    def test_est_minutes(self):
        self.assertEqual(PH.est_minutes(PH.parse_id("policy.main.stall.s0")), 50)
        self.assertEqual(PH.est_minutes(PH.parse_id(f"policy.main.no_stall.s{F0}")), 14)
        self.assertEqual(PH.est_minutes(PH.parse_id("data_main.main.stall.s0")), 3)

    def test_outputs_include_sidecars_and_stop(self):
        k = PH.parse_id(f"policy.main.stall.s{F0}")
        out = PH.outputs(k)
        for rel in (f"policy/main/stall/seed{F0}/lock.json", f"policy/main/stall/seed{F0}/lock.json.sha256",
                    f"policy/main/stall/seed{F0}/actors.msgpack", f"policy/main/stall/seed{F0}/stop.json",
                    f"stops/policy.main.stall.s{F0}.json"):
            self.assertIn(rel, out)
        self.assertIn("looks/fresh.md", PH.outputs(PH.parse_id("look.x.x.fresh")))


class TestRefusals(unittest.TestCase):
    def test_used_reserved_and_outside_seeds_for_every_kind_and_mode(self):
        for kind in K.PHASE_KINDS:
            for seed in list(K.SPENT_SEEDS) + list(K.RESERVED_SEEDS[:2]) + [5, 15, 20, 99, 100]:
                for mode in ("official", "scratch", "stub"):
                    with self.assertRaises(PH.Refused):
                        PH.check_key(PH.parse_id(f"{kind}.main.stall.s{seed}"), mode,
                                     "full" if mode == "official" else f"seeds:0,{F0}")
        for seed in list(K.SPENT_SEEDS) + list(K.RESERVED_SEEDS):
            with self.assertRaises(PH.Refused):
                PH.parse_id(f"critic.main.stall.s{seed}")

    def test_phase_main_refuses_used_and_reserved_seeds_in_every_mode(self):
        d = tmpdir(self)
        fx = d / "fx"
        fx.mkdir()
        (fx / K.TEST_MARKER).write_text(K.TEST_MARKER_TEXT)
        for seed in (K.SPENT_SEEDS[0], K.RESERVED_SEEDS[0], 5):
            for kind in ("data_main", "critic", "policy", "evaluate"):
                pid = f"{kind}.main.stall.s{seed}"
                self.assertEqual(PM.main(["--phase-id", pid, "--out", str(K.OUT), "--mode", "official"]), 97)
                self.assertEqual(PM.main(["--phase-id", pid, "--out", str(d / "scratch"), "--mode", "scratch"]), 97)
                self.assertEqual(PM.main(["--phase-id", pid, "--out", str(fx / "out"), "--mode", "stub",
                                          "--fixture", str(fx), "--plan", f"seeds:0,{seed}"]), 97)
        self.assertFalse((d / "scratch").exists())
        self.assertFalse((fx / "out").exists())

    def test_scratch_never_reaches_fresh_seeds(self):
        """Scratch mode is the plumbing on seed 0 (section 8.2): a phase of a fresh seed, or of the fresh group, is
        refused in scratch mode for every kind and plan, before any approval or data step."""
        d = tmpdir(self)
        for pid in (f"data_main.main.stall.s{F0}", f"policy.main.stall.s{F2}", f"critic_full.x.stall.s{F4}"):
            self.assertEqual(PM.main(["--phase-id", pid, "--out", str(d / "scratch"), "--mode", "scratch"]), 97)
            self.assertEqual(PM.main(["--phase-id", pid, "--out", str(d / "scratch"), "--mode", "scratch", "--plan",
                                      "smoke"]), 97)
        self.assertFalse((d / "scratch").exists())
        for kind in K.PHASE_KINDS:
            for seed in K.FRESH[:2]:
                for setting in ("main", "x"):
                    for spec in ("full", "smoke", f"seeds:0,{seed}"):
                        with self.assertRaises(PH.Refused) as cm:
                            PH.check_key(PH.make_key(kind, setting, "stall", seed), "scratch", spec)
                        self.assertTrue(str(cm.exception).startswith("refused"))
        for kind in ("lock", "look"):
            with self.assertRaises(PH.Refused):
                PH.check_key(PH.make_key(kind, group="fresh"), "scratch", "smoke")
        with self.assertRaises(PH.Refused):                       # scratch is the smoke plan only
            PH.check_key(PH.parse_id("data_main.main.stall.s0"), "scratch", "full")
        PH.check_key(PH.parse_id("data_main.main.stall.s0"), "scratch", "smoke")
        ok, why = PH.admit(PH.parse_id(f"data_main.main.stall.s{F0}"), APPROVAL, None, "scratch")
        self.assertFalse(ok)
        self.assertEqual(why, PH.SCRATCH_FRESH)
        with self.assertRaises(PH.Refused):
            R.Cfg(mode="scratch", out=d / "o", plan="full")
        self.assertEqual(R.Cfg(mode="scratch", out=d / "o").plan, "smoke")

    def test_combinations_outside_the_table(self):
        for pid in ("critic.near80.stall.s0", "v6_band.main.stall.s0", "data_band.band.stall.s0",
                    "readouts.main.stall.s0", "policy.random_rows.stall.s0", "critic_full.x.no_stall.s0",
                    "critic.main.stall.x", "lock.x.x.x", "nonsense.x.x.x", "critic.main.moon.s0", "look.x.x.band"):
            with self.assertRaises(PH.Refused):
                PH.check_key(PH.parse_id(pid))
        PH.check_key(PH.parse_id("critic_full.x.stall.s0"), "scratch", "smoke")
        with self.assertRaises(PH.Refused):
            PH.check_key(PH.parse_id("critic_full.x.stall.s0"), "official", "smoke")
        with self.assertRaises(PH.Refused):                       # the smoke seed never runs officially
            PH.check_key(PH.parse_id("critic.main.stall.s0"), "official", "full")
        with self.assertRaises(PH.Refused):                       # the official run has the fresh group only
            PH.check_key(PH.make_key("lock", group="smoke"), "official", "full")
        PH.check_key(PH.parse_id(f"critic.main.stall.s{F0}"), "official", "full")

    def test_admission(self):
        """Every phase needs the approval facts; a fresh phase is refused in scratch mode; nothing else gates a
        phase (version 8's exploratory look and its stop rules are gone)."""
        main40 = PH.parse_id(f"data_main.main.stall.s{F0}")
        self.assertFalse(PH.admit(main40, None)[0])
        self.assertFalse(PH.admit(PH.parse_id("critic.main.stall.s0"), None)[0])
        self.assertEqual(PH.admit(main40, APPROVAL), (True, None))
        self.assertEqual(PH.admit(main40, APPROVAL, None, "stub"), (True, None))
        self.assertEqual(PH.admit(PH.parse_id("critic.main.stall.s0"), APPROVAL, None, "scratch"), (True, None))
        self.assertEqual(PH.admit(PH.make_key("lock", group="fresh"), APPROVAL, None, "scratch")[1], PH.SCRATCH_FRESH)
        self.assertFalse(PH.is_skip(PH.SCRATCH_FRESH))

    def test_skip_rules_from_files(self):
        out = tmpdir(self)
        keys = PH.plan()
        # a stopped need skips its dependents, through chains
        stop = PH.parse_id(f"critic_second.main.stall.s{F1}")
        LG.write_json_once(out / PH.stop_rel(stop), dict(kind="near80-phase-stop", reason="x"))
        self.assertEqual(PH.done(stop, out), "stopped")
        self.assertIn("was stopped", PH.rule_skip(PH.parse_id(f"switch.main.stall.s{F1}"), out, keys))
        self.assertIn("was skipped", PH.rule_skip(PH.parse_id(f"policy.main.stall.s{F1}"), out, keys))
        self.assertIn("was skipped", PH.rule_skip(PH.parse_id(f"policy.main.no_stall.s{F1}"), out, keys))
        self.assertIn("was skipped", PH.rule_skip(PH.parse_id(f"evaluate.main.no_stall.s{F1}"), out, keys))
        self.assertEqual(PH.needs_state(PH.parse_id(f"evaluate.main.no_stall.s{F1}"), out, keys)[0], "skip")
        self.assertIsNone(PH.rule_skip(PH.parse_id("lock.x.x.fresh"), out, keys))
        self.assertIsNone(PH.rule_skip(PH.parse_id(f"critic.main.stall.s{F1}"), out, keys))   # not a dependent
        self.assertIsNone(PH.rule_skip(PH.parse_id(f"switch.main.stall.s{F2}"), out, keys))   # another seed
        # a failed seed: a data_main failure in either world fails every phase of the seed
        LG.write_json_once(out / K.LAYOUT["data_main"].format(world="stall", seed=F3),
                           dict(kind="near80-data-main", seed_failed_reason="the gate's checks failed"))
        for pid in (f"critic.main.no_stall.s{F3}", f"critic_full.x.stall.s{F3}", f"evaluate.main.stall.s{F3}",
                    f"policy.main.no_stall.s{F3}"):
            self.assertIn("failed the gate's checks", PH.rule_skip(PH.parse_id(pid), out, keys))
        self.assertIsNone(PH.rule_skip(PH.parse_id(f"data_main.main.no_stall.s{F3}"), out, keys))
        self.assertEqual(PH.needs_state(PH.parse_id("lock.x.x.fresh"), out, keys)[0], "wait")

    def test_out_root_rules(self):
        d = tmpdir(self)
        with self.assertRaises(PM.Refusal):
            PM.check_out_root(ROOT / "runs" / "shift_explore" / "near80_x", "scratch")
        with self.assertRaises(PM.Refusal):
            PM.check_out_root(ROOT / "runs" / "shift_explore" / "near80", "scratch")
        with self.assertRaises(PM.Refusal):
            PM.check_out_root(ROOT / "runs" / "shift_explore" / "restructure" / "step2", "official")
        with self.assertRaises(PM.Refusal):
            PM.check_out_root(d / "elsewhere", "official")
        if K.HOST == "test":
            with self.assertRaises(PM.Refusal):                       # the test host never runs an official phase
                PM.check_out_root(K.OUT, "official")
            with self.assertRaises(PH.Refused):
                R.Cfg(mode="official")
        else:
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
        self.assertTrue(str(K.OUT).replace("\\", "/").endswith("runs/shift_explore/near80_v9"))
        self.assertTrue(str(K.V8_OUT).replace("\\", "/").endswith("runs/shift_explore/near80"))

    def test_pinned_list_equals_step2s(self):
        sys.path.insert(0, str(K.S2_DIR))
        try:
            spec = importlib.util.spec_from_file_location("near80_test_step2lib", str(K.S2_DIR / "step2lib.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            sys.path.remove(str(K.S2_DIR))
        self.assertEqual(tuple(str(p) for p in mod.PINNED), tuple(str(DESKTOP_ROOT / r) for r in PM.PINNED_REL))

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
        k = PH.parse_id(f"critic.main.stall.s{F1}")
        PM.check_write_once(k, out)
        (out / f"learners/main/stall/seed{F1}").mkdir(parents=True)
        (out / f"learners/main/stall/seed{F1}/critic.msgpack").write_bytes(b"x")
        with self.assertRaises(PM.Refusal):
            PM.check_write_once(k, out)
        LG.write_json_once(out / PH.record_rel(k), dict(kind="near80-critic"))
        with self.assertRaises(PM.Refusal):
            PM.check_write_once(k, out)
        with self.assertRaises(LG.RecordError):
            LG.write_json_once(out / PH.record_rel(k), dict(kind="near80-critic"))

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

    def test_stub_refuses_repository_fixture_and_used_seeds(self):
        out = []
        self.assertEqual(R.main(["stub", "--fixture", str(ROOT / "runs" / "x")], out=out.append), 97)
        d = tmpdir(self)
        self.assertEqual(R.main(["stub", "--fixture", str(d), "--seeds", "0,7"], out=out.append), 97)
        self.assertEqual(R.main(["stub", "--fixture", str(d), "--seeds", f"0,{K.RESERVED_SEEDS[0]}"],
                                out=out.append), 97)
        self.assertEqual(R.main(["stub", "--fixture", str(d), "--seeds", "0", "--refuse", "12"], out=out.append), 97)
        self.assertFalse(any(d.iterdir()))

    def test_runner_options_for_several_runners(self):
        """--seeds, --group-only and --tag: one runner per job sharing an output root (the per-seed entry); the
        plumbing takes none of them."""
        d = tmpdir(self)
        cfg = R.Cfg(mode="stub", out=d / "fx" / "out", fixture=d / "fx", seeds=[F1, F0], tag=f"s{F0}")
        self.assertEqual(cfg.seeds, [F0, F1])
        self.assertTrue(cfg.mine(PH.parse_id(f"critic.main.stall.s{F0}")))
        self.assertTrue(cfg.mine(PH.parse_id(f"policy.main.no_stall.s{F0}")))
        self.assertFalse(cfg.mine(PH.parse_id(f"critic.main.stall.s{F2}")))
        self.assertFalse(cfg.mine(PH.parse_id("lock.x.x.fresh")))
        self.assertFalse(cfg.mine(PH.parse_id(f"evaluate.main.stall.s{F0}")))     # needs the group's lock
        g = R.Cfg(mode="stub", out=d / "fx" / "out", fixture=d / "fx", group_only=True)
        self.assertTrue(g.mine(PH.parse_id("lock.x.x.fresh")))
        self.assertTrue(g.mine(PH.parse_id("look.x.x.fresh")))
        self.assertTrue(g.mine(PH.parse_id(f"evaluate.main.no_stall.s{F1}")))
        self.assertFalse(g.mine(PH.parse_id(f"critic.main.stall.s{F0}")))
        self.assertFalse(g.mine(PH.parse_id(f"policy.main.stall.s{F0}")))
        self.assertEqual(R.runner_lock_name("group"), "runner_group.lock")
        self.assertEqual(R.status_name(None), K.LAYOUT["status"])
        with self.assertRaises(PH.Refused):
            R.Cfg(mode="stub", out=d / "fx" / "out", fixture=d / "fx", seeds=[K.SPENT_SEEDS[0]])
        with self.assertRaises(PH.Refused):
            R.Cfg(mode="stub", out=d / "fx" / "out", fixture=d / "fx", tag="a b")
        out = []
        self.assertEqual(R.main(["plumbing", "--seeds", f"{F0}"], out=out.append), 97)
        self.assertIn("one job", out[-1])

    def test_second_runner_refused(self):
        out = tmpdir(self)
        cfg = R.Cfg(mode="scratch", out=out / "o", poll=0.05, approval_fn=lambda: APPROVAL)
        (out / "o").mkdir()
        with open(out / "o" / "runner.lock", "a") as h:
            fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lines = []
            self.assertEqual(R.run_locked(cfg, [], APPROVAL, lines.append, signals=False), 97)
            self.assertIn("another near-80 runner", lines[-1])
        self.assertFalse(hasattr(R, "version7_running"))         # version 8's version-7 gating is gone

    def test_plumbing_record_required(self):
        out = tmpdir(self)
        with self.assertRaises(PH.Refused):
            R.require_plumbing(out, APPROVAL)
        p = out / K.LAYOUT["plumbing"]
        LG.write_json_once(p, dict(kind="near80-plumbing", passed=False, prereg_sha256=APPROVAL["prereg_sha256"]))
        with self.assertRaises(PH.Refused):
            R.require_plumbing(out, APPROVAL)
        out2 = tmpdir(self)
        p2 = out2 / K.LAYOUT["plumbing"]
        LG.write_json_once(p2, dict(kind="near80-plumbing", passed=True, prereg_sha256="c" * 64))
        with self.assertRaises(PH.Refused):
            R.require_plumbing(out2, APPROVAL)
        out3 = tmpdir(self)
        p3 = out3 / K.LAYOUT["plumbing"]
        LG.write_json_once(p3, dict(kind="near80-plumbing", passed=True, prereg_sha256=APPROVAL["prereg_sha256"]))
        R.require_plumbing(out3, APPROVAL)
        p3.write_text(p3.read_text().replace("true", "false"))
        with self.assertRaises(PH.Refused):
            R.require_plumbing(out3, APPROVAL)
        out4 = tmpdir(self)
        LG.write_json_once(out4 / K.LAYOUT["plumbing"], dict(kind="near80-plumbing", passed=True,
                                                              prereg_sha256=APPROVAL["prereg_sha256"]))
        LG.write_json_once(out4 / K.LAYOUT["reproduction_differs"], dict(kind="near80-reproduction-differs"))
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
        f = K.FORECAST_CPU_HOURS["fresh"]
        self.assertAlmostEqual(f, 2.2 * len(K.FRESH))
        self.assertEqual(K.CAP_CPU_HOURS, 55.0)
        paused, over = R.budget({"fresh": 2 * f + 1.0, "plumbing": 0.1})
        self.assertIn("fresh", paused)
        self.assertIn(R.PAUSE_TEXT, paused["fresh"])
        self.assertIn(f"{R.RUN_N80} budget-continue fresh", paused["fresh"])
        self.assertIsNone(over)
        paused, over = R.budget({"fresh": 2 * f - 0.1})
        self.assertEqual(paused, {})
        paused, over = R.budget({"fresh": 40.0, "plumbing": 20.0})
        self.assertIsNotNone(over)
        self.assertIn("budget-continue <stage>", over)
        # the plumbing has no phase of the plan: it never pauses (its hours still count toward the cap)
        paused, over = R.budget({"plumbing": 9.0})
        self.assertEqual((paused, over), ({}, None))
        # each typed continuation raises the stage's pause point and the cap by one forecast of the stage
        self.assertEqual(R.budget({"fresh": 2 * f + 1}, continues={"fresh": 1})[0], {})
        self.assertIn("fresh", R.budget({"fresh": 3 * f + 1}, continues={"fresh": 1})[0])
        hours = {"fresh": 40.0, "plumbing": 21.0}
        self.assertIsNotNone(R.budget(hours)[1])
        self.assertIsNone(R.budget(hours, continues={"fresh": 1})[1])

    def test_budget_continue_is_typed_and_recorded(self):
        out = tmpdir(self)
        cfg = R.Cfg(mode="scratch", out=out / "o", approval_fn=lambda: APPROVAL)
        lines = []
        self.assertEqual(R.cmd_budget_continue(cfg, "fresh", lines.append, isatty=False), 97)
        self.assertEqual(R.cmd_budget_continue(cfg, "plumbing", lines.append, isatty=True), 97)
        self.assertEqual(R.cmd_budget_continue(cfg, "fresh", lines.append, isatty=True, argv=["x"]), 0)
        self.assertEqual(R.cmd_budget_continue(cfg, "fresh", lines.append, isatty=True, argv=["x"]), 0)
        self.assertEqual(R.budget_continues(out / "o"), {"fresh": 2})
        rec = LG.read_json_verified(out / "o" / K.LAYOUT["budget_continue"].format(stage="fresh", n=2))
        self.assertEqual((rec["kind"], rec["stage"], rec["n"]), ("near80-budget-continue", "fresh", 2))
        self.assertIn(f"{R.RUN_N80} run --lanes 10", lines[-1])

    def test_cpu_hours_from_the_ledger(self):
        out = tmpdir(self)
        for pid, kind, group, sec in (("critic.main.stall.s0", "critic", "smoke", 7200),
                                      (f"critic.main.stall.s{F0}", "critic", "fresh", 3600),
                                      (f"policy.main.stall.s{F1}", "policy", "fresh", 1800)):
            n = LG.allocate_attempt(out, pid, mode="official")
            LG.write_usage(out, pid, n, sec, 0)
            LG.finish_attempt(out, dict(phase_kind=kind, setting="main", world="stall", seed=0, group=group), pid,
                              n, done=True, stopped=False, exit_code=0)
        h = LG.cpu_hours(out, plumbing_seconds=360)
        self.assertAlmostEqual(h["plumbing"], 2.1)
        self.assertAlmostEqual(h["fresh"], 1.5)
        self.assertEqual(LG.stage_of("critic", "main", "fresh"), "fresh")
        self.assertEqual(LG.stage_of("look", "x", "smoke"), "plumbing")


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
            pid = f"critic.main.stall.s{K.FRESH[i]}"
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
    LG.write_json_once(out / PH.stop_rel(key), dict(kind="near80-phase-stop", reason="a fake stop"))
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
    """A runner over fake phases (fake_phase.py stands in for phase_main.py). mode='scratch' for the smoke group's
    seed 0; the fresh group's phases (the synthetic labels) need mode='stub' with a fixture, since a scratch run
    refuses the fresh seeds (phases.admit)."""

    def __init__(self, test, keys, behavior, lanes=2, poll=0.05, mode="scratch"):
        self.d = tmpdir(test)
        self.out = self.d / "out"
        (self.out / "locks").mkdir(parents=True)
        (self.out / "behavior.json").write_text(json.dumps(behavior))
        (self.d / "meminfo").write_text(BIG_MEM)
        self.fake = self.d / "fake_phase.py"
        self.fake.write_text(f"CODE = {str(HERE)!r}\n" + FAKE_PHASE)
        extra = dict(fixture=self.d) if mode == "stub" else {}
        self.cfg = R.Cfg(mode=mode, out=self.out, plan="smoke", lanes=lanes, cpus=sorted(os.sched_getaffinity(0)),
                         meminfo=self.d / "meminfo", poll=poll, python=PY, phase_main=self.fake,
                         approval_fn=lambda: APPROVAL, **extra)
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
                         ("critic_full", "x", 0, "smoke"))
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
        self.assertEqual([s for s in fr.started() if s.startswith("critic_second")], ["critic_second.main.stall.s0.1"])

    def test_refusal_and_failure_halt(self):
        keys = self.keys(f"critic.main.stall.s{F0}", f"critic.main.stall.s{F1}")
        fr = FakeRunner(self, keys, {f"critic.main.stall.s{F0}": ["refuse"], f"critic.main.stall.s{F1}": ["sleep0.5"]},
                        lanes=1, mode="stub")
        self.assertEqual(fr.run(), 97)
        self.assertEqual(fr.started(), [f"critic.main.stall.s{F0}.1"])
        fr2 = FakeRunner(self, keys, {f"critic.main.stall.s{F0}": ["fail"], f"critic.main.stall.s{F1}": ["sleep0.5"]},
                         lanes=1, mode="stub")
        self.assertEqual(fr2.run(), 1)
        self.assertTrue(any("FAILED" in line for line in fr2.lines))

    def test_budget_pause(self):
        keys = self.keys(f"critic.main.stall.s{F0}", f"critic.main.stall.s{F1}")
        fr = FakeRunner(self, keys, {}, mode="stub")
        fr.cfg.forecast = dict(K.FORECAST_CPU_HOURS, fresh=0.0001)
        pid = f"data_main.main.stall.s{F0}"
        n = LG.allocate_attempt(fr.out, pid, mode="stub")
        LG.write_usage(fr.out, pid, n, 10.0, 0)
        LG.finish_attempt(fr.out, dict(phase_kind="data_main", group="fresh"), pid, n, done=True,
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
            self.assertEqual(R.cmd_budget_continue(fr.cfg, "fresh", [].append, isatty=True, argv=["t"]), 0)
        self.assertEqual(fr.run(), 0)
        self.assertEqual(len(fr.started()), 2)

    def test_over_the_cap_notes_every_waiting_phase(self):
        keys = self.keys(f"critic.main.stall.s{F0}")
        fr = FakeRunner(self, keys, {}, mode="stub")
        fr.cfg.cap = 0.0001
        pid = f"data_main.main.stall.s{F0}"
        n = LG.allocate_attempt(fr.out, pid, mode="stub")
        LG.write_usage(fr.out, pid, n, 10.0, 0)
        LG.finish_attempt(fr.out, dict(phase_kind="data_main", group="fresh"), pid, n, done=True,
                          stopped=False, exit_code=0)
        self.assertEqual(fr.run(until_waiting=False), R.EXIT_PAUSED)
        status = LG.read_json_verified(fr.out / "status.json")
        self.assertIn("beyond the cap", status["phases"][f"critic.main.stall.s{F0}"]["note"])
        self.assertTrue(any(line.find("PAUSED (budget): paused: the run has used") >= 0 for line in fr.lines))

    def test_a_phase_that_can_never_fit_halts(self):
        keys = self.keys("critic.main.stall.s0")
        fr = FakeRunner(self, keys, {})
        (fr.d / "meminfo").write_text("MemTotal:       12582912 kB\nMemAvailable:   12000000 kB\n")
        self.assertEqual(fr.run(until_waiting=False), 1)
        self.assertEqual(fr.started(), [])
        self.assertTrue(any("can never start" in line for line in fr.lines))

    def test_two_runners_share_an_output_root_by_seed(self):
        """The per-seed entry: a runner with --seeds starts only its seeds' phases; another runner on the other seed
        may hold the root at the same time (its own lock name)."""
        keys = self.keys(f"data_main.main.stall.s{F0}", f"data_main.main.stall.s{F1}")
        fr = FakeRunner(self, keys, {}, mode="stub")
        fr.cfg.seeds, fr.cfg.tag = [F0], f"s{F0}"
        self.assertEqual(fr.run(), 0)
        self.assertEqual(fr.started(), [f"data_main.main.stall.s{F0}.1"])
        self.assertTrue((fr.out / f"status_s{F0}.json").is_file())
        self.assertTrue(R.lock_free(fr.out / f"runner_s{F0}.lock"))
        fr.cfg.seeds, fr.cfg.tag = [F1], f"s{F1}"
        self.assertEqual(fr.run(), 0)
        self.assertEqual(len(fr.started()), 2)

    def test_safe_stop_one_ctrl_c(self):
        d = tmpdir(self)
        out = d / "out"
        (out / "locks").mkdir(parents=True)
        pids = ["critic.main.stall.s0", "critic.main.no_stall.s0", "critic_full.x.stall.s0"]   # the smoke seed
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
            f"approval_fn=lambda: dict(prereg_sha256='a'*64))\n"
            f"keys = [PH.parse_id(p) for p in {pids!r}]\n"
            "sys.exit(R.run_locked(cfg, keys, dict(prereg_sha256='a'*64)))\n")
        p = subprocess.Popen([PY, str(driver)], cwd=WORK, env=dict(TEST_ENV), stdout=subprocess.PIPE,
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
            f"approval_fn=lambda: dict(prereg_sha256='a'*64), stop_grace=5)\n"
            f"keys = [PH.parse_id({pid!r})]\n"
            "sys.exit(R.run_locked(cfg, keys, dict(prereg_sha256='a'*64)))\n")
        p = subprocess.Popen([PY, str(driver)], cwd=WORK, env=dict(TEST_ENV), stdout=subprocess.PIPE,
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
        return R.Cfg(mode="scratch", out=out / "o", approval_fn=lambda: APPROVAL), out

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
        self.assertAlmostEqual(rec["cpu_seconds"], 5 * 1.0 + 5.0 + 100.0)
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
        pid = f"evaluate.main.stall.s{F0}"
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

    def test_results_copy_records_only_and_say_git_add_f(self):
        d = tmpdir(self)
        src, dest = d / "out", d / "n80"
        LG.write_json_once(src / f"eval/main/stall/seed{F0}/values.json", dict(kind="near80-values"))
        LG.write_json_once(src / f"data/main/stall/seed{F0}/data.json", dict(kind="near80-data-main"))
        LG.write_json_once(src / K.LAYOUT["status"], dict(kind="near80-runner-status",
                                                            phases={"a": dict(state="done")}))
        (src / "looks").mkdir(parents=True)
        (src / "looks" / "fresh.md").write_text("# look\n")
        (src / "logs").mkdir()
        (src / "logs" / f"critic.main.stall.s{F0}.log").write_text("log\n")
        (src / f"learners/main/stall/seed{F0}").mkdir(parents=True)
        (src / f"learners/main/stall/seed{F0}/critic.msgpack").write_bytes(b"model")
        (src / f"data/main/stall/seed{F0}/train.hdf5").write_bytes(b"rows")
        (src / "locks").mkdir()
        (src / "locks" / "x.lock").write_text("")
        lines = []
        folder = R.make_results(src, dest, lines.append, stamp="20261016T000000Z")
        files = sorted(str(q.relative_to(folder)).replace(os.sep, "/") for q in folder.rglob("*") if q.is_file())
        self.assertEqual(files, ["RESULTS.json", "RESULTS.json.sha256", "SHA256SUMS",
                                 f"files/data/main/stall/seed{F0}/data.json",
                                 f"files/data/main/stall/seed{F0}/data.json.sha256",
                                 f"files/eval/main/stall/seed{F0}/values.json",
                                 f"files/eval/main/stall/seed{F0}/values.json.sha256",
                                 f"files/logs/critic.main.stall.s{F0}.txt", "files/looks/fresh.md",
                                 "files/status.json", "files/status.json.sha256"])
        for line in (folder / "SHA256SUMS").read_text().splitlines():
            sha, rel = line.split("  ", 1)
            self.assertEqual(LG.file_sha256(folder / rel), sha)
        self.assertTrue(LG.read_json_verified(folder / "RESULTS.json")["finished"])
        self.assertTrue(any(line.strip().startswith("git add -f ") for line in lines), lines)   # .gitignore's data/
        self.assertTrue(any(f"pull --rebase origin {K.BRANCH}" in line for line in lines))
        self.assertFalse(any("scp" in line for line in lines))
        with self.assertRaises(PH.Refused):
            R.make_results(src, dest, lines.append, stamp="20261016T000000Z")
        # on the cluster (the owner's answer to choice 9): the output root's SHA-256 list is written beside the
        # records and the owner is told to copy both folders to the desktop, check them, and commit there
        lines = []
        folder = R.make_results(src, d / "n80c", lines.append, stamp="20261016T000001Z", host="cluster")
        listed = (folder / R.OUTPUT_ROOT_SUMS).read_text(encoding="utf8").splitlines()
        self.assertEqual(len(listed), 10)                                      # every file but the lock
        for line in listed:
            sha, rel = line.split("  ", 1)
            q = Path(rel) if Path(rel).is_absolute() else K.ROOT / rel
            self.assertTrue(q.is_file(), rel)
            self.assertEqual(LG.file_sha256(q), sha)
        self.assertTrue(any("critic.msgpack" in line for line in listed))     # model files included
        self.assertFalse(any(".lock" in line for line in listed))
        text = "\n".join(lines)
        self.assertIn("Nothing is committed here", text)
        self.assertIn(f"scp -r <login node>:{src}", text)
        self.assertIn(f"sha256sum -c --quiet {K.DESKTOP_ROOT}/{K.N80_REL}/{folder.name}/{R.OUTPUT_ROOT_SUMS}", text)
        self.assertIn(f"git add -f {K.N80_REL}/{folder.name}", text)
        self.assertIn("push_runfiles_near80.sh", text)
        self.assertNotIn(f"cd {K.ROOT}\n", text)

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
        self.assertEqual(K.RUNFILES_BRANCH, "claude/near80-v9-run-files")

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
        self.assertEqual(LP.height_rank([0.1, 0.5, 2.0], 1.0), 2)
        self.assertEqual(LP.height_rank([0.1, 0.5, 2.0], 3.0), 1)
        self.assertEqual(LP.height_rank([0.1, 0.5, 2.0], 0.0), 4)
        self.assertIsNone(LP.height_rank([0.1], float("inf")))
        self.assertIsNone(LP.height_rank([], 1.0))

    def test_arm_row_with_the_per_arm_refusal(self):
        """The arm table of version 9: step 2's six arms by arm_table's rules, the five new arms by the mean
        difference, and a refusal read per arm (never a stop for a BCA-dependent arm)."""
        import policy_phase as PP
        out = tmpdir(self)

        def entry(on, height, gap=0.4):
            return dict(t_lo=0.7, t_up=0.98, certificate=dict(on=on, height=height),
                        groups=dict(n_full_throttle=50, n_rest=150, mean_full_throttle=gap + 0.1, mean_rest=0.1,
                                    q90_full_throttle=gap + 0.2, q90_rest=0.1), rank_test=dict(U=1.0, p=0.01))

        def write(seed, stall, no_stall):
            for w, e in (("stall", stall), ("no_stall", no_stall)):
                LG.write_json_once(out / K.LAYOUT["switch"].format(setting="main", world=w, seed=seed),
                                   dict(kind="near80-switch", entry=e))

        write(F0, entry(True, 2.0), entry(False, 0.05, gap=0.02))
        row = PP.arm_row("main", F0, "stall", out)
        self.assertEqual(row["arms"]["obca_k1"], dict(switch=True, h=2.0, k=1.0, refused=False, why=None))
        self.assertEqual(row["arms"]["obca_k2"]["k"], 2.0)
        self.assertEqual(row["arms"]["meandiff_k1"]["switch"], True)
        self.assertAlmostEqual(row["arms"]["meandiff_k1"]["h"], 0.4)
        self.assertAlmostEqual(row["arms"]["meandiff_q90x1"]["h"], 0.5)          # the plain 90% gap
        self.assertEqual((row["arms"]["meandiff_q90x1"]["k"], row["arms"]["meandiff_q90x2"]["k"]), (1.0, 2.0))
        self.assertAlmostEqual(row["arms"]["meandiff_q90x2"]["h"], 0.5)          # k = 2 doubles it in the program
        self.assertEqual(row["arms"]["meandiff_limit"]["kind"], "hinge")
        self.assertEqual(row["arms"][K.NOISE_ARM], dict(switch=True, h=2.0, k=1.0, refused=False, why=None))
        self.assertEqual((row["refused"], row["bca_refused"], row["meandiff_on"]), ([], False, True))
        for a in K.NEW_ARMS:
            self.assertIn(a, row["arms"])
        ns = PP.arm_row("main", F0, "no_stall", out)
        self.assertEqual(ns["arms"]["obca_k1"]["h"], 0.0)
        self.assertEqual(ns["arms"]["obca_forced"], dict(switch=True, h=2.0, k=1.0, refused=False, why=None))
        for a in ("meandiff_k1", "meandiff_q90x1", "meandiff_q90x2", "meandiff_limit"):
            self.assertFalse(ns["arms"][a]["switch"])
            self.assertEqual(ns["arms"][a]["h"], 0.0)
        self.assertEqual(ns["arms"]["meandiff_limit"]["kind"], "ramp_h0")
        # BCA on with an infinite height in the stalling world: the four BCA-dependent arms are refused, no stop; the
        # twins and the other arms stand
        write(F1, entry(True, "inf"), entry(False, 0.05))
        row = PP.arm_row("main", F1, "stall", out)
        self.assertEqual(row["refused"], sorted(K.BCA_DEPENDENT))
        self.assertTrue(row["bca_refused"])
        for a in K.BCA_DEPENDENT:
            self.assertTrue(row["arms"][a]["refused"])
            self.assertIsNone(row["arms"][a]["h"])
        for a in K.TWIN_ARMS + ("fixed_gated", "conformal_k1", "meangap_k1", "meandiff_k1"):
            self.assertFalse(row["arms"][a]["refused"], a)
        ns = PP.arm_row("main", F1, "no_stall", out)
        self.assertEqual(ns["refused"], ["obca_forced"])                      # the stalling world's height
        self.assertFalse(ns["bca_refused"])
        self.assertFalse(ns["arms"]["obca_k1"]["refused"])
        # a refusal of an arm that does not depend on BCA's height is a plumbing stop
        S = PP.load_step2()[0]
        per = {f"stall/seed{F1}": entry(True, 2.0), f"no_stall/seed{F1}": entry(False, 0.05)}
        real = PP.step2_arms

        def refuse_conformal(S_, per_, seed, world):
            arms, sources, t_lo, t_up = real(S_, per_, seed, world)
            arms["conformal_k1"].update(refused=True, why="test", h=None)
            return arms, sources, t_lo, t_up
        with unittest.mock.patch.object(PP, "step2_arms", refuse_conformal):
            with self.assertRaises(PH.PhaseStop) as cm:
                PP.arm_row_from("main", F1, "stall", per, S)
        self.assertEqual(cm.exception.reason, PH.STOP_REASONS["check"])
        # the layout's pad for a refused arm
        self.assertEqual(PP.pad_name("obca_k1"), "pad_refused_obca_k1")

    def test_ramp_programs_are_step2s_then_the_lone_calls(self):
        """BCA's switched ramp and the plumbing check train in step 2's own batches (step 2's train_policy.py: the
        stalling world runs w0_check alone, then RAMP_ARMS; the no-stall world runs [w0_check] + RAMP_ARMS); then the
        lone calls in K.LONE_CALLS' order, each a program of one; meandiff_limit runs last (train_world)."""
        import policy_phase as PP
        src = (K.S2_DIR / "train_policy.py").read_text()
        self.assertIn("trained = runner.run([MC.ramp_arm(S.W0, 0.0, 0.0)], MC.log)", src)
        self.assertIn("arms = [MC.ramp_arm(S.W0, 0.0, 0.0)] + [MC.ramp_arm(n, row[\"arms\"][n][\"h\"], "
                      "row[\"arms\"][n][\"k\"])", src)
        stall, no_stall = PP.ramp_layout("main", "stall"), PP.ramp_layout("main", "no_stall")
        self.assertEqual(stall[0], (K.W0,))
        self.assertEqual(stall[1], tuple(K.STEP2_RAMP_ARMS))
        self.assertEqual(stall[1][0], K.PRIMARY)
        self.assertEqual(no_stall[0], (K.W0,) + tuple(K.STEP2_RAMP_ARMS))
        lone = [("meandiff_k1",), ("meandiff_q90x1",), ("meandiff_q90x2",), ("obca_k1_alone",)]
        self.assertEqual(stall[2:], lone)
        self.assertEqual(no_stall[1:], lone)
        self.assertEqual(tuple(n for (n,) in lone), K.LONE_CALLS)
        self.assertNotIn("meandiff_limit", [n for c in stall for n in c])
        row = dict(arms={K.PRIMARY: dict(h=2.5, k=1.0), "meandiff_q90x2": dict(h=1.2, k=2.0),
                         "obca_k2": dict(h=None, k=2.0, refused=True)})

        class MC:
            @staticmethod
            def ramp_arm(name, h, k):
                return (name, h, k)
        self.assertEqual(PP.layout_arm(MC, row, K.PRIMARY), (K.PRIMARY, 2.5, 1.0))
        self.assertEqual(PP.layout_arm(MC, row, "meandiff_q90x2"), ("meandiff_q90x2", 1.2, 2.0))
        self.assertEqual(PP.layout_arm(MC, row, "obca_k2"), ("pad_refused_obca_k2", 0.0, 0.0))
        self.assertEqual(PP.layout_arm(MC, row, K.W0), (K.W0, 0.0, 0.0))

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
    return subprocess.run(cmd, cwd=WORK, env=phase_env(), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          timeout=900)


@unittest.skipUnless((K.CODE / "approval_check.py").is_file(), "needs approval_check.py")
class TestTrainingFileCheck(unittest.TestCase):
    def test_tampered_training_file_stops_and_skips(self):
        import h5py
        fx = _fixture(self)
        r = _phase(fx, "data_main.main.stall.s0")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rec = LG.read_json_verified(fx / "out" / K.LAYOUT["data_main"].format(world="stall", seed=0))
        self.assertTrue(rec["train_file"]["training_file_check"]["passed"])
        self.assertTrue(rec["pooled_groups"]["defined"])                       # the outcome-free count, before training
        self.assertIn("certify_chance", rec["pooled_groups"]["certification"])
        path = fx / "out" / K.LAYOUT["train_file"].format(setting="main", world="stall", seed=0)
        with h5py.File(path, "r+") as f:
            a = f["actions"][()]
            a.view("uint8")[0] ^= 1                        # one byte of the commands
            f["actions"][...] = a
        r = _phase(fx, "critic.main.stall.s0")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        k = PH.parse_id("critic.main.stall.s0")
        stop = LG.read_json_verified(fx / "out" / PH.stop_rel(k))
        self.assertEqual(stop["reason"], PH.STOP_REASONS["training_file"])
        self.assertFalse(stop["detail"]["training_file_check"]["passed"])
        self.assertFalse((fx / "out" / PH.record_rel(k)).exists())
        self.assertEqual(PH.done(k, fx / "out"), "stopped")
        keys = PH.plan(spec="seeds:0")
        for pid in ("reference_actors.main.stall.s0", "policy.main.stall.s0", "policy.main.no_stall.s0",
                    "evaluate.main.stall.s0"):
            self.assertEqual(PH.effective_state(PH.parse_id(pid), fx / "out", keys), "skipped", pid)
        self.assertEqual(PH.effective_state(PH.parse_id("critic_second.main.stall.s0"), fx / "out", keys), "missing")
        r = _phase(fx, "critic.main.stall.s0")                       # a stopped phase is never rerun
        self.assertEqual(r.returncode, 97)
        r = _phase(fx, "reference_actors.main.stall.s0")             # its dependent: skipped, nothing generated
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(PH.done(PH.parse_id("reference_actors.main.stall.s0"), fx / "out"), "skipped")
        self.assertFalse((fx / "out" / "learners/main/stall/seed0/reference_actors.msgpack").exists())

    def test_tampered_prepared_hash_stops_switch(self):
        """BCA's switch reads the setting's prepared rows: the training-file check runs for it too, so a data record
        whose prepared hash differs from the rows the switch would read is a recorded stop."""
        fx = _fixture(self)
        for pid in ("data_main.main.stall.s0", "critic_second.main.stall.s0"):
            r = _phase(fx, pid)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        p = fx / "out" / K.LAYOUT["data_main"].format(world="stall", seed=0)
        body = LG.read_json_verified(p)
        body["prepared_sha256"]["train"] = "0" * 64
        p.unlink()
        (p.parent / (p.name + ".sha256")).unlink()
        LG.write_json_once(p, body)
        r = _phase(fx, "switch.main.stall.s0")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        k = PH.parse_id("switch.main.stall.s0")
        stop = LG.read_json_verified(fx / "out" / PH.stop_rel(k))
        self.assertEqual(stop["reason"], PH.STOP_REASONS["training_file"])
        self.assertFalse((fx / "out" / PH.record_rel(k)).exists())

    def test_every_context_checks_the_prepared_rows(self):
        """settingdata.SettingContext.prepared compares the rows it serves with the data record's prepared_sha256 on
        every seed (the smoke seed included)."""
        import types
        import settingdata as SD
        out = tmpdir(self)
        good = {k: f"{k}-sha" for k in SD.PREPARED_KEYS}
        LG.write_json_once(out / K.LAYOUT["data_main"].format(world="stall", seed=0),
                           dict(kind="near80-data-main", prepared_sha256=dict(good)))
        LG.write_json_once(out / K.LAYOUT["data_main"].format(world="stall", seed=F1),
                           dict(kind="near80-data-main", prepared_sha256=dict(good, train="other")))

        def ctx(seed, prep):
            c = SD.SettingContext.__new__(SD.SettingContext)
            c.setting, c.rep, c.out_root = "main", seed, out
            c._prepared, c._prepared_checked = {"stall": prep}, set()
            return c
        prep = types.SimpleNamespace(sha256=dict(good))
        self.assertIs(ctx(0, prep).prepared("stall"), prep)
        with self.assertRaises(PH.PhaseStop) as cm:
            ctx(F1, prep).prepared("stall")
        self.assertEqual(cm.exception.reason, PH.STOP_REASONS["training_file"])
        with self.assertRaises(PH.PhaseStop):
            ctx(0, types.SimpleNamespace(sha256=dict(good, bank="x"))).prepared("stall")
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
RUN_SIZE_KEYS = ["chunk", "starts", "steps", "updates"]


@unittest.skipUnless((K.CODE / "approval_check.py").is_file(), "needs approval_check.py")
class TestFreshSeedRecordRoundTrip(unittest.TestCase):
    """Amendment 1's lesson, kept in version 9. A fresh seed's data record is written through phase_main (the wrapper
    every runner phase goes through) and read back by every later phase of that seed: the log sizes it was generated
    with live under their own name (log_sizes), apart from the wrapper's run sizes (run_sizes). Stub mode only:
    synthetic logs, nothing trained."""

    def setUp(self):
        self.fx = _fixture(self)

    def readback(self, world="stall", record=None, ok=True):
        r = subprocess.run([PY, "-c", FRESH_READBACK, str(HERE), str(self.fx), str(F0), world] +
                           (["record"] if record is not None else []),
                           input=json.dumps(record) if record is not None else None, cwd=WORK, env=phase_env(),
                           capture_output=True, text=True, timeout=900)
        if not ok:
            return None, r
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return json.loads(r.stdout.strip().splitlines()[-1]), r

    def test_record_written_by_the_wrapper_reads_back(self):
        r = _phase(self.fx, f"data_main.main.stall.s{F0}", plan=f"seeds:0,{F0}")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rec = LG.read_json_verified(self.fx / "out" / K.LAYOUT["data_main"].format(world="stall", seed=F0))
        self.assertEqual(rec["source"], "fresh")
        self.assertEqual(rec["log_sizes"], [500, 50, 50])                   # the stub's fresh_settings
        self.assertEqual(sorted(rec["run_sizes"]), RUN_SIZE_KEYS)            # the wrapper's, apart
        self.assertTrue(rec["pooled_groups"]["defined"])
        self.assertGreaterEqual(rec["pooled_groups"]["groups"]["n_full_throttle"], 29)
        got, _ = self.readback()
        self.assertEqual(got["seen"], [[[500, 50, 50], 1.5, "float"]])
        for pid in (f"critic_full.x.stall.s{F0}", f"critic.main.stall.s{F0}", f"critic_second.main.stall.s{F0}",
                    f"clone.main.stall.s{F0}"):
            r = _phase(self.fx, pid, plan=f"seeds:0,{F0}")
            self.assertEqual(r.returncode, 0, pid + "\n" + r.stdout + r.stderr)
            self.assertEqual(PH.done(PH.parse_id(pid), self.fx / "out"), "done", pid)
        crit = LG.read_json_verified(self.fx / "out" / PH.record_rel(PH.parse_id(f"critic.main.stall.s{F0}")))
        self.assertTrue(crit["training_file_check"]["passed"])

    def test_three_integers_are_used_and_anything_else_is_refused(self):
        r = _phase(self.fx, f"data_main.main.no_stall.s{F0}", plan=f"seeds:0,{F0}")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rec = LG.read_json_verified(self.fx / "out" / K.LAYOUT["data_main"].format(world="no_stall", seed=F0))
        got, _ = self.readback("no_stall", dict(rec, log_sizes=[500, 50, 50]))
        self.assertEqual(got["seen"], [[[500, 50, 50], 1.5, "float"]])
        for bad in (dict(steps=None), [500, 50], [500, 50, "50"], [500, 50, True], [0, 50, 50], [-1, 50, 50], None,
                    dict.fromkeys(RUN_SIZE_KEYS)):
            _, r2 = self.readback("no_stall", dict(rec, log_sizes=bad), ok=False)
            self.assertNotEqual(r2.returncode, 0, repr(bad))
            self.assertIn("refused: the data record's log_sizes", r2.stderr, repr(bad))

    def test_the_stub_generator_refuses_what_dataset_generate_refuses(self):
        r = subprocess.run([PY, "-c", "import sys; sys.dont_write_bytecode = True; sys.path.insert(0, sys.argv[1]);"
                                      "import dataclasses, stubs\n"
                                      "spec = stubs._recorder_spec()\n"
                                      "stubs.stub_fresh_log(0, 'stall', spec, (500, 50, 50), 1.5)\n"
                                      "other = dataclasses.replace(spec, damping=spec.damping + 0.25)\n"
                                      "ok3 = (500, 50, 50)\n"
                                      "for sp, sizes, slope in ((spec, ('chunk', 'starts', 'steps', 'updates'), 1.5),\n"
                                      "                         (spec, (500, 50), 1.5), (spec, ok3, float('nan')),\n"
                                      "                         (spec, ok3, 1.6), (other, ok3, 1.5)):\n"
                                      "    try:\n"
                                      "        stubs.stub_fresh_log(0, 'stall', sp, sizes, slope)\n"
                                      "    except TypeError:\n"
                                      "        continue\n"
                                      "    raise SystemExit('accepted ' + repr((sp, sizes, slope)))\n"
                                      "print('ok')", str(HERE)],
                           cwd=WORK, env=phase_env(), capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.strip().splitlines()[-1], "ok")


@unittest.skipUnless((K.CODE / "approval_check.py").is_file(), "needs approval_check.py")
class TestStep2Saved(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import approval_check as AC
        cls.d = Path(tempfile.mkdtemp(prefix="near80_test_step2_"))
        cls.fx = cls.d / "fx"
        AC.write_test_fixture(cls.fx)
        r = subprocess.run([PY, str(HERE / "stubs.py"), "step2-fixture", str(cls.fx)], cwd=WORK, env=phase_env(),
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise RuntimeError(r.stdout + r.stderr)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, True)

    def copy(self):
        d = tmpdir(self)
        shutil.copytree(self.fx, d / "fx")
        return d / "fx"

    def check(self, fx):
        import evaluate_phase as EP
        run = PH.Run(PH.make_key("evaluate", "main", "stall", 0), fx / "out", mode="stub", fixture=fx, starts=5)
        return EP.check_step2_saved(run, 0, "stall")

    def test_fixture_passes(self):
        lock, acts, lock_sha, vals = self.check(self.fx)
        self.assertEqual(lock["seed"], 0)
        self.assertEqual(LG.file_sha256(self.fx / "step2/policy/seed0/stall/lock.json"), lock_sha)
        for a in K.NEW_ARMS:
            self.assertIn(a, lock["arm_ids"])

    # (version 8's test_changed_point is gone with its evaluate_step2_saved phase: version 9 reproduces step 2's
    # actors bit for bit in the plumbing and never re-reads step 2's saved points; check_step2_saved checks the lock,
    # the actors and the values' lock fingerprint, as test_changed_lock_byte and test_changed_actor say)

    def test_changed_lock_byte(self):
        fx = self.copy()
        p = fx / "step2/policy/seed0/stall/lock.json"
        p.write_text(p.read_text().replace('"seed": 0', '"seed":  0'))
        with self.assertRaises(PH.PhaseStop) as cm:
            self.check(fx)
        self.assertEqual(cm.exception.reason, PH.STOP_REASONS["fingerprint"])

    def test_changed_actor(self):
        fx = self.copy()
        p = fx / "step2/policy/seed0/stall/actors.msgpack"
        b = bytearray(p.read_bytes())
        b[-1] ^= 1
        p.write_bytes(bytes(b))
        with self.assertRaises(PH.PhaseStop) as cm:
            self.check(fx)
        self.assertEqual(cm.exception.reason, PH.STOP_REASONS["fingerprint"])


# --------------------------------------------------------------------------------------------- the end-to-end stub run

@unittest.skipUnless(partners_present(), "needs approval_check.py and pooled.py")
class TestStubEndToEnd(unittest.TestCase):
    """Every phase kind of the plan on a test fixture, in stub mode, on the plumbing's seed 0 (the smoke group):
    nothing is trained (the stand-ins count their calls; the training functions are guards that raise and count
    their entries). The fresh group's stub run, with a refusing seed, is test_v9.TestStubRefuse."""

    @classmethod
    def run_stub(cls, fx, seeds):
        env = dict(TEST_ENV)
        if STUB_PATH:
            env["NEAR80_STUB_PYTHONPATH"] = STUB_PATH
        lanes = str(min(4, len(os.sched_getaffinity(0))))
        return subprocess.run([PY, str(HERE / "near80_runner.py"), "stub", "--fixture", str(fx), "--lanes", lanes,
                               "--seeds", seeds], cwd=WORK, env=env, capture_output=True, text=True, timeout=3600)

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
        cls.d = Path(tempfile.mkdtemp(prefix="near80_test_e2e_"))
        cls.fx0 = cls.fixture("fx0")
        cls.r0 = cls.run_stub(cls.fx0, "0")
        cls.look0 = LG.ok_record(cls.fx0 / "out" / K.LAYOUT["look"].format(group="smoke"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, True)

    def test_every_phase_kind_ran(self):
        out, r = self.fx0 / "out", self.r0
        self.assertEqual(r.returncode, 0, r.stdout[-4000:] + r.stderr[-4000:])
        keys = PH.plan(spec="seeds:0")
        not_done = {PH.phase_id(k): PH.done(k, out) for k in keys if PH.done(k, out) != "done"}
        self.assertEqual(not_done, {})
        self.assertEqual({k.kind for k in keys}, set(K.PHASE_KINDS))
        self.assertTrue((out / K.LAYOUT["look"].format(group="smoke")).is_file())
        self.assertTrue((out / K.LAYOUT["look_md"].format(group="smoke")).is_file())
        self.assertIsNotNone(self.look0, "the smoke look was not written")
        self.assertEqual(self.look0["kind"], "near80-v9-look-smoke")
        self.assertEqual(self.look0["claims"]["c1"]["status"], "plumbing check, not a claim")
        # the stand-ins' points are all 0 (nothing trained), so the validity check fails and every claim reads
        # 'invalid' before C2's "not read" (one seed) could be said: PREREGISTRATION.md 14.5's precedence
        self.assertEqual(self.look0["outcomes"], dict(c1="invalid", c1_by_stop=False, c2="invalid", c2_each={},
                                                      c3="invalid", c3_each={}))
        self.assertIn("c1.invalid", [r["id"] for r in self.look0["sentences"]])

    def test_records_are_stub_and_strict(self):
        out = self.fx0 / "out"
        v = LG.read_json_verified(out / PH.record_rel(PH.parse_id("evaluate.main.stall.s0")))
        self.assertEqual((v["kind"], v["mode"], v["prereg_sha256"], v["source"]), ("near80-values", "stub", None,
                                                                                  "trained"))
        self.assertEqual(set(v["arm_ids"]), set(K.ARMS["main"]))
        for a in K.TWIN_ARMS + (K.NOISE_ARM, "meandiff_k1", K.PRIMARY, "range_hinge"):
            self.assertIsNotNone(v["arm_ids"][a], a)                        # every new arm was trained and scored
        for sit in K.SITUATIONS:
            self.assertEqual(set(v["situations"][sit]), {"points", "held70_points", "capped80_points", "unseen_gain",
                                                         "overshoot_loss", "past80_share", "zone_shares",
                                                         "lower_tail_points"})
        n = LG.read_json_verified(out / PH.record_rel(PH.parse_id("evaluate.main.no_stall.s0")))
        self.assertIsNone(n["arm_ids"]["full_none"])
        pol = LG.read_json_verified(out / PH.record_rel(PH.parse_id("policy.main.stall.s0")))
        self.assertTrue(all(c.get("passed") for c in pol["checks"].values() if isinstance(c, dict)))
        self.assertIn("stub: not run", pol["standin"]["error"])
        self.assertTrue(pol["matches"]["even_pull"]["matched"])
        self.assertEqual(pol["refused_arms"], [])
        self.assertEqual(pol["meandiff_limit_kind"], "hinge")
        sw = LG.read_json_verified(out / PH.record_rel(PH.parse_id("switch.main.stall.s0")))
        self.assertTrue(sw["defined"])
        self.assertFalse(sw["refused"])
        self.assertEqual(sw["groups_rule"], "pooled")
        self.assertEqual(len(sw["pooled"]["tail_scores_sorted"]), sw["groups"]["n_full_throttle"])
        dm = LG.read_json_verified(out / PH.record_rel(PH.parse_id("data_main.main.stall.s0")))
        self.assertEqual(dm["pooled_groups"]["groups"]["n_full_throttle"], sw["groups"]["n_full_throttle"])
        self.assertEqual(dm["pooled_groups"]["groups"]["n_rest"], sw["groups"]["n_rest"])
        crit = LG.read_json_verified(out / PH.record_rel(PH.parse_id("critic.main.no_stall.s0")))
        self.assertTrue(crit["training_file_check"]["passed"])

    def test_nothing_trained(self):
        calls, hits = {}, {}
        out = self.fx0 / "out"
        for p in (out / "ledger").rglob("*.usage.json"):
            u = LG.read_json_verified(p)
            for k, v in (u.get("stub_calls") or {}).items():
                calls[k] = calls.get(k, 0) + v
            for k, v in (u.get("stub_guard_hits") or {}).items():
                hits[k] = hits.get(k, 0) + v
        self.assertEqual(hits, {})
        for name in ("fixed_critic", "clone", "actor_only_run", "run_ramp", "run_kind", "match_controls",
                     "gate_seed", "standin_matched"):
            self.assertGreater(calls.get(name, 0), 0, name)
        self.assertEqual(calls.get("fresh_log", 0), 0)                      # the smoke seed is the gate's

    def test_ramp_programs_in_the_stub_run(self):
        """Every policy phase of the stub run ran exactly the ramp programs of policy_phase.ramp_layout (step 2's
        batches, then the lone calls), and the hinge program for meandiff_limit when its switch was on."""
        import policy_phase as PP
        out = self.fx0 / "out"
        for k in PH.plan(spec="seeds:0"):
            if k.kind != "policy":
                continue
            pid = PH.phase_id(k)
            usages = [LG.read_json_verified(q) for q in (out / "ledger" / pid).glob("*.usage.json")]
            done = [u for u in usages if u.get("exit_code") == 0]
            self.assertEqual(len(done), 1, pid)
            got = [b for b in done[0]["stub_ramp_batches"] if b != ["meandiff_limit"]]
            self.assertEqual(got, [list(c) for c in PP.ramp_layout(k.setting, k.world)], pid)
            lock = LG.read_json_verified(out / PH.record_rel(k))
            self.assertEqual(lock["ramp_programs"], [list(c) for c in PP.ramp_layout(k.setting, k.world)])
            kinds = [kind for kind, names in done[0]["stub_kind_batches"] if "meandiff_limit" in names]
            if lock["meandiff_limit_kind"] == "hinge":
                self.assertEqual(kinds, ["hinge"], pid)
            else:
                self.assertEqual(kinds, [], pid)
                self.assertIn(["meandiff_limit"], done[0]["stub_ramp_batches"])

    def test_no_bytecode_in_the_code_folder(self):
        self.assertFalse((HERE / "__pycache__").exists())

    def test_used_seeds_never_appear(self):
        out = self.fx0 / "out"
        for p in out.rglob("*"):
            for s in list(K.SPENT_SEEDS) + list(K.RESERVED_SEEDS):
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
        self.assertIn("step 2's saved actors", R.default_checks(cfg, "run"))
        self.assertIn("approval", R.default_checks(cfg, "plumbing"))
        self.assertIn("step 2's saved actors", R.default_checks(cfg, "plumbing"))
        self.assertIn("the gate's saved files for the reproduction check", R.default_checks(cfg, "plumbing"))
        self.assertNotIn("step 2's saved actors", R.default_checks(cfg, "selftest"))
        self.assertIn("code manifest and pins", R.default_checks(cfg, "selftest"))
        for name in ("host", "versions", "jax", "placement", "one processor per lane", "git identity",
                     "output root writable", "gate data", "room harness", "version 6's run folder"):
            self.assertIn(name, R.default_checks(cfg, "run"))
        self.assertNotIn("version 7 not running", R.default_checks(cfg, "run"))

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
        rel = out / K.LAYOUT["plumbing"]
        rel.parent.mkdir(parents=True)
        rel.write_text('{"kind": "near80-plumbing"}')
        rel.with_name(rel.name + ".sha256").write_text("0" * 64 + f"  {rel.name}\n")
        lines = []
        self.assertEqual(R.cmd_report(out, lines.append), R.EXIT_REFUSED)
        self.assertTrue(lines and lines[-1].startswith("refused"), lines)


class TestNoBytecode(unittest.TestCase):
    def test_no_pycache(self):
        self.assertFalse((HERE / "__pycache__").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
