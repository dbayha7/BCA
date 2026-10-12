"""CPU tests of the gate (acceptance.py) and the statistics (detection.py), on the 'smoke' preset in temporary
directories: tiny logs and update counts, so every number is a test fixture, not a finding. The Stage-0 record and the
build-check record are test stubs (their real stages are tested elsewhere and take minutes); everything after them
runs for real: data, the pre-registration, H1 and H2 per seed and pooled, then (with the pooled H1 and H2 records
marked passed by the test, to reach the later stages) the room check with its lock, power, the shift-card and report
stages (the card itself stubbed for time), and the approval and BCA refusals. Nothing is written under runs/.

Also: the hash gate refuses an edited pre-registration, an edited sidecar, a body edited with a recomputed sidecar,
an edited new or reused file, an unhashed extra file, changed marks, a changed dependency directory and changed
software or numeric flags; every repository file the gate imports is hashed; the real preset is pre-registered and
loaded only in runs/shift; the pre-registration is written once and refused when learner outputs exist; a failed
placement control counts against BCA, never for it, and is rerun once; the arms of a seed must share data and
batch-row stream; a failed harm-placed arm fails the room check's seed instead of crashing it; H2(e)'s pre-registered
reading; the stop rule (alpha ladder, retune, stop reasons) and the H1 / H2 marks give known
answers on synthetic rows; the statistics (t bounds, Holm, AUC, placement efficiency, partial Spearman, pooled ratios)
and the verdict wording give known answers; world_marks equals stage0.world_card; the learners' build checks pass; and
the earlier gate's files, runs/glq and the shared environment's package list are unchanged by the suite.
JAX_PLATFORMS=cpu python -m unittest experiments.shift.test_acceptance
"""

import io
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shift import acceptance as A  # noqa: E402
from experiments.shift import dataset as DS  # noqa: E402
from experiments.shift import detection as DT  # noqa: E402
from experiments.shift import learners as L  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402
from experiments.shift import stage0 as S0  # noqa: E402
from experiments.shift import truth as TR  # noqa: E402

CFG = A.PRESETS["smoke"]
STATE = {}


def quiet(*_a, **_k):
    pass


def stub_pre(runs):
    """Test stubs for the two minutes-long stages before the data: Stage 0 (the default recording controller, passed)
    and the build checks (passed)."""
    for v in A.VARIANTS:
        A.write_json(A.stage0_path(runs, v), dict(kind="shift-stage0", variant=v, passed=True, record=None,
                                                  chosen=RC.DEFAULT_SPEC.as_dict(), test_stub=True))
    A.write_json(A.pre_dir(runs) / "build_checks.json", dict(kind="shift-build-checks", passed=True, test_stub=True))


def setUpModule():
    STATE["before"] = dict(earlier=A.earlier_gate_state(), env=A.environment_state())
    tmp = Path(tempfile.mkdtemp(prefix="shift-acceptance-"))
    STATE["tmp"] = tmp
    runs = tmp / "runs"
    stub_pre(runs)
    A.stage_data(runs, CFG, "base", log=quiet)
    A.stage_data(runs, CFG, "retune", roles=("gate", "calibration"), log=quiet)
    STATE["runs"] = runs


def tearDownModule():
    shutil.rmtree(STATE["tmp"], ignore_errors=True)


def fresh_runs(name):
    """A copy of the module's pre-registration stages (no pre-registration, no learner output)."""
    dst = STATE["tmp"] / name
    shutil.copytree(A.pre_dir(STATE["runs"]), A.pre_dir(dst))
    return dst


def prereg(name="gate"):
    """A written and loaded pre-registration of the smoke preset in its own runs directory (cached)."""
    if name not in STATE:
        runs = fresh_runs(name)
        A.write_prereg(runs, CFG, log=quiet)
        STATE[name] = A.load_prereg(runs)
    return STATE[name]


def mark(path, pr, passed, **extra):
    A.write_json(path, dict(prereg_sha256=pr.digest, passed=passed, **extra))


class DataAndPreregTest(unittest.TestCase):
    def test_data_records(self):
        idx = A.data_index(STATE["runs"])
        self.assertTrue(all(v["passed"] for v in idx.values()))
        self.assertEqual(sorted(A.required_data(CFG, STATE["runs"])), sorted(k for k in idx))
        sd = A.load_seed(STATE["runs"], CFG, "base", "stall", 0)
        rec = sd.record
        self.assertEqual(rec["checks"]["gap_rows_in_gapped_training"], 0)
        self.assertEqual(rec["checks"]["gap_rows_in_gapped_bank"], 0)
        self.assertTrue(rec["checks"]["random_same_count"])
        self.assertTrue(rec["checks"]["leak_counter_finds_every_training_gap_row"])
        self.assertEqual(sd.removals["gap"].count, sd.removals["random"].count)
        self.assertEqual(len(sd.arrays["removed_share"]), len(sd.log.split_rows("train")))
        self.assertNotIn("noise_map", sd.arrays)
        self.assertEqual(SH.leak_count(sd.removals["gap"].kept_train_rows(sd.log), sd.log, sd.gap), 0)
        self.assertEqual(sd.log.diagnostic_reads, 0)
        conf = A.load_seed(STATE["runs"], CFG, "base", "stall", 100)
        self.assertIn("removed_share", conf.arrays)
        self.assertEqual(len(conf.arrays["noise_map"]), len(conf.log.split_rows("train")))
        self.assertNotIn("pool_removed", conf.arrays)

    def test_tampered_data_is_refused(self):
        runs = fresh_runs("tamper_data")
        path = A.data_dir(runs, "base", "stall", 0) / "arrays.npz"
        with np.load(path) as z:
            arr = {k: z[k] for k in z.files}
        arr["removed_share"] = arr["removed_share"] + 1.0
        np.savez(path, **arr)
        with self.assertRaises(A.PreregError):
            A.load_seed(runs, CFG, "base", "stall", 0)
        rec_path = A.data_dir(runs, "base", "stall", 1) / "record.json"
        rec_path.write_text(rec_path.read_text().replace('"passed": true', '"passed": false'))
        with self.assertRaises(A.GateError):
            A.load_seed(runs, CFG, "base", "stall", 1)

    def test_fast_noise_map_equals_truth(self):
        sd = A.load_seed(STATE["runs"], CFG, "base", "stall", 100)
        rows = np.arange(0, 4000, 157)
        x, a = sd.log["infos/state"][rows], np.asarray(sd.log["actions"][rows], np.float64)
        np.testing.assert_allclose(A.noise_map(x, a, sd.physics, 4, 80, (1, 2)),
                                   TR.noise_map(x, a, sd.physics, draws=4, horizon=80, seed=(1, 2)), rtol=1e-9,
                                   atol=1e-11)
        np.testing.assert_array_equal(sd.arrays["noise_map"], A.noise_map(
            sd.log["infos/state"][sd.log.split_rows("train")],
            np.asarray(sd.log["actions"][sd.log.split_rows("train")], np.float64), sd.physics, CFG.noise_map_draws,
            CFG.noise_map_horizon, (DS.TC.DATA_SEED, 100, 11)))

    def test_world_marks_equal_stage0_world_card(self):
        sd = A.load_seed(STATE["runs"], CFG, "base", "stall", 0)
        with mock.patch.object(S0.DS, "generate", return_value=sd.log), \
                mock.patch.object(S0.SH, "define_gap", return_value=(sd.gap, {})):
            card, _, _ = S0.world_card(0, RC.DEFAULT_SPEC, 1.5, with_no_stall=False)
        mine, _ = A.world_marks(sd.log, sd.gap, RC.DEFAULT_SPEC, sd.physics)
        for k in ("full_throttle_commands_removed", "region_rows_kept", "kept_past80_share", "random_keeps_past80",
                  "identical_rule_share", "region_share", "full_throttle_share"):
            self.assertEqual(mine[k], card[k], k)

    def test_prereg_written_once_and_refused_with_outputs(self):
        pr = prereg()
        with self.assertRaises(A.GateError):
            A.write_prereg(pr.runs, CFG, log=quiet)
        runs = fresh_runs("outputs_exist")
        (runs / "abc" / "x").mkdir(parents=True)
        (runs / "abc" / "x" / "critic.msgpack").write_bytes(b"0")
        with self.assertRaises(A.GateError):
            A.write_prereg(runs, CFG, log=quiet)
        runs3 = fresh_runs("no_retune_stage0")
        A.stage0_path(runs3, "retune").unlink()
        with self.assertRaises(A.GateError):
            A.write_prereg(runs3, CFG, log=quiet)
        runs2 = fresh_runs("no_build_checks")
        (A.pre_dir(runs2) / "build_checks.json").unlink()
        with self.assertRaises(A.GateError):
            A.write_prereg(runs2, CFG, log=quiet)
        payload = pr.payload
        self.assertEqual(payload["alpha_ladder"], [2.5, 5.0, 10.0])
        self.assertIn("first alpha at which H1 and H2 both pass", payload["alpha_rule"])
        self.assertEqual(set(payload["code_sha256"]), set(A.NEW_FILES))
        self.assertEqual(set(payload["dependency_sha256"]), set(A.REUSED_FILES))
        for rel in A.REUSED_FILES:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_hash_gate_refuses_edits(self):
        pr = prereg()
        runs = fresh_runs("edits")
        for name in (A.PREREG_NAME, A.PREREG_NAME + ".sha256"):
            shutil.copy(pr.runs / name, runs / name)
        A.load_prereg(runs)
        path = runs / A.PREREG_NAME
        text = path.read_text()
        edited = text.replace('"harm_min": 5.0', '"harm_min": 1.0')
        self.assertNotEqual(edited, text)
        path.write_text(edited)
        with self.assertRaises(A.PreregError):
            A.load_prereg(runs)
        import hashlib
        side = runs / (A.PREREG_NAME + ".sha256")
        side.write_text(hashlib.sha256(edited.encode()).hexdigest() + "  preregistration.json\n")
        with self.assertRaises(A.PreregError) as ctx:
            A.load_prereg(runs)
        self.assertIn("pass_marks", str(ctx.exception))
        path.write_text(text)
        side.write_text("0" * 64 + "  preregistration.json\n")
        with self.assertRaises(A.PreregError):
            A.load_prereg(runs)
        shutil.copy(pr.runs / (A.PREREG_NAME + ".sha256"), side)
        A.load_prereg(runs)
        later = A.data_dir(runs, "retune", "stall", 100)
        shutil.copytree(A.data_dir(runs, "base", "stall", 100), later)
        A.load_prereg(runs)  # a data record written after the pre-registration is allowed
        rec = A.read_json(A.data_dir(runs, "base", "stall", 0) / "record.json")
        rec["seconds"] = -1.0
        A.write_json(A.data_dir(runs, "base", "stall", 0) / "record.json", rec)
        with self.assertRaises(A.PreregError):
            A.load_prereg(runs)  # a pre-registered data record may not change
        shutil.rmtree(A.data_dir(runs, "base", "stall", 0))
        shutil.copytree(A.data_dir(STATE["runs"], "base", "stall", 0), A.data_dir(runs, "base", "stall", 0))
        A.load_prereg(runs)
        with mock.patch.dict(A.PASS_MARKS["H2"], harm_min=1.0):
            with self.assertRaises(A.PreregError):
                A.load_prereg(runs)

    def test_hash_gate_refuses_edited_code(self):
        code = STATE["tmp"] / "code"
        for rel in A.NEW_FILES + A.REUSED_FILES:
            (code / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, code / rel)
        runs = fresh_runs("code_edits")
        A.write_prereg(runs, CFG, code_root=code, log=quiet)
        A.load_prereg(runs, code_root=code)
        for rel in ("experiments/shift/learners.py", "algorithms/td3_bc.py", "experiments/truth/gap.py"):
            original = (code / rel).read_bytes()
            (code / rel).write_bytes(original + b"\n# edited\n")
            with self.assertRaises(A.PreregError):
                A.load_prereg(runs, code_root=code)
            (code / rel).write_bytes(original)
        A.load_prereg(runs, code_root=code)
        extra = code / "experiments" / "shift" / "extra.py"
        extra.write_text("x = 1\n")
        with self.assertRaises(A.PreregError):
            A.load_prereg(runs, code_root=code)
        extra.unlink()
        pr = A.load_prereg(runs, code_root=code)
        (code / "experiments/shift/shift.py").write_bytes((code / "experiments/shift/shift.py").read_bytes() + b"#")
        with self.assertRaises(A.PreregError):
            A.guard(pr)

    def test_protected_paths_and_cli(self):
        with self.assertRaises(SystemExit):
            A.refuse_protected(ROOT / "runs" / "glq" / "x")
        with self.assertRaises(SystemExit):
            A.refuse_protected(ROOT / "experiments" / "shift" / "out")
        pr = prereg()
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            self.assertEqual(A.main(["bca", "--runs", str(pr.runs)]), 2)
            self.assertEqual(A.main(["approve", "--runs", str(pr.runs)]), 2)
            self.assertEqual(A.main(["room", "--runs", str(pr.runs)]), 2)
        self.assertIn("refused", err.getvalue())
        with mock.patch("sys.stderr", io.StringIO()):
            self.assertEqual(A.main(["data", "--runs", str(A.RUNS_DEFAULT), "--preset", "smoke"]), 2)
        lines = A.plan(A.PRESETS["real"])
        order = [next(i for i, ln in enumerate(lines) if f" {s}" in ln) for s in
                 ("build_checks", "stage0", "data", "prereg", "harm_critic", "harm_return", "room", "power", "report",
                  "approve", "bca", "detect", "policy", "verdict")]
        self.assertEqual(order, sorted(order))


def synth_h1(pr, variant, seeds, gap_factor=3.0, pull_gap=1.0, no_stall_factor=1.0):
    """Synthetic H1 rows (the layout h1_seed writes): full-log errors e, gapped errors gap_factor x e on the removed
    pool and e elsewhere, random and placebo equal to full, the no-stall world's gapped errors no_stall_factor x e, and a
    pull past 80% of pull_gap for the gapped critic against 0 for the full critic (true price 0.355)."""
    rng = np.random.default_rng(0)
    for s in seeds:
        arrays = {}
        for world, controllers in (("stall", ("stall_aware", "rising_thrust")), ("no_stall", ("stall_aware",))):
            for c in controllers:
                for k in A.POOLS:
                    arrays[f"{world}:{c}:episode|{k}"] = np.repeat(np.arange(30), 10)
                    if world == "stall":
                        arrays[f"{world}:{c}:removed_share|{k}"] = rng.uniform(size=300)
                arrays[f"{world}:{c}:price|region"] = np.full(300, 0.355)
        for world, removal, c, placebo in A.H1_CRITICS:
            name = A.h1_name(world, removal, c, placebo)
            for k in A.POOLS:
                e = rng.uniform(0.5, 1.5, 300)
                factor = gap_factor if (removal == "gap" and k == "removed" and world == "stall") else \
                    (no_stall_factor if (removal == "gap" and k == "removed") else 1.0)
                arrays[f"{name}|eA|{k}"] = factor * e
            arrays[f"{name}|pull|region"] = np.full(300, pull_gap if removal == "gap" else 0.0)
        d = A._variant_root(pr, "harm_critic", variant) / f"seed{s}"
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "rows.npz", **arrays)
        A.write_json(d / "values.json", dict(prereg_sha256=pr.digest,
                                             rows_sha256={k: DS.sha256_array(v) for k, v in arrays.items()}))


class ProtocolTest(unittest.TestCase):
    """Regression tests of the protocol review's findings (no learner trains here)."""

    def test_real_preset_only_in_runs_shift(self):
        runs = fresh_runs("real_elsewhere")
        with self.assertRaises(A.GateError) as ctx:
            A.write_prereg(runs, A.PRESETS["real"], log=quiet)
        self.assertIn("real preset", str(ctx.exception))
        pr = prereg()
        other = fresh_runs("real_rewritten")
        payload = dict(pr.payload, config=A.config_dict(A.PRESETS["real"]))
        text = A.canonical(payload)
        (other / A.PREREG_NAME).write_text(text)
        import hashlib
        (other / (A.PREREG_NAME + ".sha256")).write_text(hashlib.sha256(text.encode()).hexdigest() + "  x\n")
        with self.assertRaises(A.PreregError) as ctx:
            A.load_prereg(other)
        self.assertIn("official pre-registration", str(ctx.exception))

    def test_dependency_directory_and_software_are_checked(self):
        pr = prereg()
        self.assertEqual(pr.payload["dependency_directory"], A.PKG.dependency_record())
        self.assertEqual(pr.payload["software"], A.software_state())
        self.assertEqual(pr.payload["dependency_directory"]["unexpected_entries"], [])
        changed = dict(A.PKG.dependency_record(), sha256="0" * 64)
        with mock.patch.object(A.PKG, "dependency_record", return_value=changed):
            with self.assertRaises(A.PreregError):
                A.guard(pr)
            with self.assertRaises(A.PreregError):
                A.load_prereg(pr.runs)
        soft = A.software_state()
        for edit in (dict(flags=dict(soft["flags"], JAX_ENABLE_X64="1")),
                     dict(packages=dict(soft["packages"], jax="0.0.1")), dict(python="2.7.0")):
            with mock.patch.object(A, "software_state", return_value=dict(soft, **edit)):
                with self.assertRaises(A.PreregError):
                    A.guard(pr)
        rec = A.stage_record(pr, "x", 0.0)
        self.assertEqual(rec["software"], soft)

    def test_dependency_directory_cannot_shadow_or_grow(self):
        import os
        import subprocess
        deps = A.PKG.deps_dir()
        self.assertEqual(sys.path.index(str(deps)), max(i for i, p in enumerate(sys.path) if p == str(deps)))
        for earlier in (str(ROOT), next(p for p in sys.path if p.endswith("site-packages"))):
            if earlier in sys.path:
                self.assertLess(sys.path.index(earlier), sys.path.index(str(deps)))
        copy = STATE["tmp"] / "deps_copy"
        if not copy.exists():
            shutil.copytree(deps, copy, ignore=shutil.ignore_patterns("__pycache__"))
            (copy / "algorithms").mkdir()
            (copy / "algorithms" / "td3_bc_bca.py").write_text("SHADOWED = True\n")
        env = dict(os.environ, BCA_SHIFT_DEPS=str(copy), PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1",
                   JAX_PLATFORMS="cpu")
        done = subprocess.run([sys.executable, "-c", "import experiments.shift"], cwd=str(STATE["tmp"]), env=env,
                              capture_output=True, text=True, timeout=120)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("algorithms", done.stderr)
        self.assertEqual(A.PKG.dependency_record(copy)["unexpected_entries"], ["algorithms"])

    def test_every_imported_repository_file_is_hashed(self):
        import subprocess
        code = ("import experiments.shift.acceptance as A, algorithms.td3_bc_bca, runtime.networks, "
                "experiments.shift.thruster_chain, experiments.shift.stage0; print(A.unhashed_imports())")
        done = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True,
                              timeout=300, env=dict(__import__("os").environ, JAX_PLATFORMS="cpu",
                                                    PYTHONDONTWRITEBYTECODE="1"))
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertEqual(done.stdout.strip().splitlines()[-1], "[]")
        import types
        fake = types.ModuleType("shift_review_fake_module")
        fake.__file__ = str(ROOT / "experiments" / "truth" / "gap_acceptance.py")
        with mock.patch.dict(sys.modules, {"shift_review_fake_module": fake}):
            self.assertIn("experiments/truth/gap_acceptance.py", A.unhashed_imports())
            pr = prereg()
            with mock.patch("sys.stderr", io.StringIO()) as err:
                self.assertEqual(A.main(["room", "--runs", str(pr.runs)]), 2)
            self.assertIn("outside the hashed lists", err.getvalue())
        for rel in ("experiments/truth/pools.py", "experiments/truth/__init__.py", "runtime/__init__.py"):
            self.assertIn(rel, A.REUSED_FILES)

    def test_slope_report_uses_the_unstalled_side_at_80_percent(self):
        """At the stall-aware controller's capped commands the exact slope is taken at its float64 command (0.8, below
        the float32 knee), not at the float32 command (the knee itself, where autodiff averages the two sides)."""
        from experiments.shift import thruster_chain as TC

        pol = RC.stall_aware()
        x = TC.bank_starts("strong_current", 40)
        pi = pol(x)
        capped = np.abs(pi) == 0.8
        self.assertGreater(capped.sum(), 5)
        inside = TR.action_slope(pol, x, pi, horizon=300)
        at_knee = TR.action_slope(pol, x, pi.astype(np.float32).astype(np.float64), horizon=300)
        self.assertGreater(np.abs(inside - at_knee)[capped].max(), 0.1)
        h = 1e-6
        for j in range(4):
            rows = np.flatnonzero(capped[:, j])
            if not len(rows):
                continue
            back = pi[rows].copy()
            back[:, j] -= np.sign(pi[rows, j]) * h
            fd = (TR.q_value(pol, x[rows], pi[rows], horizon=300) - TR.q_value(pol, x[rows], back, horizon=300)) / h
            np.testing.assert_allclose(np.sign(pi[rows, j]) * inside[rows, j], fd, atol=1e-4)
        inputs = dict(far=dict(x=x, acts=dict(policy=pi.astype(np.float32))))
        norm = type("Norm", (), dict(normalize=staticmethod(lambda z: z)))()
        with mock.patch.object(L, "critic_rows", return_value=(None, inside)):
            rep = A.slope_report(None, norm, inputs, pol, TC.STALL, 300, rows=40)
        self.assertLess(rep["median_slope_error"], 1e-9)
        self.assertGreater(rep["median_action_curvature"], 0.0)

    def test_h2_loss_rate_rule(self):
        f = A.loss_rate_passes
        self.assertTrue(f(3.0, 1.0, 2.0))
        self.assertFalse(f(1.5, 1.0, 2.0))
        self.assertTrue(f(0.01, -0.5, 2.0))     # the other steps gain: the loss sits entirely at past-80% steps
        self.assertTrue(f(0.01, 0.0, 2.0))
        self.assertFalse(f(0.0, -0.5, 2.0))
        self.assertFalse(f(-1.0, -2.0, 2.0))
        self.assertFalse(f(None, 1.0, 2.0))
        self.assertIn("other steps lose nothing", A.MARK_LABELS["H2"]["e"])
        self.assertIn("any positive loss rate", A.MARK_TEXT["H2"])

    def test_room_match_skips_a_failed_harm_placed_arm(self):
        with mock.patch.object(L, "match_even", side_effect=AssertionError("must not match")):
            for failed, target in ((dict(harm_placed=True, none=False), 0.3), (dict(harm_placed=False), float("nan")),
                                   (dict(harm_placed=False), 0.0), (dict(none=True), 0.3)):
                match, evaluated = A.room_match(None, None, None, None, [], target, failed, None)
                self.assertIsNone(match["multiplier"])
                self.assertTrue(match["skipped"])
                self.assertEqual(evaluated, {})
        with mock.patch.object(L, "match_even", return_value=(dict(multiplier=1.3), {})) as called:
            self.assertEqual(A.room_match(None, None, None, None, [], 0.3, dict(harm_placed=False), None)[0]
                             ["multiplier"], 1.3)
            called.assert_called_once()
        # the pooled room check then fails the seed's marks instead of crashing or staying pending
        pr = prereg("room_failed_seed")
        mark(A._variant_root(pr, "harm_critic", "base") / "pooled.json", pr, True)
        mark(A._variant_root(pr, "harm_return", "base") / "alpha2.5" / "pooled.json", pr, True)
        keys = ("harm_placed_minus_matched_even", "harm_placed_minus_amount_matched", "harm_placed_minus_shuffled",
                "harm_placed_minus_none", "harm_placed_minus_best_even", "cap2_minus_matched_even",
                "cap2_minus_best_even", "reversed_minus_harm_placed")
        for i, s in enumerate(CFG.gate_seeds):
            sit = {name: dict({k: 10.0 + i for k in keys}, best_even="even_x1.25") for name in A.SITUATIONS}
            if i == 0:
                for name in A.SITUATIONS:
                    sit[name]["harm_placed_minus_matched_even"] = None
            A.write_json(A._variant_root(pr, "room", "base") / "alpha2.5" / f"seed{s}" / "values.json",
                         dict(prereg_sha256=pr.digest, situations=sit))
        room = A.h3_pooled(pr, "base", log=quiet)
        self.assertFalse(room["passed"])
        self.assertEqual(room["certified"], [])
        self.assertEqual(A.gate_outcome(pr)["reason"], "no_room")

    def test_failed_placement_control_counts_against_bca(self):
        pr = prereg("points")
        seeds = [100, 101, 102]
        arms = ("plain", "bca", "flattened", "shuffled", "noise_map", "even_x1.25", "even_x2", "matched_even",
                "cloning", "full_plain", "full_bca")
        ev = {s: dict(points=30.0, loss_from_unrecorded_throttle=4.0) for s in A.SITUATIONS}
        for i, rep in enumerate(seeds):
            scores = {a: {"100": ev, "50": ev} for a in arms}
            if i == 0:
                scores["flattened"] = {"100": None, "50": None}      # a placement control failed (after its rerun)
                scores["noise_map"] = {"100": None, "50": None}
            if i == 1:
                scores["bca"] = {"100": None, "50": None}            # BCA itself failed
            A.write_json(A._bca_dir(pr, "base", 2.5, rep) / "scores.json", dict(prereg_sha256=pr.digest, scores=scores))
        table = A._points_table(pr, "base", 2.5, seeds)
        sit = "strong_current"
        self.assertEqual(A._differences(table, "bca", "flattened", sit), [0.0, -30.0, 0.0])
        self.assertEqual(A._differences(table, "bca", "noise_map", sit), [0.0, -30.0, 0.0])
        self.assertEqual(A._differences(table, "bca", "matched_even", sit), [0.0, -30.0, 0.0])
        self.assertEqual(A._differences(table, "bca", "plain", sit), [0.0, -30.0, 0.0])
        self.assertEqual(set(A.BCA_OWN), {"bca", "full_bca", "no_stall_bca"})
        for arm in A.PLACEMENT_CONTROLS:
            self.assertIn(arm, A.BCA_FAMILY)
            self.assertNotIn(arm, A.BCA_OWN)

    def test_placement_control_is_rerun_once_and_bca_never(self):
        pr = prereg("rerun")
        data = {"stall": A.load_seed(pr.runs, CFG, "base", "stall", 100)}
        calls = []

        def fake(prep, learn, rep, approval, placement="own", **kw):   # stands in for BCA: nothing BCA runs
            calls.append(placement)
            return dict(failed=True), None, {}

        with mock.patch.object(L, "require_approval", lambda a: a), mock.patch.object(L, "bca_cotrain", fake):
            for arm in ("flattened", "shuffled", "noise_map", "bca"):
                rec = A.run_bca_arm(pr, None, "base", 2.5, 100, arm, data, log=quiet)
                self.assertTrue(rec["failed"])
                self.assertEqual(len(rec["attempts"]), 1 if arm == "bca" else 2, arm)
        self.assertEqual(calls, ["flattened"] * 2 + ["shuffled"] * 2 + ["noise_map"] * 2 + ["own"])

    def test_arms_must_share_data_and_batch_stream(self):
        good = dict(initial_sha256="i", data=dict(sha256="g"), batch_stream_sha256="b")
        recs = {a: dict(result=dict(good)) for a in ("plain", "bca", "flattened", "shuffled", "noise_map",
                                                     "even_x1.25", "even_x2")}
        recs["cloning"] = dict(result=dict(data=dict(sha256="g")))
        recs["full_plain"] = dict(result=dict(good, data=dict(sha256="f")))
        recs["full_bca"] = dict(result=dict(good, data=dict(sha256="f")))
        self.assertEqual(A.shared_inputs(recs.get)["problems"], [])
        for arm, field, value in (("shuffled", "data", dict(sha256="other")), ("noise_map", "batch_stream_sha256", "x"),
                                  ("cloning", "data", dict(sha256="other")), ("full_bca", "batch_stream_sha256", "y"),
                                  ("even_x2", "initial_sha256", "j")):
            bad = dict(recs)
            bad[arm] = dict(result=dict(recs[arm]["result"], **{field: value}))
            self.assertTrue(A.shared_inputs(bad.get)["problems"], (arm, field))


class StopRuleTest(unittest.TestCase):
    def test_h1_known_answers(self):
        pr = prereg("h1_pass")
        synth_h1(pr, "base", CFG.gate_seeds)
        rec = A.h1_pooled(pr, "base", log=quiet)
        self.assertTrue(rec["passed"], rec["marks"])
        self.assertGreater(rec["gap_error_ratio"]["pooled"], 2.0)
        pr2 = prereg("h1_exact_critic")
        synth_h1(pr2, "base", CFG.gate_seeds, gap_factor=1.0, pull_gap=0.0)
        rec2 = A.h1_pooled(pr2, "base", log=quiet)
        self.assertFalse(rec2["marks"]["a"])
        self.assertFalse(rec2["marks"]["f"])
        self.assertTrue(rec2["marks"]["b"] and rec2["marks"]["c"] and rec2["marks"]["d"])
        pr3 = prereg("h1_no_stall_too")
        synth_h1(pr3, "base", CFG.gate_seeds, no_stall_factor=3.0)
        rec3 = A.h1_pooled(pr3, "base", log=quiet)
        self.assertTrue(rec3["marks"]["a"])
        self.assertFalse(rec3["marks"]["e"])

    def test_alpha_ladder_and_outcomes(self):
        pr = prereg("ladder")
        root = lambda stage, v: A._variant_root(pr, stage, v)  # noqa: E731
        with self.assertRaises(A.GateError):
            A.require_alpha_allowed(pr, "base", 5.0)
        with self.assertRaises(A.GateError):
            A.require_alpha_allowed(pr, "base", 3.0)
        with self.assertRaises(A.GateError):
            A.variant_allowed(pr, "retune")
        mark(root("harm_critic", "base") / "pooled.json", pr, True)
        self.assertEqual(A.harm_state(pr, "base")["next_alpha"], 2.5)
        mark(root("harm_return", "base") / "alpha2.5" / "pooled.json", pr, False)
        st = A.harm_state(pr, "base")
        self.assertEqual(st["next_alpha"], 5.0)
        A.require_alpha_allowed(pr, "base", 5.0)
        with self.assertRaises(A.GateError):
            A.require_alpha_allowed(pr, "base", 10.0)
        mark(root("harm_return", "base") / "alpha5" / "pooled.json", pr, True)
        self.assertEqual(A.chosen_alpha(pr, "base"), 5.0)
        self.assertEqual(A.gate_outcome(pr)["state"], "pending")
        mark(root("room", "base") / "alpha5" / "pooled.json", pr, False, certified=[])
        o = A.gate_outcome(pr)
        self.assertEqual((o["state"], o["reason"]), ("stop", "no_room"))
        mark(root("room", "base") / "alpha5" / "pooled.json", pr, True, certified=["strong_current"])
        mark(root("power", "base") / "alpha5" / "pooled.json", pr, False, N=40, smallest_effect_at_20_seeds=4.2)
        self.assertEqual(A.gate_outcome(pr)["reason"], "underpowered")
        mark(root("power", "base") / "alpha5" / "pooled.json", pr, True, N=12)
        o = A.gate_outcome(pr)
        self.assertEqual((o["state"], o["N"], o["alpha"], o["certified"]), ("ready", 12, 5.0, ["strong_current"]))

    def test_retune_and_no_harm(self):
        pr = prereg("retune")
        root = lambda stage, v: A._variant_root(pr, stage, v)  # noqa: E731
        mark(root("harm_critic", "base") / "pooled.json", pr, False)
        self.assertTrue(A.harm_state(pr, "base")["failed"])
        with self.assertRaises(A.GateError):
            A.require_alpha_allowed(pr, "base", 5.0)
        self.assertEqual(A.active_variant(pr), "retune")
        self.assertTrue(A.variant_allowed(pr, "retune"))
        self.assertEqual(A.gate_outcome(pr)["state"], "pending")
        mark(root("harm_critic", "retune") / "pooled.json", pr, True)
        for a in (2.5, 5.0, 10.0):
            mark(root("harm_return", "retune") / f"alpha{a:g}" / "pooled.json", pr, False)
        o = A.gate_outcome(pr)
        self.assertEqual((o["state"], o["reason"]), ("stop", "no_harm"))
        self.assertIn("does not produce critic-caused harm", A.STOP_MESSAGES[o["reason"]])

    def test_h2_known_answers(self):
        pr = prereg("h2")
        seeds = CFG.gate_seeds
        for i, s in enumerate(seeds):
            sit = {}
            for name in A.SITUATIONS:
                p = dict(full=60.0 + i, gapped=48.0 - i, random=58.0, no_stall_full=70.0, no_stall_gapped=69.0)
                sit[name] = dict(harm=p["full"] - p["gapped"], specific=p["random"] - p["gapped"],
                                 harm_minus_no_stall_harm=(p["full"] - p["gapped"]) - 1.0, host_swap_harm=None,
                                 td3bc_minus_cloning=None, no_stall_harm=1.0, random_harm=2.0,
                                 unrecorded_throttle_share=dict(gapped=0.3 + 0.01 * i, full=0.1),
                                 attribution=dict(sum_loss_past_80=30.0, steps_past_80=10, sum_loss_other=10.0,
                                                  steps_other=90))
            A.write_json(A._variant_root(pr, "harm_return", "base") / "alpha2.5" / f"seed{s}" / "values.json",
                         dict(prereg_sha256=pr.digest, situations=sit))
        rec = A.h2_pooled(pr, "base", 2.5, log=quiet)
        self.assertTrue(rec["passed"], rec["marks"])
        sc = rec["situations"]["strong_current"]
        self.assertAlmostEqual(sc["harm"]["mean"], 13.0)
        self.assertAlmostEqual(sc["loss_rate"]["loss_rate_past_80"], 3.0)
        p = A._variant_root(pr, "harm_return", "base") / "alpha2.5" / f"seed{seeds[0]}" / "values.json"
        body = A.read_json(p)
        body["situations"]["strong_current"]["harm"] = None
        A.write_json(p, body)
        self.assertFalse(A.h2_pooled(pr, "base", 2.5, log=quiet)["marks"]["a"])


class PipelineTest(unittest.TestCase):
    """The smoke pipeline from the pre-registration to the gate report and the refusals that follow it."""

    def test_pipeline(self):
        pr = prereg("pipeline")
        for s in CFG.gate_seeds:
            rec = A.h1_seed(pr, "base", s, log=quiet)
            self.assertEqual(len(rec["critics"]), len(A.H1_CRITICS))
            self.assertFalse(any(c["failed"] for c in rec["critics"].values()))
        h1 = A.h1_pooled(pr, "base", log=quiet)
        self.assertIn("a", h1["marks"])
        self.assertEqual(h1["gap_error_ratio"]["replicates"], CFG.bootstrap_replicates)
        for s in CFG.gate_seeds:
            rec = A.h2_seed(pr, "base", 2.5, s, log=quiet)
            self.assertEqual(rec["situations"]["strong_current"]["attribution"]["starts"], CFG.attribution_starts)
            self.assertLessEqual(rec["situations"]["strong_current"]["attribution"]["max_sum_error"], 1e-8)
        A.h2_pooled(pr, "base", 2.5, log=quiet)
        self.assertIsNotNone(A.read_json(A.critic_dir(pr, "base", "stall", "gap", "stall_aware", False, 0)
                                         / "record.json"))
        h2s = A.read_json(A._variant_root(pr, "harm_return", "base") / "alpha2.5" / "seed0" / "values.json")
        self.assertIsNotNone(h2s["host_critic_errors"]["gapped"]["removed"])
        # reach the later stages: the test marks the pooled H1 and H2 records passed
        for path in (A._variant_root(pr, "harm_critic", "base") / "pooled.json",
                     A._variant_root(pr, "harm_return", "base") / "alpha2.5" / "pooled.json"):
            body = A.read_json(path)
            body.update(passed=True, marked_passed_by_test=True)
            A.write_json(path, body)
        self.assertEqual(A.chosen_alpha(pr, "base"), 2.5)
        checked = []
        real_eval = L.evaluate_policy

        def eval_after_lock(*a, **k):
            seed = STATE["room_seed"]
            lock = A._variant_root(pr, "room", "base") / "alpha2.5" / f"seed{seed}" / "room_blind.json"
            self.assertTrue(lock.exists(), "a return was computed before the lock")
            checked.append(seed)
            return real_eval(*a, **k)

        with mock.patch.object(L, "evaluate_policy", side_effect=eval_after_lock):
            for s in CFG.gate_seeds:
                STATE["room_seed"] = s
                rec = A.h3_seed(pr, "base", s, log=quiet)
                self.assertIsNotNone(rec["situations"]["strong_current"]["harm_placed_minus_matched_even"])
        self.assertEqual(sorted(set(checked)), sorted(CFG.gate_seeds))
        blind = A.read_json(A._variant_root(pr, "room", "base") / "alpha2.5" / "seed0" / "room_blind.json")
        self.assertIsNotNone(blind["match"]["multiplier"])
        room = A.h3_pooled(pr, "base", log=quiet)
        if not room["passed"]:
            room.update(passed=True, certified=["strong_current"], marked_passed_by_test=True)
            A.write_json(A._variant_root(pr, "room", "base") / "alpha2.5" / "pooled.json", room)
        for s in CFG.gate_seeds:
            A.h4_seed(pr, "base", s, log=quiet)
        power = A.h4_pooled(pr, "base", log=quiet)
        self.assertEqual(power["N"], A.required_seeds(power["s"]))
        self.assertNotEqual(A.gate_outcome(pr)["state"], "pending")
        with self.assertRaises(A.GateError) as ctx:
            A.stage_report(pr, log=quiet)
        self.assertIn("shift cards", str(ctx.exception))
        stub = lambda pol, s, *a, **k: dict(situation=s, policy=getattr(pol, "name", "?"), test_stub=True)  # noqa
        with mock.patch.object(SH, "shift_card", side_effect=stub):
            for s in CFG.gate_seeds:
                cards = A.report_cards(pr, s, log=quiet)
                self.assertIn("actor of the gapped critic (H2)", cards["cards"])
        rep = A.stage_report(pr, log=quiet)
        md = (pr.root / "report" / "report.md").read_text()
        for banned in (" twin", "operator", "trap", "habit", "lure", "button"):
            self.assertNotIn(banned, md.lower())
        if not rep["bca_allowed"]:
            with self.assertRaises(A.GateError):
                A.stage_approve(pr, input_fn=lambda q: "x", stdin=io.StringIO(""))
            body = A.read_json(pr.root / "report" / "report.json")
            body["bca_allowed"] = True
            A.write_json(pr.root / "report" / "report.json", body)
        with self.assertRaises(A.GateError) as ctx:
            A.stage_approve(pr, input_fn=lambda q: "x", stdin=io.StringIO(""))
        self.assertIn("terminal", str(ctx.exception))
        self.assertFalse(A.approval_path(pr).exists())
        with self.assertRaises(L.ApprovalError):
            A.load_approval(pr)
        forged = dict(kind="bca-shift-approval", report_path=str(pr.root / "report" / "report.json"),
                      report_sha256="f" * 64, prereg_sha256=pr.digest)
        A.write_json(A.approval_path(pr), forged)
        with self.assertRaises(L.ApprovalError):
            A.load_approval(pr)
        for fn in (lambda: A.stage_bca(pr, None, 100), lambda: A.detect_seed(pr, None, 100),
                   lambda: A.stage_policy(pr, None), lambda: A.stage_verdict(pr, None),
                   lambda: A.bca_scores(pr, None, 100), lambda: A.detect_pooled(pr, None)):
            with self.assertRaises(L.ApprovalError):
                fn()
        self.assertFalse((pr.root / "bca").exists())


class StatisticsTest(unittest.TestCase):
    def test_bounds_and_holm(self):
        self.assertAlmostEqual(DT.t_lower([1.0, 2.0, 3.0]), 2.0 - 2.919986 / np.sqrt(3), places=5)
        self.assertAlmostEqual(DT.t_upper([1.0, 2.0, 3.0]), 2.0 + 2.919986 / np.sqrt(3), places=5)
        self.assertAlmostEqual(DT.half_width([1.0, 2.0, 3.0]), 4.302653 / np.sqrt(3), places=5)
        self.assertIsNone(DT.t_lower([1.0]))
        adj = DT.holm(dict(a=0.01, b=0.04, c=0.03))
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["c"], 0.06)
        self.assertAlmostEqual(adj["b"], 0.06)
        self.assertEqual(DT.need(0.8, 5), 4)
        self.assertEqual(A.required_seeds(0.0), 10)
        self.assertEqual(A.required_seeds(5.0), 39)

    def test_detection_known_answers(self):
        rng = np.random.default_rng(0)
        n = 2000
        D = rng.exponential(size=n)
        tie = rng.permutation(n)
        dose = L.harm_doses(D)
        self.assertAlmostEqual(DT.auc(dose, DT.top_mask(D, 0.1, tie)), 1.0)
        self.assertAlmostEqual(DT.placement_efficiency(dose, D), 1.0)
        self.assertAlmostEqual(DT.placement_efficiency(DT.place_by(dose, -D, tie), D), -1.0)
        self.assertEqual(DT.placement_efficiency(np.full(n, 1.2), D), 0.0)
        shuffled = dose[rng.permutation(n)]
        self.assertLess(abs(DT.auc(shuffled, DT.top_mask(D, 0.1, tie)) - 0.5), 0.07)
        self.assertLess(abs(DT.placement_efficiency(shuffled, D)), 0.1)
        nm = rng.uniform(size=n)
        region = rng.uniform(size=n) < 0.3
        self.assertEqual(DT.partial_spearman(nm, D, [nm, region]), 0.0)
        self.assertGreater(DT.partial_spearman(dose, D, [nm, region]), 0.9)
        self.assertAlmostEqual(DT.concentration(np.where(region, 1.4, 1.0), region), 1.4)
        comp = DT.comparator_scores(dose, D, dict(itself=D, noise=nm), tie)
        self.assertAlmostEqual(comp["itself"]["placement_efficiency"], 1.0)
        rows = DT.state_map(nm, dict(D=D), tie, groups=10)
        self.assertEqual(sum(r["rows"] for r in rows), n)
        self.assertTrue(all(rows[i]["removed_share_mean"] <= rows[i + 1]["removed_share_mean"] for i in range(9)))

    def test_detection_family_and_pooled_ratio(self):
        marks = A.PASS_MARKS["B1"]
        good = [dict(d1_auc=0.8 + 0.01 * i, concentration_ratio=1.2 + 0.01 * i, d3_partial_spearman=0.3 + 0.01 * i)
                for i in range(10)]
        self.assertTrue(DT.detection_family(good, marks)["passed"])
        bad = [dict(g, d3_partial_spearman=0.0 + 0.01 * (i % 3 - 1)) for i, g in enumerate(good)]
        fam = DT.detection_family(bad, marks)
        self.assertFalse(fam["D3"]["passed"])
        self.assertFalse(fam["passed"])
        failed_seed = good[:-1] + [dict(d1_auc=None, concentration_ratio=None, d3_partial_spearman=None)]
        self.assertEqual(DT.detection_family(failed_seed, marks)["D1"]["values"][-1], 0.5)
        num = {0: np.array([2.0, 4.0, 6.0]), 1: np.array([3.0, 3.0, 3.0])}
        den = {0: np.array([1.0, 2.0, 3.0]), 1: np.array([1.0, 1.0, 1.0])}
        blocks = {0: np.array([0, 1, 2]), 1: np.array([0, 1, 2])}
        r = DT.pooled_ratio(num, den, blocks, 50, 0)
        self.assertEqual(r["per_seed"], {"0": 2.0, "1": 3.0})
        self.assertAlmostEqual(r["pooled"], np.sqrt(6.0))

    def test_verdict_wording(self):
        ok = dict(detection_passed=True, level1=dict(strong_current=True), level2=dict(strong_current=True), p2=True,
                  p3=True, p4=True, harm_present=True, half_width_ok=True, cost=False, noise_rider=False)
        self.assertEqual(A.decide(**ok), ("V1", A.VERDICTS["V1"]))
        code, text = A.decide(**dict(ok, noise_rider=True))
        self.assertEqual(code, "V1")
        self.assertIn("target noise alone", text)
        self.assertEqual(A.decide(**dict(ok, level2=dict(strong_current=False)))[0], "V2")
        self.assertEqual(A.decide(**dict(ok, level1=dict(strong_current=False), level2={}))[0], "V3")
        self.assertEqual(A.decide(**dict(ok, detection_passed=False, level1=dict(strong_current=False),
                                         level2={}))[0], "V4")
        self.assertEqual(A.decide(**dict(ok, p2=False))[0], "V5")
        self.assertEqual(A.decide(**dict(ok, cost=True))[0], "V5")
        self.assertEqual(A.decide(**dict(ok, harm_present=False))[0], "V6")
        self.assertEqual(A.decide(**dict(ok, half_width_ok=False))[0], "V6")
        # levels 1 and 2 pass but detection or P3 fails: never 'no detectable benefit'
        self.assertEqual(A.decide(**dict(ok, detection_passed=False)), ("V7", A.VERDICTS["V7"]))
        code, text = A.decide(**dict(ok, detection_passed=False, p3=False))
        self.assertEqual(code, "V7")
        self.assertIn(A.VERDICTS["V7_recording_rider"], text)
        self.assertEqual(A.decide(**dict(ok, p3=False)), ("V8", A.VERDICTS["V8"]))
        self.assertIn("target noise alone", A.decide(**dict(ok, p3=False, noise_rider=True))[1])
        self.assertEqual(A.decide(**dict(ok, p3=False, level2=dict(strong_current=False)))[0], "V2")
        for code in ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"):
            self.assertIn(f"({code})", A.VERDICT_ORDER)
        for text in A.VERDICTS.values():
            for banned in ("twin", "operator", "trap", "habit", "lure", "button"):
                self.assertNotIn(banned, text.lower())


class PlainWordsTest(unittest.TestCase):
    def test_no_nicknames_in_reader_visible_files(self):
        banned = re.compile(r"\b(twins?|operators?|traps?|habits?|lures?|buttons?|false preference|linear belief)\b")
        for name in ("learners.py", "detection.py", "acceptance.py", "SHIFT_ENV.md"):
            text = (ROOT / "experiments" / "shift" / name).read_text().lower()
            self.assertIsNone(banned.search(text), f"{name}: {banned.search(text)}")
        texts = list(A.MARK_TEXT.values()) + list(A.STOP_MESSAGES.values()) + list(A.STOP_RULE.values()) + [
            A.ALPHA_RULE, A.VERDICT_ORDER, L.BCA_REFUSAL, A.PASS_MARKS["S1"]["fallback_grid"]] + [
            v for d in A.MARK_LABELS.values() for v in d.values()] + list(A.VERDICTS.values()) + list(
            S0.COMPUTED.values()) + list(S0.STAND_INS.values())
        for text in texts:
            self.assertNotIn("twin", text.lower())
            self.assertIsNone(re.search(r"\b(press(es)?|hand|amendment a1|holm family)\b", text.lower()), text)
        keys = list(S0.MARKS) + list(RC.RecorderSpec().as_dict())
        for key in keys:
            self.assertIsNone(re.search(r"press|hand|drafts", key), key)
        # the CLI names marks in words, not letter codes
        pr = prereg("h1_words")
        synth_h1(pr, "base", CFG.gate_seeds)
        lines = []
        A.h1_pooled(pr, "base", log=lines.append)
        self.assertIn(A.MARK_LABELS["H1"]["a"], lines[0])
        self.assertNotIn("a=P", lines[0])


class BuildChecksTest(unittest.TestCase):
    def test_learner_checks(self):
        out = A.learner_checks(updates=200, log=quiet)
        self.assertTrue(out["passed"], out)


class ZZEarlierGateUntouchedTest(unittest.TestCase):
    def test_earlier_gate_and_environment_unchanged(self):
        self.assertEqual(STATE["before"]["earlier"], A.earlier_gate_state())
        self.assertEqual(STATE["before"]["env"], A.environment_state())
        self.assertFalse((ROOT / "runs" / "shift").exists())


if __name__ == "__main__":
    unittest.main()
