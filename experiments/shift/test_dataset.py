"""CPU tests of the recorded log and the recording controller (no learning; every number is a test fixture, not a
finding). Small logs (40 + 10 + 10 episodes) keep it fast; the rules are the same as for the 1,000-episode logs.

Checks: regeneration gives identical arrays and a byte-identical HDF5 file; every row replays through
reset(options={'state'}) + step() bit for bit; D4RL keys and dtypes; timeouts at t = 99 only and terminals all False;
next_observations[i] equals observations[i + 1] inside an episode; splits by episode; the calibration bank (5 rows of
every calibration episode); a tampered HDF5 array is refused; the full log file says it holds every split and the D4RL
training files hold only the six D4RL arrays of the training rows a removal keeps; the recording controller's mode flag is guarded and the
removal rule, pools and shift card never read it; the vectorised simulation equals the env.step path bit for bit;
the recording controller: normal-driving commands never reach 80%, full-throttle commands are exactly
clip(P + 0.03 n) and within 5 SD of the pattern, the full-throttle rate in the region is 0.5 +- 3 SE, and the trigger
is exogenous (both worlds give the full-throttle command at identical rows).
JAX_PLATFORMS=cpu python -m unittest experiments.shift.test_dataset
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shift import dataset as DS  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402

SMALL = (40, 10, 10)
LOGS = {}


def small_log(rep=900, world="stall"):
    key = (rep, world)
    if key not in LOGS:
        LOGS[key] = DS.generate(rep, world, sizes=SMALL)
    return LOGS[key]


class LogTest(unittest.TestCase):
    def test_regeneration_identical_and_hdf5_bytes(self):
        a, b = small_log(), DS.generate(900, "stall", sizes=SMALL)
        self.assertEqual(a.array_hashes(), b.array_hashes())
        self.assertEqual(a.meta, b.meta)
        tmp = tempfile.mkdtemp()
        try:
            m1 = DS.write_hdf5(a, Path(tmp) / "one.hdf5")
            m2 = DS.write_hdf5(b, Path(tmp) / "two.hdf5")
            self.assertEqual(m1["file_sha256"], m2["file_sha256"])
            self.assertEqual(m1["arrays"], m2["arrays"])
            back = DS.read_hdf5(Path(tmp) / "one.hdf5")
            self.assertEqual(back.array_hashes(), a.array_hashes())
            self.assertEqual(back.meta, a.meta)
            import h5py

            with h5py.File(Path(tmp) / "two.hdf5", "r+") as f:
                f["rewards"][0] = f["rewards"][0] + 1.0
            with self.assertRaises(ValueError):
                DS.read_hdf5(Path(tmp) / "two.hdf5")
        finally:
            shutil.rmtree(tmp)

    def test_training_files_hold_only_the_kept_training_rows(self):
        """The full log file holds every split (and says so); a standard D4RL loader must instead get a training
        file: only the six D4RL arrays of the training rows a removal keeps (the gapped set for the test)."""
        import h5py

        log = small_log()
        gap, _ = SH.define_gap(log)
        tmp = tempfile.mkdtemp()
        try:
            DS.write_hdf5(log, Path(tmp) / "full.hdf5")
            with h5py.File(Path(tmp) / "full.hdf5", "r") as f:
                self.assertIn("infos/split", f.attrs["about"])
                self.assertIn("not a training set", f.attrs["about"])
            for kind in ("none", "gap", "random"):
                removal = SH.make_removal(log, kind, gap)
                rows = removal.kept_train_rows(log)
                m = DS.write_d4rl_training(log, removal, Path(tmp) / f"train_{kind}.hdf5")
                self.assertEqual(m["rows"], len(rows))
                with h5py.File(Path(tmp) / f"train_{kind}.hdf5", "r") as f:
                    self.assertEqual(sorted(f.keys()), sorted(DS.D4RL_KEYS))
                    for k in DS.D4RL_KEYS:
                        np.testing.assert_array_equal(f[k][()], log[k][rows])
                    self.assertIn("next_observations", f.attrs["about"])
                self.assertTrue(np.all(log["infos/split"][rows] == 0))
            self.assertEqual(SH.leak_count(SH.make_removal(log, "gap", gap).kept_train_rows(log), log, gap), 0)
            self.assertLess(DS.write_d4rl_training(log, SH.make_removal(log, "gap", gap),
                                                   Path(tmp) / "g.hdf5")["rows"], len(log.split_rows("train")))
        finally:
            shutil.rmtree(tmp)

    def test_every_row_replays_bitwise(self):
        for world in ("stall", "no_stall"):
            log = small_log(901, world)
            result = DS.replay(log)
            self.assertEqual(result["rows"], log.n_rows)
            self.assertEqual(result["mismatches"], 0, result["first"])
        log = small_log()
        forged = dict(log._arrays)
        forged["rewards"] = forged["rewards"].copy()
        forged["rewards"][5] += 1e-3
        self.assertEqual(DS.replay(DS.Log(forged, log.meta), rows=np.arange(10))["mismatches"], 1)

    def test_d4rl_fields(self):
        log = small_log()
        n = log.n_rows
        self.assertEqual(n, sum(SMALL) * 100)
        dtypes = dict(observations=np.float32, actions=np.float32, rewards=np.float32, next_observations=np.float32,
                      terminals=np.bool_, timeouts=np.bool_)
        for k, dt in dtypes.items():
            self.assertEqual(log[k].dtype, dt, k)
            self.assertEqual(len(log[k]), n)
        self.assertEqual(log["observations"].shape, (n, 13))
        self.assertEqual(log["actions"].shape, (n, 4))
        self.assertEqual(log["infos/state"].dtype, np.float64)
        self.assertFalse(log["terminals"].any())
        self.assertTrue(np.array_equal(log["timeouts"], log["infos/t"] == 99))
        self.assertTrue(np.array_equal(log["observations"], log["infos/state"].astype(np.float32)))
        self.assertTrue(np.array_equal(log["next_observations"], log["infos/next_state"].astype(np.float32)))
        same_episode = log["infos/episode"][1:] == log["infos/episode"][:-1]
        nxt, now = log["next_observations"][:-1], log["observations"][1:]
        self.assertTrue(np.array_equal(nxt[same_episode], now[same_episode]))
        nxt, now = log["infos/next_state"][:-1], log["infos/state"][1:]
        self.assertTrue(np.array_equal(nxt[same_episode], now[same_episode]))
        self.assertTrue(np.all(np.abs(log["actions"]) <= 1.0))
        d4 = DS.d4rl_dict(log, np.arange(0, n, 3))
        self.assertEqual(sorted(d4), sorted(DS.D4RL_KEYS))
        self.assertTrue(np.array_equal(d4["next_observations"], log["next_observations"][::3]))

    def test_splits_by_episode_and_bank(self):
        log = small_log()
        ep = log["infos/episode"]
        sets = {s: set(np.unique(ep[log.split_rows(s)]).tolist()) for s in DS.SPLITS}
        self.assertEqual(sets["train"], set(range(0, 40)))
        self.assertEqual(sets["calibration"], set(range(40, 50)))
        self.assertEqual(sets["evaluation"], set(range(50, 60)))
        bank = log.bank_rows
        self.assertEqual(len(bank), 10 * DS.BANK_ROWS_PER_EPISODE)
        self.assertTrue(np.all(log["infos/split"][bank] == 1))
        self.assertTrue(np.array_equal(np.bincount(ep[bank] - 40), np.full(10, 5)))
        self.assertEqual(len(np.unique(bank)), len(bank))

    def test_simulate_equals_env_path(self):
        log = small_log()
        E = sum(SMALL)
        starts, noise, coin = DS.draw_streams(900, E)
        sim = DS.simulate(RC.Recorder(), TC.STALL, starts, noise, coin)
        self.assertTrue(np.array_equal(sim["states"][:, :-1].reshape(-1, 13), log["infos/state"]))
        self.assertTrue(np.array_equal(sim["commands"].reshape(-1, 4), log["actions"]))
        self.assertTrue(np.array_equal(sim["full_throttle"].reshape(-1), log.diagnostic("infos/full_throttle_command")))
        self.assertTrue(np.array_equal(sim["rewards"].reshape(-1).astype(np.float32), log["rewards"]))


class GuardTest(unittest.TestCase):
    def test_mode_flag_is_guarded_and_never_read_by_removal_pools_or_card(self):
        log = DS.generate(902, "stall", sizes=(60, 10, 10))
        with self.assertRaises(KeyError):
            log["infos/full_throttle_command"]
        self.assertNotIn("infos/full_throttle_command", list(log))
        self.assertEqual(log.diagnostic_reads, 0)
        gap, record = SH.define_gap(log)
        self.assertIsNotNone(gap, record.get("reason"))
        rec = RC.Recorder()
        removals = {k: SH.make_removal(log, k, gap) for k in SH.REMOVALS}
        pools = SH.make_pools(log, gap, rec, size=50, ms_draws=256)
        ref = SH.card_reference(log, removals["gap"], rec, heldout_rows=200)
        SH.card_measures(log["infos/state"][:500], log["actions"][:500], ref, TC.STALL, ms_states=50, knn_states=100)
        SH.removed_share(log["infos/state"][:20], removals["gap"], rec, draws=64)
        self.assertEqual(log.diagnostic_reads, 0)
        self.assertEqual(len(pools["removed"]), 50)
        log.diagnostic("infos/full_throttle_command")
        self.assertEqual(log.diagnostic_reads, 1)


class RecorderTest(unittest.TestCase):
    def test_textbook_gain_and_pattern(self):
        K = RC.textbook_gain()
        np.testing.assert_allclose(K[:, TC.CURRENT], [0.910, 0.125, 0.532, 0.370], atol=5e-4)
        P = RC.full_throttle_pattern(K)
        np.testing.assert_allclose(P, [-1.0, -0.137, -0.585, -0.406], atol=5e-4)
        self.assertTrue(np.array_equal(RC.textbook_gain(TC.NO_STALL), K))

    def test_recording_controller_modes(self):
        log = small_log()
        a = np.asarray(log["actions"], np.float64)
        press = log.diagnostic("infos/full_throttle_command")
        x = log["infos/state"]
        self.assertTrue(np.all(np.abs(a[~press]) < 0.8))
        self.assertTrue(np.all(TC.past_knee(a[press])))
        rec = RC.Recorder()
        _, noise, coin = DS.draw_streams(900, sum(SMALL))
        n = noise.reshape(-1, 4)
        exact = np.clip(rec.pattern + 0.03 * n[press], -1, 1).astype(np.float32)
        self.assertTrue(np.array_equal(exact, log["actions"][press]))
        self.assertLess(np.abs(a[press] - rec.pattern).max(), 5 * 0.03)
        region = rec.in_region(x)
        self.assertTrue(np.array_equal(press, region & (coin.reshape(-1) < 0.5)))
        rate = press[region].mean()
        se = np.sqrt(0.25 / region.sum())
        self.assertLess(abs(rate - 0.5), 3 * se)

    def test_trigger_is_exogenous(self):
        stall, no_stall = small_log(903, "stall"), small_log(903, "no_stall")
        p1, p2 = stall.diagnostic("infos/full_throttle_command"), no_stall.diagnostic("infos/full_throttle_command")
        self.assertTrue(np.array_equal(p1, p2))
        x1, x2 = stall["infos/state"], no_stall["infos/state"]
        self.assertTrue(np.array_equal(TC.push(x1), TC.push(x2)))
        first = stall["infos/t"] == 0
        self.assertTrue(np.array_equal(x1[first], x2[first]))
        self.assertFalse(np.array_equal(x1, x2))

    def test_reference_controllers(self):
        x = TC.bank_starts("strong_current", 300)
        aw, lin, hand = RC.stall_aware(), RC.rising_thrust(), RC.hand_only()
        self.assertLessEqual(np.abs(aw(x)).max(), 0.8)
        self.assertGreater(np.abs(lin(x)).max(), 0.8)
        self.assertLess(np.abs(hand(x)).max(), 0.8)
        self.assertTrue(np.array_equal(np.stack([aw(x[i]) for i in range(len(x))]), aw(x)))
        self.assertTrue(np.array_equal(np.clip(lin(x), -0.8, 0.8), aw(x)))
        streamed = RC.StreamedRecorder.on_bank(RC.Recorder(), n=300)
        with self.assertRaises(TypeError):
            streamed(x)
        self.assertEqual(streamed.act(x, 0).shape, (300, 4))

    def test_cli_refuses_protected_directories(self):
        for bad in (ROOT / "runs" / "glq" / "x", ROOT / "runs" / "shift", ROOT / "experiments" / "shift" / "out"):
            with self.assertRaises(SystemExit):
                DS.refuse_protected(bad)
        self.assertEqual(DS.refuse_protected(tempfile.gettempdir()), Path(tempfile.gettempdir()).resolve())


if __name__ == "__main__":
    unittest.main()
