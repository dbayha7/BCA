"""CPU tests of the action-gap protocol (gap.py, gap_train.py, gap_acceptance.py) on synthetic D4RL-format data.

Every number here is a test fixture on synthetic data or fabricated arrays, NOT A FINDING. Checks:
  - the rule: a planted dense ball is recovered next to a corner-saturated duplicate block (mass in band, c* within
    0.05, deterministic); the tie-block fallback, the band failure and the point-mass guard; the inclusive ball;
  - the data (one synthetic Hopper file with a planted action cluster): the gap holds 15% +- 1% of the training rows
    and is defined identically twice; G0a's integrity checks all pass (the gapped conversion is native minus gap
    array for array, per-key hashes, the projected reservation passes runtime validate_prepared); no gap row reaches
    the gapped host's training rows, withheld rows or calibration bank, while a leaked row is detected (negative
    control: a copy with one gap row's timeout flag cleared is refused by the projected reservation and counted by
    the leak detector); the matched kept pool is matched (other episode, kept, inside the caliper, no reuse) and
    K_far's neighbourhoods hold no gap row; freeze refuses a failed G0 without the synthetic override;
  - gap_train: --prepare-only through the real train.py on a synthetic file (test-only reservation override that
    refuses real data); --method bca refuses with the band wording and no BCA flag exists; a wrong gap hash refuses;
    restoring a gapped run refuses without its manifest;
  - known-answer scorers on fabricated labels: G0b, G1 (pass, no effect, failed placebo), G2, G3, G4b's placement
    verdicts, CNG and the out-of-gap action;
  - end to end (plan 'full', profile synthetic_smoke, two seeds, four random-init hosts): gap define/check/freeze,
    preregister, gap_train, stage0 + H0, freeze, run. Every stage writes its JSON under one pre-registration sha256,
    marked SYNTHETIC; the hash gate refuses a wrong hash, an edited pre-registration, an edited gap.json and a forged
    body; a second run recomputes nothing; the enforced stop rule blocks G4 when G1 fails.
JAX_PLATFORMS=cpu LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin python -m unittest experiments.truth.test_gap
"""

import contextlib
import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.truth import gap as GAP
from experiments.truth import gap_acceptance as GA
from experiments.truth import gap_train as GT
from experiments.truth import pools as PL
from experiments.truth import state_restore as SR
from experiments.truth import synthetic as SY

SEEDS = tuple(GA.SMOKE["seeds"])
# Near the random-init hosts' actions on this file, so the gapped hosts visit the gap and every in-gap path runs.
PLANTED = dict(center=[-0.12, 0.18, -0.25], share=0.25, sd=0.06)
FIX = {}


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def silent(*_):
    pass


def setUpModule():
    tmp = Path(tempfile.mkdtemp(prefix="truth-gap-"))
    physics = SR.Physics("hopper")
    (tmp / "data").mkdir()
    meta = SY.write_synthetic(tmp / "data", "hopper", n_rows=1500, seed=5, physics=physics, planted=PLANTED)
    physics.close()
    file = Path(meta["with_infos"])
    world = tmp / "world"
    with quiet():
        manifest = GAP.define(world, "hopper", file=file, profile="synthetic_smoke")
        gap_sha = SR.file_sha256(world / "gap" / "gap.json")
        g0 = GAP.check(world, gap_sha)
        refused = None
        if not g0["passed"]:
            try:
                GAP.freeze(world, gap_sha)
            except ValueError as error:
                refused = str(error)
        GAP.freeze(world, gap_sha, exercise_failed_g0=not g0["passed"])
        prereg_gap = GA.preregister(world, "hopper", gap_sha, profile="synthetic_smoke", plan="full")
        hosts, train_records = {}, {}
        for s in SEEDS:
            run = tmp / f"run_gap_{s}"
            train_records[s] = GT.run("hopper", world / "gap" / "gap.json", gap_sha, s, run, device="cpu",
                                      prepare_only=True, synthetic_test=True)
            GT.attach_random_checkpoint(run)
            hosts[GA.host_key("gap", s)] = str(run)
            hosts[GA.host_key("full", s)] = PL.write_synthetic_run(tmp / f"run_full_{s}", "hopper", file, s)["run_dir"]
        summary, h0 = GA.stage0(world, gap_sha, prereg_gap, hosts, rows=200)
        sha = GA.freeze(world, gap_sha256=gap_sha, gap_prereg_sha256=prereg_gap, hosts=hosts)
        verdict = GA.run_all(world, sha, log=silent)
    FIX.update(tmp=tmp, file=file, world=world, manifest=manifest, gap_sha=gap_sha, g0=g0, freeze_refused=refused,
               prereg_gap=prereg_gap, hosts=hosts, train=train_records, stage0=summary, h0=h0, sha=sha,
               verdict=verdict)


def tearDownModule():
    shutil.rmtree(FIX["tmp"], ignore_errors=True)


# ---------------------------------------------------------------- the rule (NumPy only)

class GapRuleTest(unittest.TestCase):

    def planted(self, seed=0):
        g = np.random.default_rng(seed)
        corner = np.tile([1.0, 1.0, -1.0], (480, 1))
        ball = np.clip(np.array([0.2, -0.3, 0.1]) + 0.04 * g.normal(size=(1500, 3)), -1, 1)
        rest = g.uniform(-1, 1, size=(4020, 3))
        return g.permutation(np.concatenate([corner, ball, rest])).astype(np.float32)

    def test_planted_cluster_recovered_and_deterministic(self):
        a = self.planted()
        first = GAP.find_gap(a, GAP.SMOKE | dict(search=dict(GAP.SMOKE["search"], score_rows=None)))
        second = GAP.find_gap(a, GAP.SMOKE | dict(search=dict(GAP.SMOKE["search"], score_rows=None)))
        self.assertTrue(first["defined"])
        self.assertEqual((first["centre"], first["rho"]), (second["centre"], second["rho"]))
        self.assertLess(np.linalg.norm(np.asarray(first["centre"]) - [0.2, -0.3, 0.1]), 0.05)
        self.assertTrue(0.14 <= first["mass_train"] <= 0.16 and first["rho"] >= 0.05)
        d = GAP.descriptors(a.astype(np.float64), first["centre"], first["rho"], GAP.SMOKE)
        self.assertAlmostEqual(d["largest_duplicate_mass"], 0.08)
        self.assertGreater(d["density_contrast"], 5.0)
        self.assertAlmostEqual(d["exactly_one_share_per_coordinate"][0], 0.08, delta=0.002)

    def test_radius_rule_ties_and_band(self):
        base = np.r_[np.linspace(0.01, 0.15, 150), np.linspace(0.3, 1.0, 850)]
        self.assertEqual(GAP.radius_rule(base, 0.15, [0.14, 0.16])["rule"], "base")
        tie = np.r_[np.linspace(0.01, 0.14, 140), np.full(30, 0.2), np.linspace(0.3, 1.0, 830)]
        r = GAP.radius_rule(tie, 0.15, [0.14, 0.16])
        self.assertEqual((r["rule"], r["rho"], r["mass"]), ("tie_block_fallback", 0.14, 0.14))
        self.assertAlmostEqual(r["inclusive_mass_at_k"], 0.17)
        wide = np.r_[np.linspace(0.01, 0.1, 100), np.full(100, 0.2), np.linspace(0.3, 1.0, 800)]
        self.assertEqual(GAP.radius_rule(wide, 0.15, [0.14, 0.16])["rule"], "none")

    def test_point_mass_and_spike_fail(self):
        g = np.random.default_rng(1)
        spike = np.r_[np.tile([0.3, 0.3, 0.3], (500, 1)), g.uniform(-1, 1, (1500, 3))]
        r = GAP.find_gap(spike.astype(np.float32), GAP.SMOKE | dict(search=dict(GAP.SMOKE["search"], score_rows=None)))
        self.assertFalse(r["defined"])
        tight = np.r_[0.005 * g.normal(size=(400, 3)) + 0.2, g.uniform(-1, 1, (1600, 3))]
        r = GAP.find_gap(tight.astype(np.float32), GAP.SMOKE | dict(search=dict(GAP.SMOKE["search"], score_rows=None)))
        self.assertFalse(r["defined"])
        self.assertIn("point-mass", r["reason"])

    def test_inclusive_ball_and_out_of_gap_action(self):
        c = np.zeros(3)
        self.assertTrue(GAP.in_gap(np.array([[0.25, 0.0, 0.0]]), c, 0.25)[0])  # inclusive at rho
        self.assertFalse(GAP.in_gap(np.array([[0.2500001, 0.0, 0.0]]), c, 0.25)[0])
        out = GA.out_of_gap_actions(np.array([[0.1, 0, 0], [0, 0, 0]]), c, 0.2, 1.1)
        np.testing.assert_allclose(out, [[0.22, 0, 0], [0.22, 0, 0]], atol=1e-7)
        edge = GA.out_of_gap_actions(np.array([[0.99, 0, 0]]), np.array([0.95, 0, 0]), 0.2, 1.1)
        self.assertEqual(float(edge[0, 0]), 1.0)


# ---------------------------------------------------------------- the data and the gapped hosts

class GapDataTest(unittest.TestCase):

    def setUp(self):
        self.m = GAP.read_manifest(FIX["world"] / "gap" / "gap.json", FIX["gap_sha"])

    def test_mass_band_and_determinism(self):
        gap = self.m.data["gap"]
        self.assertTrue(0.14 <= gap["masses"]["training"] <= 0.16)
        self.assertLess(np.linalg.norm(np.asarray(gap["centre"]) - PLANTED["center"]), 0.05)
        with quiet():
            again = GAP.define(FIX["tmp"] / "world_again", "hopper", file=FIX["file"], profile="synthetic_smoke",
                               gapped_dir=FIX["tmp"] / "gapped_again")
        self.assertEqual(again["definition_sha256"], self.m.data["definition_sha256"])
        self.assertEqual(again["gapped"]["per_key_sha256"], self.m.data["gapped"]["per_key_sha256"])
        for name in ("gap_rows", "pool_G", "pool_G_held", "pool_K_match", "pool_K_far"):
            self.assertEqual(again["side_files"][name]["sha256"], self.m.data["side_files"][name]["sha256"])

    def test_g0a_integrity_and_freeze_refusal(self):
        g0a = FIX["g0"]["g0a"]
        self.assertTrue(g0a["passed"], g0a["passes"])
        self.assertEqual(g0a["values"]["leaks"], dict(training=0, withheld=0, bank=0))
        self.assertGreater(g0a["values"]["negative_control_leaks"], 0)
        self.assertEqual(set(FIX["g0"]["g0b"]["passes"]), {"i_dense", "ii_thin_under_kept", "iii_neighbours_lost",
                                                          "iv_states_covered", "v_matched"})
        if not FIX["g0"]["passed"]:  # a failed G0 stops before training unless the synthetic override is used
            self.assertIn("G0 failed", FIX["freeze_refused"])
            self.assertTrue(json.loads((FIX["world"] / "gap" / "freeze.json").read_text())["synthetic_g0_override"])

    def test_no_gap_row_reaches_training_or_bank(self):
        host = PL.restore_host(FIX["hosts"][GA.host_key("gap", SEEDS[0])], data_dir=FIX["file"].parent)
        native = self.m.native_split_rows()
        gap_rows = self.m.array("gap_rows")
        for name, ids in (("training", host.training_ids), ("withheld", host.withheld_ids), ("bank", host.bank_ids)):
            rows = host.rows[ids]
            self.assertEqual(int(np.isin(rows, gap_rows).sum()), 0, name)
            np.testing.assert_array_equal(rows, native[name][~np.isin(native[name], gap_rows)])
        self.assertGreater(len(host.bank_ids), 0)
        # thinness and blocks use ORIGINAL episodes, never the fragments the flags create
        np.testing.assert_array_equal(host.episodes, self.m.original_episodes(host.rows))
        self.assertEqual(GAP.leak_counts(self.m, {"t": native["training"]})["t"],
                         self.m.data["gap"]["counts"]["training"])

    def test_leaked_row_is_detected(self):
        import h5py
        from runtime import td3_bc as R

        leaky = FIX["tmp"] / "leaky" / self.m.data["gapped"]["file"]
        leaky.parent.mkdir(exist_ok=True)
        shutil.copy(self.m.gapped_path(), leaky)
        row = int(self.m.array("gap_rows")[3])
        with h5py.File(leaky, "r+") as f:
            t = f["timeouts"][()]
            t[row] = False
            f["timeouts"][...] = t
        raw = PL.read_raw(leaky)
        converted, rows, _ = SR.converted_rows({k: raw[k] for k in PL.RAW_KEYS})
        self.assertEqual(GAP.leak_counts(self.m, {"training": rows})["training"], 1)
        maps = R.dependency_maps(rows, converted["terminals"])
        reserve = GAP.make_projected_reserve(self.m, R._reserve)
        reservation = R.Reservation(**self.m.data["reservation"])
        with self.assertRaisesRegex(ValueError, "leaks gap transitions"):
            reserve(maps, reservation, None)

    def test_matched_and_far_pools(self):
        m = self.m
        native_rows = m.array("native_raw_current")
        training_rows = native_rows[m.array("native_training_ids")]
        raw = PL.read_raw(FIX["file"])
        mean, std = (np.asarray(m.data["native"][k], np.float32) for k in ("obs_mean", "obs_std"))
        s = lambda r: GAP.standardize(raw["observations"][r], mean, std)
        G, K, dist = m.array("pool_G"), m.array("pool_K_match_for_G"), m.array("pool_K_match_distance")
        hit = K >= 0
        self.assertGreaterEqual(hit.mean(), 0.5)
        self.assertEqual(len(np.unique(K[hit])), int(hit.sum()))  # without replacement
        self.assertFalse(np.isin(K[hit], m.array("gap_rows")).any())
        self.assertTrue(np.isin(K[hit], training_rows).all())
        self.assertTrue(np.all(m.original_episodes(K[hit]) != m.original_episodes(G[hit])))
        np.testing.assert_allclose(np.linalg.norm(s(K[hit]) - s(G[hit]), axis=1), dist[hit], rtol=1e-9)
        self.assertTrue(np.all(dist[hit] <= m.data["geometry"]["r_s"]))
        self.assertTrue(np.all(m.d_over_rho(raw["actions"][K[hit]]) > 1.0))
        # K_far: no gap row among the 10 nearest native-training rows in x (brute force)
        x = lambda r: np.concatenate([s(r), raw["actions"][r].astype(np.float64)], axis=1)
        xt, gap_t = x(training_rows), np.isin(training_rows, m.array("gap_rows"))
        for r in m.array("pool_K_far"):
            d = np.linalg.norm(xt - x(np.array([r])), axis=1)
            d[training_rows == r] = np.inf
            self.assertFalse(gap_t[np.argsort(d, kind="stable")[:10]].any())

    def test_gap_train_record_and_refusals(self):
        rec = FIX["train"][SEEDS[0]]
        self.assertTrue(rec["synthetic"] and rec["prepare_only"])
        self.assertEqual(rec["projected_counts"]["training"] + self.m.data["gap"]["counts"]["training"],
                         rec["projected_counts"]["native_training"])
        resolved = json.loads((Path(FIX["hosts"][GA.host_key("gap", SEEDS[0])]) / "resolved.json").read_text())
        self.assertTrue(resolved["run_id"].endswith(f"-gap15-{FIX['gap_sha'][:8]}"))
        self.assertEqual(resolved["cache"]["sha256"], self.m.data["gapped"]["sha256"])
        args = ["--task", "hopper", "--gap", str(FIX["world"] / "gap" / "gap.json"), "--gap-sha256",
                FIX["gap_sha"], "--seed", str(SEEDS[0]), "--output", str(FIX["tmp"] / "never")]
        with self.assertRaisesRegex(GT.BandFitDisabled, "allow-bca-band"):
            GT.main(args + ["--method", "bca"])
        with self.assertRaisesRegex(GT.BandFitDisabled, "allow-bca-band"):
            GT.run("hopper", FIX["world"] / "gap" / "gap.json", FIX["gap_sha"], SEEDS[0], FIX["tmp"] / "never",
                   method="bca")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            GT.main(args + ["--allow-bca-band"])  # no such flag
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            GT.run("hopper", FIX["world"] / "gap" / "gap.json", "0" * 64, SEEDS[0], FIX["tmp"] / "never",
                   prepare_only=True, synthetic_test=True)
        with self.assertRaisesRegex(ValueError, "synthetic"):
            GT.run("hopper", FIX["world"] / "gap" / "gap.json", FIX["gap_sha"], SEEDS[0], FIX["tmp"] / "never",
                   prepare_only=True)
        self.assertFalse((FIX["tmp"] / "never").exists())

    def test_restore_refuses_without_manifest(self):
        src = Path(FIX["hosts"][GA.host_key("gap", SEEDS[0])])
        dst = FIX["tmp"] / "no_manifest_run"
        shutil.copytree(src, dst)
        row = json.loads((dst / "resolved.json").read_text())
        row["action_gap"]["manifest"] = str(FIX["tmp"] / "missing" / "gap.json")
        (dst / "resolved.json").write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "cannot be restored without it"):
            PL.restore_host(dst, data_dir=FIX["file"].parent)
        host = PL.restore_host(dst, data_dir=FIX["file"].parent, gap_manifest=FIX["world"] / "gap" / "gap.json")
        self.assertEqual(host.identity["action_gap"]["manifest_sha256"], FIX["gap_sha"])


# ---------------------------------------------------------------- known-answer scorers (fabricated labels)

class ScorerTest(unittest.TestCase):
    seeds = [11, 12, 13]

    def g1_fixture(self, gap_factor=(3.0, 2.5, 1.0, 1.0), full_scale=None):
        g = np.random.default_rng(7)
        n = 400
        code = np.repeat(np.arange(4), n)
        tables = {}
        for s in self.seeds:
            for arm in GA.ARMS:
                scale = np.ones(4) if arm == "full" else np.asarray(gap_factor)
                if arm == "full" and full_scale is not None:
                    scale = scale * full_scale.get(s, 1.0)
                ea = g.normal(size=4 * n) * scale[code]
                tables[GA.host_key(arm, s)] = dict(ea=ea, e_logged=ea + 0.1, length_pi=g.integers(1, 1000, 4 * n)
                                                   .astype(np.float64))
        common = dict(code=code, block=np.arange(4 * n) // 4, d_logged=g.uniform(0, 1, 4 * n))
        return tables, common

    def test_g0b(self):
        v = dict(c_full_median=12, thin_kept_share=0.6, kept_full_ratio_median=0.1, state_kept_median=15,
                 state_ratio_median=0.8, match_rate=0.95)
        marks = GAP.DESIGN["g0_marks"]
        self.assertTrue(all(GAP.g0b_passes(v, marks).values()))
        self.assertFalse(GAP.g0b_passes(dict(v, thin_kept_share=0.3), marks)["ii_thin_under_kept"])
        self.assertFalse(GAP.g0b_passes(dict(v, state_ratio_median=0.3), marks)["iv_states_covered"])

    def test_g1_known_answers(self):
        marks = GA.PASS_MARKS["G1"]
        values, passes = GA.g1_score(*self.g1_fixture(), self.seeds, marks, 200, 0)
        self.assertTrue(all(passes.values()), (passes, values["pooled"], values["intervals95"]))
        self.assertAlmostEqual(values["pooled"]["G"], 3.0, delta=0.4)
        _, passes = GA.g1_score(*self.g1_fixture(gap_factor=(1.0, 1.0, 1.0, 1.0)), self.seeds, marks, 100, 0)
        self.assertFalse(passes["a"] or passes["d"])
        self.assertTrue(passes["b"])
        _, passes = GA.g1_score(*self.g1_fixture(full_scale={12: 2.5}), self.seeds, marks, 100, 0)
        self.assertFalse(passes["e"])

    def test_g2_known_answers(self):
        g = np.random.default_rng(3)
        visits = {}
        for s in self.seeds:
            for arm, p in (("gap", 0.1), ("full", 0.02)):
                ep = np.repeat(np.arange(20), 100)
                in_gap = g.random(len(ep)) < p
                visits[GA.host_key(arm, s)] = dict(in_gap=in_gap, near=in_gap, episode=ep,
                                                   d_pi=g.uniform(0, 5, len(ep)))
        values, passes = GA.g2_score(visits, self.seeds, GA.PASS_MARKS["G2"], 100, 0, 0.15)
        self.assertTrue(all(passes.values()), values)
        self.assertGreater(values["ratio_gapped_over_full"][str(self.seeds[0])], 2.0)
        for s in self.seeds:
            visits[GA.host_key("gap", s)]["in_gap"][:] = False
        _, passes = GA.g2_score(visits, self.seeds, GA.PASS_MARKS["G2"], 100, 0, 0.15)
        self.assertFalse(passes["share"] or passes["hosts"])

    def test_g3_known_answers(self):
        g = np.random.default_rng(5)
        returns, v3, sigma = {}, {}, {}
        for s in self.seeds:
            full = 60 + 10 * g.normal(size=100)
            returns[GA.host_key("full", s)] = full
            returns[GA.host_key("gap", s)] = full - 6 + 0.5 * g.normal(size=100)
            for arm in GA.ARMS:
                n = 100
                in_gap = np.r_[np.ones(50, bool), np.zeros(50, bool)]
                qpi = np.full((n, 2), 100.0) + g.normal(size=(n, 2))
                qpi[:, 1] = np.where(in_gap, qpi[:, 0] + 10, np.nan)  # a_out is truly better
                q1 = qpi.copy()
                if arm == "gap":
                    q1[in_gap, 0] += 20  # overestimates its in-gap actions ...
                    q1[in_gap, 1] = q1[in_gap, 0] - 5  # ... and prefers them to a_out
                v3[GA.host_key(arm, s)] = dict(in_gap=in_gap, q1=q1, qpi=qpi, episode=np.arange(n) // 5)
            sigma[str(s)] = 0.1
        values, passes = GA.g3_score(returns, v3, sigma, self.seeds, GA.PASS_MARKS["G3"], GA.DESIGN, 100, 0)
        self.assertTrue(all(passes.values()) and values["mechanism"], (passes, values["return_cost"]))
        self.assertAlmostEqual(values["return_cost"]["pooled_delta_J"], -6.0, delta=0.5)
        self.assertEqual(values["misled"]["share"], 1.0)
        for s in self.seeds:
            returns[GA.host_key("gap", s)] = returns[GA.host_key("full", s)] + g.normal(size=100)
        values, passes = GA.g3_score(returns, v3, sigma, self.seeds, GA.PASS_MARKS["G3"], GA.DESIGN, 100, 0)
        self.assertFalse(passes["a_return_cost"])
        self.assertTrue(values["mechanism"])

    def g4_fixture(self, gap_star=56.0, abs_star=50.0):
        ex = GA.DESIGN["g4"]["extraction"]
        g = np.random.default_rng(9)
        J, matched = {}, {}
        for s in self.seeds:
            for unit in ("1", "2", "3", "4"):
                j = {"NONE": 50.0, **{f"P0({x})": 49.0 for x in ex["p0_ladder"]}}
                j[f"P0({ex['level']})"] = 50.0
                j.update({f"SHUF_{k}*": 50.0 for k in range(ex["shuf"])})
                j["OR_GAP*"] = gap_star + 0.3 * g.normal()
                j["OR_abs*"] = abs_star + 0.3 * g.normal()
                j.update({f"OR_GAP@L{i}": gap_star - 1 for i in range(5)})
                j.update({f"OR_abs@L{i}": abs_star - 1 for i in range(5)})
                j["OR_signed*"], j["ANTI_signed*"] = 60 + g.normal(), 45 + g.normal()
                J[(s, unit)] = j
                matched[(s, unit)] = dict(matched={a: dict(fidelity=0.01) for a in ("OR_abs", "OR_GAP", "SHUF_0")})
        cos = {str(s): {"OR_abs": 0.5, "OR_GAP": 0.4} for s in self.seeds}
        return J, matched, cos, ex

    def test_g4b_placement_verdicts(self):
        marks, m7, r7 = GA.PASS_MARKS["G4b"], GA.PASS_MARKS["R7"], GA.DESIGN["r7"]
        J, matched, cos, ex = self.g4_fixture()
        (v4, p4), (v7, p7) = GA.g4b_score(J, matched, cos, {"OR_abs": 1.0, "OR_GAP": 2.0}, self.seeds, ex, marks, m7,
                                          r7, 200, 0)
        self.assertEqual(p4, {"OR_abs": False, "OR_GAP": True})
        self.assertEqual(v4["verdict"], "room near the gap")
        self.assertTrue(all(p7.values()), (p7, v7["mde"]))
        J, matched, cos, ex = self.g4_fixture(gap_star=50.0, abs_star=56.0)
        verdict = lambda c: GA.g4b_score(J, matched, cos, {"OR_abs": c, "OR_GAP": 2.0}, self.seeds, ex, marks, m7,
                                         r7, 200, 0)[0][0]["verdict"]
        self.assertEqual(verdict(1.2), "room, not at the gap")
        self.assertEqual(verdict(1.8), "room near the gap")
        J, matched, cos, ex = self.g4_fixture(gap_star=50.0, abs_star=50.0)
        self.assertEqual(verdict(1.8), "no room")

    def test_cng(self):
        self.assertAlmostEqual(GA.cng([3.0, 1.0, 0.5, 0.5], [0.5, 1.0, 2.0, 3.0], 1.5), 1.6)
        self.assertIsNone(GA.cng([1.0, 1.0], [2.0, 3.0], 1.5))


# ---------------------------------------------------------------- end to end and the hash gate

class EndToEndTest(unittest.TestCase):

    def test_every_stage_written_and_marked(self):
        v, world, sha = FIX["verdict"], FIX["world"], FIX["sha"]
        self.assertTrue(v["synthetic"] and v["meaning"].startswith("SYNTHETIC TEST RUN, NOT A FINDING"))
        self.assertEqual(v["statuses"]["H0"], "pass")
        self.assertEqual(v["statuses"]["G0A"], "pass")
        for req in GA.PLANS["full"]:
            self.assertIn(v["statuses"][req], ("pass", "fail"), req)
        records = list((world / "scores").glob("*.json")) + list(world.glob("host_*/scores/G4a.json"))
        self.assertEqual(len(records), len(GA.PLANS["full"]) - 1 + len(SEEDS))
        for path in records:
            r = json.loads(path.read_text())
            self.assertEqual((r["prereg_sha256"], r["synthetic"], r["gap_sha256"]), (sha, True, FIX["gap_sha"]))
            self.assertIn("NOT A FINDING", r["note"])
        for s in SEEDS:
            out = world / "g4" / f"critic_{s}"
            self.assertEqual(SR.file_sha256(out / "matched.json"),
                             (out / "matched.json.sha256").read_text().split()[0])
            self.assertTrue(json.loads((out / "evaluation.json").read_text())["matched_lock_verified"])
        self.assertTrue((world / "verdict.md").read_text().startswith("**SYNTHETIC TEST RUN. NOT A FINDING.**"))
        g2 = json.loads((world / "scores" / "G2.json").read_text())
        self.assertGreater(g2["values"]["pooled_S_gap_gapped"], 0.0)  # the in-gap paths of G3 ran
        g3 = json.loads((world / "scores" / "G3.json").read_text())
        self.assertGreater(g3["values"]["misled"]["states"], 0)
        h0 = FIX["h0"]
        for key, rec in h0["hosts"].items():
            self.assertTrue(rec["passed"], (key, rec["checks"]))
            if key.startswith("gap-"):
                self.assertEqual(rec["details"]["gap_rows_inside"], dict(training=0, withheld=0, bank=0))

    def test_resume_recomputes_nothing(self):
        before = {p: p.stat().st_mtime_ns for p in FIX["world"].rglob("*.npz")}
        with quiet():
            again = GA.run_all(FIX["world"], FIX["sha"], log=silent)
        self.assertEqual(again["statuses"], FIX["verdict"]["statuses"])
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in FIX["world"].rglob("*.npz")})

    def copy_world(self, name):
        target = FIX["tmp"] / name
        target.mkdir()
        for f in ("prereg.json", "prereg.json.sha256", "gap_prereg.json", "gap_prereg.json.sha256"):
            shutil.copy(FIX["world"] / f, target / f)
        shutil.copytree(FIX["world"] / "gap", target / "gap")
        return target

    def test_hash_gate_refuses_edits(self):
        world, sha = FIX["world"], FIX["sha"]
        with self.assertRaises(GA.PreregError):
            GA.load_prereg(world, "0" * 64)
        body, digest, _ = GA.load_prereg(world, sha)
        self.assertEqual(digest, sha)
        edited = self.copy_world("edited_prereg")  # an edited mark, sidecar made consistent with the edit
        document = json.loads((edited / "prereg.json").read_text())
        document["body"]["pass_marks"]["G1"]["ratio_min"] = 1.0
        text = json.dumps(document).encode()
        (edited / "prereg.json").write_bytes(text)
        new = GA.sha256_hex(text)
        (edited / "prereg.json.sha256").write_text(f"{new}  prereg.json\n")
        with self.assertRaises(GA.PreregError):
            GA.load_prereg(edited, sha)
        with self.assertRaisesRegex(GA.PreregError, "embedded hash"):
            GA.load_prereg(edited, new)
        moved = self.copy_world("edited_gap")  # the gap itself edited after the freeze
        gap = json.loads((moved / "gap" / "gap.json").read_text())
        gap["gap"]["rho"] *= 1.01
        (moved / "gap" / "gap.json").write_text(json.dumps(gap))
        with self.assertRaisesRegex(GA.PreregError, "gap"):
            GA.load_prereg(moved, sha)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            GT.verify(moved / "gap" / "gap.json", FIX["gap_sha"], "hopper")
        forged = copy.deepcopy(body)
        forged["pass_marks"]["G3"]["return_cost_min"] = 0.0
        for call in (lambda: GA.stage_score(world, forged, sha), lambda: GA.verdict(world, forged, sha),
                     lambda: GA.stage_pools(world, body, "0" * 64, GA.host_key("gap", SEEDS[0]))):
            with self.assertRaises(GA.PreregError):
                call()

    def test_smoothed_truth_path(self):
        """When G1-pre requires the smoothed truth, labels adds G1_smooth on every host and G1 reads its means."""
        world = FIX["tmp"] / "smoothed"
        shutil.copytree(FIX["world"], world, ignore=shutil.ignore_patterns("g4"))
        body, sha, _ = GA.load_prereg(world, FIX["sha"])
        pre = json.loads((world / "scores" / "G1_pre.json").read_text())
        pre["values"]["smoothed_truth_required"] = True
        (world / "scores" / "G1_pre.json").write_text(json.dumps(pre))
        key = GA.host_key("gap", SEEDS[0])
        self.assertIn("G1_smooth", GA.label_jobs(body, key, smooth=True))
        self.assertFalse(GA.labels_complete(world, body, key))
        with quiet():
            for k in GA.planned_keys(body["protocol"]):
                GA.stage_labels(world, body, sha, k, jobs=["G1_smooth"], log=silent)
        self.assertTrue(GA.labels_complete(world, body, key))
        smooth, _ = GA.g1_tables(world, body, smooth=True)
        single, _ = GA.g1_tables(world, body, smooth=False)
        self.assertEqual(set(smooth), set(single))
        self.assertFalse(np.allclose(smooth[key]["ea"], single[key]["ea"], rtol=0, atol=0))
        self.assertLess(np.median(np.abs(smooth[key]["ea"] - single[key]["ea"])), 1.0)  # jitter: 1e-4 of the data SD

    def test_order_refusals_and_stop_rule(self):
        world = FIX["world"]
        with self.assertRaisesRegex(ValueError, "exists"):
            GA.preregister(world, "hopper", FIX["gap_sha"], profile="synthetic_smoke")
        with self.assertRaises(ValueError):
            GA.freeze(world, gap_sha256=FIX["gap_sha"], gap_prereg_sha256=FIX["prereg_gap"], hosts=FIX["hosts"])
        with self.assertRaises(ValueError):
            GAP.freeze(world, FIX["gap_sha"])  # frozen once
        body, _, _ = GA.load_prereg(world, FIX["sha"])
        enforced = copy.deepcopy(body)
        enforced["protocol"]["stop_rule"] = "enforced"
        g1 = json.loads((world / "scores" / "G1.json").read_text())
        allowed, reason = GA.g4_gate(world, enforced)
        if not g1["passed"]:
            self.assertFalse(allowed)
            self.assertIn("G1", reason)


if __name__ == "__main__":
    unittest.main()
