"""Regression tests for the adversarial review of experiments/truth (C-review, 2026-10-11).

Each test pins one defect that a probe confirmed before the fix (probe numbers are synthetic fixtures, POST HOC):
  1. Hash gate bypass. The stage functions trusted the body and hash they were given: stage_pools ran with a forged
     body (a loosened R1 pass mark) and the hash '0' * 64, and wrote pools stamped prereg_sha256 = '000...0'. Every
     stage function now calls verify_stage, which re-reads W/prereg.json; an unacknowledged source change refuses.
  2. freeze accepted tolerances under which the misaligned-by-one negative control passes. Synthetic Stage 0: the
     negative control's restore miss max 3.48, next-observation p99 3.01, reward p99 0.48, terminal agreement 0.94
     all pass at tolerances (10, 50, 50, 0.5), and freeze wrote a pre-registration. It now refuses.
  3. R4b's best-strength margin left NONE (m = 1) out of even caution. With NONE 70, best P0 60 and best OR-L-abs 65,
     the margin was +5 (a pass) although NONE beats every OR-L-abs arm. NONE is now in the P0 set: margin -5, fail.
  4. R7's null pairs: paired_z gives +-inf when every unit has the same nonzero difference, and strict JSON
     (allow_nan=False) refused to write it, so extraction-score crashed. z is now stored as None plus a flag, and
     the pair still counts as fired.
  5. One pool-T row without truth (a simulator error, status 2, q NaN) aborted R4a and the extraction. The oracle now
     drops such rows, and the screen and the extraction use the same kept rows.
  6. Labels made for another pool file were scored silently. load_job now checks the label manifest's input hashes
     against the pool's states and actions.
  7. The host's discount and target-smoothing noise were never checked against the pre-registered label settings.
  8. freeze --state-source rebuilt: pool states rebuilt from the observation when infos/* fail the gate.
  9. Per-job rotation of the label shards over workers still labels every shard once (synthetic Hopper graft, three
     workers), and a synthetic verdict says so in its meaning line.
Synthetic files and randomly initialised hosts only: no number here is a finding.
JAX_PLATFORMS=cpu LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin python -m unittest experiments.truth.test_review
"""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.truth import acceptance as A
from experiments.truth import labeller as LB
from experiments.truth import pools as PL
from experiments.truth import state_restore as SR
from experiments.truth import synthetic as SY

SEED = 202609171


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def silent(*_):
    pass


class ReviewFixture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="truth-review-"))
        physics = SR.Physics("hopper")
        (cls.tmp / "data").mkdir()
        meta = SY.write_synthetic(cls.tmp / "data", "hopper", n_rows=1500, seed=5, physics=physics)
        physics.close()
        cls.file = Path(meta["with_infos"])
        with quiet():
            cls.host_run = PL.write_synthetic_run(cls.tmp / "run", "hopper", cls.file, SEED)["run_dir"]
            cls.stage0 = A.stage0(cls.tmp / "s0", "hopper", file=cls.file, run_dir=cls.host_run, rows=200,
                                  eps_draws=4)
        cls.stage0_dir = cls.tmp / "s0" / "stage0"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def world(self, name, plan="full", **kwargs):
        world = self.tmp / name
        with quiet():
            sha = A.freeze(world, "hopper", hosts={SEED: self.host_run}, stage0_dir=self.stage0_dir,
                           profile="synthetic_smoke", plan=plan, **kwargs)
        body, _, _ = A.load_prereg(world, sha)
        return world, sha, body


class GateTest(ReviewFixture):

    def test_stage_functions_refuse_a_forged_body_or_hash(self):
        world, sha, body = self.world("gate")
        forged = json.loads(json.dumps(body))
        forged["pass_marks"]["R1"]["pi_thin_share_min"] = 0.0
        with self.assertRaises(A.PreregError):
            A.stage_pools(world, forged, sha, SEED)  # right hash, edited body
        wrong = "0" * 64
        calls = [lambda: A.stage_pools(world, body, wrong, SEED),
                 lambda: A.stage_labels(world, body, wrong, SEED, log=silent),
                 lambda: A.stage_score(world, body, wrong, SEED),
                 lambda: A.stage_screen(world, body, wrong, SEED),
                 lambda: A.stage_extract(world, body, wrong, SEED, log=silent),
                 lambda: A.stage_evaluate(world, body, wrong, SEED, log=silent),
                 lambda: A.stage_extraction_score(world, body, wrong),
                 lambda: A.verdict(world, body, wrong)]
        for call in calls:
            with self.assertRaises(A.PreregError):
                call()
        self.assertFalse(A.critic_dir(world, SEED).exists())
        self.assertFalse((world / "extraction").exists())
        self.assertTrue(A.verify_stage(world, body, sha))

    def test_unacknowledged_code_change_refuses(self):
        world, sha, body = self.world("code")
        doc = json.loads((world / "prereg.json").read_text())
        doc["body"]["code_sha256"]["experiments/truth/pools.py"] = "0" * 64
        doc["body_sha256"] = A.canonical_sha256(doc["body"])
        text = json.dumps(doc).encode()
        (world / "prereg.json").write_bytes(text)
        digest = A.sha256_hex(text)
        (world / "prereg.json.sha256").write_text(f"{digest}  prereg.json\n")
        with self.assertRaises(A.PreregError):
            A.verify_stage(world, doc["body"], digest)
        with self.assertRaises(A.PreregError):
            A.stage_score(world, doc["body"], digest, SEED)
        self.assertTrue(A.verify_stage(world, doc["body"], digest, code_changed=True))

    def test_freeze_refuses_tolerances_that_pass_the_negative_control(self):
        neg = self.stage0["gate_negative"]
        loose = dict(tol_restore=10.0, tol_next_obs=50.0, tol_reward=50.0, min_terminal_agreement=0.5)
        self.assertTrue(A.gate_passes(neg, loose))  # the probe: the misaligned states pass these tolerances
        with self.assertRaises(ValueError) as caught:
            with quiet():
                A.freeze(self.tmp / "loose", "hopper", hosts={SEED: self.host_run}, stage0_dir=self.stage0_dir,
                         profile="synthetic_smoke", plan="graft", tolerances=loose)
        self.assertIn("negative control", str(caught.exception))
        self.assertFalse((self.tmp / "loose" / "prereg.json").exists())
        self.assertTrue(A.gate_passes(self.stage0["gate"], SR.DEFAULT_TOLERANCES))
        self.assertFalse(A.gate_passes(neg, SR.DEFAULT_TOLERANCES))

    def test_state_source_rebuilt(self):
        world, _, body = self.world("rebuilt", plan="graft", state_source="rebuilt")
        self.assertEqual(body["protocol"]["state_source"], "rebuilt")
        self.assertEqual(body["stage0"]["gate_pilot"]["source"], "rebuilt")
        _, _, auto = self.world("auto", plan="graft")
        self.assertEqual((auto["protocol"]["state_source"], auto["stage0"]["gate_pilot"]["source"]),
                         ("auto", "infos"))
        with quiet():
            host = PL.restore_host(self.host_run, data_dir=self.file.parent)
        ids = host.training_ids[:50]
        _, rebuilt, _ = PL._restorable(host, ids, "rebuilt")
        _, infos, _ = PL._restorable(host, ids, "auto")
        self.assertEqual((rebuilt.source, infos.source), ("rebuilt", "infos"))
        np.testing.assert_array_equal(rebuilt.qpos[:, 0], 0.0)
        np.testing.assert_array_equal(rebuilt.qpos[:, 1:], infos.qpos[:, 1:])
        np.testing.assert_array_equal(rebuilt.qvel, infos.qvel)

    def test_host_settings_must_match_label_settings(self):
        world, _, body = self.world("settings", plan="graft")
        with quiet():
            self.assertEqual(A.host_for(body, SEED).seed, SEED)
        for path, value in ((("gamma",), 0.98), (("gap", "noise_sd"), 0.1), (("gap", "noise_clip"), 0.3)):
            changed = json.loads(json.dumps(body))
            target = changed["protocol"]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError):
                with quiet():
                    A.host_for(changed, SEED)


def unit_scores(u):
    """Evaluation means of one unit: NONE is the best even-caution arm and beats every OR-L-abs arm."""
    J = {"NONE": 70.0 + 0.3 * u, "P0(0.25)": 55.0 + u, "P0(0.5)": 58.0 - 0.2 * u, "P0(2.0)": 59.5 + 0.1 * u,
         "P0(4.0)": 50.0, "OR_abs@L0": 62.0 + 0.5 * u, "OR_abs*": 65.0 + 0.2 * u * u, "OR_signed*": 66.0 + u,
         "ANTI_signed*": 54.0 - u * u, "SHUF_0*": 60.0 + 2 * u, "SHUF_1*": 61.0 + 2 * u,
         "SHUF_2*": 59.0 + u * u, "SHUF_3*": 60.5 - u}
    return {k: dict(mean=v) for k, v in J.items()}


class ExtractionScoreTest(ReviewFixture):

    def test_best_margin_includes_none_and_infinite_null_z_is_written(self):
        world, sha, body = self.world("xscore")
        units = [1, 2, 3]
        scores = A.critic_dir(world, SEED) / "scores"
        scores.mkdir(parents=True)
        (scores / "R4a.json").write_text(json.dumps(dict(prereg_sha256=sha, values=dict(arms=dict(
            OR_abs=dict(cos_with_p0=0.5))))))
        out = world / "extraction" / f"critic_{SEED}"
        out.mkdir(parents=True)
        arms = ["OR_abs", "OR_signed", "ANTI_signed"] + [f"SHUF_{k}" for k in range(4)]
        matched = {str(u): dict(policies={"NONE": dict(q1_mean_withheld=100.0), "P0(0.25)": dict(
            q1_mean_withheld=110.0)}, matched={a: dict(fidelity=0.01) for a in arms}) for u in units}
        (out / "matched.json").write_text(json.dumps(dict(prereg_sha256=sha, simulator_forbidden_during_matching=True,
                                                          units=matched)))
        (out / "evaluation.json").write_text(json.dumps(dict(prereg_sha256=sha, matched_lock_verified=True,
                                                             units={str(u): unit_scores(u) for u in units})))
        written = A.stage_extraction_score(world, body, sha)
        v4 = written["R4b"]["values"]
        self.assertEqual(v4["p0_best_arm"], "NONE")
        self.assertAlmostEqual(v4["p0_best_mean"], 70.6)
        self.assertLess(v4["best_margin"], 0.0)  # it was +5 with NONE left out
        self.assertFalse(written["R4b"]["passes"]["best"])
        pair = next(p for p in written["R7"]["values"]["null_pairs"] if p["pair"] == [0, 1])
        self.assertEqual((pair["z"], pair["z_infinite"], pair["fires"]), (None, True, True))
        self.assertEqual(written["R7"]["values"]["null_pairs_fired"],
                         sum(p["fires"] for p in written["R7"]["values"]["null_pairs"]))
        json.loads((world / "extraction" / "R7.json").read_text())  # strict JSON was written


class OracleAndLabelsTest(ReviewFixture):

    def test_oracle_drops_rows_without_truth(self):
        _, _, body = self.world("oracle", plan="graft")
        g = np.random.default_rng(0)
        n = 40
        poolT = dict(q=g.normal(size=(n, 2, 2)).astype(np.float32))
        qpi = g.normal(size=(n, 2))
        qpi[3, 1] = np.nan
        arms, e_a, keep = A.oracle_arms(body, poolT, qpi, 4)
        self.assertEqual((int(keep.sum()), bool(keep[3])), (n - 1, False))
        self.assertTrue(all(len(w) == n - 1 for w in arms.values()))
        self.assertEqual(len(e_a), n - 1)
        with self.assertRaises(ValueError):
            A.oracle_arms(body, poolT, np.full((n, 2), np.nan), 4)

    def test_labels_for_another_pool_are_refused(self):
        world, _, _ = self.world("labels", plan="graft")
        with quiet():
            host = PL.restore_host(self.host_run, data_dir=self.file.parent)
        ids = host.training_ids[:4]
        states = SR.logged_states(host.task, host.raw, host.rows[ids])
        policy = host.policy(8)
        obs = SR.observation(host.task, states.qpos, states.qvel)
        pi = policy.actions(obs)
        pool = dict(qpos=states.qpos, qvel=states.qvel,
                    actions=np.stack([pi, host.converted["actions"][ids].astype(np.float32)], axis=1))
        physics = SR.Physics("hopper")
        try:
            spec = LB.LabelSpec(host.task.dataset, horizon=50, shard_size=3)
            LB.run_labels(A.critic_dir(world, SEED) / "labels" / "L", physics, policy, spec, PL.pool_items(pool))
        finally:
            physics.close()
        q, _, status = A.load_job(world, SEED, "L", 4, 2, pool)
        self.assertTrue(np.isfinite(q).all() and np.all(status >= 0))
        other = dict(pool, actions=pool["actions"].copy())
        other["actions"][2, 1] = np.clip(other["actions"][2, 1] + 0.01, -1, 1)
        with self.assertRaises(ValueError):
            A.load_job(world, SEED, "L", 4, 2, other)
        moved = dict(pool, qvel=pool["qvel"] + 1e-6)
        with self.assertRaises(ValueError):
            A.load_job(world, SEED, "L", 4, 2, moved)


class GraftWorkersTest(ReviewFixture):
    """Synthetic Hopper graft end to end with labels split over three workers (rotated per job)."""

    def test_three_workers_cover_every_shard_and_verdict_is_marked(self):
        world, sha, body = self.world("graft3", plan="graft")
        with quiet():
            A.stage_pools(world, body, sha, SEED)
            per_worker = [A.stage_labels(world, body, sha, SEED, worker=w, workers=3, log=silent) for w in range(3)]
        self.assertTrue(A.labels_complete(world, body, SEED))
        for job in A.label_jobs(body):
            manifest = json.loads((A.critic_dir(world, SEED) / "labels" / job / "manifest.json").read_text())
            handled = sum(rec[job]["shards"] for rec in per_worker)
            self.assertEqual(handled, manifest["manifest"]["n_shards"], job)  # every shard exactly once
        one_shard = [j for j in A.label_jobs(body)
                     if json.loads((A.critic_dir(world, SEED) / "labels" / j / "manifest.json").read_text())
                     ["manifest"]["n_shards"] == 1]
        owners = {j: next(w for w, rec in enumerate(per_worker) if rec[j]["shards"]) for j in one_shard}
        self.assertGreater(len(set(owners.values())), 1, owners)  # not all on worker 0
        with quiet():
            v = A.run_all(world, sha, log=silent)
        self.assertEqual(set(v["statuses"]), {"R1", "R2", "R5_world", "R6"})
        self.assertTrue(v["complete"] and v["synthetic"])
        self.assertTrue(v["meaning"].startswith("SYNTHETIC TEST RUN, NOT A FINDING"))
        for req in ("R1", "R2", "R5_world", "R6"):
            r = A.read_json(A.critic_dir(world, SEED) / "scores" / f"{req}.json")
            self.assertEqual((r["prereg_sha256"], r["synthetic"]), (sha, True))


if __name__ == "__main__":
    unittest.main()
