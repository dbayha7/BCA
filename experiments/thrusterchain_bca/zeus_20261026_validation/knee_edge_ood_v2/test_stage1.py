"""Stage 1 (A4) tests on Windows: the decision rules on constructed anchors (including Part 8.2(iii)'s example), the
evaluation computations with a stand-in actor, the worker's control flow with a fake rerun1 runtime, and the
supervisor end to end with a fake dispatcher (sealing, R0 stops, re-attempts). No checkpoint is opened, no model is
trained and no benchmark seed is used."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import numpy as np

import fixtures as F
import qpi
import stage1
import stage1_eval as E
import stage1_math as SM
import stage1_worker as W
import step03
import step04
import units as U
from core import sha_file
from knee_region import GATED_PAST_KNEE, KNEE, S9, SLOTS


# ------------------------------------------------------------------------------------------------ decision rules

def seed_inputs(n=48, *, kc_c05=1.0, kc_r05=0.0, rho05=1.0, rho10=1.5, kc_c10=1.5, item5=True, valid=True,
                r_deep_c=0.30, r_deep_r=0.05, dc_c=3.0, dc_r=0.5, dc_swap=0.2):
    def run(kc05, kc10, r_deep, dc):
        return {"valid": np.full(n, valid), "item5_ok": np.full(n, item5), "rho_s05": np.full(n, rho05),
                "rho_s10": np.full(n, rho10), "rho_max_p10": np.full(n, -0.5), "kc_s05": np.full(n, kc05, float),
                "kc_s10": np.full(n, kc10, float), "r_deep_rate": r_deep, "gain_e1": 120.0, "delta_clip": dc,
                "a123_past_rate": 0.0}
    return run(kc_c05, kc_c10, r_deep_c, dc_c), run(kc_r05, 0.0, r_deep_r, dc_r), {"delta_clip": dc_swap, "a123_past_rate": 0.0}


class Rules(unittest.TestCase):
    def test_clean_pass(self):
        out = SM.seed_rules(*seed_inputs())
        self.assertTrue(out["validity_ok"], out)
        self.assertTrue(all(out[k]["ok"] for k in SM.CRITERIA), out)
        self.assertEqual(SM.label(v1_reasons=[], r0={"adequacy_ok": True, "fail_early": False},
                                  per_seed={s: out for s in range(3)})[0], "PASS")

    def test_part_8_2_iii_example_withholds_reading_2(self):
        # 47 common anchors with rho_hat_C(s05) = 1.0: 20 (KC_R -0.2, KC_C 0.9), 4 (0, 0.4), 23 (+0.45, 0.9); one invalid.
        c, r, swap = seed_inputs()
        kc_r = np.r_[np.full(20, -0.2), np.zeros(4), np.full(23, 0.45), 0.0]
        kc_c = np.r_[np.full(20, 0.9), np.full(4, 0.4), np.full(23, 0.9), 0.0]
        c.update(kc_s05=kc_c, valid=np.r_[np.ones(47, bool), False])
        r.update(kc_s05=kc_r, rho_s05=np.full(48, 1.0))
        out = SM.seed_rules(c, r, swap)
        self.assertAlmostEqual(out["V0d"]["k1a_reach_i"], 0.9)
        self.assertAlmostEqual(out["V0d"]["k1a_reach_ii"], 0.9)
        self.assertAlmostEqual(out["K1a"]["median_kc_s05_C_minus_R"], 0.45)
        self.assertFalse(out["K1a"]["ok"])
        self.assertAlmostEqual(out["companions"]["k1a_with_positive_random_kc_zeroed"], 0.90)
        self.assertFalse(out["reading2"]["b"])
        per = {s: out for s in range(3)}
        label, _, reading = SM.label(v1_reasons=[], r0={"adequacy_ok": True, "fail_early": False}, per_seed=per)
        self.assertEqual(label, "FAIL-consistent")
        self.assertIsNone(reading)                                   # K1c passes, K1a fails, but (b) withholds

    def test_minus_infinity_convention_and_floor(self):
        c, r, swap = seed_inputs()
        c["item5_ok"] = np.r_[np.ones(20, bool), np.zeros(28, bool)]           # 28 non-C-valid enter as -inf
        out = SM.seed_rules(c, r, swap)
        self.assertEqual(out["V0d"]["minus_infinity_entries"], 28)
        self.assertEqual(out["V0d"]["k1c_reach_median_rho_C_s10"], float("-inf"))
        self.assertFalse(out["V0d"]["ok"] or out["V0c"]["ok"])
        label = SM.label(v1_reasons=[], r0={"adequacy_ok": True, "fail_early": False}, per_seed={s: out for s in range(3)})[0]
        self.assertEqual(label, "INCONCLUSIVE")

    def test_v0b_random_kc_band(self):
        out = SM.seed_rules(*seed_inputs(kc_r05=0.5))                 # 0.5 > 0.4 x (1.0 + 0.10)
        self.assertFalse(out["V0b"]["ok"])

    def test_k1c_relative_clause_uses_c_valid(self):
        c, r, swap = seed_inputs(kc_c10=0.6, rho10=1.5)               # absolute passes; 0.6 < 0.5 x 1.5 fails
        out = SM.seed_rules(c, r, swap)
        self.assertEqual(out["K1c"]["failed_clauses"], ["relative"])

    def test_precedence_and_readings(self):
        r0 = {"adequacy_ok": True, "fail_early": False}
        good = SM.seed_rules(*seed_inputs())
        self.assertEqual(SM.label(v1_reasons=["x"], r0=r0, per_seed={})[0], "INCOMPLETE")
        self.assertEqual(SM.label(v1_reasons=[], r0={"adequacy_ok": False, "fail_early": False}, per_seed={})[0], "INCONCLUSIVE")
        self.assertEqual(SM.label(v1_reasons=[], r0={"adequacy_ok": True, "fail_early": True}, per_seed={})[0], "FAIL-early")
        k1c_fail = SM.seed_rules(*seed_inputs(kc_c10=0.1))
        self.assertEqual(SM.label(v1_reasons=[], r0=r0, per_seed={s: k1c_fail for s in range(3)})[2], 1)
        k2_fail = SM.seed_rules(*seed_inputs(r_deep_c=0.1))
        self.assertEqual(SM.label(v1_reasons=[], r0=r0, per_seed={s: k2_fail for s in range(3)})[:3:2], ("FAIL-consistent", 3))
        mixed = {0: k2_fail, 1: good, 2: good}
        self.assertEqual(SM.label(v1_reasons=[], r0=r0, per_seed=mixed), ("FAIL-mixed", ["seed 0: K2 fails"], None))
        # K2 fails everywhere but K1a fails in one seed: the precondition for reading 3 is not met.
        k1a_k2 = SM.seed_rules(*seed_inputs(kc_c05=0.2, r_deep_c=0.1))
        self.assertIsNone(SM.label(v1_reasons=[], r0=r0, per_seed={0: k1a_k2, 1: k2_fail, 2: k2_fail})[2])
        k3c = SM.seed_rules(*seed_inputs(dc_swap=2.5))
        self.assertEqual(SM.label(v1_reasons=[], r0=r0, per_seed={s: k3c for s in range(3)})[2], 6)


# ------------------------------------------------------------------------------------------------ evaluation

class StandIn:
    """A stand-in policy: the fixture actor and a smooth fake critic."""

    def __init__(self):
        self.act = F.toy_actor()
        self.calls = 0

    def q(self, obs, actions):
        a = np.asarray(actions, np.float64)
        q1 = -30 + 2 * a[:, 0] - a[:, 1] ** 2
        return np.stack([q1, q1 - 0.1], axis=1).astype(np.float32)

    def assert_unchanged(self):
        pass


class Evaluation(unittest.TestCase):
    def test_deploy_variants(self):
        raw = np.asarray([[-0.95, 0.9, -0.3, 0.1]], np.float32)
        self.assertEqual(E.deploy(raw, "clip")[0, 0], np.float32(-KNEE))
        self.assertEqual(E.deploy(raw, "clip")[0, 1], np.float32(0.9))
        self.assertTrue(np.all(np.abs(E.deploy(raw, "clipall")) <= np.float32(KNEE)))
        with self.assertRaises(ValueError):
            E.deploy(raw.astype(np.float64), "pi")

    def test_bank_states_match_the_environment_reset(self):
        states = E.bank_states("r0_e2_starts")[:4]
        obs = step04.bank_starts("r0_e2_starts")[:4]
        self.assertTrue(np.array_equal(states.astype(np.float32), obs[:, :13]))

    def test_rollout_return_equals_qpi_of_the_first_action(self):
        budget = W.Budget(10 ** 7, 600)
        states = E.bank_states("r0_e2_starts")[:3]
        forward = F.toy_actor()
        roll = E.rollouts(states, forward, "stall3", "pi", budget)
        obs0 = np.concatenate([states, np.ones((3, 1))], axis=1).astype(np.float32)
        first = forward(obs0)[:, None, :]
        values, _, _ = qpi.raw_values_from_states(states, np.zeros(3, np.int64), first, forward, "stall3", clip_all=False)
        self.assertTrue(np.allclose(roll["returns"], values[:, 0], rtol=1e-12, atol=1e-9))
        self.assertTrue(np.array_equal(roll["executed"][:, 0], first[:, 0]))

    def test_clip_changes_only_past_knee_commands(self):
        budget = W.Budget(10 ** 7, 600)
        states = E.bank_states("r0_e2_starts")[:2]
        forward = lambda obs: np.tile(np.asarray([[-0.95, 0.2, 0.0, -0.1]], np.float32), (len(obs), 1))
        pi = E.rollouts(states, forward, "stall3", "pi", budget)
        clip = E.rollouts(states, forward, "stall3", "clip", budget)
        rates = E.step_rates(pi["executed"], pi["obs"])
        self.assertEqual(rates["r_deep_minus"], 1.0)
        self.assertEqual(E.step_rates(clip["executed"], clip["obs"])["r_deep_minus"], 0.0)
        self.assertTrue(np.all(clip["returns"] > pi["returns"]))              # stall slope 3 thrust loss
        ns = E.rollouts(states, forward, "nostall", "pi", budget)
        self.assertFalse(np.allclose(ns["returns"], pi["returns"]))

    def test_visited_rule(self):
        picks = E.visited_indices(12345)
        self.assertEqual({t: len(v) for t, v in picks.items()}, {0: 16, 25: 16, 50: 16})
        self.assertTrue(all(len(set(v)) == 16 and v.max() < 128 for v in picks.values()))
        self.assertTrue(all(np.array_equal(picks[t], E.visited_indices(12345)[t]) for t in picks))

    def test_region_disagreement_stops_the_unit(self):
        a = np.asarray([[-0.95, 0.75, 0.0, 0.85]], np.float32)
        mask, side, deep = E.classify(a)
        self.assertTrue(mask[0] and side[0] == -1 and deep[0])
        self.assertTrue(E.a123_past(a)[0])                                  # a3 = 0.85 is past the knee
        self.assertFalse(E.a123_past(np.asarray([[-0.95, 0.75, 0.0, 0.8]], np.float32))[0])
        with mock.patch.object(E.second, "withheld_region", lambda x: (np.zeros(x.shape[:-1], bool), np.zeros(x.shape[:-1], np.int8))):
            with self.assertRaises(ValueError):
                E.step_rates(a[None], np.zeros((1, 1, 14), np.float32))

    def test_stall_equals_nostall_detects_a_difference(self):
        names = list(E.GATED_SLOTS)
        q = np.tile(np.arange(len(names), dtype=float), (3, 1))
        out = {"gated_slots": names, "nostall_slots": names, "qpi_clip": q, "qpi_nostall": q.copy()}
        self.assertTrue(E.stall_equals_nostall(out)[0])
        out["qpi_nostall"][1, names.index("mid_m40") if "mid_m40" in names else 0] += 1e-6
        self.assertFalse(E.stall_equals_nostall(out)[0])

    def test_target_smoothing_rate_matches_monte_carlo(self):
        rng = np.random.Generator(np.random.PCG64(3))
        mu = rng.uniform(-1, 1, 400)
        noise = np.clip(rng.normal(0, 0.2, (4000, 400)), -0.5, 0.5)
        mc = float(np.mean(np.abs(np.clip(mu + noise, -1, 1)) > KNEE))
        self.assertAlmostEqual(E.target_smoothing_rate(mu, 0.2, 0.5), mc, delta=3e-3)

    def test_anchor_values_and_the_equality_check(self):
        anchors = [step04.build_anchor("stage1_e2_anchors", i) for i in (0, 20)]
        budget = W.Budget(10 ** 7, 600)
        out = E.anchor_values(StandIn(), anchors, "stall3", budget, stall_trained=True)
        self.assertEqual(out["qpi_clip"].shape, (2, len(E.GATED_SLOTS)))
        self.assertEqual(out["own_qpi_clip"].shape, (2, len(E.GATED_SLOTS) + 1))
        self.assertEqual(out["a1_q1"].shape, (2, 4))
        self.assertTrue(out["valid"].all())
        entry = W.sanitize({"anchors_e2": out, "anchors_e1": out})
        self.assertEqual(stage1.equality_check(entry)[0], True)          # the clipped continuation never passes the knee
        idx = out["gated_slots"].index("s10")
        kc = SM.kc(out["q1"], out["qpi_clip"], out["gated_slots"], "s10")
        k = out["gated_slots"].index("edge_knee")
        self.assertTrue(np.allclose(kc, (out["q1"][:, idx] - out["q1"][:, k]) - (out["qpi_clip"][:, idx] - out["qpi_clip"][:, k])))


# ------------------------------------------------------------------------------------------------ worker control flow

class FakeRuntime:
    """Counts like rerun1's runtime: one critic update per step, phase = critic % 2, calibration at 10k and 15k."""

    def __init__(self):
        self.steps = 0

    @staticmethod
    def host(value):
        return json.loads(json.dumps(value, default=step03._plain))

    def state(self, critic):
        return SimpleNamespace(critic=critic, microstep=SimpleNamespace(phase=critic % 2, rng=critic,
                               agent_state=SimpleNamespace(actor=SimpleNamespace(params=f"actor{critic}"))),
                               aux=SimpleNamespace(posterior="ref"))

    def step(self, adapter, state):
        self.steps += 1
        nxt = self.state(state.critic + 1)
        refresh = int(nxt.critic >= 10_000) + int(nxt.critic >= 15_000)
        return nxt, {"counts": {"critic": nxt.critic, "refresh_accepted": refresh}, "refresh_due": nxt.critic in (10_000, 15_000)}

    def validate(self, adapter, state):
        return {"counts": {"critic": state.critic, "refresh_accepted": int(state.critic >= 10_000) + int(state.critic >= 15_000)},
                "checks": {"ok": True}}

    def tree_digest(self, value):
        return hashlib.sha256(repr(getattr(value, "critic", value)).encode()).hexdigest()

    def match(self, adapter, state):
        return {"distance_coefficient": 1.5, "reference_digest": self.tree_digest("ref"),
                "anchor_actor_digest": self.tree_digest(state.microstep.agent_state.actor.params)}, None, None

    def restore_exact(self, adapter, template, raw):
        return self.state(int(raw.decode()))

    def branch(self, adapter, state, protocol, seed_index, arm, coefficient):
        assert arm == "native" and coefficient == 1.5
        return SimpleNamespace(name="branch", sample_indices=adapter.sample_indices, context=adapter.context), state


class WorkerFlow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rt = FakeRuntime()
        context = SimpleNamespace(initial_parameter_hashes={"actor": self.rt.tree_digest("actor0")}, normalization={"m": 1},
                                  runtime={"device_kinds": [step03.K.RECORDED_DEVICE_KIND]})
        adapter = SimpleNamespace(context=context, sample_indices=lambda rng: np.arange(4) + int(rng))
        audits = []
        audit = SimpleNamespace(audit=lambda a, s, p: audits.append(s.critic) or {"per_row": {}, "batches": [], "critic": s.critic})
        self.audits = audits

        def save_state(runtime, out, name, state):
            raw = str(state.critic).encode()
            return {"state": W.write_bytes(out / f"state_{name}.msgpack", raw), "state_digest": runtime.tree_digest(state),
                    "actor_digest": "x"}
        self.patches = [mock.patch.object(step03, "rerun1", lambda: (self.rt, None, audit, None)),
                        mock.patch.object(W, "load_world", lambda job, world: (None, None)),
                        mock.patch.object(W, "bind", lambda *a: (adapter, self.rt.state(0), ["asserted"], {"world": "stall3"}, np.arange(768))),
                        mock.patch.object(W, "save_state", save_state),
                        mock.patch.object(W, "audit_summary", lambda result: {"critic": result["critic"]}),
                        mock.patch.object(W.B, "assert_adapter", lambda adapter, expected: ["branch asserted"])]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_warmup_then_branch(self):
        warm_dir = self.root / "warm"
        warm_dir.mkdir()
        job = {"id": "warmup_s0_cutoff_stall3", "kind": "warmup", "seed_index": 0, "arm": "cutoff", "world": "stall3"}
        warm = W.warmup(job, warm_dir, W.Deadline(600))
        self.assertEqual((warm["counts"]["critic"], self.audits), (10_000, [10_000]))
        self.assertEqual(warm["minibatches"]["phase0_batches"], 5_000)
        (warm_dir / "report.json").write_text(json.dumps({**W.sanitize(warm), "status": "completed"}), encoding="utf-8")
        out = self.root / "branch"
        out.mkdir()
        rep = W.branch({**job, "id": "branch_s0_cutoff_stall3", "kind": "branch", "warmup_dir": str(warm_dir)}, out, W.Deadline(600))
        self.assertEqual(self.audits, [10_000, 15_000, 20_000])
        self.assertEqual(sorted(rep["checkpoints"]), ["015000", "020000"])
        self.assertNotEqual(rep["minibatches"]["phase0_minibatch_sha256"], warm["minibatches"]["phase0_minibatch_sha256"])
        self.assertEqual(rep["branch_assertions"], ["branch asserted"])

    def test_tampered_checkpoint_refused(self):
        warm_dir = self.root / "warm"
        warm_dir.mkdir()
        warm = W.warmup({"id": "w", "kind": "warmup", "seed_index": 0, "arm": "cutoff", "world": "stall3"}, warm_dir, W.Deadline(600))
        (warm_dir / "report.json").write_text(json.dumps({**W.sanitize(warm), "status": "completed"}), encoding="utf-8")
        (warm_dir / "state_010000.msgpack").write_bytes(b"9999")
        out = self.root / "b"
        out.mkdir()
        with self.assertRaises(ValueError):
            W.branch({"id": "b", "kind": "branch", "seed_index": 0, "arm": "cutoff", "world": "stall3",
                      "warmup_dir": str(warm_dir)}, out, W.Deadline(600))

    def test_fingerprint_covers_exactly_the_code_a_kind_runs(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            for m in U.WORKER_MODULES:
                shutil.copy(W.HERE / f"{m}.py", Path(tmp) / f"{m}.py")
            with mock.patch.object(U, "HERE", Path(tmp)):
                before = {k: U.fingerprint("stage1_worker", k) for k in ("warmup", "branch", "evaluate", "r0")}
                before2 = {k: U.fingerprint("stage2_worker", k) for k in ("precheck", "branch", "evaluate")}
                path = Path(tmp) / "stage1_worker.py"
                text = path.read_text(encoding="utf-8")
                path.write_text(text.replace('result["transitions"] = budget.used\n    return result',
                                             'result["transitions"] = int(budget.used)\n    return result'), encoding="utf-8")
                after = {k: U.fingerprint("stage1_worker", k) for k in before}
                self.assertNotEqual(after["evaluate"], before["evaluate"])
                self.assertEqual([after[k] == before[k] for k in ("warmup", "branch", "r0")], [True, True, True])
                path.write_text(text.replace('    digest, batches, metrics = hashlib.sha256(), 0, None',
                                             '    digest, batches, metrics = hashlib.sha256(), int(0), None'), encoding="utf-8")
                after = {k: U.fingerprint("stage1_worker", k) for k in before}
                self.assertNotEqual(after["warmup"], before["warmup"])         # train() is shared by warmup and branch
                self.assertNotEqual(after["branch"], before["branch"])
                self.assertEqual(after["r0"], before["r0"])
                self.assertNotEqual(U.fingerprint("stage2_worker", "branch"), before2["branch"])   # reaches W1.train
                self.assertEqual(U.fingerprint("stage2_worker", "precheck"), before2["precheck"])
                path.write_text(text.replace('def deploy(raw, variant):', 'def deploy(raw, variant):  # comment only')
                                if False else text + "\n# a trailing comment\n", encoding="utf-8")
                self.assertEqual({k: U.fingerprint("stage1_worker", k) for k in before}, before)   # comments are not code
                path.write_text(text.replace('    """JSON-safe: arrays to lists', '    """JSON safe: arrays to lists'), encoding="utf-8")
                changed = {k: U.fingerprint("stage1_worker", k) for k in before}
                self.assertTrue(all(changed[k] != before[k] for k in before))   # run_job/sanitize write every report
                path.write_text(text.replace("def audit_summary(result):", "@staticmethod\ndef audit_summary(result):"),
                                encoding="utf-8")
                self.assertNotEqual(U.fingerprint("stage1_worker", "warmup"), before["warmup"])   # decorators count
                path.write_text(text.replace("    digest, batches, metrics = hashlib.sha256(), 0, None",
                                             "    digest, batches, metrics = hashlib.sha256(), 0, None\n    _n = [progress for progress in ()]"),
                                encoding="utf-8")
                base = U.fingerprint("stage1_worker", "warmup")
                p2 = path.read_text(encoding="utf-8")
                path.write_text(p2.replace("def progress(out, record):\n", "def progress(out, record):\n    record = dict(record)\n"),
                                encoding="utf-8")
                self.assertNotEqual(U.fingerprint("stage1_worker", "warmup"), base)   # a comprehension local hides nothing

    def test_bc_source_nodes_and_sanitize(self):
        import ast
        tree = ast.parse(W.BC_SOURCE.read_text(encoding="utf-8"))
        names = {getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and hasattr(n.targets[0], "id") else None)
                 for n in tree.body}
        self.assertTrue(set(W.BC_NODES) <= names)
        self.assertEqual(W.sanitize({"a": np.asarray([1.0, np.inf]), 2: np.float32(np.nan), "b": np.bool_(True)}),
                         {"a": [1.0, "positive_infinity"], "2": "nan", "b": True})

    def test_audit_summary(self):
        rows = {"high_load": [True, True, False], "actor_actions": [[-0.9, 0, 0, 0], [0.1, 0, 0, 0], [0, 0, 0, 0]],
                "action_loss_gradients": {"q": [[0.2, 0, 0, 0], [-0.1, 0, 0, 0], [0, 0, 0, 0]],
                                          "bc": [[-0.05, 0, 0, 0], [0.0, 0, 0, 0], [0, 0, 0, 0]]}}
        with mock.patch.object(W, "audit_summary", W.audit_summary.__wrapped__ if hasattr(W.audit_summary, "__wrapped__") else W.audit_summary):
            pass
        out = self.patches[4].temp_original({"per_row": rows, "batches": [{"lambda_from_unpenalized_Q1": 0.01}]})
        self.assertAlmostEqual(out["median_reach_radius_high_load"], 0.3)
        self.assertEqual(out["actor_past_knee_minus_high_load"], 1)
        self.assertEqual(out["share_bc_pulls_back_when_past_knee"], 1.0)


# ------------------------------------------------------------------------------------------------ supervisor

def synthetic_anchors(arm, world):
    """Gated E2 anchors where CUTOFF over-values past the knee and RANDOM does not (a clean PASS)."""
    names, ns = list(E.GATED_SLOTS), list(E.NOSTALL_SLOTS)
    drop = {"s05": 1.0, "s10": 1.5, "s15": 2.0, "s20": 2.5, "p10": -0.5}
    smooth = {n: -30 + 2 * SLOTS[n] for n in names}
    qpi = np.asarray([[smooth[n] - drop.get(n, 0.0) for n in names]] * 64)
    q1 = (np.asarray([[smooth[n] for n in names]] * 64) if arm == "cutoff" else qpi + 25.0)
    out = {"gated_slots": names, "qpi_clip": qpi, "q1": q1, "q2": q1 - 0.1, "qmin": q1 - 0.1, "valid": np.ones(64, bool),
           "times": np.repeat([0, 25, 50, 75], 16)}
    if world == "stall3":
        out.update(nostall_slots=ns, qpi_nostall=np.asarray([[smooth[n] for n in ns]] * 64))
    return out


def synthetic_eval(i):
    runs = {}
    for world in stage1.WORLDS:
        for arm in stage1.ARMS:
            stall = world == "stall3"
            rates = {"r_deep_minus": 0.3 if arm == "cutoff" else 0.05, "a123_past_knee": 0.0}
            dc = 3.0 if arm == "cutoff" else 0.5
            own = {"e2/pi": {"J": 100.0}, "e1/pi": {"J": 200.0}, "e2/delta_clip": dc, "e2/delta_clipall": dc, "e2_rates": rates}
            entry = {"own_world": own, "anchors_e2": synthetic_anchors(arm, world), "visited": {
                "slots": list(E.VISITED_SLOTS), "qpi_clip": np.zeros((48, 6)), "q1": np.zeros((48, 6))},
                "target_smoothing_w_rate": {"rate": 0.3}, "visited_vs_training_smd": [0.0] * 14}
            entry["anchors_e1"] = entry["anchors_e2"]
            if stall:
                entry["nostall_world"] = {"e2/pi": {"J": 104.0}, "e2/delta_clip": 0.2, "e2_rates": {"a123_past_knee": 0.0}}
            runs[stage1.run_name(i, arm, world)] = {"J_E1_untrained": 50.0, "020000": entry,
                                                    "015000": {"own_world": own}}
    return W.sanitize({"kind": "evaluate", "seed_index": i, "runs": runs, "transitions": 1000})


class Integrity(unittest.TestCase):
    def test_record_and_verify(self):
        import integrity
        import stage1_worker
        record = U.integrity_record()
        self.assertIn(str(Path(stage1_worker.__file__).resolve()), record["files"])
        self.assertIn(str(step03.RERUN1 / "runtime.py"), record["files"])
        with tempfile.TemporaryDirectory() as root:
            job = Path(root) / "job.json"
            job.write_text(json.dumps({"id": "x", "integrity": record}), encoding="utf-8")
            argv = ["stage1_worker.py", "--job", str(job)]
            self.assertEqual(integrity.verify(argv, stage1_worker.__file__), len(record["files"]))
            self.assertIsNone(integrity.verify(["stage1_worker.py"], stage1_worker.__file__))
            bad = {**record, "files": {**record["files"], str(Path(stage1_worker.__file__).resolve()): "0" * 64}}
            job.write_text(json.dumps({"id": "x", "integrity": bad}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                integrity.verify(argv, stage1_worker.__file__)
            job.write_text(json.dumps({"id": "x", "integrity": {**record, "package_dir": root}}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                integrity.verify(argv, stage1_worker.__file__)
            job.write_text(json.dumps({"id": "x"}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                integrity.verify(argv, stage1_worker.__file__)


class FakeDispatch:
    def __init__(self, fail=(), r0=None, lie=(), extra=None):
        self.fail, self.r0, self.seen = set(fail), r0 or {"adequacy_ok": True, "fail_early": False}, []
        self.lie, self.extra = set(lie), extra or {}          # lie: the worker writes COMPLETE but exits nonzero

    def __call__(self, job_dir, cap):
        job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        self.seen.append(job["id"])
        body = {"preflight": {}, "warmup": {"gradient_audits": {}, "device_kinds": [step03.K.RECORDED_DEVICE_KIND]},
                "branch": {"gradient_audits": {}}, "bc_only": {},
                "r0": {**self.r0, "seeds": {}, "transitions": 100}}.get(job["kind"])
        if job["kind"] == "evaluate":
            body = synthetic_eval(job["seed_index"])
        ok = job["id"] not in self.fail
        report = {**body, **self.extra.get(job["id"], {}), "job": job["id"], "kind": job["kind"],
                  "status": "completed" if ok else "failed", "unit_label": "COMPLETE" if ok else "INCOMPLETE",
                  "elapsed_seconds": 1.0, "producer_sha256": U.fingerprint("stage1_worker", job["kind"])}
        step03._write_new(job_dir / "report.json", report)
        exit_ok = ok and job["id"] not in self.lie
        return {"exit_code": 0 if exit_ok else 1, "timed_out": False, "interrupted": False, "elapsed_seconds": 1.0}


def S_COUNT(name):
    import seeds
    return seeds.BANKS[name]["count"]


class Supervisor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        a3 = root / "step04_run" / "attempt_1"
        a3.mkdir(parents=True)
        step03._write_new(a3 / "collection.json", {"unit_label": "COMPLETE"})
        step03._write_new(a3 / "report.json", {"label": "PASS", "approval": {"sealed_units": {},
                                                                             "manifest_sha256": stage1.A3_MANIFEST_SHA256}})
        manifest = {"files": {n: sha_file(stage1.HERE / n) for n in sorted(stage1.A4_REQUIRED_FILES)}}
        (root / "A4_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.root = root
        self.patches = [mock.patch.object(stage1, "RUN", root / "stage1_run"), mock.patch.object(stage1, "MANIFEST", root / "A4_MANIFEST.json"),
                        mock.patch.object(stage1, "APPROVAL", root / "A4_APPROVAL.json"), mock.patch.object(step04, "RUN", root / "step04_run")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def approve(self, attempt, ack=True):
        words = stage1.APPROVAL_WORDS + ", attempt " + str(attempt) + (". " + stage1.PREDICTION_ACK if ack else "")
        (self.root / "A4_APPROVAL.json").write_text(json.dumps({"approval": words, "attempt": attempt,
                                                                "package_manifest_sha256": sha_file(self.root / "A4_MANIFEST.json")}),
                                                    encoding="utf-8")

    def test_pass_end_to_end(self):
        self.approve(1)
        dispatch = FakeDispatch()
        report = stage1.execute(provenance="smoke", dispatch=dispatch)
        self.assertEqual(report["label"], "PASS", report.get("reasons"))
        self.assertEqual(dispatch.seen, ["preflight"] + stage1.JOB_IDS)
        self.assertTrue(report["a5_precondition"]["ok"])

    def test_missing_acknowledgement_refused(self):
        self.approve(1, ack=False)
        with self.assertRaises(ValueError):
            stage1.execute(provenance="smoke", dispatch=FakeDispatch())

    def test_fail_early_skips_branches(self):
        self.approve(1)
        dispatch = FakeDispatch(r0={"adequacy_ok": True, "fail_early": True})
        report = stage1.execute(provenance="smoke", dispatch=dispatch)
        self.assertEqual((report["label"], report["reading"]), ("FAIL-early", SM.FAIL_EARLY_READING))
        self.assertFalse(any(j.startswith("branch_") for j in dispatch.seen))

    def test_reattempt_runs_only_unsealed_units_and_second_failure_is_inconclusive(self):
        self.approve(1)
        first = stage1.execute(provenance="smoke", dispatch=FakeDispatch(fail={"branch_s1_random_nostall"}))
        self.assertEqual((first["label"], first["failed_unit"]), ("INCOMPLETE", "branch_s1_random_nostall"))
        self.approve(2)
        again = FakeDispatch(fail={"branch_s1_random_nostall"})
        second = stage1.execute(provenance="smoke", dispatch=again)
        self.assertEqual(again.seen, ["preflight", "branch_s1_random_nostall"])
        self.assertEqual(second["label"], "INCONCLUSIVE")
        with self.assertRaises(ValueError):                           # no attempt after INCONCLUSIVE
            self.approve(3)
            stage1.execute(provenance="smoke", dispatch=FakeDispatch())

    def test_reattempt_completes(self):
        self.approve(1)
        stage1.execute(provenance="smoke", dispatch=FakeDispatch(fail={"evaluate_s2"}))
        self.approve(2)
        dispatch = FakeDispatch()
        report = stage1.execute(provenance="smoke", dispatch=dispatch)
        self.assertEqual(dispatch.seen, ["preflight", "evaluate_s2"])
        self.assertEqual(report["label"], "PASS", report.get("reasons"))

    def test_smoke_driver_runs_every_worker_kind_with_test_seeds(self):
        import stage1_smoke
        dispatch = FakeDispatch()
        out = self.root / "smoke"
        result = stage1_smoke.main(out, dispatch=dispatch, collect_fn=lambda d: None)
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(dispatch.seen, ["preflight"] + [j["id"] for j in stage1_smoke.JOBS])
        job = json.loads((out / "evaluate_s0" / "job.json").read_text(encoding="utf-8"))
        self.assertEqual((job["provenance"], job["smoke"]["anchors"]), ("smoke", stage1_smoke.ANCHORS))
        self.assertTrue(result["parse_checks"]["v1_stall_equals_nostall"] and result["device_kinds_ok"])
        self.assertEqual(result["parse_checks"]["anchor_times"], [0, 25, 50, 75])
        self.assertTrue(result["a4_training_projection"]["within_training_cap"])
        self.assertNotIn("seeds", result.get("r0_structure", {}))             # blinded: no outcome values
        self.assertTrue((out / "SMOKE_RESULT.json").is_file())
        with self.assertRaises(ValueError):
            stage1_smoke.main(stage1.HERE / "smoke_inside", dispatch=dispatch, collect_fn=lambda d: None)

    def test_smoke_streams_are_test_only_with_the_sealed_structure(self):
        import stage1_smoke
        for name in ("stage1_e2_bank", "stage1_e1_bank", "r0_e2_starts", "stage1_e2_anchors", "stage1_e1_anchors"):
            for i in (0, 5, S_COUNT(name) - 1):
                sealed, smoke = E.stream_item(name, i), E.stream_item(name, i, stage1_smoke.STREAM_SEED)
                self.assertNotEqual(sealed["reset_seed"], smoke["reset_seed"])
                self.assertEqual(sealed.get("t"), smoke.get("t"))
                if name.startswith(("stage1_e2", "r0_e2")):
                    self.assertEqual(tuple(smoke["current"]), (0.9, 1.1))
        anchor = E.build_anchor("stage1_e2_anchors", 16, stage1_smoke.STREAM_SEED)
        self.assertEqual((anchor["t"], anchor["prefix_actions"].shape), (25, (25, 4)))
        self.assertTrue(np.array_equal(qpi.raw_anchor_state(anchor), anchor["state"]))
        self.assertTrue(np.array_equal(E.build_anchor("stage1_e2_anchors", 3)["state"], step04.build_anchor("stage1_e2_anchors", 3)["state"]))
        self.assertTrue(stage1_smoke.STREAM_SEED not in stage1_smoke.sealed_values())

    def test_worker_refuses_smoke_reductions_in_research(self):
        with self.assertRaises(ValueError):
            W.smoke_options({"provenance": "research", "smoke": {"seeds": [0]}})
        with self.assertRaises(ValueError):
            W.smoke_options({"provenance": "research", "visited_seed": 99})
        with self.assertRaises(ValueError):
            W.smoke_options({"provenance": "smoke", "smoke": {"anchors": 4}})          # smoke needs test seeds
        seeded = {"provenance": "smoke", "test_seeds": F.TEST_SEEDS, "training_seeds": [{}, {}, {}], "visited_seed": 1}
        with self.assertRaises(ValueError):
            W.smoke_options({**seeded, "kind": "evaluate", "smoke": {"runs": []}})    # evaluation needs test-only streams
        self.assertEqual(W.smoke_options({**seeded, "kind": "evaluate", "smoke": {"stream_seed": 5}}), {"stream_seed": 5})
        self.assertEqual(W.smoke_options({"provenance": "research"}), {})

    def test_supervisor_verdict_governs_sealing(self):
        self.approve(1)
        first = stage1.execute(provenance="smoke", dispatch=FakeDispatch(lie={"warmup_s0_random_stall3"}))
        self.assertEqual(first["failed_unit"], "warmup_s0_random_stall3")
        self.approve(2)
        dispatch = FakeDispatch()
        stage1.execute(provenance="smoke", dispatch=dispatch)
        self.assertIn("warmup_s0_random_stall3", dispatch.seen)          # not sealed: the worker's own COMPLETE is not enough

    def test_analysis_crash_is_counted_not_exempt(self):
        self.approve(1)
        with mock.patch.object(stage1, "decide", side_effect=KeyError("e2/delta_clip")):
            report = stage1.execute(provenance="smoke", dispatch=FakeDispatch())
        self.assertEqual((report["label"], report["failed_unit"], report["counts_under_5_0"]), ("INCOMPLETE", "decision", True))
        self.assertIn("traceback", report)

    def test_sealed_unit_from_changed_code_must_reproduce(self):
        self.approve(1)
        stage1.execute(provenance="smoke", dispatch=FakeDispatch(fail={"evaluate_s2"}))
        sealed = stage1.RUN / "attempt_1" / "warmup_s1_cutoff_stall3" / "report.json"
        old = json.loads(sealed.read_text(encoding="utf-8"))
        old["producer_sha256"] = {"stage1_worker.py": "older code"}
        sealed.write_text(json.dumps(old), encoding="utf-8")
        self.approve(2)
        differs = FakeDispatch(extra={"warmup_s1_cutoff_stall3": {"minibatches": "different"}})
        report = stage1.execute(provenance="smoke", dispatch=differs)
        self.assertEqual(differs.seen, ["preflight", "warmup_s1_cutoff_stall3"])
        self.assertEqual((report["label"], report["failed_unit"]), ("INCOMPLETE", "warmup_s1_cutoff_stall3"))
        self.assertIn("does not reproduce", report["reasons"][0])

    def test_sealed_unit_that_reproduces_is_accepted(self):
        self.approve(1)
        stage1.execute(provenance="smoke", dispatch=FakeDispatch(fail={"evaluate_s2"}))
        sealed = stage1.RUN / "attempt_1" / "warmup_s1_cutoff_stall3" / "report.json"
        old = json.loads(sealed.read_text(encoding="utf-8"))
        old["producer_sha256"] = {"stage1_worker.py": "older code"}
        sealed.write_text(json.dumps(old), encoding="utf-8")
        self.approve(2)
        dispatch = FakeDispatch()
        report = stage1.execute(provenance="smoke", dispatch=dispatch)
        self.assertEqual(report["label"], "PASS", report.get("reasons"))
        self.assertEqual(report["reproduced_units"], ["warmup_s1_cutoff_stall3"])
        self.assertEqual(report["caps"]["training_seconds"], 27.0)          # each training unit charged once
        self.assertEqual(set(report["unit_dirs"]), set(stage1.JOB_IDS))

    def test_failed_reproduction_is_never_sealed_and_a_repeat_is_inconclusive(self):
        self.test_sealed_unit_from_changed_code_must_reproduce()            # attempt 2: INCOMPLETE, marker written
        marker = stage1.RUN / "attempt_2" / "warmup_s1_cutoff_stall3" / "NOT_REPRODUCED.json"
        self.assertTrue(marker.is_file())
        self.approve(3)
        again = FakeDispatch(extra={"warmup_s1_cutoff_stall3": {"minibatches": "different"}})
        report = stage1.execute(provenance="smoke", dispatch=again)
        self.assertEqual(again.seen, ["preflight", "warmup_s1_cutoff_stall3"])   # compared with the original again
        self.assertEqual(report["label"], "INCONCLUSIVE")                   # second failure, unchanged code

    def test_rejected_reproduction_keeps_the_reference(self):
        self.approve(1)
        stage1.execute(provenance="smoke", dispatch=FakeDispatch(fail={"evaluate_s2"}))
        sealed = stage1.RUN / "attempt_1" / "warmup_s1_cutoff_stall3" / "report.json"
        old = json.loads(sealed.read_text(encoding="utf-8"))
        old["producer_sha256"] = "older code"
        sealed.write_text(json.dumps(old), encoding="utf-8")
        self.approve(2)                                     # the worker writes COMPLETE, the supervisor rejects the run
        stage1.execute(provenance="smoke", dispatch=FakeDispatch(lie={"warmup_s1_cutoff_stall3"},
                                                                 extra={"warmup_s1_cutoff_stall3": {"minibatches": "x"}}))
        self.approve(3)
        again = FakeDispatch(extra={"warmup_s1_cutoff_stall3": {"minibatches": "x"}})
        report = stage1.execute(provenance="smoke", dispatch=again)
        self.assertIn("does not reproduce", " ".join(report["reasons"]))          # still compared with attempt 1
        self.assertIn("minibatches", report["reasons"][0])                        # the differing key is reported

    def test_changed_job_inputs_require_reproduction(self):
        self.approve(1)
        stage1.execute(provenance="smoke", dispatch=FakeDispatch(fail={"evaluate_s2"}))
        job_path = stage1.RUN / "attempt_1" / "warmup_s0_cutoff_stall3" / "job.json"
        job = json.loads(job_path.read_text(encoding="utf-8"))
        job_path.write_text(json.dumps({**job, "seed_index": 1}), encoding="utf-8")       # as if built by other code
        self.approve(2)
        dispatch = FakeDispatch()
        report = stage1.execute(provenance="smoke", dispatch=dispatch)
        self.assertIn("warmup_s0_cutoff_stall3", dispatch.seen)
        self.assertIn("warmup_s0_cutoff_stall3", report["reproduced_units"])

    def test_interrupted_rerun_keeps_the_reference(self):
        self.approve(1)
        stage1.execute(provenance="smoke", dispatch=FakeDispatch(fail={"evaluate_s2"}))
        sealed = stage1.RUN / "attempt_1" / "warmup_s1_cutoff_stall3" / "report.json"
        sealed.write_text(json.dumps({**json.loads(sealed.read_text(encoding="utf-8")), "producer_sha256": "older"}), encoding="utf-8")
        self.approve(2)
        base = FakeDispatch(extra={"warmup_s1_cutoff_stall3": {"minibatches": "x"}})

        def interrupt(job_dir, cap):
            base(job_dir, cap)
            if job_dir.name == "warmup_s1_cutoff_stall3":
                raise KeyboardInterrupt
            return {"exit_code": 0, "timed_out": False, "interrupted": False, "elapsed_seconds": 1.0}
        with self.assertRaises(KeyboardInterrupt):
            stage1.execute(provenance="smoke", dispatch=interrupt)
        self.assertTrue((stage1.RUN / "attempt_2" / "warmup_s1_cutoff_stall3" / "NOT_REPRODUCED.json").is_file())

    def test_differences_report_type_and_sign(self):
        diff = U.differences({"a": 100, "b": -0.0, "c": [1.0, 2.0]}, {"a": 100.0, "b": 0.0, "c": [1.0, 2.5]})
        self.assertEqual(set(diff), {"a", "b", "c"})
        self.assertEqual(diff["c"]["max_abs_difference"], 0.5)
        self.assertIsNotNone(diff["b"]["note"])

    def test_alternating_packages_is_refused(self):
        self.approve(1)
        stage1.execute(provenance="smoke", dispatch=FakeDispatch(fail={"evaluate_s2"}))
        original = (self.root / "A4_MANIFEST.json").read_text(encoding="utf-8")
        (self.root / "A4_MANIFEST.json").write_text(original.replace("{", '{"fix": 1, ', 1), encoding="utf-8")
        self.approve(2)
        stage1.execute(provenance="smoke", dispatch=FakeDispatch(fail={"evaluate_s2"}))
        (self.root / "A4_MANIFEST.json").write_text(original, encoding="utf-8")   # back to the original package
        self.approve(3)
        with self.assertRaises(ValueError):
            stage1.execute(provenance="smoke", dispatch=FakeDispatch())

    def test_preflight_failure_is_exempt(self):
        self.approve(1)
        report = stage1.execute(provenance="smoke", dispatch=FakeDispatch(fail={"preflight"}))
        self.assertEqual((report["label"], report["counts_under_5_0"]), ("INCOMPLETE", False))


if __name__ == "__main__":
    unittest.main()
