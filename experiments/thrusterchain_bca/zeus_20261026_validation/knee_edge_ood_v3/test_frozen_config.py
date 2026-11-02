"""4.8 comparison rules on the saved JSON (no checkpoint is opened), 4.6 host config and cadence, the probe builder,
seeds (derivation, Stage 2 forks, audit can fail) and the sealed expected values."""
from importlib.metadata import PackageNotFoundError
from pathlib import Path
import ast
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import numpy as np

import frozen_checks as K
import probes as P
import reconcile as C
import seeds as S
import worlds
from core import DESIGN, PAYLOAD, RERUN1
from knee_region import SLOTS


class FrozenRules(unittest.TestCase):
    def test_slots_1_to_5_match_every_saved_file(self):
        for path in K.reference_files():
            saved = K.read_reference(path)
            self.assertTrue(K.check_slots_1_to_5(saved["actions"])[0], path.name)
            self.assertTrue(np.all(np.abs(saved["actions"][:, 0]) <= 1))

    def test_saved_lambda_every_cell(self):
        for cell in K.CELLS:
            record = K.saved_lambdas(cell)
            self.assertTrue(np.all((record["values"] > 0) & (record["values"] < 1)), cell)

    def test_identity_comparison_passes_and_perturbations_fail(self):
        saved = K.read_reference(K.reference_files()[0])
        lam = K.saved_lambdas("seed0_full")["values"]
        args = dict(slot0=saved["actions"][:, 0], q1=saved["q1"], twin_min=saved["twin_min"],
                    adjusted=saved["adjusted"], qpi=saved["qpi"], lam=lam, saved_lambda=lam)
        self.assertTrue(K.check_cell(saved, **args)[0])
        self.assertFalse(K.check_cell(saved, **dict(args, qpi=saved["qpi"] * (1 + 1e-8)))[0])
        self.assertFalse(K.check_cell(saved, **dict(args, q1=saved["q1"] + 1e-3 * np.maximum(1, np.abs(saved["q1"]))))[0])
        self.assertFalse(K.check_cell(saved, **dict(args, lam=lam * (1 + 1e-5)))[0])
        with self.assertRaises(ValueError):
            K.check_cell(saved, **dict(args, saved_lambda=None))
        with self.assertRaises(ValueError):
            K.check_cell(dict(saved, adjusted=None), **args)
        self.assertTrue(np.all(np.isfinite(K.width_from_adjusted(saved["q1"], saved["adjusted"]))))

    def test_preflight_exception_rule(self):
        good = lambda native: {"device_kinds": [K.RECORDED_DEVICE_KIND]}
        failures = [lambda native: {"device_kinds": ["CPU"]}, lambda native: {},
                    lambda native: (_ for _ in ()).throw(PackageNotFoundError("jax-cuda12-plugin")),
                    lambda native: (_ for _ in ()).throw(ValueError("Frozen runtime versions differ"))]
        ledger = {"reproduction_values": 0}
        self.assertEqual(K.preflight(good, None, ledger)[0], "PASS")
        for fn in failures:
            label, reason = K.preflight(fn, None, ledger)
            self.assertEqual(label, "INCOMPLETE_PREFLIGHT", reason)
            self.assertFalse(K.failure_counts_under_5_0(label, ledger))
        self.assertEqual(ledger["preflight_failures"], len(failures))
        self.assertTrue(K.failure_counts_under_5_0("INCOMPLETE_PREFLIGHT", {"reproduction_values": 3}))
        self.assertTrue(K.failure_counts_under_5_0("INCOMPLETE", ledger))
        with self.assertRaises(ValueError):
            K.preflight(good, None, {"reproduction_values": 1})


class Probes(unittest.TestCase):
    def test_gated_takes_part0_names_and_catches_swaps(self):
        P.build(3, ("s05", "s10", "s20", "edge_knee", "zero"))
        swapped = P.build(2, ("s10", "s20"))
        with self.assertRaises(ValueError):
            P.assert_probes(swapped, ("s20", "s10"))
        self.assertEqual(P.receipt(("s05",), "gated")["bits"], ["0xbf59999a"])
        for bad in (["s12"], [-0.85]):                                  # unknown names and bare values are refused
            with self.assertRaises(ValueError):
                P.build(3, bad)

    def test_g05_and_reach_forecast_point_sets(self):
        g05 = P.build(2, P.G05_POINTS, kind="g05")
        self.assertEqual(g05.shape, (2, 7, 4))
        self.assertEqual(float(g05[0, 4, 0]), SLOTS["s05"])
        with self.assertRaises(ValueError):
            P.build(2, [-0.86], kind="g05")
        own = np.full((2, 3), 0.3, np.float32)
        P.build(2, P.REACH_POINTS, a123=own, kind="reach_forecast")
        with self.assertRaises(ValueError):
            P.build(2, [-0.95], a123=own, kind="reach_forecast")

    def test_a123_rules(self):
        probes = P.build(2, ("s10",))
        probes[0, 0, 2] = np.float32(0.1)
        with self.assertRaises(ValueError):
            P.assert_probes(probes, ("s10",))
        own = np.full((2, 3), 0.25, np.float32)
        out = P.build(2, ("s05",), a123=own, kind="actor_own")
        self.assertTrue(np.all(out[..., 1:] == np.float32(0.25)))
        with self.assertRaises(ValueError):
            P.build(2, ("s05",), a123=own)

    def test_identity_images_and_known_answer(self):
        built = P.build(1, ("s10",), kind="identity_image", world="stall3")
        image = worlds.identity_shift(np.asarray([SLOTS["s10"]]), "stall3")[0]
        self.assertEqual(built[0, 0, 0], image[0])
        with self.assertRaises(ValueError):
            P.assert_probes(built, ("s10",), kind="identity_image", world="stall15_identity")
        with self.assertRaises(ValueError):
            P.build(1, ("edge_knee",), kind="identity_image", world="stall3")   # inside the knee: no identity
        self.assertEqual(P.known_answer_slots_1_to_5(4).shape, (4, 5, 4))


class HostConfig(unittest.TestCase):
    def test_upstream_defaults_target_and_bindings(self):
        source = (PAYLOAD / "unifloral_td3bc_validation_v1/upstream/algorithms/td3_bc.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        defaults = {n.target.id: ast.literal_eval(n.value) for n in ast.walk(tree)
                    if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None
                    and n.target.id in ("td3_alpha", "policy_noise", "noise_clip", "gamma")}
        self.assertEqual((defaults["td3_alpha"], defaults["policy_noise"], defaults["noise_clip"]), (2.5, 0.2, 0.5))
        self.assertEqual(defaults["gamma"], 0.99)
        self.assertIn("next_q_value = (1.0 - transition.done) * jnp.min(target_q)", source)   # no bootstrap when done
        runtime = (RERUN1 / "runtime.py").read_text(encoding="utf-8").splitlines()
        call = runtime[136] + runtime[137]
        self.assertIn("gamma=.98", call)
        self.assertNotIn("td3_alpha", call)                          # alpha 2.5 and noise 0.2/0.5 stay upstream defaults
        self.assertNotIn("noise", call)
        pin = json.loads((PAYLOAD / "unifloral_td3bc_validation_v1/provenance/unifloral.json").read_text(encoding="utf-8"))
        self.assertTrue(pin["revision"].startswith("f2dc1278"))

    def test_cadence_from_saved_evidence(self):
        protocol = json.loads((RERUN1 / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual((protocol["branch_actor_updates"], protocol["branch_critic_updates"]), (5000, 10000))
        self.assertEqual(protocol["gamma"], 0.98)


class Sealed(unittest.TestCase):
    def test_seeds_rederive_and_stage2_forks(self):
        record = S.load()
        self.assertEqual(len(record["numpy"]), len(S.NUMPY_NAMES))
        training = [v for r in record["training"] for k, v in r.items() if k != "index"]
        forks = [r[k] for r in record["stage2_forks"] for k in ("branch_native_seed", "branch_aux_seed")]
        self.assertTrue(all(1 <= v < 2 ** 31 for v in training + forks))
        self.assertFalse(set(forks) & set(training))
        view = S.stage2_protocol_seeds(record, 1, "cutoff", "stall3")
        fork = [r for r in record["stage2_forks"] if (r["seed_index"], r["arm"], r["world"]) == (1, "cutoff", "stall3")][0]
        self.assertEqual((view[1]["branch_native_seed"], view[1]["branch_aux_seed"]),
                         (fork["branch_native_seed"], fork["branch_aux_seed"]))
        self.assertEqual(view[1]["native_seed"], record["training"][1]["native_seed"])
        self.assertNotEqual(view[1]["branch_native_seed"], record["training"][1]["branch_native_seed"])

    def test_bank_rules_sealed(self):
        item = S.bank_item("stage1_e2_anchors", 40)
        self.assertEqual((item["t"], item["current"]), (50, (0.9, 1.1)))
        self.assertEqual(S.prefix_uniforms("stage1_e2_anchors", 40, 50).shape, (50, 5))
        self.assertIn(S.bank_item("stage1_e1_bank", 3)["current"], ((0.1, 0.7), (0.9, 1.1)))

    def test_audit_can_fail(self):
        record = S.build()
        planted = record["numpy"]["random_control"]
        self.assertIn(planted, S.audit(record, prior={planted}, episodes=5)["raw_collisions"])
        derived = S.audit(record, prior={record["numpy"]["dataset_entropy"]}, episodes=5)
        self.assertTrue(derived["derived_reset_seed_collisions"])
        self.assertFalse(derived["clean"])

    def test_expected_values_are_protocol_literals_and_outward_of_mc(self):
        bounds = C.sealed_bounds()
        b = bounds["bounds"]
        literal = {"share_high": [0.503, 0.647], "share_deep": [0.036, 0.120], "episodes_with_deep": [0.087, 0.260],
                   "cutoff_removed": [26643, 30052], "cutoff_removed_fraction_high": [0.479, 0.509],
                   "cutoff_removed_fraction_low": [0.187, 0.214], "cutoff_high_load_anchor": [-0.733, -0.720],
                   "random_w_retention": [0.460, 0.504], "random_high_load_anchor": [-0.781, -0.768],
                   "l14_pooled_high": [2.00, 2.17], "l14_mid": [2.00, 2.18], "l14_deep": [1.82, 2.24],
                   "calibration_replacements": [17, 78]}
        self.assertEqual(b, literal)
        table = (DESIGN / "screens/random_control_mc_output.txt").read_text(encoding="utf-8").splitlines()[2:]
        rows = {l.split()[0]: [float(x) for x in l.split()[1:]] for l in table if l and not l.startswith("first")}
        for name, mc in bounds["mc_names"].items():
            _, _, lo, hi = rows[mc]
            self.assertLessEqual(b[name][0], lo, name)
            self.assertGreaterEqual(b[name][1], hi, name)
        self.assertEqual(bounds["label_on_miss"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
