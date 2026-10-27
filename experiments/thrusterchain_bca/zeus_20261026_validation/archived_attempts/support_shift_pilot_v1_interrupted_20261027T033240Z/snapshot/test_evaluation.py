"""Synthetic evaluator fixtures: no actual networks, canonical dynamics or resets."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import evaluate_worker as ev


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((Path(__file__).parent / "protocol.json").read_text())
        self.temp = tempfile.TemporaryDirectory(prefix="support_eval_fixture_", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def memory_ledger(self, cap=51200):
        with patch.object(ev.prior.Ledger, "persist", lambda self: None):
            ledger = ev.prior.Ledger(self.path / "progress.json", cap, 600, clock=lambda: 0.)
        ledger.persist = lambda: None
        return ledger

    def data(self):
        returns = {cell+"__"+arm: np.full(32, {"untrained":-10.,"native":-5.,"distance":-4.,"bca":-2.}[arm])
                   for cell in ev.CELLS for arm in ("untrained",)+ev.ARMS}
        actions = np.zeros((32,100,4), np.float32)
        actions[0,0,0] = .75
        exposures = {key:ev.gap_exposure(actions) for key in returns}
        return returns,exposures

    def test_exact_protocol_budget_and_seed_contract(self):
        ev.validate_protocol(self.protocol)
        self.assertEqual(len(ev.POLICIES), 16)
        for key,value in (("simulator_transition_cap",51201),("evaluation_episodes",384),("gamma",.99)):
            changed = copy.deepcopy(self.protocol); changed[key] = value
            with self.assertRaises(ValueError): ev.validate_protocol(changed)

    def test_all_gates_and_bootstrap_are_reproducible(self):
        returns, exposures = self.data()
        report = ev.decision_report(returns, exposures, self.protocol)
        self.assertTrue(report["primary_screen_passed"])
        self.assertTrue(report["learning_adequacy_passed"])
        self.assertEqual(len(report["learning_comparisons"]), 4)
        self.assertEqual(sum(len(v) for v in report["bca_comparisons"].values()), 8)
        self.assertEqual(report, ev.decision_report(returns, exposures, self.protocol))
        self.assertEqual(report["bca_comparisons"][ev.CELLS[0]]["native"]["mean_gain"], 3.)
        self.assertEqual(report["restricted_minus_broad_relative_benefit"]["0"]["native"]["mean_relative_benefit_change"], 0.)

    def test_learning_gate_failure_cannot_pass_on_bca_alone(self):
        returns, exposures = self.data()
        returns[ev.CELLS[0]+"__native"] = returns[ev.CELLS[0]+"__untrained"].copy()
        report = ev.decision_report(returns, exposures, self.protocol)
        self.assertTrue(report["all_bca_comparisons_positive"])
        self.assertFalse(report["primary_screen_passed"])
        self.assertEqual(report["decision"], "stop_inadequate_learning_without_extension")

    def test_one_control_failure_stops_and_records_harms(self):
        returns, exposures = self.data()
        returns[ev.CELLS[1]+"__distance"] = np.zeros(32)
        report = ev.decision_report(returns, exposures, self.protocol)
        self.assertFalse(report["primary_screen_passed"])
        self.assertEqual(report["bca_comparisons"][ev.CELLS[1]]["distance"]["harmed_episodes"], 32)

    def test_missing_or_nonfinite_policy_refused(self):
        returns, exposures = self.data()
        changed = returns.copy(); changed.pop(ev.POLICIES[0])
        with self.assertRaises(ValueError): ev.decision_report(changed, exposures, self.protocol)
        returns[ev.POLICIES[-1]][0] = np.nan
        with self.assertRaises(ValueError): ev.decision_report(returns, exposures, self.protocol)

    def test_gap_exposure_threshold_and_no_exposure_interpretation(self):
        action = np.zeros((32,100,4), np.float64)
        action[0,0,0] = .5+1e-6
        action[0,1,0] = .5+2e-6
        result = ev.gap_exposure(action)
        self.assertEqual(result["above_cutoff_action_rows"], 1)
        self.assertEqual(result["episodes_with_gap_action"], 1)
        returns, exposures = self.data()
        exposures = {key:ev.gap_exposure(np.zeros((32,100,4))) for key in returns}
        report = ev.decision_report(returns, exposures, self.protocol)
        self.assertTrue(report["primary_screen_passed"])
        self.assertFalse(report["extrapolation_interpretation"]["seed0_restricted"]["baseline_gap_exposure_observed"])

    def test_full_cap_accounting_and_no_extra_transition(self):
        ledger = self.memory_ledger()
        for _ in range(32): ledger.before_reset(); ledger.after_reset()
        for policy in ev.POLICIES:
            for step in range(100): ledger.before(32,policy,step); ledger.after(32)
            ledger.data["completed_episodes"] += 32
        self.assertEqual(ledger.data["completed_transition_rows"],51200)
        self.assertEqual(ledger.data["completed_episodes"],512)
        self.assertEqual(ledger.data["reset_calls_completed"],32)
        with self.assertRaises(RuntimeError): ledger.before(32,"duplicate",0)

    def test_short_rollout_uses_live_actions_and_final_reward(self):
        ledger = self.memory_ledger(cap=64)
        seen = []
        def actor(obs):
            seen.append(obs.copy())
            return np.repeat(obs[:,:1],4,axis=1).astype(np.float32)
        def dynamics(states, actions):
            return states+.1, np.full(32,2.,np.float64)
        result = ev.prior.rollout(actor,dynamics,np.zeros((32,13),np.float64),2,.98,ledger,"synthetic")
        np.testing.assert_allclose(result["returns"],2+2*.98)
        self.assertEqual(float(seen[0][0,0]),0.)
        self.assertAlmostEqual(float(seen[1][0,0]),.1)
        self.assertEqual(ledger.data["completed_transition_rows"],64)

    def test_failed_callback_remains_charged_and_unknown(self):
        ledger = self.memory_ledger()
        def fail(states,actions): raise RuntimeError("synthetic callback failed")
        with self.assertRaises(RuntimeError):
            ev.prior.rollout(lambda obs:np.zeros((32,4),np.float32),fail,np.zeros((32,13),np.float64),100,.98,ledger,"synthetic")
        self.assertEqual(ledger.data["attempted_transition_rows"],32)
        self.assertEqual(ledger.data["inflight_transition_rows"],32)
        self.assertFalse(ledger.data["execution_counts_known"])

    def test_native_action_dtype_and_normalization_validation(self):
        value=np.zeros((32,4),np.float32)
        np.testing.assert_array_equal(ev.native_action_rows(value),value)
        with self.assertRaises(ValueError): ev.native_action_rows(value.astype(np.float64))
        norm={"mean":[0.]*14,"std":[1.]*14,"epsilon":.001}
        self.assertEqual(ev.normalization_checked(norm),norm)
        norm["std"][0]=-1
        with self.assertRaises(ValueError): ev.normalization_checked(norm)

    def test_actor_manifest_counts_and_source_are_enforced(self):
        manifest={"cell":"seed0_restricted","condition":"restricted","seed_index":0,
            "protocol_sha256":"fixture","completed":True,"kind":"branch","arm":"bca",
            "native_source_sha256":self.protocol["native_source_sha256"],"native_revision":self.protocol["native_revision"],
            "counts":{"critic":12000,"actor":6000,"auxiliary":12000,"actor_target":6000,"critic_target":6000,
                "refresh_accepted":2,"last_refresh_critic":11000,"reference_ready":True,"phase":0},
            "normalization":{"mean":[0.]*14,"std":[1.]*14,"epsilon":.001}}
        ev.validate_actor_manifest(manifest,"seed0_restricted","bca",self.protocol,"fixture")
        manifest["counts"]["critic"]=11999
        with self.assertRaises(ValueError): ev.validate_actor_manifest(manifest,"seed0_restricted","bca",self.protocol,"fixture")

    def test_atomic_ledger_file_records_before_dispatch(self):
        ledger=ev.prior.Ledger(self.path / "progress.json",51200,600,clock=lambda:0.)
        ledger.before(32,"fixture",0)
        saved=json.loads((self.path / "progress.json").read_text())
        self.assertEqual(saved["inflight_transition_rows"],32)
        self.assertFalse(saved["execution_counts_known"])
        ledger.after(32)

    def fixture_exports(self):
        def save(directory,name,value):
            raw = json.dumps(value,sort_keys=True).encode()
            (directory/name).write_bytes(raw)
            return ev.digest(raw)
        for cell in ev.CELLS:
            seed,condition = int(cell[4]),cell[6:]
            for stage in ("warmup",)+ev.ARMS:
                directory=self.path/cell/stage; directory.mkdir(parents=True)
                state_bytes=b"synthetic state"; (directory/"final_state.msgpack").write_bytes(state_bytes)
                state_artifact={"path":"final_state.msgpack","bytes":len(state_bytes),"sha256":ev.digest(state_bytes)}
                kind="warmup" if stage=="warmup" else "branch"
                counts={"critic":10000 if kind=="warmup" else 12000,"actor":5000 if kind=="warmup" else 6000,
                    "auxiliary":10000 if kind=="warmup" else 12000,"actor_target":5000 if kind=="warmup" else 6000,
                    "critic_target":5000 if kind=="warmup" else 6000,"refresh_accepted":1 if kind=="warmup" else 2,
                    "last_refresh_critic":10000 if kind=="warmup" else 11000,"reference_ready":True,"phase":0}
                def actor_manifest(untrained=False):
                    filename="untrained_actor.msgpack" if untrained else "final_actor.msgpack"
                    raw=("synthetic actor"+str(seed)).encode(); (directory/filename).write_bytes(raw)
                    result={"cell":cell,"condition":condition,"seed_index":seed,"protocol_sha256":"fixture",
                        "completed":True,"kind":"untrained" if untrained else kind,
                        "arm":"native" if untrained or stage=="warmup" else stage,
                        "native_source_sha256":self.protocol["native_source_sha256"],"native_revision":self.protocol["native_revision"],
                        "actor_params":{"path":filename,"bytes":len(raw),"sha256":ev.digest(raw)},
                        "counts":{"critic":0,"actor":0,"auxiliary":0,"refresh_accepted":0} if untrained else counts,
                        "normalization":{"mean":[float(condition=="restricted")]*14,"std":[1.]*14,"epsilon":.001},
                        "parameter_leaves":[{"seed":seed}]}
                    if not untrained: result["final_state"]=state_artifact
                    return result
                report={"status":"completed_support_"+kind,"protocol_sha256":"fixture","cell":cell,
                    "execution_counts_known":True,
                    "actor_manifest_sha256":save(directory,"actor_manifest.json",actor_manifest()),
                    "state_manifest_sha256":save(directory,"state_manifest.json",{"artifact":state_artifact,
                        "protocol_sha256":"fixture","cell":cell,"kind":kind})}
                if stage=="warmup": report["untrained_actor_manifest_sha256"]=save(directory,"untrained_actor_manifest.json",actor_manifest(True))
                report_hash=save(directory,"report.json",report)
                save(directory,"SUPERVISOR_RESULT.json",{"job":cell+"/"+stage,"completed":True,"exit_code":0,
                    "timed_out":False,"interrupted":False,"report_sha256":report_hash})

    def test_complete_export_graph_handoff_and_paired_normalization(self):
        self.fixture_exports()
        exports,identities,hashes,norms=ev.assemble_exports(self.protocol,self.path,"fixture")
        self.assertEqual(set(exports),set(ev.POLICIES))
        self.assertEqual(len(identities),16)
        self.assertEqual(set(norms),{"restricted","broad_count_matched"})
        self.assertTrue(hashes)

    def test_sealed_actor_tampering_refused_before_evaluation(self):
        self.fixture_exports()
        (self.path/ev.CELLS[0]/"bca"/"final_actor.msgpack").write_bytes(b"changed")
        with self.assertRaises(ValueError): ev.assemble_exports(self.protocol,self.path,"fixture")


if __name__ == "__main__": unittest.main()
