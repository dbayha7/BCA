"""Synthetic-only saved-prediction utility tests; no saved outcomes are read."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class DecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np=np
        spec=importlib.util.spec_from_file_location("synthetic_decision",ROOT/"decision.py")
        cls.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.m)
        cls.p=json.loads((ROOT/"protocol.json").read_text())

    def fixture(self,n=128):
        queries={};predictions=[]
        for i in range(n):
            identity={"group_id":i,"batch":self.p["batch_names"][i//64],"time":[0,25,50,75][i%4],"episode_seed":1000+i}
            queries[i]={**identity,"q1":[0.,2.,1.]+[-9.]*6,"u":[0.,1.,.3]+[0.]*6,
                        "actor_distance":[0.,1.,1.]+[1.]*6,"values":[0.,2.,4.]+[-10.]*6,"raw_q_index":1}
            predictions.append({**identity,"fold":i%8,"predictions":{"base":[0.]*9,
                              "base_plus_residual_width":[0.,2.,-2.]+[0.]*6}})
        return queries,predictions

    def data(self):
        queries,predictions=self.fixture()
        batches=[{"name":name,"queries":[query for query in queries.values() if query["batch"]==name]} for name in self.p["batch_names"]]
        transfers=[]
        for train,test in (self.p["batch_names"],self.p["batch_names"][::-1]):
            subset=copy.deepcopy([row for row in predictions if row["batch"]==test])
            for row in subset:
                row["predictions"]["base_plus_residual_width"]=[0.]*9 if test==self.p["batch_names"][1] else [-4.,2.,2.]+[0.]*6
            transfers.append({"training_batch":train,"heldout_batch":test,"episodes":subset})
        return {"batches":batches,"prediction_receipt":{"cv":{"episodes":predictions},"cross_batch_transfer":transfers}}

    def test_five_exact_formulas(self):
        query,pred=self.fixture(1);q=query[0]
        scores=self.m.score_methods(q["q1"],q["u"],q["actor_distance"],pred[0]["predictions"],self.p)
        self.assertEqual(tuple(scores),self.m.METHODS)
        self.assertEqual(scores["corrected_base"]["scores"],q["q1"])
        self.assertEqual(scores["corrected_width"]["scores"][:3],[0.,0.,3.])
        self.assertEqual(scores["bca_kappa_3"]["scores"],[a-3*b for a,b in zip(q["q1"],q["u"])])
        self.assertEqual(scores["contrast_actor_distance_kappa_0.1"]["scores"],[a-.1*19.29619572569095*b for a,b in zip(q["q1"],q["actor_distance"])])

    def test_perfect_centered_error_correction_selects_value_best(self):
        queries,preds=self.fixture(1);q=queries[0]
        error=self.np.asarray(q["q1"])-self.np.asarray(q["values"])
        preds[0]["predictions"]["base_plus_residual_width"]=(error-error.mean()).tolist()
        rows,_=self.m.evaluate_saved_predictions(queries,preds,self.p,1)
        self.assertEqual(rows[0]["methods"]["corrected_width"]["selected_index"],2)
        self.assertEqual(rows[0]["methods"]["corrected_width"]["candidate_bank_regret"],0.)

    def test_values_cannot_change_score_selections_or_seal(self):
        queries,preds=self.fixture(4)
        before,seal=self.m.evaluate_saved_predictions(queries,preds,self.p,4)
        altered=copy.deepcopy(queries)
        for row in altered.values():row["values"]=[99.,100.,-100.]+[-9.]*6
        after,other=self.m.evaluate_saved_predictions(altered,preds,self.p,4)
        self.assertEqual(seal,other)
        for first,second in zip(before,after):
            for name in self.m.METHODS:
                self.assertEqual(first["methods"][name]["selected_index"],second["methods"][name]["selected_index"])
        self.assertNotEqual(before[0]["primary"]["gain"],after[0]["primary"]["gain"])

    def test_identical_corrections_zero_event_conditional_undefined(self):
        queries,preds=self.fixture(4)
        for row in preds:row["predictions"]["base_plus_residual_width"]=[0.]*9
        rows,_=self.m.evaluate_saved_predictions(queries,preds,self.p,4)
        summary=self.m.summarize(rows)
        self.assertEqual(summary["primary_mean_gain"],0.)
        self.assertEqual(summary["decision_disagreement"]["prevalence"],0.)
        self.assertIsNone(summary["decision_disagreement"]["conditional_mean_gain"])

    def test_changed_indices_equal_values_not_score_ties(self):
        queries,preds=self.fixture(1);queries[0]["values"][2]=2.
        rows,_=self.m.evaluate_saved_predictions(queries,preds,self.p,1)
        self.assertTrue(rows[0]["primary"]["different_indices"])
        self.assertFalse(rows[0]["primary"]["score_ambiguous"])
        self.assertEqual(rows[0]["primary"]["return_direction"],"near_equal")
        self.assertEqual(rows[0]["primary"]["gain"],0.)

    def test_decomposition_and_regret_identity(self):
        queries,preds=self.fixture(4);queries[1]["values"][2]=1.
        for row in preds[2:]:row["predictions"]["base_plus_residual_width"]=[0.]*9
        rows,_=self.m.evaluate_saved_predictions(queries,preds,self.p,4);s=self.m.summarize(rows)
        self.assertEqual(s["decision_disagreement"]["prevalence"],.5)
        self.assertEqual(s["decision_disagreement"]["conditional_mean_gain"],.5)
        self.assertEqual(s["primary_mean_gain"],.25)
        for row in rows:self.assertEqual(row["methods"]["corrected_base"]["candidate_bank_regret"]-row["methods"]["corrected_width"]["candidate_bank_regret"],row["primary"]["gain"])

    def test_near_equal_gains_are_not_zeroed_in_primary(self):
        queries,preds=self.fixture(1);queries[0]["values"][2]=2.+1e-7
        rows,_=self.m.evaluate_saved_predictions(queries,preds,self.p,1)
        self.assertEqual(rows[0]["primary"]["return_direction"],"near_equal")
        self.assertGreater(self.m.summarize(rows)["primary_mean_gain"],0.)

    def test_score_ties_preserve_first_index_without_value_tiebreak(self):
        queries,preds=self.fixture(1)
        queries[0]["q1"]=[2.,2.+1e-6,1.]+[-9.]*6;queries[0]["raw_q_index"]=0
        queries[0]["values"]=[0.,100.,1.]+[-10.]*6
        rows,_=self.m.evaluate_saved_predictions(queries,preds,self.p,1)
        self.assertEqual(rows[0]["methods"]["corrected_base"]["selected_index"],0)
        self.assertTrue(rows[0]["methods"]["corrected_base"]["selection"]["score_ambiguous"])
        self.assertTrue(self.m.summarize(rows)["all_episodes_retained"])

    def test_score_offsets_can_change_numerical_near_max_set(self):
        scores=[0.,1.]+[-9.]*7
        self.assertEqual(self.m.selection(scores,1e-6,1e-6)["index"],1)
        self.assertEqual(self.m.selection([x+1e9 for x in scores],1e-6,1e-6)["index"],0)

    def test_mismatched_missing_duplicate_nonfinite_predictions_rejected(self):
        queries,preds=self.fixture(2)
        altered=copy.deepcopy(preds);altered[0]["episode_seed"]+=1
        with self.assertRaises(ValueError):self.m.evaluate_saved_predictions(queries,altered,self.p,2)
        with self.assertRaises(ValueError):self.m.evaluate_saved_predictions(queries,preds[:1],self.p,2)
        with self.assertRaises(ValueError):self.m.evaluate_saved_predictions(queries,[preds[0],preds[0]],self.p,2)
        altered=copy.deepcopy(preds);altered[0]["predictions"]["base"][0]=float("nan")
        with self.assertRaises(ValueError):self.m.evaluate_saved_predictions(queries,altered,self.p,2)

    def test_primary_bootstrap_keeps_cells_and_pairing(self):
        queries,preds=self.fixture();rows,_=self.m.evaluate_saved_predictions(queries,preds,self.p,128)
        result,indices=self.m.bootstrap_primary(rows,self.p)
        self.assertEqual(result["conditional_percentile_ci95"],[2.,2.])
        for cell in result["cells"]:self.assertTrue(self.np.isin(indices[:,cell["positions"]],cell["positions"]).all())
        self.assertEqual(result["resample_indices_sha256"],self.m.bootstrap_primary(rows,self.p)[0]["resample_indices_sha256"])

    def test_end_to_end_uses_directional_transfer_predictions_and_no_work(self):
        result=self.m.analyze(self.data(),self.p)
        self.assertEqual(result["cv"]["overall"]["primary_mean_gain"],2.)
        self.assertEqual([tr["summary"]["primary_mean_gain"] for tr in result["cross_batch_transfer"]],[0.,-2.])
        self.assertEqual(len(result["cv"]["episodes"]),128)
        for transfer in result["cross_batch_transfer"]:self.assertFalse(transfer["interval_computed"])
        for key in ("new_diagnostic_fits","new_simulator_calls","new_gpu_work","policy_optimizer_updates","calibrator_updates","confirmation_files_read"):
            self.assertEqual(result[key],0)
        for key in ("new_model_fitted","new_policy_training","automatic_training_approval","method_or_strength_selected","production_method_approved"):
            self.assertIs(result[key],False)
        self.assertIs(result["full_training_hold"],True)
        self.assertEqual(json.loads(json.dumps(result,allow_nan=False))["status"],"pass")


if __name__=="__main__":unittest.main(verbosity=2)
