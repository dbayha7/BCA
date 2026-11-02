"""Synthetic-only diagnostic-fit tests. Never reads saved outcomes/checkpoints."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class PredictiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np=np
        spec=importlib.util.spec_from_file_location("synthetic_predictive",ROOT/"predictive.py")
        cls.m=importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.m)
        cls.p=json.loads((ROOT/"protocol.json").read_text())

    def fixture(self,n=8):
        np=self.np; q=np.arange(-4.,5.); d=q*q; d-=d.mean()
        z=q**3-(np.sum(q**4)/np.sum(q*q))*q
        base=np.tile(np.column_stack((q,d)),(n,1,1))
        width=np.tile(q+.2*d+z,(n,1)); y=np.tile(.5*q-.3*d+2*z,(n,1))
        perm=np.asarray([np.roll(row,i+1) for i,row in enumerate(width)])
        return base,width,perm,y,z

    def fit(self,data=None,test_base=None,test_width=None):
        base,width,perm,y,_=data or self.fixture()
        return self.m.fit_models(base,width,perm,y,base[:2] if test_base is None else test_base,
            width[:2] if test_width is None else test_width,perm[:2],self.p)

    def batches(self):
        np=self.np; base,width,_,target,_=self.fixture(128)
        batches=[]
        for b,name in enumerate(self.p["batch_names"]):
            queries=[]
            for i in range(64):
                g=b*64+i; q=50+.01*g+base[g,:,0]
                queries.append({"group_id":g,"episode_seed":1000+g,"time":[0,25,50,75][i%4],
                    "q1":q,"actor_distance":base[g,:,1]+20,"u":width[g]+100,"values":q-target[g]})
            batches.append({"name":name,"queries":queries})
        return batches

    def test_ridge_objective_no_half_or_feature_count_factor(self):
        beta=self.m.ridge_solve([[-1.],[1.]],[-2.,2.],1.)
        self.np.testing.assert_array_equal(beta,[1.])

    def test_known_residual_signal_and_training_orthogonality(self):
        out=self.fit(); name=self.m.MODELS[1]
        receipt=out["fits"][name]
        self.assertTrue(receipt["residual_feature_active"])
        self.assertLess(max(x for x in receipt["training_residual_normalized_dots"] if x is not None),1e-12)
        self.assertLess(receipt["base_coefficient_max_abs_difference"],1e-12)
        base,width,perm,y,z=self.fixture()
        improvement=self.np.mean((out["predictions"]["base"]-y[:2])**2)-self.np.mean((out["predictions"][name]-y[:2])**2)
        self.assertAlmostEqual(float(improvement),float(3*self.np.mean(z*z)),places=10)

    def test_affine_duplicate_and_constant_width_exact_base_collapse(self):
        base,width,perm,y,z=self.fixture()
        for duplicate in (2*base[:,:,0]+3*base[:,:,1],self.np.zeros_like(width)):
            out=self.fit((base,duplicate,duplicate,y,z))
            for name in self.m.MODELS[1:]:
                self.assertFalse(out["fits"][name]["residual_feature_active"])
                self.assertTrue(out["fits"][name]["reused_exact_base_prediction"])
                self.np.testing.assert_array_equal(out["predictions"][name],out["predictions"]["base"])
            self.assertEqual(out["fit_counts"]["ridge_system_solves"],1)

    def test_subfloor_component_inactive_and_larger_component_active(self):
        base,width,perm,y,z=self.fixture()
        for scale,active in ((1e-14,False),(1e-8,True)):
            residualized=base[:,:,0]+scale*self.np.tile(z,(len(base),1))
            out=self.fit((base,residualized,perm,y,z))
            self.assertEqual(out["fits"][self.m.MODELS[1]]["residual_feature_active"],active)

    def test_constant_and_collinear_base_columns_stay_finite(self):
        base,width,perm,y,z=self.fixture()
        for altered in (self.np.zeros_like(base),self.np.stack((base[:,:,0],2*base[:,:,0]),axis=2)):
            out=self.fit((altered,width,perm,y,z))
            for predicted in out["predictions"].values(): self.assertTrue(self.np.isfinite(predicted).all())
        zero=self.fit((self.np.zeros_like(base),width,perm,y,z))
        self.assertFalse(zero["fits"]["base"]["feature_active"].any())

    def test_heldout_covariates_never_change_training_receipts(self):
        data=self.fixture(); original=self.fit(data)
        changed=self.fit(data,test_base=data[0][:2]*100+13,test_width=data[1][:2]*50-9)
        self.assertEqual(self.m._plain(original["fits"]),self.m._plain(changed["fits"]))
        self.assertFalse(self.np.array_equal(original["predictions"]["base"],changed["predictions"]["base"]))

    def test_group_balance_and_heldout_target_isolation(self):
        batches=self.batches(); prepared=self.m._prepare(batches,self.p)
        folds,cells=self.m.assign_folds(prepared["metadata"],self.p)
        self.assertEqual(folds,self.m.assign_folds(prepared["metadata"],self.p)[0])
        for fold in range(8):
            self.assertEqual(folds.count(fold),16)
            for cell in cells: self.assertEqual(sum(folds[i]==fold for i in cell["indices"]),2)
        changed=copy.deepcopy(batches)
        for b in changed:
            for row in b["queries"]:
                if folds[row["group_id"]]==0: row["values"]=row["values"]+self.np.arange(9)*100
        changed_prepared=self.m._prepare(changed,self.p)
        train=self.np.asarray([i for i,f in enumerate(folds) if f!=0])
        self.np.testing.assert_array_equal(prepared["target"][train],changed_prepared["target"][train])
        self.np.testing.assert_array_equal(prepared["base"],changed_prepared["base"])
        self.np.testing.assert_array_equal(prepared["width"],changed_prepared["width"])

    def test_fixed_within_state_permutation_preserves_multiset(self):
        data=self.m._prepare(self.batches(),self.p)
        a,orders=self.m.permute_width(data["width"],data["metadata"],self.p["permutation_seed"])
        b,again=self.m.permute_width(data["width"],data["metadata"],self.p["permutation_seed"])
        self.assertEqual(orders,again);self.np.testing.assert_array_equal(a,b)
        self.np.testing.assert_array_equal(self.np.sort(a,axis=1),self.np.sort(data["width"],axis=1))

    def test_ranking_target_ties_excluded_prediction_ties_half_credit(self):
        for prediction,expected in (([0.,1.,2.],1.),([2.,1.,0.],0.),([0.,0.,0.],.5)):
            self.assertEqual(self.m.ranking_concordance(prediction,[0.,1.,2.],1e-6,1e-6)["score"],expected)
        tied=self.m.ranking_concordance([0.,1.,2.],[1.,1.,1.],1e-6,1e-6)
        self.assertFalse(tied["defined"]);self.assertIsNone(tied["score"])
        self.assertEqual(tied["target_tied_pairs"],3)

    def test_bootstrap_cells_pairing_and_constant_effect(self):
        data=self.m._prepare(self.batches(),self.p);_,cells=self.m.assign_folds(data["metadata"],self.p)
        receipt,indices=self.m.bootstrap_primary(self.np.full(128,2.),cells,self.p)
        self.np.testing.assert_array_equal(receipt["conditional_percentile_ci95"],[2.,2.])
        for cell in cells: self.assertTrue(self.np.isin(indices[:,cell["indices"]],cell["indices"]).all())
        self.assertEqual(receipt["resample_indices_sha256"],self.m.bootstrap_primary(self.np.full(128,2.),cells,self.p)[0]["resample_indices_sha256"])

    def test_nonfinite_inputs_rejected(self):
        bad=self.fixture();bad[0][0,0,0]=float("nan")
        with self.assertRaises(ValueError): self.fit(bad)
        with self.assertRaises(ValueError): self.m.ridge_solve([[1.],[float("inf")]],[1.,2.])

    def test_end_to_end_synthetic_counts_scope_and_no_policy_training(self):
        result=self.m.analyze(self.batches(),self.p)
        self.assertEqual(result["status"],"pass")
        self.assertEqual(result["diagnostic_regression_fits"],30)
        self.assertEqual(result["fit_counts"]["width_projection_lstsq_fits"],20)
        self.assertLessEqual(result["fit_counts"]["total_linear_fitting_operations"],50)
        self.assertEqual(len(result["fit_receipts"]),10)
        self.assertEqual(len(result["cv"]["episodes"]),128)
        self.assertEqual(len(result["cross_batch_transfer"]),2)
        for row in result["cross_batch_transfer"]:
            self.assertEqual(row["summary"]["episodes"],64);self.assertFalse(row["interval_computed"])
        for key in ("new_policy_training","automatic_training_approval","method_or_strength_selected","action_policy_changed"):
            self.assertIs(result[key],False)
        self.assertIs(result["full_training_hold"],True)
        for key in ("new_simulator_calls","new_gpu_work","confirmation_files_read","optimizer_updates","calibrator_updates"):
            self.assertEqual(result[key],0)
        self.assertEqual(json.loads(json.dumps(result,allow_nan=False))["status"],"pass")


if __name__=="__main__": unittest.main(verbosity=2)
