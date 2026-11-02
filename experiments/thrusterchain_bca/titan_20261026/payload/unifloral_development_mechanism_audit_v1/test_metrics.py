"""Synthetic-only CPU tests; no development or confirmation input files read."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("mechanism_test_metrics", ROOT / "metrics.py")
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)
        cls.protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
        amendment = json.loads((ROOT / "PROTOCOL_AMENDMENT_01.json").read_text(encoding="utf-8"))
        cls.protocol.update(amendment["additions"])

    def state(self, group=0, direction=1, constant=False, event=False):
        q = [10.-i for i in range(9)]
        u = [2.]*9 if constant else [1.+i for i in range(9)]
        error = [direction*float(i) for i in range(9)]
        d = [i*i/64. for i in range(9)]
        values = [a-b for a,b in zip(q,error)]
        raw = 0
        if event:
            q, u, d = [1.,10.,8.]+[-100.]*6, [0.,3.,0.]+[10.]*6, [0.,1.,1.]+[1.]*6
            values, raw = [0.,0.,2.]+[-100.]*6, 1
        return {"group_id": group, "time": [0,25,50,75][group % 4], "episode_seed": 1000+group,
                "q1":q,"u":u,"actor_distance":d,"values":values,"raw_q_index":raw}

    def batches(self):
        return [{"name":name,"queries":[self.state(batch*64+i) for i in range(64)]}
                for batch,name in enumerate(self.protocol["batch_names"])]

    def test_perfect_and_reverse_correlations(self):
        x = list(range(9))
        for sign in (1,-1):
            y = [sign*v for v in x]
            self.assertAlmostEqual(self.m.pearson(x,y)["value"],sign,places=14)
            self.assertAlmostEqual(self.m.spearman(x,y)["value"],sign,places=14)

    def test_spearman_average_ties_and_nonlinear_monotonicity(self):
        self.assertEqual(self.m.average_ranks([1,1,3,4]),[.5,.5,2.,3.])
        self.assertAlmostEqual(self.m.spearman([1,1,3,4],[1,1,9,16])["value"],1.,places=14)

    def test_constants_and_nonfinite_are_not_zero_association(self):
        for first,second in (([.1]*9,list(range(9))),(list(range(9)),[0.]*9)):
            out=self.m.pearson(first,second)
            self.assertFalse(out["defined"])
            self.assertIsNone(out["value"])
        for value in (float("nan"),float("inf"),True):
            with self.assertRaises(ValueError): self.m.pearson([value,1.],[0.,1.])

    def test_positive_optimism_constant_when_all_pessimistic(self):
        query=self.state(direction=-1)
        query["values"]=[q+i+1 for i,q in enumerate(query["q1"])]
        record=self.m._state(query,"synthetic",self.protocol)
        self.assertFalse(record["relationships"]["u"]["positive_optimism"]["pearson"]["defined"])
        self.assertTrue(record["relationships"]["u"]["signed_error"]["pearson"]["defined"])

    def confounded(self):
        d=[i/10. for i in range(9)]
        z=[.1,-.2,.1]+[0.]*6
        return [1+a+b for a,b in zip(d,z)], [a-b for a,b in zip(d,z)], d

    def test_partial_removes_distance_and_can_reverse_association(self):
        u,e,d=self.confounded()
        self.assertAlmostEqual(self.m.pearson(u,e)["value"],9/11.,places=13)
        self.assertAlmostEqual(self.m.partial_pearson(u,e,d,1e-12)["value"],-1.,places=13)

    def test_distance_only_and_constant_control_undefined(self):
        d=list(range(9))
        for u,e,covariate in (([1+2*x for x in d],[3*x for x in d],d),
                              (d,[x*x for x in d],[1.]*9)):
            self.assertFalse(self.m.partial_pearson(u,e,covariate,1e-12)["defined"])

    def test_positive_affine_invariance_of_correlations_not_tie_rules(self):
        u,e,d=self.confounded()
        converted=([3*x+10 for x in u],[7*x-100 for x in e],[2*x+1 for x in d])
        self.assertAlmostEqual(self.m.pearson(u,e)["value"],self.m.pearson(*converted[:2])["value"],places=12)
        self.assertAlmostEqual(self.m.partial_pearson(u,e,d,1e-12)["value"],self.m.partial_pearson(*converted,1e-12)["value"],places=12)

    def test_partial_relative_floor_inclusive_boundary(self):
        u,_,d=self.confounded()
        reference=self.m.partial_pearson(u,u,d,1e-12)
        floor=reference["predictor_residual_relative_norm"]
        self.assertFalse(self.m.partial_pearson(u,u,d,floor)["defined"])
        self.assertTrue(self.m.partial_pearson(u,u,d,math.nextafter(floor,0.))["defined"])

    def two_control_fixture(self):
        d=[float(i) for i in range(-4,5)]
        q=[x*x for x in d]
        coefficient=math.fsum(x**4 for x in d)/math.fsum(x*x for x in d)
        z=[x**3-coefficient*x for x in d]
        u=[a+b+c for a,b,c in zip(d,q,z)]
        e=[3*a-2*b+2*c for a,b,c in zip(d,q,z)]
        return u,e,d,q

    def test_two_control_partial_incremental_signal(self):
        result=self.m.partial_pearson_two_controls(*self.two_control_fixture(),1e-12,1e-12)
        self.assertTrue(result["defined"])
        self.assertEqual(result["control_rank"],2)
        self.assertAlmostEqual(result["value"],1.,places=12)

    def test_two_control_rank_deficiency_is_undefined(self):
        u,e,d,_=self.two_control_fixture()
        result=self.m.partial_pearson_two_controls(u,e,d,[2*x+3 for x in d],1e-12,1e-12)
        self.assertFalse(result["defined"])
        self.assertEqual(result["reason"],"control_rank_deficient")

    def test_pair_correct_wrong_and_action_orientation(self):
        u=list(range(9)); e=[x*x for x in u]
        correct=self.m.pairwise_concordance(u,e,1e-6,1e-6)
        self.assertEqual(correct["correct_pairs"],36)
        wrong=self.m.pairwise_concordance(u,[-x for x in e],1e-6,1e-6)
        self.assertEqual(wrong["wrong_pairs"],36)
        self.assertEqual(correct,self.m.pairwise_concordance(u[::-1],e[::-1],1e-6,1e-6))

    def test_pair_ties_and_no_comparable_denominator(self):
        result=self.m.pairwise_concordance([1.]*9,list(range(9)),1e-6,1e-6)
        self.assertEqual(result["signal_only_tied_pairs"],36)
        self.assertIsNone(result["correct_rate_among_comparable"])
        self.assertFalse(result["correct_rate_defined"])
        self.assertEqual(result["tie_fraction"],1.)

    def test_pair_tolerance_boundary_and_offset_caveat(self):
        self.assertEqual(self.m._direction(1.,0.,1.,0.),0)
        self.assertEqual(self.m._direction(1.000001,0.,1.,0.),1)
        self.assertEqual(self.m._direction(1.,0.,1e-6,1e-6),1)
        self.assertEqual(self.m._direction(1e9+1,1e9,1e-6,1e-6),0)

    def test_best_ambiguity_and_exact_regret(self):
        result=self.m.decision_direction([0.,0.,2.],[1.,3.,2.],[0.,1.,2.],[10.,9.999995,8.],2,1e-6,1e-6)
        self.assertEqual(result["best_value_near_indices"],[0,1])
        self.assertEqual(result["exact_candidate_bank_regret"],2.)
        self.assertTrue(result["regret_above_band"])
        self.assertFalse(result["eligible_unambiguous_best"])
        self.assertEqual(result["u_direction"],"supports_better_candidate")

    def test_saved_raw_choice_must_follow_rule(self):
        with self.assertRaises(ValueError):
            self.m.decision_direction([1.,2.],[1.,2.],[0.,1.],[0.,1.],0,1e-6,1e-6)

    def test_equal_state_weighting_ignores_undefined_not_batch_size(self):
        batches=self.batches()
        for i in range(1,64): batches[0]["queries"][i]["u"]=[2.]*9
        for i in range(64): batches[1]["queries"][i]=self.state(64+i,direction=-1)
        result=self.m.analyze(batches,self.protocol)
        summary=result["overall"]["relationships"]["u"]["signed_error"]["pearson"]
        self.assertEqual(summary["defined_states"],65)
        self.assertEqual(summary["undefined_states"],63)
        self.assertAlmostEqual(summary["mean_over_defined_states"],-63/65.,places=13)
        self.assertEqual(len(result["by_batch_time"]),2)
        self.assertEqual(result["by_time"]["25"]["state_count"],32)
        self.assertEqual(json.loads(json.dumps(result,allow_nan=False))["status"],"pass")

    def test_pair_common_defined_subset(self):
        first=self.m._state(self.state(),"synthetic",self.protocol)
        second=self.m._state(self.state(1,constant=True),"synthetic",self.protocol)
        result=self.m.summarize([first,second])["pairwise"]["common_defined_u_minus_distance"]
        self.assertEqual(result["defined_states"],1)
        self.assertEqual(result["equal_state_mean_accuracy_difference"],0.)

    def test_event_frequency_conditional_unconditional_identity(self):
        event=self.m._state(self.state(event=True),"synthetic",self.protocol)
        same=self.m._state(self.state(1),"synthetic",self.protocol)
        result=self.m.summarize([event,same])["fixed_selector_event"]
        self.assertEqual(result["disagreement_frequency"],.5)
        self.assertEqual(result["conditional_mean_gain_when_different"],2.)
        self.assertEqual(result["unconditional_mean_gain"],1.)
        self.assertTrue(result["identity_verified"])

    def test_no_different_indices_conditional_undefined(self):
        result=self.m.summarize([self.m._state(self.state(),"synthetic",self.protocol)])["fixed_selector_event"]
        self.assertIsNone(result["conditional_mean_gain_when_different"])
        self.assertEqual(result["unconditional_mean_gain"],0.)
        self.assertEqual(result["frequency_times_conditional_gain"],0.)

    def test_zero_observed_variance_not_zero_sample_recommendation(self):
        result=self.m.analyze(self.batches(),self.protocol)["precision_planning"]
        self.assertEqual(result["mean_within_time_sample_variance"],0.)
        for row in result["estimates"]:
            self.assertFalse(row["estimable"])
            self.assertIsNone(row["normal_approximation_80pct_power_total_episodes"])

    def test_precision_formula_equal_time_variance_round_up_four(self):
        states=[]
        for i in range(128):
            states.append({"time":[0,25,50,75][i%4],"fixed_selector_event":{"bca_minus_control_value":float((i//4)%2)}})
        result=self.m.precision_planning(states,self.protocol)
        variance=8/31.
        self.assertAlmostEqual(result["mean_within_time_sample_variance"],variance,places=14)
        for row in result["estimates"]:
            target=row["illustrative_effect_or_half_width"]
            self.assertEqual(row["normal_approximation_80pct_power_total_episodes"],4*math.ceil(((1.96+.84)**2*variance/target**2)/4))
            self.assertFalse(result["sample_size_recommendation"])

    def test_duplicate_groups_or_wrong_strata_rejected(self):
        batches=self.batches()
        batches[1]["queries"][0]["group_id"]=0
        with self.assertRaises(ValueError): self.m.analyze(batches,self.protocol)
        batches=self.batches(); batches[0]["queries"][0]["time"]=25
        with self.assertRaises(ValueError): self.m.analyze(batches,self.protocol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
