import unittest
from aggregate_results import estimate, summarize_group, METHODS, METRICS, COUNT_KEYS


class AnalysisTests(unittest.TestCase):
    def test_constant_pair_has_zero_interval_width(self):
        got=estimate([0.]*200,200)
        self.assertEqual(got['ci95'],[0.,0.])
        self.assertEqual(got['n_defined'],200)

    def test_undefined_and_missing_are_distinct(self):
        got=estimate([1.,None,'inf',3.],6)
        self.assertEqual((got['n_defined'],got['n_undefined'],got['n_missing']),(2,2,2))
        self.assertEqual(got['mean'],2.)

    def test_no_finite_metric_is_not_zero(self):
        got=estimate([None]*200,200)
        self.assertIsNone(got['mean'])
        self.assertIsNone(got['ci95'])
        self.assertEqual(got['n_undefined'],200)

    def test_paired_intervals_use_within_replication_difference(self):
        rows=[]
        for i in range(5):
            methods={}
            for j,method in enumerate(METHODS):
                data={key:0 for key in COUNT_KEYS}
                data.update({key:None for key in METRICS})
                data.update(coverage_including_infinite=float(i+j),refusal_counts={},subgroups={})
                methods[method]=data
            rows.append(dict(methods=methods,n_eff=128,positive_weight_count=128,
                projected_actor_hits=1,no_positive_weight_bank=False,
                calibration_weights=dict(mean=1,max=1,p99=1),query_mass=dict(mean=1,max=1,p99=1),
                paired=dict(delta_coverage_including_infinite=1.)))
        got=summarize_group(rows,5)['paired_same_metric']['coverage_including_infinite']
        self.assertEqual(got['mean'],1.)
        self.assertEqual(got['ci95'],[1.,1.])


if __name__=='__main__':unittest.main()
