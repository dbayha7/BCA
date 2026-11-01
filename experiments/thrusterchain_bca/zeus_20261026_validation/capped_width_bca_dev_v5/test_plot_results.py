"""Low-cost figure and decision semantics tests; no simulator or fitting."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import plot_results as p


def fixture():
    data={}
    for cx in p.CONTEXTS:
        for arm in p.ARMS:
            for seed in range(3):
                q=np.tile(np.arange(5,dtype=float),(16,1))
                first=np.zeros((5,16,4),np.float32)
                first[3,0,0]=.1; first[3,1,0]=.2; first[3,2,0]=.3
                q[0,3]=-.5; q[1,3]=1.5; q[2,3]=4.
                row={'returns':np.tile(-100+np.arange(5,dtype=float)[:,None],(1,16))+seed,
                     'first_qpi':q,'first_actions':first,
                     'clip_saturated':np.zeros((4,16,100),bool),'guards':np.ones((4,16,100),bool),
                     'reset_seeds':np.arange(16)}
                data[(cx,arm,seed)]=row
    refits={'cases':[{'seed':seed,'contexts':{cx:{'summary':{'actor_visited':
                     {'rms_paired_mean_contrast':.2+seed*.1,'rms_across_refit_sd':.08,
                      'rms_native_refit_sd':.07,'rms_bca_refit_sd':.09}}} for cx in p.CONTEXTS}}
                    for seed in range(3)]}
    return data,refits


class PlotTests(unittest.TestCase):
    def test_individual_contrast_sd_is_per_checkpoint(self):
        w=np.zeros((4,2,32,15),np.float32)
        for i in range(4):
            w[i,0,:,1]=i;w[i,1,:,1]=2*i
        source=['random_prefix']*16+['native_visited']*8+['bca_visited']*8
        r=p.individual_refit_sd(w,source)
        self.assertGreater(r['rms_native_refit_sd'],0)
        self.assertAlmostEqual(r['rms_bca_refit_sd'],2*r['rms_native_refit_sd'])
    def test_changed_action_and_qpi_gain_counts(self):
        data,_=fixture(); r=p.first_action_counts(data[('ordinary','native',0)],1)
        self.assertEqual([r[k] for k in ('harmed','near_tied','improved','unchanged')],[1,0,2,13])
        self.assertAlmostEqual(r['mean_gain'],float(np.mean(data[('ordinary','native',0)]['first_qpi'][:,3]
                                                               -data[('ordinary','native',0)]['first_qpi'][:,1])))
        # An unchanged action is not categorized as an improvement even if
        # a contrived inconsistent reference value differs.
        self.assertEqual(sum(r[k] for k in ('harmed','near_tied','improved','unchanged')),16)

    def test_all_figures_and_csv_render_from_aligned_values(self):
        data,refits=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            p.fig_paired_returns(data,out);p.fig_absolute_returns(data,out);p.fig_first_actions(data,out)
            p.fig_telemetry(data,out);p.fig_refits(refits,out);p.fig_actor_gain(data,out);p.write_csv(data,refits,out)
            for stem in ('01_paired_return_gains','02_complete_returns','03_first_action_changes',
                         '04_cap_and_guard_activity','05_paired_refit_variability','06_capped_vs_actor'):
                self.assertEqual((out/(stem+'.png')).read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
                self.assertTrue((out/(stem+'.pdf')).read_bytes().startswith(b'%PDF'))
            with (out/'episode_values.csv').open(encoding='utf8') as f:
                self.assertEqual(sum(1 for _ in f),961)


if __name__=='__main__':
    unittest.main()
