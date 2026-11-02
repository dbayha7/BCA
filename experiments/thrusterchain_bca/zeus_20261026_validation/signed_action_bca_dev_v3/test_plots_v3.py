"""Synthetic rendering smoke; temporary artifacts never represent run results."""
import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location('plot_results_expanded_v3', Path(__file__).with_name('plot_results.py'))
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)


class PlotTest(unittest.TestCase):
    def test_temporary_synthetic_results_render(self):
        def score():
            return {'mean_regret':.3,'ranking_accuracy':.6,'ranking_informative_pairs':50,
                'changed':2,'improved':1,'harmed':1,'changed_near_tie':0,
                'all_action_error':{'mae':10.},'high_load_fixed_band_error':{'mae':9.}}
        summary={s:{'states':16,'high_load_states':10,'informative_families':12,
                    'raw_mistake_families':5,'scores':{m:score() for m in p.METHODS}} for s in p.STRATA}
        probe={'qpi':[[0.]*15 for _ in range(32)],'actions':[[[0.]*4 for _ in range(15)] for _ in range(32)],
               'strata': ['natural']*16+['ambiguous_q1']*16,
               'components':{'reliable':[[True]*15 for _ in range(32)]}}
        cases=[]
        for arm in p.ARMS:
            for seed in range(3):
                cases.append({'arm':arm,'seed':seed,'check':{'zero_mae':2.,'signed_mae':1.9},
                    'match':{'lambda_mean':1.5,'lambda_gradient':2.},
                    'contexts':{ctx:{'summary':summary,'per_probe':probe} for ctx in p.CONTEXTS}})
        ledger={'transition_steps_attempted':377704,'transition_steps_completed':377704,
                'continuation_rows_completed':5760,'calibration_refreshes':18,
                'auxiliary_updates':0,'actor_updates':0,'critic_updates':0,
                'actor_target_updates':0,'critic_target_updates':0}
        screen={ctx:{s:{'status':'inconclusive','qualifying_seeds':{'native':0,'bca':0}} for s in p.STRATA} for ctx in p.CONTEXTS}
        result={'status':'completed_expanded_frozen_diagnostic','ledger':ledger,'cases':cases,'screen':screen}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'synthetic.json';source.write_text(json.dumps(result))
            with patch.object(p,'SOURCE',source), patch.object(p,'HERE',root),patch.object(sys,'argv',['plot_results.py']):
                p.run()
            output=root/'plots_v1'
            self.assertEqual(len(list(output.glob('*.png'))),4)
            self.assertEqual(len(list(output.glob('*.pdf'))),4)
            self.assertTrue((output/'all_metrics.csv').exists())
            self.assertTrue((output/'PLOT_RECEIPT.json').exists())


if __name__=='__main__': unittest.main()
