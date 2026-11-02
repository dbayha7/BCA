"""Synthetic rendering only; temporary outputs are not experimental evidence."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('common_action_plot_v4',Path(__file__).with_name('plot_results.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def cell(seed,critic,continuation,src):
    n=16 if src=='random_prefix' else 8
    delta=.1*(seed-1)+(.04 if critic==continuation else -.03)
    def comparison(x):
        return {'mean_gain':x,'changed':3,'improved':2,'harmed':1,'near_tied':0,
                'per_state_gain':[x]*n}
    groups={}
    for group in ('all','guard_on','guard_off'):
        groups[group]={'states':n if group=='all' else n//2,
            'comparisons':{'raw_to_symmetric':comparison(-delta),
                'symmetric_to_guarded':comparison(delta),'guarded_to_unguarded':comparison(.02)}}
    return groups


class PlotResultsTests(unittest.TestCase):
    def test_synthetic_render_and_source_receipt(self):
        cases=[]
        for seed in range(3):
            contexts={}
            for cx in p.CONTEXTS:
                summaries={a:{b:{s:cell(seed,a,b,s) for s in p.SOURCES} for b in p.ARMS} for a in p.ARMS}
                contexts[cx]={'summaries':summaries}
            cases.append({'seed':seed,'contexts':contexts})
        screen={cx:{'status':'inconclusive','qualifying_seeds':{'native':0,'bca':0},
                    'eligible_seeds':{'native':1,'bca':1}} for cx in p.CONTEXTS}
        ledger={'transition_steps_attempted':364904,'transition_steps_completed':364904,
                'continuation_rows_completed':5760,'reconstruction_refreshes':18,
                **{k:0 for k in ('actor_updates','critic_updates','actor_target_updates','critic_target_updates','auxiliary_updates')}}
        value={'status':'completed_common_action_continuation_diagnostic','cases':cases,'ledger':ledger,'screen':screen}
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'SYNTHETIC.json';raw=json.dumps(value).encode();source.write_bytes(raw)
            out=p.render(source,Path(tmp)/'synthetic_plots')
            self.assertEqual(len(list(out.glob('*.png'))),5)
            self.assertEqual(len(list(out.glob('*.pdf'))),5)
            receipt=json.loads((out/'PLOT_RECEIPT.json').read_bytes())
            self.assertEqual(receipt['source_sha256'],hashlib.sha256(raw).hexdigest())
            rows=(out/'paired_comparisons.csv').read_text().splitlines()
            self.assertEqual(len(rows),1+2*3*2*2*3*3*3)
            self.assertIn('cross-policy',(out/'FIGURE_GUIDE.md').read_text(encoding='utf8'))


if __name__=='__main__': unittest.main()
