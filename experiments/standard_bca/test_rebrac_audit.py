"""Known-answer/refusal checks of pure audit guards, without loading checkpoints."""
import ast,copy,hashlib,json,unittest
from pathlib import Path
import numpy as np
source=Path(__file__).with_name('audit_first_rebrac_bca.py').read_text()
tree=ast.parse(source); ns=dict(np=np,hashlib=hashlib,json=json)
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('require','digest','array_hash','check_radii','expected_events','event_identity')]
nodes += [n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RADIUS_FIELDS' for t in n.targets)]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<audit-pure-guards>','exec'),ns)
class Tests(unittest.TestCase):
    def radii(self):
        return dict(radius=[121.],conformal_radius=[100.],bayesian_radius=[121.],posterior_quantiles=[list(range(128))],
                    effective_sample_size=[1103.],supported_count=[1103],inputs_valid=[True],finite=[True])
    def test_known_rank_maximum(self): ns['check_radii'](self.radii())
    def test_wrong_rank(self):
        r=self.radii(); r['bayesian_radius']=[120.]
        with self.assertRaises(ValueError): ns['check_radii'](r)
    def test_wrong_combination(self):
        r=self.radii(); r['radius']=[122.]
        with self.assertRaises(ValueError): ns['check_radii'](r)
    def test_wrong_support(self):
        r=self.radii(); r['effective_sample_size']=[1102.]
        with self.assertRaises(ValueError): ns['check_radii'](r)
    def test_nonfinite(self):
        r=self.radii(); r['posterior_quantiles'][0][0]=float('nan')
        with self.assertRaises(ValueError): ns['check_radii'](r)
    def test_missing_shape(self):
        r=self.radii(); r['posterior_quantiles'][0].pop()
        with self.assertRaises(ValueError): ns['check_radii'](r)
    def test_exact_phase_sequence(self):
        p=dict(refresh_events=[dict(step=i) for i in range(10000,1000000,5000)],
            evaluation_events=[dict(kind='periodic',step=i) for i in range(5000,1000001,5000)]+[dict(kind='final',step=1000000)])
        s=ns['expected_events'](p); self.assertEqual(len(s),2802)
        i=s.index(('accepted_scan',10000,None))
        self.assertEqual(s[i:i+5],[('accepted_scan',10000,None),('phase',10000,'refresh'),('refresh',10000,None),('phase',10000,'evaluation'),('periodic',10000,None)])
        self.assertEqual(s[-5:],[('phase',1000000,'evaluation'),('periodic',1000000,None),('phase',1000000,'evaluation'),('final',1000000,None),('completed',1000000,None)])
    def test_zero_refresh_refused(self):
        with self.assertRaises(ValueError): ns['expected_events'](dict(refresh_events=[dict(step=0)],evaluation_events=[]))
if __name__=='__main__': unittest.main()
