"""Synthetic capsule receipts never authorize actual execution."""
import copy,hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from execution_capsule import EngineeringCapsule,CPU_ENV,GATES,ROLES,PLAN,maximum_caps,identity,pin_file

def save(p,value):p.write_bytes((json.dumps(value,sort_keys=True)+'\n').encode())
def pin(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),identity=identity(p))
class Fixture:
    def __init__(self,root):
        self.root=Path(root);self.inputs=self.root/'inputs';self.inputs.mkdir();self.parent=self.root/'runs';self.parent.mkdir()
        self.ancestor=self.root/'synthetic_ancestor';self.ancestor.write_bytes(b'synthetic; not SQLite')
        self.lock=self.root/'synthetic_shared_lock';self.lock.write_bytes(b'not acquired')
        self.assets={}
        for role in sorted(ROLES):
            p=self.inputs/role
            p.write_bytes(role.encode())
            if role=='plan':p.write_bytes((Path(__file__).resolve().parents[2]/'docs/superpowers/plans/2026-09-28-ood-robustness-v2.md').read_bytes())
            if role=='ancestor_binding':save(p,dict(path=str(self.ancestor),identity=identity(self.ancestor),header_sha256='1'*64,entries=1,last_sha256='2'*64,reserved={'global':[60,240]},audit_receipt_sha256='3'*64))
            self.assets[role]=pin(p)
        self.declaration=self.inputs/'declaration.json';self.report=self.inputs/'review.json';self.actual=self.inputs/'actual.json';self.reviewer=self.inputs/'reviewer.py'
        self.reviewer.write_bytes(b'pass\n');self.output=self.parent/'new-run'
        self.spec=dict(schema='ood-v2-engineering-capsule-v1',phase='engineering',plan_sha256=PLAN,pair=['td3_bc','hopper',202609171],assets=self.assets,
            paths=dict(parent=str(self.parent),run_root=str(self.output),extension=str(self.output/'extension.sqlite'),archive=str(self.output/'archive.sqlite'),dispatch=str(self.output/'dispatch.json'),lock=str(self.lock),ancestor=str(self.ancestor)),
            command=[self.assets['interpreter']['path'],'-I',self.assets['entrypoint']['path'],'--execution-declaration',str(self.declaration)],cwd=str(self.parent),
            environment=copy.deepcopy(CPU_ENV),caps=maximum_caps(),combined_global_cap=[39998400,159993600],one_worker=True,automatic_retry=False)
        self.rebuild()
    def rebuild(self):
        save(self.declaration,self.spec);self.declaration_pin=pin(self.declaration)
        self.review=dict(schema='ood-v2-engineering-independent-review-v1',phase='engineering',declaration_sha256=self.declaration_pin['sha256'],assets={k:v['sha256'] for k,v in self.spec['assets'].items()},gates={k:True for k in GATES},reviewer_source_sha256=pin(self.reviewer)['sha256'])
        self.command=[self.assets['interpreter']['path'],'-I',str(self.reviewer),'--execution-declaration',str(self.declaration),'--output',str(self.report)]
        self.close_review()
    def close_review(self):
        save(self.report,self.review)
        self.receipt=dict(schema='ood-v2-independent-review-process-v1',command=self.command,actual_returncode=0,timeout=False,interruption_signal=None,
            started='2020-01-01T00:00:00+00:00',ended='2020-01-01T00:00:01+00:00',pid=os.getpid()+100000,
            declaration_sha256=self.declaration_pin['sha256'],reviewer_source_sha256=pin(self.reviewer)['sha256'],report_sha256=pin(self.report)['sha256'])
        self.close_actual()
    def close_actual(self):
        save(self.actual,self.receipt)
        self.trusted=dict(report=pin(self.report),actual_exit=pin(self.actual),reviewer_source=pin(self.reviewer),interpreter=self.assets['interpreter'],command=self.command)
    def build(self):return EngineeringCapsule(declaration_pin=self.declaration_pin,trusted_review=self.trusted)

class Tests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.f=Fixture(self.tmp.name)
    def refuse_spec(self,change):change(self.f.spec);self.f.rebuild();self.assertRaises((ValueError,KeyError),self.f.build)
    def test_bootstrap_no_creation(self):
        c=self.f.build();r=c.take_bootstrap();self.assertEqual(r['permitted_scope'],'engineering/td3_bc/hopper');self.assertFalse(r['science_accepted']);self.assertFalse(self.f.output.exists())
        self.assertEqual(self.f.ancestor.read_bytes(),b'synthetic; not SQLite');self.assertEqual(self.f.lock.read_bytes(),b'not acquired')
    def test_single_consumption(self):
        c=self.f.build();c.take_bootstrap();self.assertRaises(ValueError,c.take_bootstrap)
    def test_stream_hold_refused(self):
        self.f.review['gates']['fresh_streams']=False;self.f.close_review();self.assertRaises(ValueError,self.f.build)
    def test_missing_gate(self):
        self.f.review['gates'].pop('native_factory_imports_wrappers');self.f.close_review();self.assertRaises(ValueError,self.f.build)
    def test_truthy_gate_not_boolean(self):
        self.f.review['gates']['training_pair']=1;self.f.close_review();self.assertRaises(ValueError,self.f.build)
    def test_unknown_exit(self):
        self.f.receipt['actual_returncode']=None;self.f.close_actual();self.assertRaises(ValueError,self.f.build)
    def test_nonzero_exit(self):
        self.f.receipt['actual_returncode']=23;self.f.close_actual();self.assertRaises(ValueError,self.f.build)
    def test_false_exit_not_integer_zero(self):
        self.f.receipt['actual_returncode']=False;self.f.close_actual();self.assertRaises(ValueError,self.f.build)
    def test_timeout(self):
        self.f.receipt['timeout']=True;self.f.close_actual();self.assertRaises(ValueError,self.f.build)
    def test_stale_report(self):
        self.f.receipt['report_sha256']='a'*64;self.f.close_actual();self.assertRaises(ValueError,self.f.build)
    def test_wrong_review_command(self):
        self.f.receipt['command']=self.f.command+['--bypass'];self.f.close_actual();self.assertRaises(ValueError,self.f.build)
    def test_same_process_reviewer(self):
        self.f.receipt['pid']=os.getpid();self.f.close_actual();self.assertRaises(ValueError,self.f.build)
    def test_future_review(self):
        self.f.receipt['ended']='2099-01-01T00:00:00+00:00';self.f.close_actual();self.assertRaises(ValueError,self.f.build)
    def test_unzoned_review(self):
        self.f.receipt['ended']='2020-01-01T00:00:01';self.f.close_actual();self.assertRaises(ValueError,self.f.build)
    def test_science_phase_refused(self):self.refuse_spec(lambda s:s.update(phase='science'))
    def test_gpu_environment_refused(self):self.refuse_spec(lambda s:s['environment'].update(CUDA_VISIBLE_DEVICES='0'))
    def test_inherited_extra_environment_refused(self):self.refuse_spec(lambda s:s['environment'].update(PYTHONPATH='/tmp'))
    def test_retry_refused(self):self.refuse_spec(lambda s:s.update(automatic_retry=True))
    def test_seed_substitution_refused(self):self.refuse_spec(lambda s:s.update(pair=['td3_bc','hopper',3]))
    def test_cap_expansion_refused(self):self.refuse_spec(lambda s:s['caps']['global'].__setitem__(0,40000000))
    def test_missing_asset_refused(self):self.refuse_spec(lambda s:s['assets'].pop('supervisor_acceptance'))
    def test_command_declaration_mismatch(self):self.refuse_spec(lambda s:s['command'].__setitem__(-1,str(self.f.report)))
    def test_output_escape(self):self.refuse_spec(lambda s:s['paths'].update(archive=str(self.f.root/'elsewhere')))
    def test_existing_run_preserved(self):
        self.f.output.mkdir();(self.f.output/'unresolved').write_bytes(b'keep');self.assertRaises(ValueError,self.f.build);self.assertEqual((self.f.output/'unresolved').read_bytes(),b'keep')
    def test_output_alias(self):self.refuse_spec(lambda s:s['paths'].update(archive=s['paths']['extension']))
    def test_symlink_source_refused(self):
        p=Path(self.f.assets['entrypoint']['path']);other=self.f.inputs/'redirect';p.rename(other);p.symlink_to(other);self.assertRaises(ValueError,self.f.build)
    def test_hardlink_source_refused(self):
        os.link(self.f.assets['entrypoint']['path'],self.f.inputs/'alias');self.assertRaises(ValueError,self.f.build)
    def test_reviewer_source_changed(self):
        self.f.reviewer.write_bytes(b'print(1)\n');self.assertRaises(ValueError,self.f.build)
    def test_post_binding_input_change_poisoned(self):
        c=self.f.build();Path(self.f.assets['entrypoint']['path']).write_bytes(b'changed');self.assertRaises(ValueError,c.take_bootstrap);self.assertTrue(c.failed)
    def test_post_binding_review_change_poisoned(self):
        c=self.f.build();self.f.report.write_bytes(b'{}');self.assertRaises(ValueError,c.assert_unchanged);self.assertTrue(c.failed)
    def test_mutated_contract_poisoned(self):
        c=self.f.build();c.spec['pair'][2]+=1;self.assertRaises(ValueError,c.take_bootstrap);self.assertTrue(c.failed)
    def test_fork_refused(self):
        c=self.f.build()
        with patch('execution_capsule.os.getpid',return_value=c.pid+1):self.assertRaises(ValueError,c.take_bootstrap)
        self.assertTrue(c.failed)
    def test_ancestor_change(self):
        c=self.f.build();self.f.ancestor.write_bytes(b'changed');self.assertRaises(ValueError,c.take_bootstrap)
    def test_ancestor_sidecar(self):
        Path(str(self.f.ancestor)+'-wal').write_bytes(b'unknown');self.assertRaises(ValueError,self.f.build)
    def test_oversized_asset_limit(self):
        self.assertRaises(ValueError,pin_file,self.f.assets['entrypoint'],limit=1)
    def test_report_asset_omission(self):
        self.f.review['assets'].pop('native_schema');self.f.close_review();self.assertRaises(ValueError,self.f.build)
    def test_review_is_mutable_lock(self):
        self.f.spec['paths']['lock']=str(self.f.report);self.f.rebuild();self.assertRaises(ValueError,self.f.build)
    def test_source_is_mutable_lock(self):
        self.f.spec['paths']['lock']=str(self.f.reviewer);self.f.rebuild();self.assertRaises(ValueError,self.f.build)
    def test_shared_interpreter_exact_pin(self):
        self.assertEqual(self.f.trusted['interpreter'],self.f.assets['interpreter']);self.f.build().take_bootstrap()
    def test_changed_parent_identity(self):
        c=self.f.build();self.f.parent.rename(self.f.root/'old-parent');self.f.parent.mkdir();self.assertRaises(ValueError,c.take_bootstrap)
    def test_post_binding_root_created(self):
        c=self.f.build();self.f.output.mkdir();self.assertRaises(ValueError,c.take_bootstrap);self.assertTrue(self.f.output.exists())
    def test_equal_bytes_replaced_input(self):
        c=self.f.build();p=Path(self.f.assets['entrypoint']['path']);raw=p.read_bytes();p.rename(self.f.inputs/'old-entry');p.write_bytes(raw)
        self.assertRaises(ValueError,c.take_bootstrap)
    def test_ancestor_combined_overflow(self):
        p=Path(self.f.assets['ancestor_binding']['path']);v=json.loads(p.read_bytes());v['reserved']['global']=[4000000,16000000];save(p,v)
        self.f.spec['assets']['ancestor_binding']=pin(p);self.f.rebuild();self.assertRaises(ValueError,self.f.build)
    def test_ancestor_path_substitution(self):
        self.refuse_spec(lambda s:s['paths'].update(ancestor=str(self.f.root/'different')))
    def test_declaration_changed_after_external_pin(self):
        self.f.declaration.write_bytes(b'{}');self.assertRaises(ValueError,self.f.build)
    def test_duplicate_json_key(self):
        self.f.declaration.write_bytes(b'{"schema":"a","schema":"b"}')
        self.f.declaration_pin=pin(self.f.declaration);self.assertRaises(ValueError,self.f.build)
if __name__=='__main__':unittest.main()
