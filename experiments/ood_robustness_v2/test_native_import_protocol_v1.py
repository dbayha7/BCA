"""New actual process protocol fixtures; never full native target imports."""
import ast,base64,hashlib,importlib.util,json,os,stat,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
W=HERE.parent/'standard_bca_noiw_campaign_v1/monitor_20260929T044506Z'
BASE=Path('/home/dbayha/bca-work/ood-native-v4-capability-fixtures/monitor_20260929T044506Z')
CASES=('complete','failure_after_create','exit_without_completion','death_after_fsync','timeout_before_ready','stdout_limit','stderr_limit','wrong_ready','fake_completion','reject_completion_ack','method_override','parent_spec_mutated','production_purpose','changed_environment','changed_source','preexisting_scratch','repeat_parent','completion_state_mutated','completion_before_capability','repeat_release','wrong_capability_result','profile_removal','parent_input_hash_changed')
WORKER='''import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
p=Path(sys.argv[1]);control=json.loads((p/'fixture_control.json').read_bytes());case=control['case']
if case=='timeout_before_ready':time.sleep(10)
if case in ('stdout_limit','stderr_limit'):
    stream=sys.stdout.buffer if case=='stdout_limit' else sys.stderr.buffer
    for unused in range(129):stream.write(b'x'*65535+b'\\n');stream.flush()
m=load('protocol',p/'protocol.py');d=load('dispatcher',p/'dispatcher.py')
g=d.NativeImportDispatcherV4((p/'declaration.json').read_bytes(),(p/'binding.json').read_bytes(),m.DISPATCHER)
proto=m.WorkerProtocol(g,hashlib.sha256((p/'protocol.py').read_bytes()).hexdigest(),hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
try:
    g.install()
    if case=='wrong_ready':
        proto.emit(dict(stage='ready',identity={},environment={}));sys.stdin.readline();raise ValueError('Unexpected release')
    proto.ready_release();g.create();g.write_diagnostic('events.jsonl',b'{"event":"bootstrap_complete"}\\n')
    if case=='failure_after_create':raise ValueError('Declared synthetic post-create failure')
    if case=='exit_without_completion':os._exit(0)
    if case=='death_after_fsync':os._exit(23)
    if case=='fake_completion':
        proto.emit(dict(stage='completion_pending',identity=proto.owner));sys.stdin.readline();raise ValueError('Unexpected completion ack')
    if case=='method_override':proto.emit=lambda packet:None;proto.check()
    if case=='completion_before_capability':proto.prepare_completion({})
    if case=='repeat_release':proto.ready_release()
    if case=='profile_removal':sys.setprofile(None)
    config=json.loads((p/'capability_config.json').read_bytes())
    g.bind_capability(str(p/'capability.py'),'v4_fixture_capability',config['source_sha256'],config['definition_sha256'])
    load('v4_fixture_capability',p/'capability.py');result=g.capability_result()
    if case=='wrong_capability_result':result['root_retained_empty']=False
    proto.prepare_completion(result)
    if case=='completion_state_mutated':proto.pending['profile_present']=False
    if case=='reject_completion_ack':
        actual=proto.receive();assert actual['stage']=='completion_ack';raise ValueError('Declared synthetic rejection after actual parent acknowledgment')
    proto.accept_completion()
except BaseException as error:proto.abort(error)
'''

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def run(label):
    assert label in ('protocol_tests_v1','protocol_tests_v2','protocol_tests_v3')
    root=BASE/label;root.mkdir(parents=True,exist_ok=False)
    protocol=load('parent_protocol',HERE/'native_import_protocol_v1.py')
    code=(HERE/'native_import_protocol_v1.py').read_bytes();dispatcher=(HERE/'native_import_dispatch_v4.py').read_bytes();declaration=(HERE/'native_import_contract_v3.json').read_bytes();decl=json.loads(declaration)
    sha=lambda b:hashlib.sha256(b).hexdigest();assert sha(dispatcher)==protocol.DISPATCHER
    saved=json.loads((W.parent/'monitor_20260929T004802Z/scratch_source_inspection.json').read_bytes());item=next(v for k,v in saved['sources'].items() if k.endswith('/filelock/_strict.py'))
    strict=base64.b64decode(item['base64'],validate=True);assert sha(strict)==item['sha256']=='379d37ee7580cf6005e7d2e5c8f32c3a0e08c062f5dc3eb1eb88614049f7bd04'
    node=next(n for n in ast.parse(strict).body if isinstance(n,ast.FunctionDef) and n.name=='_probe_link_follow_symlinks');definition=ast.get_source_segment(strict.decode(),node)
    capability=('import os\nimport tempfile\nfrom pathlib import Path\n_HAS_LINK = hasattr(os, "link")\n\n'+definition+'\n\n_LINK_HONORS_FOLLOW_SYMLINKS = _probe_link_follow_symlinks()\n').encode()
    config=json.dumps(dict(source_sha256=sha(capability),definition_sha256=sha(definition.encode()))).encode();worker=WORKER.encode();results=[]
    for case in CASES:
        p=root/case;p.mkdir(mode=0o700);parent_id=protocol.directory_id(p.stat());env=dict(decl['environment'],TMPDIR=str(p/'scratch'))
        binding=json.dumps(dict(schema='native-import-v4-capability-fixture-v1',purpose='capability_fixture_only',declaration_sha256=protocol.CONTRACT,parent=str(p),parent_id=parent_id,scratch=str(p/'scratch'),diagnostics=str(p/'diagnostics'),environment=env,fixture_source_sha256=sha(worker))).encode()
        control=json.dumps(dict(case=case)).encode()
        inputs={'worker.py':worker,'protocol.py':code,'dispatcher.py':dispatcher,'declaration.json':declaration,'binding.json':binding,'capability.py':capability,'capability_config.json':config,'fixture_control.json':control}
        for n,b in inputs.items():(p/n).write_bytes(b)
        spec=dict(purpose='native_import_completion_fixture_only',parent=str(p),parent_id=parent_id,environment=env,protocol_sha256=sha(code),dispatcher_sha256=sha(dispatcher),entry_sha256=sha(worker),binding_sha256=sha(binding),capability_sha256=sha(capability),capability_config_sha256=sha(config),control_sha256=sha(control),command=[sys.executable,'-I',str(p/'worker.py'),str(p)],fixture_timeout_seconds=.5 if case=='timeout_before_ready' else 180,stdout_limit=8388608,stderr_limit=8388608)
        if case=='production_purpose':spec['purpose']='production_native_import'
        if case=='changed_environment':spec['environment']=dict(env,EXTRA='1')
        if case=='changed_source':(p/'worker.py').write_bytes(worker+b'\n# changed before launch\n')
        if case=='preexisting_scratch':(p/'scratch').mkdir(mode=0o700)
        (p/'parent_spec.json').write_text(json.dumps(spec,indent=2))
        receipt=None;exception=None;repeat_refused=False
        try:
            parent=protocol.NativeImportParent(spec)
            if case=='parent_spec_mutated':parent.spec['fixture_timeout_seconds']=1
            if case=='parent_input_hash_changed':(p/'binding.json').write_bytes(binding+b' ')
            receipt=parent.run()
            if case=='repeat_parent':
                try:parent.run()
                except ValueError:repeat_refused=True
        except BaseException as e:exception=type(e).__name__+': '+str(e)
        r=dict(case=case,parent_fixture=str(p),receipt=receipt,prelaunch_exception=exception,actual_child_exit=None if receipt is None else receipt['actual_child_exit'],repeat_refused=repeat_refused,input_sha256={n:sha(b) for n,b in inputs.items()})
        results.append(r);(W/(label+'_children.json')).write_text(json.dumps(results,indent=2));(p/'test_result.json').write_text(json.dumps(r,indent=2))
        if case in ('complete','repeat_parent'):
            assert exception is None and receipt['completed'] and receipt['actual_child_exit']==0 and receipt['release_sent'] and receipt['completion_ack_sent'] and receipt['worker_ack_received'],r
            assert receipt['observed_before_release']==receipt['observed_before_completion_ack'] and receipt['diagnostics_readback']
            assert repeat_refused==(case=='repeat_parent')
        elif case in ('parent_spec_mutated','production_purpose','changed_environment','changed_source','preexisting_scratch','parent_input_hash_changed'):
            assert exception is not None and receipt is None and not (p/'supervision').exists(),r
        else:
            assert exception is None and receipt is not None and not receipt['completed'],r
            if case=='death_after_fsync':assert receipt['actual_child_exit']==23
            if case=='exit_without_completion':assert receipt['actual_child_exit']==0 and not receipt['completion_ack_sent']
            if case in ('stdout_limit','stderr_limit'):assert receipt['overflow']['stream']==case.split('_')[0] and receipt['log_bytes'][case.split('_')[0]]==8388608
            if case=='reject_completion_ack':assert receipt['completion_ack_sent'] and not receipt['worker_ack_received']
            if case in ('failure_after_create','reject_completion_ack','method_override','completion_state_mutated','completion_before_capability','repeat_release','wrong_capability_result','profile_removal'):assert receipt['actual_child_exit']==1,r
        assert not any(n in sys.modules for n in ('filelock','numpy','jax','mujoco_py','glfw','gym','d4rl'))
        print(case,'child',r['actual_child_exit'],'completed',None if receipt is None else receipt['completed'],flush=True)
    print(json.dumps(dict(tests=len(results),skipped=0,full_native_route_accepted=False,production_dispatch=False)))
if __name__=='__main__':run(sys.argv[1])
