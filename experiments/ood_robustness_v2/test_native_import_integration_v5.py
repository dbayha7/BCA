"""OPEN unified composition fixtures; no full filelock or scientific import."""
import ast,base64,hashlib,importlib.util,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
C=Path(__file__).resolve().parent;W=C.parent/'standard_bca_noiw_campaign_v1/monitor_20260929T054337Z'
BASE=Path('/home/dbayha/bca-work/ood-native-v5-integration-fixtures/monitor_20260929T054337Z')
CASES=('complete','wrong_order','wrong_keyword_order','wrong_callback','production_route','direct_alias','wrapper_changed','captured_default_changed','native_owner_changed','premature_completion','repeat_registration','callback_after_capability','threading_prior_changed','partial_after_return','registry_after_return','extra_native_registration','profile_removal','wrong_combined_result','reject_completion_ack','profile_internal_error','production_parent')
DRIVER='''import os,sys,atexit,threading,random,logging
import concurrent.futures.thread as thread
import binding_fixture_api as api
import binding_fixture_rw as rw
import binding_fixture_soft as soft
TOKEN=object();FIRST=object();SECOND=object()
def run_rest():
    os.register_at_fork(after_in_child=rw._abort_forked_sqlite_transition) # rw fork
    atexit.register(soft._cleanup_all_instances) # soft exit
    os.register_at_fork(after_in_child=random._inst.seed) # random fork
    os.register_at_fork(before=thread._global_shutdown_lock.acquire,after_in_child=thread._global_shutdown_lock._at_fork_reinit,after_in_parent=thread._global_shutdown_lock.release) # native fork
    os.register_at_fork(before=logging._acquireLock,after_in_child=logging._after_at_fork_child_reinit_locks,after_in_parent=logging._releaseLock) # logging fork
    atexit.register(logging.shutdown) # logging exit
    threading._register_atexit(api._thread_callback,TOKEN,first=FIRST,second=SECOND) # internal threading
'''
sha=lambda b:hashlib.sha256(b).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def run(label):
    assert label in ('integration_tests_v1','integration_tests_v2','integration_tests_v3','integration_tests_v4','integration_tests_v5');root=BASE/label;root.mkdir(parents=True,exist_ok=False)
    protocol=load('integration_parent',C/'native_import_protocol_v2.py')
    helper_raw=(C/'test_native_registration_bindings_v1.py').read_bytes();assert sha(helper_raw)=='cb8384c6abfd8e5f03ac4498f08065aa930d6c5c12ac7f75ae276f0f508e46f6'
    helper=load('closed_source_fixture_helper',C/'test_native_registration_bindings_v1.py');fixture,defs=helper.sources();fixture['driver.py']=DRIVER.encode()
    saved=json.loads((W.parent/'monitor_20260929T004802Z/scratch_source_inspection.json').read_bytes());item=next(v for k,v in saved['sources'].items() if k.endswith('/filelock/_strict.py'));strict=base64.b64decode(item['base64']);assert sha(strict)==item['sha256']=='379d37ee7580cf6005e7d2e5c8f32c3a0e08c062f5dc3eb1eb88614049f7bd04'
    node=next(n for n in ast.parse(strict).body if isinstance(n,ast.FunctionDef) and n.name=='_probe_link_follow_symlinks');definition=ast.get_source_segment(strict.decode(),node)
    capability=('import os\nimport tempfile\nfrom pathlib import Path\n_HAS_LINK=hasattr(os,"link")\n\n'+definition+'\n\n_LINK_HONORS_FOLLOW_SYMLINKS=_probe_link_follow_symlinks()\n').encode()
    dispatcher=(C/'native_import_dispatch_v5.py').read_bytes();protocol_raw=(C/'native_import_protocol_v2.py').read_bytes();worker=(C/'native_import_integration_entry_v5.py').read_bytes();binder=(C/'native_registration_bindings_v1.py').read_bytes();declaration=(C/'native_import_contract_v3.json').read_bytes();decl=json.loads(declaration)
    assert sha(dispatcher)==protocol.DISPATCHER and sha(binder)=='997fdc4653f204df3d799f47c604937075a15d001a2cb99b5200009cf384f251'
    config=json.dumps(dict(source_sha256=sha(capability),definition_sha256=sha(definition.encode()))).encode();results=[]
    for case in CASES:
        p=root/case;p.mkdir(mode=0o700);parent_id=protocol.directory_id(p.stat());env=dict(decl['environment'],TMPDIR=str(p/'scratch'))
        binding=json.dumps(dict(schema='native-import-v5-integration-fixture-v1',purpose='integration_fixture_only',declaration_sha256=protocol.CONTRACT,parent=str(p),parent_id=parent_id,scratch=str(p/'scratch'),diagnostics=str(p/'diagnostics'),environment=env,fixture_source_sha256=sha(worker))).encode()
        control=json.dumps(dict(case=case,stdlib_sha256=helper.HASHES,fixture_sha256={n:sha(b) for n,b in fixture.items()})).encode()
        inputs=dict(fixture,**{'binder.py':binder,'worker.py':worker,'protocol.py':protocol_raw,'dispatcher.py':dispatcher,'declaration.json':declaration,'binding.json':binding,'capability.py':capability,'capability_config.json':config,'fixture_control.json':control})
        for n,b in inputs.items():(p/n).write_bytes(b)
        spec=dict(purpose='native_import_integration_fixture_only',parent=str(p),parent_id=parent_id,environment=env,protocol_sha256=sha(protocol_raw),dispatcher_sha256=sha(dispatcher),entry_sha256=sha(worker),binding_sha256=sha(binding),capability_sha256=sha(capability),capability_config_sha256=sha(config),control_sha256=sha(control),integration_inputs={n:sha(inputs[n]) for n in ('binder.py','api.py','rw.py','soft.py','driver.py')},command=[sys.executable,'-I',str(p/'worker.py'),str(p)],fixture_timeout_seconds=180,stdout_limit=8388608,stderr_limit=8388608)
        if case=='production_parent':spec['purpose']='production_native_import'
        (p/'parent_spec.json').write_text(json.dumps(spec,indent=2));receipt=None;error=None
        try:receipt=protocol.NativeImportParent(spec).run()
        except BaseException as e:error=type(e).__name__+': '+str(e)
        row=dict(case=case,parent_fixture=str(p),receipt=receipt,prelaunch_exception=error,actual_child_exit=None if receipt is None else receipt['actual_child_exit'],input_sha256={n:sha(b) for n,b in inputs.items()})
        results.append(row);(p/'test_result.json').write_text(json.dumps(row,indent=2));(W/(label+'_children.json')).write_text(json.dumps(results,indent=2));print(case,row['actual_child_exit'],error,flush=True)
        if case=='production_parent':assert error and receipt is None and not (p/'supervision').exists()
        elif case=='complete':assert error is None and receipt['completed'] and receipt['actual_child_exit']==0 and receipt['release_sent'] and receipt['completion_ack_sent'] and receipt['worker_ack_received'],row
        else:assert error is None and not receipt['completed'] and receipt['actual_child_exit']==1,row
        if case=='reject_completion_ack':assert receipt['completion_ack_sent'] and not receipt['worker_ack_received']
        if receipt is not None:
            assert b'UNEXPECTED_NORMAL_SHUTDOWN' not in (p/'supervision/stdout').read_bytes()
            assert (p/'supervision/stderr').read_bytes()==b''
            if case!='complete':
                failure=next(x for x in receipt['packets'] if x['stage']=='worker_failure');assert failure['profile_present'] is True
                if case=='profile_internal_error':assert failure['origin']=='profile_callback' and failure['error_type']=='AttributeError' and not failure['diagnostic_write_attempted']
    print(json.dumps(dict(tests=len(results),skipped=0,full_actual_route_accepted=False,draft_open=True)))
if __name__=='__main__':run(sys.argv[1])
