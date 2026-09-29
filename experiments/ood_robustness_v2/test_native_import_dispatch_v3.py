"""New persistent isolated bootstrap fixtures; no target imports or callback replay."""
import hashlib,importlib.util,json,os,select,stat,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
MONITOR=HERE.parent/'standard_bca_noiw_campaign_v1/monitor_20260929T034335Z'
BASE=Path('/home/dbayha/bca-work/ood-native-v3-bootstrap-fixtures/monitor_20260929T034335Z')
CASES=('complete','before_release','wrong_release','repeat_release','repeat_install','repeat_create',
       'outside_write','extra_directory','direct_ticket','direct_native_alias','environment',
       'additional_hook','synthetic_target_event','bad_diagnostic_name','byte_ceiling',
       'binding_changed','identity_changed','method_inventory','preexisting_scratch','preexisting_diagnostics',
       'root_replaced','root_mode_changed','diagnostic_hardlink','diagnostic_appended','source_changed',
       'mutation_after_refusal','observer_suppressed','observer_install_exception','death_after_fsync',
       'module_helper_changed','module_limit_changed','namespace_changed','method_defaults_changed',
       'native_alias_keyword_call','synthetic_rename_event')


def emit(v):
    print(json.dumps(v,sort_keys=True),flush=True)


def child(case,parent):
    p=Path(parent);binding=(p/'binding.json').read_bytes();b=json.loads(binding)
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==b['fixture_source_sha256']
    module_path=p/'dispatcher.py';spec=importlib.util.spec_from_file_location('fixture_dispatcher',module_path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    assert not [n for n in sys.modules if n.split('.')[0] in ('filelock','numpy','jax','mujoco_py','glfw','gym','d4rl')]
    g=module.NativeImportDispatcherV3((p/'declaration.json').read_bytes(),binding,hashlib.sha256(module_path.read_bytes()).hexdigest())
    if case in ('observer_suppressed','observer_install_exception'):
        def prior(event,args):
            if event=='sys.addaudithook':
                if case=='observer_suppressed':raise RuntimeError('Expected own-hook suppression fixture')
                raise KeyboardInterrupt('Expected own-hook installation exception fixture')
        sys.addaudithook(prior)
    expected=case not in ('complete','death_after_fsync')
    caught=False;diagnostic_after_refusal=False;creation_ack=None;write_ack=None
    try:
        g.install()
        emit(dict(stage='ready',identity=g.identity,environment=g.environment))
        observed=json.loads(sys.stdin.readline())
        if case=='before_release':g.create()
        g.release(observed)
        if case=='wrong_release':raise AssertionError('Wrong parent release was accepted')
        if case=='repeat_release':g.release(observed)
        if case=='repeat_install':g.install()
        g.create();creation_ack=g.summary()
        write_ack=g.write_diagnostic('events.jsonl',b'{"event":"worker_bootstrap_fsynced"}\n')
        emit(dict(stage='created',creation=creation_ack,write_ack=write_ack))
        assert sys.stdin.readline()=='continue\n'
        if case=='death_after_fsync':os._exit(23)
        if case=='repeat_create':g.create()
        elif case=='outside_write':open(str(p/'outside'), 'wb')
        elif case=='extra_directory':os.mkdir(str(p/'extra'),0o700)
        elif case=='direct_ticket':g._native('write',1,b'FORBIDDEN_NATIVE_PAYLOAD')
        elif case=='direct_native_alias':g.api['write'](1,b'FORBIDDEN_NATIVE_PAYLOAD')
        elif case=='native_alias_keyword_call':os.mkdir(str(p/'extra'),mode=0o700)
        elif case=='synthetic_rename_event':sys.audit('os.rename','synthetic-source','synthetic-target',-1,-1)
        elif case=='environment':os.environ['V3_UNDECLARED']='1'
        elif case=='additional_hook':sys.addaudithook(lambda *args:None)
        elif case=='synthetic_target_event':sys.audit('import','filelock',None,[],[],[])
        elif case=='bad_diagnostic_name':g.write_diagnostic('../outside',b'x')
        elif case=='byte_ceiling':
            g.write_diagnostic('maps.txt',b'x'*module.MAX_FILE)
            g.write_diagnostic('maps.txt',b'y')
        elif case=='binding_changed':g.binding['purpose']='changed';g.check()
        elif case=='identity_changed':g.identity['pid']+=1;g.check()
        elif case=='method_inventory':g.methods.pop('_admit');g.check()
        elif case=='module_helper_changed':module.process_identity=lambda:g.identity;g.check()
        elif case=='module_limit_changed':module.MAX_FILE*=2;g.check()
        elif case=='namespace_changed':module.__file__='different.py';g.check()
        elif case=='method_defaults_changed':type(g).check.__defaults__=(True,);g.check()
        elif case=='mutation_after_refusal':
            try:g.write_diagnostic('../outside',b'x')
            except module.DispatchRefusal:pass
            open(str(p/'outside'),'wb')
        elif case in ('root_replaced','root_mode_changed','diagnostic_hardlink','diagnostic_appended','source_changed'):g.check()
        if expected:raise AssertionError('Expected fixture refusal not observed')
    except module.DispatchRefusal:
        caught=True;first=g.first_refusal
        if case in ('direct_native_alias','native_alias_keyword_call','synthetic_rename_event'):
            assert first['caller_source']==__file__ and first['caller_line']>0
        if case in ('direct_native_alias','native_alias_keyword_call'):
            assert first['actual_positional_metadata'] is None and first['ordered_keyword_metadata'] is None
        if case=='synthetic_rename_event':
            assert first['actual_positional_metadata']==[{'type':'str','value':'synthetic-source'},{'type':'str','value':'synthetic-target'},{'type':'int','value':-1},{'type':'int','value':-1}]
        try:g.check()
        except module.DispatchRefusal:pass
        assert first==g.first_refusal,'First refusal changed on poisoned recheck'
        if set(g.descriptors)==set(module.FILES):
            try:
                g.write_diagnostic('final.json',(json.dumps(dict(first_refusal=first,expected_fixture_refusal=True))+'\n').encode())
                diagnostic_after_refusal=True
            except module.DispatchRefusal:
                assert first==g.first_refusal
    except BaseException as e:
        emit(dict(stage='unexpected',case=case,error_type=type(e).__name__,message=str(e),first_refusal=g.first_refusal))
        os._exit(2)
    if caught!=expected:
        emit(dict(stage='unexpected',case=case,caught=caught,expected=expected,first_refusal=g.first_refusal));os._exit(3)
    if not expected:
        g.write_diagnostic('final.json',b'{"bootstrap_only":true,"native_import_accepted":false}\n')
        g.check()
    result=dict(stage='result',case=case,expected_refusal=expected,caught=caught,first_refusal=g.first_refusal,
                diagnostic_after_refusal=diagnostic_after_refusal,profile_present=sys.getprofile() is g.profile_hook,
                own_audit_canary_seen=g.install_seen,phase=g.phase,native_events=g.events,
                child_identity=g.identity,creation_ack=creation_ack,initial_write_ack=write_ack,
                diagnostic_bytes=g.written,total_diagnostic_bytes=g.total,
                target_modules=[n for n in sys.modules if n.split('.')[0] in ('filelock','numpy','jax','mujoco_py','glfw','gym','d4rl')],
                actual_production_profile_executed=False,full_route_implemented=False,scientific_execution_accepted=False)
    emit(result);os._exit(0)


def readline(p,timeout=20):
    ready,_,_=select.select([p.stdout],[],[],timeout)
    if not ready:raise TimeoutError('Fixture handshake timed out')
    return p.stdout.readline()


def inspect_process(pid):
    q=Path('/proc')/str(pid)
    tail=(q/'stat').read_bytes().rsplit(b') ',1)[1].split()
    raw=(q/'environ').read_bytes();assert raw.endswith(b'\0')
    pairs=[x.decode().split('=',1) for x in raw[:-1].split(b'\0')];assert len(pairs)==len(dict(pairs))
    return dict(identity=dict(pid=pid,parent_pid=int(tail[1]),session=int(tail[3]),group=int(tail[2]),start_ticks=int(tail[19]),argv_hex=(q/'cmdline').read_bytes().hex()),environment=dict(pairs))


def run_all(label):
    root=BASE/label;root.mkdir(parents=True,exist_ok=False)
    source=(HERE/'native_import_dispatch_v3.py').read_bytes();test=Path(__file__).read_bytes()
    declaration=(HERE/'native_import_contract_v3.json').read_bytes();d=json.loads(declaration)
    results=[]
    for case in CASES:
        parent=root/case;parent.mkdir(mode=0o700);s=parent.stat()
        b=dict(schema='native-import-v3-bootstrap-fixture-v1',purpose='bootstrap_fixture_only',declaration_sha256=hashlib.sha256(declaration).hexdigest(),parent=str(parent),parent_id=[s.st_dev,s.st_ino,s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode)],
               scratch=str(parent/'scratch'),diagnostics=str(parent/'diagnostics'),environment=dict(d['environment'],TMPDIR=str(parent/'scratch')),fixture_source_sha256=hashlib.sha256(test).hexdigest())
        (parent/'binding.json').write_text(json.dumps(b,indent=2));(parent/'declaration.json').write_bytes(declaration)
        (parent/'dispatcher.py').write_bytes(source);(parent/'fixture.py').write_bytes(test)
        if case.startswith('preexisting_'):(parent/case.removeprefix('preexisting_')).mkdir(mode=0o700)
        before={n:(parent/n).exists() for n in ('scratch','diagnostics')}
        cmd=[sys.executable,'-I',str(parent/'fixture.py'),'--child',case,str(parent)]
        started=time.time();p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=b['environment'],start_new_session=True,bufsize=0)
        lines=[];observed=None;created=None;mutation=None
        try:
            first=readline(p);lines.append(first);item=json.loads(first)
            if item.get('stage')=='ready':
                observed=inspect_process(p.pid)
                assert observed['identity']['parent_pid']==os.getpid() and observed['identity']['pid']==observed['identity']['session']==observed['identity']['group']
                assert observed==dict(identity=item['identity'],environment=b['environment'])
                release=json.loads(json.dumps(observed))
                if case=='wrong_release':release['identity']['start_ticks']+=1
                p.stdin.write((json.dumps(release)+'\n').encode());p.stdin.flush()
                second=readline(p);lines.append(second);item=json.loads(second)
                if item.get('stage')=='created':
                    created=item
                    if case=='root_replaced':
                        os.rename(parent/'scratch',parent/'scratch-retained');(parent/'scratch').mkdir(mode=0o700);mutation='replaced_scratch_directory'
                    elif case=='root_mode_changed':os.chmod(parent/'scratch',0o750);mutation='scratch_mode_changed'
                    elif case=='diagnostic_hardlink':os.link(parent/'diagnostics/maps.txt',parent/'maps-alias');mutation='synthetic_diagnostic_hardlink'
                    elif case=='diagnostic_appended':
                        with (parent/'diagnostics/maps.txt').open('ab') as f:f.write(b'parent-fixture-change')
                        mutation='synthetic_parent_append'
                    elif case=='source_changed':
                        with (parent/'dispatcher.py').open('ab') as f:f.write(b'\n# synthetic source mutation\n')
                        mutation='synthetic_source_append'
                    p.stdin.write(b'continue\n');p.stdin.flush()
            tail,err=p.communicate(timeout=25);out=b''.join(lines)+tail
        except BaseException:
            if p.poll() is None:os.killpg(p.pid,15)
            tail,err=p.communicate(timeout=5);out=b''.join(lines)+tail
            (parent/'parent_failed.stdout').write_bytes(out);(parent/'parent_failed.stderr').write_bytes(err)
            raise
        rows=[json.loads(x) for x in out.splitlines()]
        record=dict(case=case,command=cmd,actual_exit=p.returncode,elapsed_seconds=time.time()-started,parent_pid=os.getpid(),observed_before_release=observed,
                    roots_existed_before_launch=before,creation_ack=created,parent_fixture_mutation=mutation,
                    stdout_sha256=hashlib.sha256(out).hexdigest(),stderr_sha256=hashlib.sha256(err).hexdigest(),rows=rows,
                    retained_paths=[str(x.relative_to(parent)) for x in sorted(parent.rglob('*'))],fixture_parent=str(parent))
        (parent/'child.stdout').write_bytes(out);(parent/'child.stderr').write_bytes(err);(parent/'child_exit.json').write_text(json.dumps(record,indent=2))
        results.append(record)
        (MONITOR/(label+'_children.json')).write_text(json.dumps(results,indent=2))
        assert p.returncode==(23 if case=='death_after_fsync' else 0),(case,p.returncode,out.decode(errors='replace'),err.decode(errors='replace'))
        assert b'FORBIDDEN_NATIVE_PAYLOAD' not in out
        if case!='death_after_fsync':
            final=rows[-1];assert final['stage']=='result' and not final['target_modules'],(case,final)
            assert final['expected_refusal']==(case!='complete')
        if created:
            assert (parent/'diagnostics/events.jsonl').read_bytes()==b'{"event":"worker_bootstrap_fsynced"}\n'
        assert not (parent/'outside').exists() and not (parent/'extra').exists()
        print(case,p.returncode,flush=True)
    print(json.dumps(dict(tests=len(results),skipped=0,ordinary_child_zero=sum(r['actual_exit']==0 for r in results),deliberate_death23=sum(r['actual_exit']==23 for r in results),target_imports=False)))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--child':child(sys.argv[2],sys.argv[3])
    else:run_all(sys.argv[1])
