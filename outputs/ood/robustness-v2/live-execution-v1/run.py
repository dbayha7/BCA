"""One real, immutable TD3 Hopper v2 attempt; no training or automatic retry."""
from pathlib import Path
import datetime, hashlib, importlib.metadata, json, os, sys, time, traceback

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
REPO = Path('/mnt/c/Users/David Bayha/Documents/GitHub/BCA')
CORE = WORK / 'ood_robustness_v2'
AMENDED = REPO / 'experiments/ood_robustness_v2'
sys.path[:0] = [str(CORE), str(AMENDED), str(REPO)]
PAIR = ('td3_bc', 'hopper', 202609171)
LOCK = Path('/home/dbayha/bca-work/resource-locks/local-rtx5070ti.lock')
LANE = Path('/home/dbayha/bca-work/standard-noiw-v1/queue-local-v1')
ROOT = Path('/home/dbayha/bca-work/ood-live-v2')
KEYROOT = WORK / 'standard_bca_noiw_campaign_v1/monitor_20260928T230801Z'
ANCESTOR = WORK / 'standard_bca_noiw_campaign_v1/monitor_20260928T182657Z/real_ancestor_binding_v2.json'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return hashlib.sha256(raw).hexdigest()

def status(attempt, stage, **values):
    v = dict(utc=now(), pid=os.getpid(), pair=list(PAIR), stage=stage,
             scientific_complete=False, **values)
    tmp = attempt / 'status.tmp'
    with tmp.open('w') as f:
        json.dump(v, f, indent=2); f.flush(); os.fsync(f.fileno())
    tmp.replace(attempt / 'status.json')
    print(json.dumps(v), flush=True)

def accepted_inputs():
    """Retain the closed scientific checks; bind the known Git-attributes append."""
    from experiments.ood.collect import read_artifact, require
    from experiments.ood.real_connection import linux_path
    from experiments.ood.protocol import digest, load_config
    accepted=REPO/'runs/ood/td3-hopper-s202609171-connection-v2'
    closed=json.loads((REPO/'docs/validation/ood-trained-td3-hopper-connection.json').read_bytes())
    pre=read_artifact(accepted/'trained-connection-v1/precommit.json',closed['artifacts_sha256'][
        'outputs/ood/td3_bc/hopper/s202609171/trained-connection-v2/precommit.json'])
    require(closed['trained_cpu_query_gate_accepted'] is True,'Missing trained query gate')
    binding=read_artifact(accepted/'training-binding.json',pre['binding_sha256'])
    full=read_artifact(accepted/'full-declaration.json',pre['declaration_sha256'])
    require(binding['pair_id']=='td3_bc/hopper/s202609171' and binding['training_accepted'] is True,'Wrong accepted pair')
    require(digest({k:v for k,v in full.items() if k!='manifest_sha256'})==full['manifest_sha256']
            and full['config']==load_config(),'Original scientific declaration changed')
    for rel,expected in binding['frozen_source_sha256'].items():
        if rel=='.gitattributes':
            # Only the already reviewed advisor-artifact Git metadata append.
            expected='96b2d7289922f3cf451efd7a6a4ebda265439f4387c9d4ed5d1f96773929054f'
        require(sha(REPO/rel)==expected,'Frozen scientific source changed: '+rel)
    for item in binding['inputs'].values():
        require(sha(linux_path(item['path']))==item['sha256'],'Training audit input changed')
    for rel,expected in pre['source_sha256'].items():
        require(sha(REPO/rel)==expected,'Accepted query source changed: '+rel)
    real=json.loads((REPO/'docs/validation/ood-real-state-connection.json').read_bytes())
    require(real['accepted_engineering'] is True and real['actual_worker_exit']['actual_returncode']==0
            and real['supervisor_actual_exit']==0,'No accepted original simulator connection')
    for rel,expected in real['source_sha256'].items():
        require(sha(REPO/rel)==expected,'Accepted simulator source changed: '+rel)
    pair,=[p for p in full['pairs'] if p['pair_id']==binding['pair_id']]
    return binding,full,pair,pre

def run(attempt):
    import numpy as np
    from ancestor_guard import AncestorGuard, ExclusiveLease
    from extension_ledger import ExtensionLedger, declaration
    from recorded_step import EncodedStore, RecordedStep, content_hash, require
    from native_constructor import construct_recorded
    from collector_driver import collect_states
    from outcome_driver import run_panel
    from precommit_bank_amended_v1 import capture_schedule, build_precommit, PIN_NAMES
    from precommit_file import canonical
    from support_reader import decode_projection
    from experiments.ood.collect import read_artifact, write_artifact
    from experiments.ood.production import paired_reset
    from experiments.ood.real_connection import _load_pair
    from experiments.ood.streaming import ArtifactArchive, ArchiveDirectory, FrozenPolicy

    require(os.environ.get('JAX_PLATFORMS')=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU launch required')
    attempt.mkdir(parents=True, exist_ok=False)
    status(attempt, 'startup')
    # Source/data identity before any model or simulator is used.
    binding, full, old_pair, old_precommit = accepted_inputs()
    files = [HERE/'run.py', HERE/'README.md', ANCESTOR,
             CORE/'training_feasibility.json', KEYROOT/'saved_key_review.json',
             KEYROOT/'saved_key_review_actual_exit.json',
             REPO/'docs/superpowers/plans/2026-09-28-ood-robustness-v2.md']
    files += [CORE/(n+'.py') for n in ('ancestor_guard','extension_ledger','recorded_step','native_constructor',
              'collector_driver','outcome_driver','precommit_bank','precommit_file','candidate_design','support_reader')]
    files += [AMENDED/(n+'.py') for n in ('precommit_bank_amended_v1','candidate_design_amended_v1')]
    sources = {str(p):sha(p) for p in files}
    plan_hash = 'f11ebe8f8e3ccc4af10511ed1a241e8f7c5f0aa9f84770103dd6928e59b60756'
    require(sources[str(files[6])]==plan_hash, 'Scientific plan changed')
    prep = json.loads((CORE/'preparation_validation.json').read_bytes())
    for n in ('candidate_design.py','support_reader.py'):
        require(sha(CORE/n)==prep['code_sha256'][n], 'Frozen scientific source changed: '+n)
    feasibility = json.loads((CORE/'training_feasibility.json').read_bytes())
    require(sha(CORE/'training_feasibility.json')==prep['result_sha256']['training_feasibility.json'], 'Feasibility changed')
    review = json.loads((KEYROOT/'saved_key_review.json').read_bytes())
    ex = json.loads((KEYROOT/'saved_key_review_actual_exit.json').read_bytes())
    require(ex['actual_exit']==0 and review['all_key_values_arithmetic_equal'] is True
            and review['producer_actual_exit']==1 and review['old_v2_overlaps']==0, 'Key review not accepted')
    record, = [r for r in review['records'] if r['kind']=='v2' and r['pair']==list(PAIR)]
    keypath = KEYROOT/record['file']
    require(sha(keypath)==record['sha256'], 'Saved continuation bytes changed')
    keys = np.frombuffer(keypath.read_bytes(), dtype='<u4').reshape(256,250,2).copy()
    key_receipt = review['amended_precommit_bindings'][record['index']]
    support_path = Path('/home/dbayha/bca-work/ood-execution-v1/td3-hopper-s202609171-candidates-v1/support-bank.json')
    support = decode_projection(support_path.read_bytes(), feasibility['support_bank_sha256'])
    schedule = capture_schedule(*PAIR)
    execution = dict(schema='ood-live-execution-v2', pair=list(PAIR), sources=sources,
        scientific_plan_sha256=plan_hash, training_binding_sha256=old_precommit['binding_sha256'],
        checkpoints=old_precommit['checkpoint_sha256'], support_bank_sha256=sha(support_path),
        key_file_sha256=sha(keypath), key_review_sha256=sha(KEYROOT/'saved_key_review.json'),
        key_disposition='exact independently verified numeric bytes accepted under new startup contract; original producer exit1 preserved',
        runtime_policy='ordinary installed library startup; CPU enforced; no Python audit/profile hooks',
        action_absolute_tolerance=1e-6, reward_absolute_tolerance=1e-7, full_state_tolerance=0,
        captures=[100,300], reset_blocks=64, horizon=250, max_slots=14, schedule=schedule,
        phase_order=['engineering','collection','whole_pair_precommit','outcomes','independent_saved_review'],
        attempt=str(attempt), lock=str(LOCK), original_ledger_read_only=True,
        scientific_acceptance=False, no_training=True, no_automatic_retry=True)
    declaration_hash = put(attempt/'declaration.json', execution)
    put(attempt/'environment-before.json', dict(os.environ))
    sim = None
    with ExclusiveLease(LOCK) as lease:
        guard = AncestorGuard(json.loads(ANCESTOR.read_bytes()))
        ledgerpath = ROOT/'extension.sqlite'
        # Only the first attempt may create the campaign ledger. Existing ledgers
        # require an explicit follow-up route; never silently create a fresh budget.
        spec = declaration(ledgerpath, LOCK, guard.binding, declaration_hash)
        put(attempt/'extension-declaration.json', spec)
        with ExtensionLedger.create(ledgerpath, guard, lease, spec) as ledger:
            try:
                from runtime.environment import setup
                setup()
                import jax, jax.numpy as jnp
                from gym.envs.mujoco.hopper import HopperEnv
                from gym.envs.mujoco.mujoco_env import MujocoEnv
                from experiments.ood.simulator import LocomotionAdapter
                require(jax.default_backend()=='cpu', 'Unexpected GPU backend')
                prior_path=REPO/'outputs/ood/td3_bc/hopper/s202609171/trained-connection-v2/runtime.json'
                prior=read_artifact(prior_path,sha(prior_path))
                packages={p:importlib.metadata.version(p) for p in prior['packages']}
                require(packages==prior['packages'], 'Accepted numerical runtime changed')
                runtime_hash=put(attempt/'runtime.json',dict(packages=packages,backend=jax.default_backend(),
                    devices=[str(d) for d in jax.devices()],environment=dict(os.environ)))
                models=_load_pair(REPO,LANE,binding)
                policies={k:FrozenPolicy(v) for k,v in models.items()}
                status(attempt,'models_loaded')
                with ArtifactArchive(attempt/'evidence.sqlite') as archive:
                    store=EncodedStore(archive)
                    sim,constructor=construct_recorded(
                        lambda:LocomotionAdapter('hopper-medium-v2',max_transitions=1847568,evidence_dir=attempt/'native-reference'),
                        HopperEnv,MujocoEnv,ledger,store,PAIR,attempt.name+'-constructor')
                    sim.evidence_dir=ArchiveDirectory(archive)
                    recorded=RecordedStep(sim,ledger,store,archive.read,PAIR,attempt.name)
                    last=[time.monotonic()]
                    def step(phase,owner,action):
                        result=recorded(phase,owner,action)
                        if time.monotonic()-last[0]>30:
                            status(attempt,phase,transitions=sim.transitions,current=owner)
                            last[0]=time.monotonic()
                        return result
                    # Live integration check on fixed engineering resets, with
                    # real input/applied/control/full-state bytes saved first.
                    engineering=[]
                    for method,seed in zip(('host','bca'),full['engineering_streams']['td3_bc/hopper'][6:8]):
                        obs=paired_reset(sim,int(seed));snapshot=sim.capture()
                        direct=models[method].actions(obs[None,:])[0]
                        fast=policies[method].actions(obs[None,:])[0]
                        error=np.abs(direct.astype(np.float64)-fast.astype(np.float64))
                        store.put('engineering/'+method+'/parity',dict(observation=obs,direct=direct,fast=fast,error=error))
                        require((error<=1e-6).all(), 'Actor parity gate failed')
                        first=step('engineering','engineering/'+method+'/first',direct)
                        sim.restore(snapshot,expected_sha256=sim.snapshot_hash(snapshot))
                        repeat=step('engineering','engineering/'+method+'/repeat',fast)
                        require(content_hash(first['record'])==content_hash(repeat['record']) and
                                content_hash(first['after'])==content_hash(repeat['after']), 'Exact first-repeat gate failed')
                        policies[method].seal()
                        engineering.append(dict(method=method,max_action_error=float(error.max()),repeat_exact=True))
                    put(attempt/'engineering.json',dict(live_gate_passed=True,comparisons=engineering,
                        constructor=constructor,environment_transitions=sim.transitions+1,independent_review_pending=True))
                    status(attempt,'collection',transitions=sim.transitions)
                    states=collect_states(sim,policies,PAIR,step,store,paired_reset)
                    write_artifact(attempt/'states.json',states)
                    status(attempt,'precommit',captured=states['captured'],missing=states['missing'],transitions=sim.transitions)
                    pins=dict(state_snapshot='0'*64,training_acceptance=old_precommit['binding_sha256'],
                        host_checkpoint=binding['checkpoints']['host']['checkpoint_sha256'],
                        bca_checkpoint=binding['checkpoints']['bca']['checkpoint_sha256'],
                        prepared_data=content_hash(binding['checkpoints']['host']['preparation_input_hashes']),
                        support_bank=sha(support_path),source_manifest=content_hash(sources),runtime=runtime_hash,
                        execution_declaration=declaration_hash,stream_inventory=sha(KEYROOT/'stream_input_review.json'))
                    require(set(pins)==set(PIN_NAMES),'Pin interface mismatch')
                    banks={};manifest=[]
                    for row in states['rows']:
                        i=row['state_index']
                        if row['status']=='missing':
                            manifest.append(dict(index=i,status='missing',state_artifact_sha256=row['artifact_sha256']));continue
                        obs=row['observation'];normalized=np.asarray((jnp.asarray(obs)-models['host'].mean)/models['host'].std)
                        sq=np.square(support['observations'].astype(np.float64)-normalized).mean(axis=1)
                        order=np.argsort(sq,kind='stable')[:32];neighbors=support['actions'][order]
                        actors={k:p.actions(obs[None,:])[0] for k,p in policies.items()}
                        statepins=dict(pins,state_snapshot=row['snapshot_content_sha256'])
                        bank=build_precommit(pair=PAIR,state_index=i,pins=statepins,key_receipt=key_receipt,keys=keys[i],
                            anchor_sent=neighbors[0],neighboring_actions=neighbors,host_sent=actors['host'],bca_sent=actors['bca'],
                            q95=feasibility['action_q95'],q99=feasibility['action_q99'],
                            state_distance=float(np.sqrt(sq[order[0]])),state_q95=feasibility['state_q95'])
                        # All coordinates are irrevocably committed before width queries.
                        bankhash=hashlib.sha256(canonical(bank)).hexdigest()
                        store.put(f'precommit/state{i:03d}/bank',bank)
                        slots=[r for r in bank['slots'] if r['status']=='present']
                        applied=np.stack([np.frombuffer(bytes.fromhex(r['applied']['hex']),dtype=np.float32) for r in slots])
                        score=models['bca'].score(np.repeat(obs[None,:],len(slots),axis=0),applied)
                        random=np.random.default_rng(row['random_score_seed']).random(14)
                        scorehash=store.put(f'precommit/state{i:03d}/scores',dict(slots=[r['slot'] for r in slots],
                            bca=score,host_width=None,random=random,constant=np.zeros(14)))
                        banks[i]=(bank,bankhash)
                        manifest.append(dict(index=i,status='captured',bank_sha256=bankhash,scores_sha256=scorehash,
                            state_artifact_sha256=row['artifact_sha256']))
                    for p in policies.values():p.seal()
                    require(len(manifest)==256 and len(banks)==states['captured'],'Incomplete whole-pair precommit')
                    require(all(sha(p)==h for p,h in sources.items()),'Pinned execution source changed')
                    put(attempt/'whole-pair-precommit.json',dict(rows=manifest,keys_sha256=sha(keypath),
                        before_outcomes=True,outcome_calls=0,declaration_sha256=declaration_hash))
                    status(attempt,'outcomes',captured=states['captured'],missing=states['missing'],panels_completed=0,transitions=sim.transitions)
                    panels=[]
                    for row in states['rows']:
                        i=row['state_index']
                        if i not in banks:continue
                        bank,bankhash=banks[i]
                        for continuation in ('host','bca'):
                            result=run_panel(sim=sim,policy=policies[continuation],snapshot=row['snapshot'],bank=bank,
                                bank_sha256=bankhash,snapshot_sha256=row['snapshot_content_sha256'],keys=keys[i],
                                continuation=continuation,step=step,store=store)
                            result_hash=write_artifact(attempt/f'panel-{i:03d}-{continuation}.json',result)
                            panels.append(dict(index=i,continuation=continuation,sha256=result_hash))
                            status(attempt,'outcomes',panels_completed=len(panels),panels_expected=2*states['captured'],transitions=sim.transitions)
                    for p in policies.values():p.seal()
                    require(all(sha(p)==h for p,h in sources.items()),'Execution sources changed')
                    resources=ledger.audit()
                    put(attempt/'execution-complete.json',dict(panels=panels,resources=resources,
                        archive_records=archive.index,archive_head=archive.previous,declaration_sha256=declaration_hash,
                        execution_finished=True,scientific_acceptance=False,independent_review_required=True))
                    status(attempt,'execution_finished_pending_independent_review',panels_completed=len(panels),transitions=sim.transitions)
            except BaseException as exc:
                put(attempt/'failure.json',dict(utc=now(),error=repr(exc),traceback=traceback.format_exc(),
                    simulator_transitions=None if sim is None else sim.transitions,no_retry=True))
                status(attempt,'failed',error=repr(exc));raise
            finally:
                if sim is not None:sim.close()

if __name__=='__main__':
    run(Path(sys.argv[1]))
