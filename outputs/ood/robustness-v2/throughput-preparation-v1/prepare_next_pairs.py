from pathlib import Path
import ast,datetime,hashlib,json
W=Path(__file__).resolve().parent;R=Path('C:/Users/David Bayha/Documents/GitHub/BCA')
L=W.parents[1]/'ood_live_v2'
sha=lambda b:hashlib.sha256(b).hexdigest()
definitions=[('hopper',202609172,'standard-second-rebrac-hopper-pair'),
             ('hopper',202609173,'standard-third-rebrac-hopper-pair'),
             ('walker2d',202609171,'standard-first-rebrac-walker-pair'),
             ('walker2d',202609172,'standard-second-rebrac-walker-pair'),
             ('walker2d',202609173,'standard-third-rebrac-walker-pair')]
pairs=[]
for env,seed,name in definitions:
    acceptance=R/f'docs/validation/{name}.json';accepted=json.loads(acceptance.read_bytes())
    assert accepted['accepted_for_recorded_training_comparison'] and accepted['actual_audit_exits']==dict(host=0,bca=0)
    methods={}
    for method,arm in [('host','host'),('bca','bca_noiw')]:
        path=R/f'outputs/standard_bca/rebrac/{env}/{arm}/s{seed}/verified-v1/audit.json'
        raw=path.read_bytes();audit=json.loads(raw)
        assert sha(raw)==accepted['exact_export_hashes'][method+'-verified/audit.json']
        assert audit['accepted'] and audit['host_updates']==1000000 and audit['actor_updates']==500000
        assert audit['actual_worker_exit']['actual_returncode']==0 and not audit['actual_worker_exit']['timeout']
        assert audit['run_id']==f'rebrac-{env}-'+('host' if method=='host' else 'bca-noiw')+f'-s{seed}'
        final,=[c for c in audit['checkpoints'] if c['step']==1000000]
        methods[method]=dict(audit_path=str(path),audit_sha256=sha(raw),run_id=audit['run_id'],checkpoint_sha256=final['sha256'])
    pairs.append(dict(pair=['rebrac',env,seed],acceptance_path=str(acceptance),acceptance_sha256=sha(acceptance.read_bytes()),
                      methods=methods,query_root=f'/users/dbayha/bca-ood-v2/rebrac-{env}-s{seed}-query-throughput-v1',
                      proposed_outcome_lease=f'/users/dbayha/bca-ood-v2/pair-leases/rebrac-{env}-s{seed}.lock',
                      engineering_cap_proposed=2000 if env=='walker2d' else 0,
                      phase_caps=dict(collection=38400,repeat_checks=7168,outcomes=1792000),
                      physics_dispatch_accepted=False,query_preparation_only=True))
parent=(L/'cluster_query_gate.py').read_bytes().decode()
source=parent.replace("    manifest=json.loads", "    intent=json.loads((ROOT/'pair-intent.json').read_bytes())\n    pair=intent['pair']\n    assert pair[0]=='rebrac' and pair[1] in ('hopper','walker2d') and pair[2] in range(202609171,202609176)\n    manifest=json.loads",1)
source=source.replace("        run=LANE/'runs'/audit['run_id'];", "        assert audit['run_id']==intent['methods'][method]['run_id']\n        assert sha(ROOT/(method+'-audit.json'))==intent['methods'][method]['audit_sha256']\n        run=LANE/'runs'/audit['run_id'];",1)
source=source.replace("pair=['rebrac','hopper',202609171]",'pair=pair')
source=source.replace("np.savez(f,**score)","np.savez(f,**score,direct_width=np.asarray(direct.width),direct_dose=np.asarray(direct.dose))")
assert source!=parent and "pair=['rebrac','hopper',202609171]" not in source
ast.parse(source)
(W/'query_next_pair.py').write_bytes(source.encode())
manifest=dict(schema='ood-accepted-next-pair-preparation-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              pairs=pairs,source_parent_sha256=sha(parent.encode()),source_sha256=sha(source.encode()),
              model_weights_transferred=False,scientific_sources_changed=False,training_reaudited=False)
(W/'next_pair_manifest.json').write_bytes(json.dumps(manifest,indent=2).encode())
allocation=dict(schema='ood-parallel-allocation-proposal-v1',status='PROPOSED_NOT_PHYSICS_DISPATCH_ACCEPTED',
                original_environment_cap=39998400,closed_ancestor_reserved=1298353,
                local_owner='td3_bc',local_half_max=18395680,cluster_owner='rebrac',cluster_half_max=18395680,
                cluster_phase_max_per_pair=1837568,cluster_phase_max_all_ten=18375680,
                engineering_by_cell={'hopper':10000,'walker2d':10000},
                legacy_hopper171_engineering_entitlement_retained=10000,
                unstarted_hopper_engineering_currently_allocated=0,
                walker_engineering_proposed_per_seed=2000,walker_seeds=[202609171,202609172,202609173,202609174,202609175],
                duplicate_pair_blocked=['rebrac','hopper',202609171],
                currently_training_accepted_unstarted=[p['pair'] for p in pairs],
                required_before_hopper_parallel_physics='Independently bind first worker engineering phase closed and its actual reserved engineering totals; explicitly repartition unused entitlement without refunding calls.',
                required_before_any_parallel_physics=['Atomic persistent allocation claim and duplicate exclusion across legacy/new routes.',
                    'Pair lease exclusion/death tests and cumulative per-pair cap; failed claims are never silently reopened.',
                    'Accepted checkpoint query, selected storage integration and live actor/restore/reward gates.'])
assert allocation['closed_ancestor_reserved']+allocation['local_half_max']+allocation['cluster_half_max']==38089713<39998400
assert 10*1837568+10000+5*2000==18395680
assert len(set(tuple(p['pair']) for p in pairs))==5
(W/'parallel_allocation_proposal.json').write_bytes(json.dumps(allocation,indent=2).encode())
print(json.dumps(dict(prepared_pairs=len(pairs),query_source_sha256=sha(source.encode()),physics_dispatch_accepted=False)))
