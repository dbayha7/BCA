"""One reserved central-quartile job; source-level execution hold is enforced."""
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import os
import time

from common import HERE, PAYLOAD, read, sha, digest, atomic_json, atomic_bytes, module, require


def resume_step(job):
    """Only the specifically approved durable checkpoint may be continued."""
    if not any(k in job for k in ('resume_from_step','resume_source_job')):
        return None
    require(job.get('resume_from_step') == 16000
        and type(job.get('resume_from_step')) is int
        and job.get('resume_source_job') == 'seed2_hole/bca'
        and job.get('kind') == 'branch' and job.get('cell') == 'seed2_hole'
        and job.get('seed_index') == 2 and job.get('condition') == 'hole'
        and job.get('arm') == 'bca', 'Undeclared continuation source or boundary')
    return 16000


def validate_job(protocol,job):
    fixed = {"schema":"unifloral.central_quartile_training.v1", "warmup_critic_updates":10000,
        "branch_critic_updates":10000,"warmup_actor_updates":5000,"branch_actor_updates":5000,
        "warmup_auxiliary_updates":10000,"branch_auxiliary_updates":10000,"bca_kappa":3.,
        "actor_distance_factor":19.29619572569095,"batch_size":256,"gamma":.98,
        "progress_every":20,"warmup_timeout_seconds":1800,"branch_timeout_seconds":1800,
        "checkpoint_every":1000,"calibration_boundaries":[10000,15000],"final_critic_step":20000,
        "initial_residual_scale":1.,"automatic_restart":False,"resume":False}
    require(all(protocol.get(k) == v and not (type(v) is not bool and isinstance(protocol.get(k),bool))
                for k,v in fixed.items()),"Worker protocol constants differ")
    require(job["kind"] in ("warmup","branch") and type(job["seed_index"]) is int and job["seed_index"] in (0,1,2),"Invalid job")
    require(job["condition"] in ("full","hole","random")
        and job["cell"] == f"seed{job['seed_index']}_{job['condition']}","Cell identity mismatch")
    if job["kind"] == "branch":
        require(job["arm"] in ("native","bca","distance"),"Invalid arm")
    step = resume_step(job)
    return 20000-step if step is not None else protocol[job['kind'] + '_critic_updates']


def expected_segment_updates(count_keys,job,budget):
    refreshes = 0 if resume_step(job) is not None else 1
    return dict(zip(count_keys,(budget,budget//2,budget//2,budget//2,budget//2,budget,budget,refreshes,refreshes)))


def immutable_bytes(path, raw):
    """Exclusive evidence: a partial write remains visible and cannot be replaced."""
    path = Path(path)
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {'path':path.name, 'sha256':digest(raw), 'bytes':len(raw)}


def immutable_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    return immutable_bytes(path,raw)


def save_checkpoint(output, raw_state, raw_actor, *, counts, job, protocol_sha,
                    normalization, matching_receipt_sha256=None):
    """Seal the full training state at a declared boundary; never authorize resume."""
    step = counts['critic']
    require(type(step) is int and step in range(0,20001,1000), 'Invalid immutable checkpoint boundary')
    folder = Path(output) / 'checkpoints' / f'step_{step:05d}'
    folder.mkdir(parents=True,exist_ok=False)
    state = immutable_bytes(folder/'full_state.msgpack',raw_state)
    actor = immutable_bytes(folder/'actor.msgpack',raw_actor) if raw_actor is not None else None
    manifest = {'schema':'unifloral.central_quartile.checkpoint.v1',
        'protocol_sha256':protocol_sha,'cell':job['cell'],'condition':job['condition'],
        'seed_index':job['seed_index'],'kind':job['kind'],'arm':job.get('arm','native'),
        'absolute_critic_step':step,'counts':counts,'artifact':state,'actor_params':actor,
        'normalization':normalization,'complete_optimizer_target_rng_state':True,
        'matching_receipt_sha256':matching_receipt_sha256,'resume_authorized':False}
    seal = immutable_json(folder/'manifest.json',manifest)
    return {'path':folder.relative_to(output).as_posix()+'/manifest.json',**{k:seal[k] for k in ('sha256','bytes')}}


def verify_warmup(path,protocol_sha,job):
    from recovery_inputs import job_directory, prior_root, authenticate_prior_file
    path = Path(path).resolve()
    require(path == job_directory(job['cell'],'warmup').resolve(),"Warmup source outside declared cell")
    def authenticated(name):
        return authenticate_prior_file((path/name).relative_to(prior_root()).as_posix())
    raw_report = authenticated('report.json')
    supervisor,report = json.loads(authenticated('SUPERVISOR_RESULT.json')),json.loads(raw_report)
    require(supervisor.get("completed") and supervisor.get("report_sha256") == digest(raw_report),"Warmup supervisor/report seal mismatch")
    require(report["status"] == "completed_central_quartile_warmup" and report["protocol_sha256"] == protocol_sha
        and report["cell"] == job["cell"] and report["condition"] == job["condition"]
        and report["seed_index"] == job["seed_index"] and report["execution_counts_known"],"Warmup not eligible")
    artifacts = {name:authenticated(name+'.json') for name in ('actor_manifest','state_manifest','matching_receipt')}
    for name,raw in artifacts.items():
        require(digest(raw) == report[name+'_sha256'],'Warmup artifact seal differs: '+name)
    actor,state_manifest,matching = (json.loads(artifacts[name]) for name in ('actor_manifest','state_manifest','matching_receipt'))
    artifact = state_manifest["artifact"]
    raw_state = authenticated('final_state.msgpack')
    require(artifact == actor["final_state"] and artifact["path"] == "final_state.msgpack"
        and digest(raw_state) == artifact["sha256"]
        and len(raw_state) == artifact["bytes"],"Warmup full state mismatch")
    return report,actor,matching,raw_state


def import_resume_prefix(output,job,protocol_sha,normalization,matching_sha):
    """Copy only authenticated durable evidence; no inference or optimizer calls."""
    from recovery_inputs import authenticate_prior_file
    require(resume_step(job) == 16000,'Resume prefix requires the approved branch')
    source = 'run/'+job['resume_source_job']+'/'
    raw_progress = authenticate_prior_file(source+'progress.json')
    progress = json.loads(raw_progress)
    require(progress['job']['cell'] == job['cell'] and progress['job']['arm'] == job['arm'],
            'Prior progress identifies another job')
    raw_matching = authenticate_prior_file(source+'matching_receipt.json')
    require(digest(raw_matching) == matching_sha and (Path(output)/'matching_receipt.json').read_bytes() == raw_matching,
            'Prior matching receipt differs from restored warmup')
    checkpoints,gradients = {},{}
    resume_raw,resume_manifest = None,None
    for step in range(11000,16001,1000):
        seal = progress['checkpoint_manifests'][str(step)]
        relative = f'checkpoints/step_{step:05d}/manifest.json'
        require(seal['path'] == relative,'Prior checkpoint path differs')
        raw_manifest = authenticate_prior_file(source+relative)
        require(digest(raw_manifest) == seal['sha256'] and len(raw_manifest) == seal['bytes'],
                'Prior checkpoint progress seal differs')
        manifest = json.loads(raw_manifest)
        require(all(manifest.get(k) == v for k,v in dict(protocol_sha256=protocol_sha,cell=job['cell'],
            condition=job['condition'],seed_index=job['seed_index'],kind='branch',arm='bca',
            absolute_critic_step=step,normalization=normalization,matching_receipt_sha256=matching_sha,
            complete_optimizer_target_rng_state=True).items()), 'Prior checkpoint identity differs')
        require(manifest['counts']['critic'] == step and manifest['counts']['phase'] == 0,
                'Prior checkpoint counter boundary differs')
        folder = Path(output)/Path(relative).parent
        folder.mkdir(parents=True,exist_ok=False)
        for key,name in (('artifact','full_state.msgpack'),('actor_params','actor.msgpack')):
            artifact = manifest[key]
            if artifact is None:
                require(key == 'actor_params' and step != 15000,'Missing required prior actor export')
                continue
            require(artifact['path'] == name,'Prior checkpoint artifact path differs')
            raw = authenticate_prior_file(source+Path(relative).parent.as_posix()+'/'+name)
            require(digest(raw) == artifact['sha256'] and len(raw) == artifact['bytes'],
                    'Prior checkpoint payload differs')
            immutable_bytes(folder/name,raw)
            if step == 16000 and key == 'artifact': resume_raw = raw
        immutable_bytes(folder/'manifest.json',raw_manifest)
        checkpoints[str(step)] = seal
        if step == 16000: resume_manifest = manifest
    gradient_name = 'gradient_audit_015000.json'
    raw_gradient = authenticate_prior_file(source+gradient_name)
    gradient = json.loads(raw_gradient)
    require(all(gradient.get(k) == v for k,v in dict(absolute_critic_step=15000,protocol_sha256=protocol_sha,
        cell=job['cell'],condition=job['condition'],seed_index=job['seed_index'],arm='bca').items()),
        'Prior gradient audit identity differs')
    gradients['15000'] = immutable_bytes(Path(output)/gradient_name,raw_gradient)
    raw_refresh = authenticate_prior_file(source+'refresh_receipt.json')
    require(json.loads(raw_refresh)['absolute_critic'] == 15000,'Prior calibration boundary differs')
    refresh = immutable_bytes(Path(output)/'refresh_receipt.json',raw_refresh)
    lineage = {'source_job':job['resume_source_job'],'resume_from_step':16000,
        'source_checkpoint':{'path':source+checkpoints['16000']['path'],'sha256':checkpoints['16000']['sha256']},
        'source_progress_sha256':digest(raw_progress),'imported_checkpoints':checkpoints.copy(),
        'imported_gradient_audits':gradients.copy(),'imported_refresh_receipt':refresh,
        'prefix_unique_replay_source_rows':None,
        'prefix_unique_row_note':'The original full-state checkpoint omits the Python seen-row bitmap. New row identities and ledger counts cover only the resumed segment.',
        'prior_interruption_tail_counts_known':False,'reseeded':False,'recalibrated_before_updates':False}
    return resume_raw,resume_manifest,checkpoints,gradients,lineage


def restore_resume_state(runtime,adapter,template,raw,manifest):
    """Exact decoded complete-state and scheduler validation before any dispatch."""
    diagnostic = {}
    restored = runtime.restore_exact(adapter,template,raw,diagnostic=diagnostic)
    checked = runtime.validate(adapter,restored)
    require(runtime.host(checked['counts']) == manifest['counts'],'Restored continuation counts differ')
    return restored,{'status':'passed_exact_continuation_restore','exact_state':diagnostic,
        'counts':runtime.host(checked['counts']),'checks':runtime.host(checked['checks']),
        'optimizer_updates':0,'calibration_updates':0,'simulator_transitions':0}


def run(protocol,output_dir,job):
    import launch
    launch.require_execution_authorization(protocol)
    started = time.monotonic(); output = Path(output_dir).resolve()
    require(output.is_dir() and not Path(output_dir).is_symlink(),"Reserved output directory required")
    require(not any((output / name).exists() for name in ("progress.json","report.json","final_state.msgpack","partial_state.msgpack","untrained_actor.msgpack")),"Existing scientific evidence; no retry")
    runtime = module("_quartile_runtime",HERE / "runtime.py")
    budget = validate_job(protocol,job)
    require(read(HERE / "protocol.json") == protocol,"Passed protocol differs from frozen file")
    protocol_sha = sha(HERE / "protocol.json")
    cap = min(protocol[job['kind']+'_timeout_seconds'],job.get('timeout_seconds',0))
    require(math.isfinite(cap) and cap > 0,'Finite supervisor worker budget required')
    hard_deadline = min(started+cap, job.get('deadline_monotonic',0))
    require(math.isfinite(hard_deadline) and hard_deadline > started,'Active supervisor deadline required')
    stage,state,adapter = "initialization",None,None
    initial,last_metrics,attempted,inflight = None,None,{},False
    logical_initial,logical_initial_rng,continuation_lineage,restore_receipt = None,None,None,None
    is_resumed = resume_step(job) is not None
    checkpoint_manifests = {}
    gradient_artifacts = {}
    anchor_params = None
    led = {"committed_additional_updates":{k:0 for k in runtime.COUNT_KEYS},"attempted_uncommitted_updates":{},
        "replay_row_draws":0,"critic_row_uses":0,"auxiliary_row_uses":0,"realized_unique_replay_source_rows":0,
        "simulator_transitions":0,"evaluation_episodes":0,"checkpoint_selection_reads":0,
        "gradient_audit_snapshots_attempted":0,"gradient_audit_snapshots_completed":0,
        "gradient_audit_probe_rows_attempted":0,"gradient_audit_probe_rows_completed":0,
        "forced_kill_counts":"Last progress is a committed lower bound; up to20 later critic/auxiliary dispatches may be unrecorded."}
    if is_resumed:
        led.update(accounting_scope='New 16000-to-20000 segment only; imported evidence is not recomputed',
            prior_unique_replay_source_rows=None,imported_gradient_snapshots=1,
            imported_gradient_probe_rows=768,prior_execution_counts_known=False)
    def deadline():
        require(time.monotonic() < hard_deadline,"Fixed worker deadline reached")
    def journal(status="running"):
        atomic_json(output / "progress.json",runtime.host({"status":status,"job":job,"stage":stage,
            "ledger":led,"checkpoint_manifests":checkpoint_manifests,
            "elapsed_seconds":time.monotonic()-started,"updated_utc":datetime.now(timezone.utc).isoformat()}))
    def deltas(final):
        return {k:final[k]-initial[k] for k in runtime.COUNT_KEYS}
    def gradient_snapshot(absolute_step):
        nonlocal stage
        deadline()
        require(absolute_step in (10000,15000,20000), 'Undeclared gradient snapshot')
        require(str(absolute_step) not in gradient_artifacts, 'Gradient snapshot already recorded')
        expected_calls = 1 if job['kind']=='warmup' or is_resumed else 2
        require(led['gradient_audit_snapshots_attempted'] < expected_calls, 'Gradient diagnostic cap exceeded')
        stage = 'fixed_training_gradient_diagnostic'
        led['gradient_audit_snapshots_attempted'] += 1
        led['gradient_audit_probe_rows_attempted'] += 768
        def update(counts):
            led['current_gradient_dispatch_counts'] = counts
            journal()
            deadline()
        journal()
        helper=module('_central_gradient_audit', HERE/'gradient_audit.py')
        result=helper.audit(adapter,state,context.data.receipt['gradient_probe_source_row_ids'],
                            anchor_params=anchor_params,progress=update)
        result.update(absolute_critic_step=absolute_step,protocol_sha256=protocol_sha,
                      cell=job['cell'],condition=job['condition'],seed_index=job['seed_index'])
        artifact=immutable_json(output/f'gradient_audit_{absolute_step:06d}.json',runtime.host(result))
        gradient_artifacts[str(absolute_step)]=artifact
        led['gradient_audit_snapshots_completed'] += 1
        led['gradient_audit_probe_rows_completed'] += 768
        stage = 'training'
        journal()
    def manifest(params,counts,kind,arm,final_state=None):
        artifact = atomic_bytes(output / ("untrained_actor.msgpack" if kind=="untrained" else "final_actor.msgpack"),serialization.to_bytes(params))
        result = {"schema":"unifloral.central_quartile.actor.v1","completed":True,"kind":kind,"arm":arm,
            "cell":job["cell"],"condition":job["condition"],"seed_index":job["seed_index"],
            "protocol_sha256":protocol_sha,"native_source_sha256":protocol["native_source_sha256"],
            "native_revision":protocol["native_revision"],"actor_params":artifact,
            "parameter_leaves":runtime.parameter_leaves(params),"normalization":context.normalization,
            "counts":counts,"final_counts":counts,"initial_parameter_hashes":context.initial_parameter_hashes,
            "architecture":{"class":"DeterministicTanhActor","observation_dim":14,"action_dim":4}}
        if final_state is not None:
            result["final_state"] = final_state
        return atomic_json(output / ("untrained_actor_manifest.json" if kind=="untrained" else "actor_manifest.json"),result)
    try:
        journal(); deadline()
        parent = verify_warmup(job["warmup_path"],protocol_sha,job) if job["kind"]=="branch" else None
        adapter,state = runtime.create(protocol,job["seed_index"],job["condition"])
        context = adapter.context
        import numpy as np
        from flax import serialization
        atomic_json(output / "runtime_receipt.json",runtime.host(context.runtime))
        data_artifact = atomic_json(output / "data_receipt.json",context.data.receipt)
        require(context.data.receipt["data_plan_sha256"] == protocol["data_plan_sha256"],"Data plan identity changed")
        fresh_counts = context.training.counts(state)
        untrained_manifest = None; matching = None; matching_artifact = None
        if parent is not None:
            warmup_report,warmup_actor,matching,raw_state = parent
            require(context.initial_parameter_hashes == warmup_report["initial_parameter_hashes"],"Fresh template initialization differs from warmup")
            require(context.normalization == warmup_actor["normalization"] and data_artifact["sha256"] == warmup_report["data_receipt_sha256"],"Warmup data/normalization mismatch")
            state = runtime.restore_exact(adapter,state,raw_state)
            require(runtime.tree_digest(state.aux.posterior) == matching["reference_digest"]
                and runtime.tree_digest(state.microstep.agent_state.actor.params) == matching["anchor_actor_digest"],"Matching reference/anchor differs")
            anchor_params = state.microstep.agent_state.actor.params
            adapter,state = runtime.branch(adapter,state,protocol,job["seed_index"],job["arm"],matching["distance_coefficient"])
            matching_artifact = immutable_json(output/'matching_receipt.json',matching)
            logical_initial = context.training.counts(state)
            logical_initial_rng = {"native":runtime.host(state.microstep.rng),"auxiliary":runtime.host(state.aux_rng)}
            if is_resumed:
                stage = 'authenticated_continuation_restore'; journal(); deadline()
                raw_resume,resume_manifest,checkpoint_manifests,gradient_artifacts,continuation_lineage = import_resume_prefix(
                    output,job,protocol_sha,context.normalization,matching_artifact['sha256'])
                # branch() ran only on the authenticated 10k warmup. Restoring 16k
                # overwrites every dynamic field, including both RNG streams and
                # the already accepted 15k calibration. Never branch/reseed again.
                state,restore_receipt = restore_resume_state(runtime,adapter,state,raw_resume,resume_manifest)
                restore_receipt['source_checkpoint'] = continuation_lineage['source_checkpoint']
                immutable_json(output/'continuation_restore_receipt.json',runtime.host(restore_receipt))
                immutable_json(output/'continuation_lineage.json',continuation_lineage)
        else:
            untrained_manifest = manifest(state.microstep.agent_state.actor.params,fresh_counts,"untrained","native")
            checkpoint_manifests['0'] = save_checkpoint(output,serialization.to_bytes(state),
                serialization.to_bytes(state.microstep.agent_state.actor.params),counts=fresh_counts,
                job=job,protocol_sha=protocol_sha,normalization=context.normalization)
        initial = context.training.counts(state)
        initial_rng = {"native":runtime.host(state.microstep.rng),"auxiliary":runtime.host(state.aux_rng)}
        if logical_initial is None:
            logical_initial,logical_initial_rng = initial,initial_rng
        seen = np.zeros(len(context.data.source_row_ids),bool)
        stage = "training"; journal(); deadline()
        for index in range(budget):
            deadline(); phase = int(np.asarray(state.microstep.phase))
            sampled = np.asarray(adapter.sample_indices(state.microstep.rng)) if phase==0 else None
            inflight = True
            state,last_metrics = runtime.step(adapter,state)
            inflight = False
            if sampled is not None:
                # Reconstruct the pure pinned phase0 sampler without changing its RNG.
                seen[sampled] = True
                led["replay_row_draws"] += len(sampled)
            led["critic_row_uses"] += 256; led["auxiliary_row_uses"] += 256
            led["realized_unique_replay_source_rows"] = int(seen.sum())
            led["committed_additional_updates"] = deltas(last_metrics["counts"])
            if last_metrics["refresh_due"]:
                atomic_json(output / "refresh_receipt.json",runtime.host({"absolute_critic":last_metrics["counts"]["critic"],
                    "counts":last_metrics["counts"],"refresh":last_metrics["refresh"],
                    "ordering":"Boundary actor uses old reference; subsequent actor uses new reference"}))
            absolute_step = int(last_metrics['counts']['critic'])
            if absolute_step == 15000:
                gradient_snapshot(absolute_step)
            if absolute_step % protocol['checkpoint_every'] == 0 and index+1 < budget:
                checkpoint_manifests[str(absolute_step)] = save_checkpoint(output,serialization.to_bytes(state),
                    serialization.to_bytes(state.microstep.agent_state.actor.params) if absolute_step == 15000 else None,
                    counts=runtime.host(last_metrics['counts']),job=job,protocol_sha=protocol_sha,
                    normalization=context.normalization,
                    matching_receipt_sha256=sha(output/'matching_receipt.json') if (output/'matching_receipt.json').exists() else None)
            if (index+1)%protocol['progress_every'] == 0:
                journal()
        deadline(); final = runtime.validate(adapter,state)
        expected = expected_segment_updates(runtime.COUNT_KEYS,job,budget)
        require(deltas(final["counts"]) == expected,"Committed budget mismatch")
        unique_rows_artifact = atomic_bytes(output / "unique_replay_source_row_ids.i64",np.asarray(context.data.source_row_ids[seen],dtype="<i8").tobytes())
        matching_gradients_artifact = None
        if job["kind"] == "warmup":
            stage = "fixed_training_probe_match"; journal(); deadline()
            before = runtime.tree_digest(state)
            matching,bca_grad,distance_grad = runtime.match(adapter,state)
            require(before == runtime.tree_digest(state),"Matching mutated complete warmup state")
            matching.update(cell=job["cell"],condition=job["condition"],seed_index=job["seed_index"],
                protocol_sha256=protocol_sha,data_plan_sha256=protocol["data_plan_sha256"],
                probe_source_row_ids_sha256=context.data.receipt["gradient_probe_source_row_ids_sha256"])
            matching_artifact = atomic_json(output / "matching_receipt.json",matching)
            # Cache exact gradients for arithmetic review; no post-hoc rematching.
            import io
            buffer = io.BytesIO(); np.savez(buffer,bca=bca_grad,distance=distance_grad)
            matching_gradients_artifact = atomic_bytes(output / "matching_gradients.npz",buffer.getvalue())
        else:
            require(matching_artifact is not None and sha(output/'matching_receipt.json') == matching_artifact['sha256'],
                    'Frozen branch matching receipt changed')
        gradient_snapshot(int(final['counts']['critic']))
        deadline(); stage = "sealed_exports"
        absolute_step = int(final['counts']['critic'])
        checkpoint_manifests[str(absolute_step)] = save_checkpoint(output,serialization.to_bytes(state),
            serialization.to_bytes(state.microstep.agent_state.actor.params),counts=runtime.host(final['counts']),
            job=job,protocol_sha=protocol_sha,normalization=context.normalization,
            matching_receipt_sha256=matching_artifact['sha256'])
        state_artifact = atomic_bytes(output / "final_state.msgpack",serialization.to_bytes(state))
        state_manifest = atomic_json(output / "state_manifest.json",{"schema":"unifloral.central_quartile.state.v1",
            "artifact":state_artifact,"protocol_sha256":protocol_sha,"cell":job["cell"],"condition":job["condition"],
            "seed_index":job["seed_index"],"kind":job["kind"],"counts":final["counts"],
            "state_leaves":runtime.parameter_leaves(state),"normalization":context.normalization,
            "resume_authorized":False,"complete_optimizer_target_rng_state":True})
        actor_manifest = manifest(state.microstep.agent_state.actor.params,final["counts"],job["kind"],job.get("arm","native"),state_artifact)
        report = {"schema":"unifloral.central_quartile.training_result.v1","status":"completed_central_quartile_"+job["kind"],
            "kind":job["kind"],"cell":job["cell"],"condition":job["condition"],"seed_index":job["seed_index"],
            "arm":job.get("arm","native"),"protocol_sha256":protocol_sha,"execution_counts_known":True,
            "initial_counts":logical_initial,"final_counts":final["counts"],
            "additional_updates":{k:final['counts'][k]-logical_initial[k] for k in runtime.COUNT_KEYS},
            "segment_initial_counts":initial,"segment_additional_updates":deltas(final['counts']),
            "segment_initial_rng":initial_rng,"continuation_lineage":continuation_lineage,
            "continuation_restore_receipt":restore_receipt,
            "logical_branch_counts_known":True,"prior_physical_execution_counts_known":False if is_resumed else True,
            "initial_parameter_hashes":context.initial_parameter_hashes,"initial_optimizer_steps":context.initial_optimizer_steps,
            "initial_optimizer_hashes":context.initial_optimizer_hashes,"initial_fork_rng":logical_initial_rng,
            "final_rng":{"native":runtime.host(state.microstep.rng),"auxiliary":runtime.host(state.aux_rng)},
            "actor_manifest_sha256":actor_manifest["sha256"],"state_manifest_sha256":state_manifest["sha256"],
            "matching_receipt_sha256":matching_artifact["sha256"],"data_receipt_sha256":data_artifact["sha256"],
            "untrained_actor_manifest_sha256":untrained_manifest["sha256"] if untrained_manifest else None,
            "unique_replay_source_rows":unique_rows_artifact,"matching_gradients":matching_gradients_artifact,
            "unique_replay_source_rows_scope":"New resumed segment only; prior unique identities unknown" if is_resumed else "Entire job",
            "checkpoint_manifests":checkpoint_manifests,
            "gradient_audits":gradient_artifacts,
            "checks":final["checks"],"ledger":led,"last_update_metrics":last_metrics,
            "distance_coefficient":matching["distance_coefficient"],"elapsed_seconds":time.monotonic()-started,
            "initialization":("Static adapter built from authenticated 10k warmup and original branch RNG fold, then exact complete 16k restore; no reseed or repeated calibration"
                if is_resumed else "fresh parameters/optimizers; branch reconstructs static template then exactly restores its10k complete state"),
            "scale_bank_role":protocol["scale_bank_role"],"error":None,"full_training_hold":True}
        atomic_json(output / "report.json",runtime.host(report)); journal(report["status"])
        return runtime.host(report)
    except BaseException as exc:
        if hasattr(exc,"last_committed_state"):
            state = exc.last_committed_state; attempted = exc.attempted_counts; stage = exc.stage
        if hasattr(exc,'gradient_audit_ledger'):
            led['current_gradient_dispatch_counts'] = exc.gradient_audit_ledger
        led["attempted_uncommitted_updates"] = attempted
        if state is not None and adapter is not None:
            try:
                from flax import serialization
                if initial is not None:
                    led["committed_additional_updates"] = deltas(adapter.context.training.counts(state))
                atomic_bytes(output / "partial_state.msgpack",serialization.to_bytes(state))
            except Exception as save_error:
                led["partial_state_error"] = str(save_error)
        report = {"status":"failed_central_quartile_training","job":job,"stage":stage,"error":type(exc).__name__+": "+str(exc),
            "execution_counts_known":not inflight and not any(attempted.values())
                and led['gradient_audit_snapshots_attempted']==led['gradient_audit_snapshots_completed'],"ledger":led,
            "elapsed_seconds":time.monotonic()-started,"protocol_sha256":protocol_sha,"automatic_restart":False,
            "diagnostics":getattr(exc,"diagnostics",None)}
        report['checkpoint_manifests'] = checkpoint_manifests
        report['gradient_audits'] = gradient_artifacts
        report['continuation_lineage'] = continuation_lineage
        report['segment_initial_counts'] = initial
        atomic_json(output / "report.json",runtime.host(report)); journal(report["status"])
        return runtime.host(report)
