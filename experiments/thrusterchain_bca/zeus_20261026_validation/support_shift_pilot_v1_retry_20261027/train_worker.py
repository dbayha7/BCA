"""One reserved support-pilot job; no imports initialize models or dispatch work."""
from datetime import datetime, timezone
from pathlib import Path
import time

from common import HERE, PAYLOAD, read, sha, digest, atomic_json, atomic_bytes, module, require


def validate_job(protocol,job):
    fixed = {"schema":"unifloral.support_shift_pilot.v1", "warmup_critic_updates":10000,
        "branch_critic_updates":2000,"warmup_actor_updates":5000,"branch_actor_updates":1000,
        "warmup_auxiliary_updates":10000,"branch_auxiliary_updates":2000,"bca_kappa":3.,
        "actor_distance_factor":19.29619572569095,"batch_size":256,"gamma":.98,
        "progress_every":20,"warmup_timeout_seconds":1800,"branch_timeout_seconds":300,
        "initial_residual_scale":1.,"automatic_restart":False,"resume":False}
    require(all(protocol.get(k) == v for k,v in fixed.items()),"Worker protocol constants differ")
    require(job["kind"] in ("warmup","branch") and type(job["seed_index"]) is int and job["seed_index"] in (0,1),"Invalid job")
    require(job["condition"] in ("restricted","broad_count_matched")
        and job["cell"] == f"seed{job['seed_index']}_{job['condition']}","Cell identity mismatch")
    if job["kind"] == "branch":
        require(job["arm"] in ("native","bca","distance"),"Invalid arm")
    return 10000 if job["kind"] == "warmup" else 2000


def verify_warmup(path,protocol_sha,job):
    path = Path(path).resolve()
    require(path == (HERE / "run" / job["cell"] / "warmup").resolve(),"Warmup source outside declared cell")
    supervisor,report = read(path / "SUPERVISOR_RESULT.json"),read(path / "report.json")
    require(supervisor.get("completed") and supervisor.get("report_sha256") == sha(path / "report.json"),"Warmup supervisor/report seal mismatch")
    require(report["status"] == "completed_support_warmup" and report["protocol_sha256"] == protocol_sha
        and report["cell"] == job["cell"] and report["condition"] == job["condition"]
        and report["seed_index"] == job["seed_index"] and report["execution_counts_known"],"Warmup not eligible")
    for name in ("actor_manifest","state_manifest","matching_receipt"):
        require(sha(path / (name+".json")) == report[name+"_sha256"],"Warmup artifact seal differs: "+name)
    actor,state_manifest,matching = read(path / "actor_manifest.json"),read(path / "state_manifest.json"),read(path / "matching_receipt.json")
    artifact = state_manifest["artifact"]
    require(artifact == actor["final_state"] and artifact["path"] == "final_state.msgpack"
        and sha(path / artifact["path"]) == artifact["sha256"]
        and (path / artifact["path"]).stat().st_size == artifact["bytes"],"Warmup full state mismatch")
    return report,actor,matching,(path / artifact["path"]).read_bytes()


def run(protocol,output_dir,job):
    started = time.monotonic(); output = Path(output_dir).resolve()
    require(output.is_dir() and not Path(output_dir).is_symlink(),"Reserved output directory required")
    require(not any((output / name).exists() for name in ("progress.json","report.json","final_state.msgpack","partial_state.msgpack","untrained_actor.msgpack")),"Existing scientific evidence; no retry")
    runtime = module("_support_runtime",HERE / "runtime.py")
    budget = validate_job(protocol,job)
    require(read(HERE / "protocol.json") == protocol,"Passed protocol differs from frozen file")
    protocol_sha = sha(HERE / "protocol.json")
    cap = 1800 if job["kind"] == "warmup" else 300
    stage,state,adapter = "initialization",None,None
    initial,last_metrics,attempted,inflight = None,None,{},False
    led = {"committed_additional_updates":{k:0 for k in runtime.COUNT_KEYS},"attempted_uncommitted_updates":{},
        "replay_row_draws":0,"critic_row_uses":0,"auxiliary_row_uses":0,"realized_unique_replay_source_rows":0,
        "simulator_transitions":0,"evaluation_episodes":0,"checkpoint_selection_reads":0,
        "forced_kill_counts":"Last progress is a committed lower bound; up to20 later critic/auxiliary dispatches may be unrecorded."}
    def deadline():
        require(time.monotonic()-started < cap,"Fixed worker deadline reached")
    def journal(status="running"):
        atomic_json(output / "progress.json",runtime.host({"status":status,"job":job,"stage":stage,
            "ledger":led,"elapsed_seconds":time.monotonic()-started,"updated_utc":datetime.now(timezone.utc).isoformat()}))
    def deltas(final):
        return {k:final[k]-initial[k] for k in runtime.COUNT_KEYS}
    def manifest(params,counts,kind,arm,final_state=None):
        artifact = atomic_bytes(output / ("untrained_actor.msgpack" if kind=="untrained" else "final_actor.msgpack"),serialization.to_bytes(params))
        result = {"schema":"unifloral.support_pilot.actor.v1","completed":True,"kind":kind,"arm":arm,
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
        untrained_manifest = None; matching = None
        if parent is not None:
            warmup_report,warmup_actor,matching,raw_state = parent
            require(context.initial_parameter_hashes == warmup_report["initial_parameter_hashes"],"Fresh template initialization differs from warmup")
            require(context.normalization == warmup_actor["normalization"] and data_artifact["sha256"] == warmup_report["data_receipt_sha256"],"Warmup data/normalization mismatch")
            state = serialization.from_bytes(state,raw_state)
            require(serialization.to_bytes(state) == raw_state,"Complete warmup restore not exact")
            runtime.validate(adapter,state)
            require(runtime.tree_digest(state.aux.posterior) == matching["reference_digest"]
                and runtime.tree_digest(state.microstep.agent_state.actor.params) == matching["anchor_actor_digest"],"Matching reference/anchor differs")
            adapter,state = runtime.branch(adapter,state,protocol,job["seed_index"],job["arm"],matching["distance_coefficient"])
        else:
            untrained_manifest = manifest(state.microstep.agent_state.actor.params,fresh_counts,"untrained","native")
        initial = context.training.counts(state)
        initial_rng = {"native":runtime.host(state.microstep.rng),"auxiliary":runtime.host(state.aux_rng)}
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
            if (index+1)%20 == 0:
                journal()
        deadline(); final = runtime.validate(adapter,state)
        expected = dict(zip(runtime.COUNT_KEYS,(budget,budget//2,budget//2,budget//2,budget//2,budget,budget,1,1)))
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
            matching_artifact = atomic_json(output / "matching_receipt.json",matching)
        deadline(); stage = "sealed_exports"
        state_artifact = atomic_bytes(output / "final_state.msgpack",serialization.to_bytes(state))
        state_manifest = atomic_json(output / "state_manifest.json",{"schema":"unifloral.support_pilot.state.v1",
            "artifact":state_artifact,"protocol_sha256":protocol_sha,"cell":job["cell"],"condition":job["condition"],
            "seed_index":job["seed_index"],"kind":job["kind"],"counts":final["counts"],
            "state_leaves":runtime.parameter_leaves(state),"normalization":context.normalization,
            "resume_authorized":False,"complete_optimizer_target_rng_state":True})
        actor_manifest = manifest(state.microstep.agent_state.actor.params,final["counts"],job["kind"],job.get("arm","native"),state_artifact)
        report = {"schema":"unifloral.support_pilot.training_result.v1","status":"completed_support_"+job["kind"],
            "kind":job["kind"],"cell":job["cell"],"condition":job["condition"],"seed_index":job["seed_index"],
            "arm":job.get("arm","native"),"protocol_sha256":protocol_sha,"execution_counts_known":True,
            "initial_counts":initial,"final_counts":final["counts"],"additional_updates":deltas(final["counts"]),
            "initial_parameter_hashes":context.initial_parameter_hashes,"initial_optimizer_steps":context.initial_optimizer_steps,
            "initial_optimizer_hashes":context.initial_optimizer_hashes,"initial_fork_rng":initial_rng,
            "final_rng":{"native":runtime.host(state.microstep.rng),"auxiliary":runtime.host(state.aux_rng)},
            "actor_manifest_sha256":actor_manifest["sha256"],"state_manifest_sha256":state_manifest["sha256"],
            "matching_receipt_sha256":matching_artifact["sha256"],"data_receipt_sha256":data_artifact["sha256"],
            "untrained_actor_manifest_sha256":untrained_manifest["sha256"] if untrained_manifest else None,
            "unique_replay_source_rows":unique_rows_artifact,"matching_gradients":matching_gradients_artifact,
            "checks":final["checks"],"ledger":led,"last_update_metrics":last_metrics,
            "distance_coefficient":matching["distance_coefficient"],"elapsed_seconds":time.monotonic()-started,
            "initialization":"fresh parameters/optimizers; branch reconstructs static template then exactly restores its10k complete state",
            "scale_bank_role":protocol["scale_bank_role"],"error":None,"full_training_hold":True}
        atomic_json(output / "report.json",runtime.host(report)); journal(report["status"])
        return runtime.host(report)
    except Exception as exc:
        if hasattr(exc,"last_committed_state"):
            state = exc.last_committed_state; attempted = exc.attempted_counts; stage = exc.stage
        led["attempted_uncommitted_updates"] = attempted
        if state is not None and adapter is not None:
            try:
                from flax import serialization
                if initial is not None:
                    led["committed_additional_updates"] = deltas(adapter.context.training.counts(state))
                atomic_bytes(output / "partial_state.msgpack",serialization.to_bytes(state))
            except Exception as save_error:
                led["partial_state_error"] = str(save_error)
        report = {"status":"failed_support_training","job":job,"stage":stage,"error":type(exc).__name__+": "+str(exc),
            "execution_counts_known":not inflight and not any(attempted.values()),"ledger":led,
            "elapsed_seconds":time.monotonic()-started,"protocol_sha256":protocol_sha,"automatic_restart":False,
            "diagnostics":getattr(exc,"diagnostics",None)}
        atomic_json(output / "report.json",runtime.host(report)); journal(report["status"])
        return runtime.host(report)
