"""Single bounded runtime gate; importing this file performs no scientific work."""
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
PAYLOAD = HERE.parents[1] / "titan_20261026/payload"
BUNDLE = PAYLOAD / "transfer_bundle"
EXPORT = PAYLOAD / "unifloral_export_recovery_v2/recovery_20261023T214750_256386Z/inference_export/eligible_checkpoint.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _module(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        value = sys.modules[name]
        require(Path(value.__file__).resolve() == path, "Loaded module path differs: "+name)
        return value
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return value


def _host(value):
    if isinstance(value, dict):
        return {str(k): _host(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_host(v) for v in value]
    if hasattr(value, "tolist"):
        return _host(value.tolist())
    return value


def _json(path, value, *, replace=False):
    raw = (json.dumps(_host(value), indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    if replace:
        temp = path.with_name(path.name+".tmp."+str(os.getpid()))
        with temp.open("wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    else:
        with path.open("xb") as stream:
            stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def _runtime(contract):
    actual = {name: importlib.metadata.version(name) for name in contract["runtime_packages"]}
    require(actual == contract["runtime_packages"], "Frozen runtime package versions differ")
    import jax
    require(jax.default_backend() == "gpu" and not jax.config.jax_disable_jit
            and not jax.config.jax_enable_x64
            and str(jax.config.jax_default_matmul_precision) == "highest"
            and os.environ.get("XLA_FLAGS") == "--xla_gpu_triton_gemm_any=True",
            "Expected frozen GPU/JIT/float32/highest-precision runtime")
    return {"python": sys.version, "executable": sys.executable,
            "packages": actual, "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "device_kinds": [d.device_kind for d in jax.devices()],
            "upstream_constraint_exception": "Same disclosed pinned runtime exception as Titan and prior Zeus validation"}


def _core():
    manifest = BUNDLE / "BUNDLE_MANIFEST.json"
    raw = manifest.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == "3443280fd4340ba64aef4d9601136257014bf2a6c61ca8c890eba06afa76690a",
            "Transfer core manifest differs")
    files = json.loads(raw)["files"]
    selected = {k: v for k, v in files.items() if k.startswith("core_v4/")}
    require(selected, "Core source manifest is empty")
    for name, expected in selected.items():
        path = (BUNDLE/name).resolve()
        require(path.is_relative_to(BUNDLE) and path.is_file()
                and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                "Core source identity differs: "+name)
    root = BUNDLE / "core_v4"
    for name, loaded in list(sys.modules.items()):
        if name == "thrusterchain" or name.startswith("thrusterchain."):
            if getattr(loaded, "__file__", None):
                require(Path(loaded.__file__).resolve().is_relative_to(root), "Foreign ThrusterChain module loaded")
    sys.path.insert(0, str(root))
    core = importlib.import_module("thrusterchain.environment")
    return core, {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "verified_files": len(selected)}


def run(outputdir, contract, progresscallback, deadline):
    """Execute exactly one gate. Exceptions preserve the live attempted ledger."""
    output = Path(outputdir).resolve()
    require(output.is_dir(), "Supervisor must reserve and create the output directory")
    result = {"status": "running", "decision": None, "started_utc": datetime.now(timezone.utc).isoformat(),
              "execution_counts_known": True, "full_training_hold": True,
              "automatic_retry": False, "learner_updates": 0, "calibration_updates": 0}
    ledger = {"transition_steps_attempted": 0, "transition_steps_completed": 0}
    last_progress = [0.]
    phase = ["verify_prepared"]

    def progress(_ignored=None, force=False):
        result["ledger"] = dict(ledger)
        now = time.monotonic()
        if force or now-last_progress[0] >= 5:
            message = {"saved_utc": datetime.now(timezone.utc).isoformat(),
                       "phase": phase[0], "status": result["status"],
                       "execution_counts_known": result["execution_counts_known"],
                       "ledger": dict(ledger)}
            _json(output/"progress.json", message, replace=True)
            if progresscallback is not None:
                progresscallback(message)
            last_progress[0] = now

    def stage(name):
        require(time.monotonic() < deadline, "Global gate deadline reached")
        phase[0] = name
        progress(force=True)

    try:
        import launch
        result["prepared_before"] = launch.verify_prepared()
        require(contract == json.loads((HERE/"contract.json").read_text()), "Caller contract differs")
        result["contract_sha256"] = hashlib.sha256((HERE/"contract.json").read_bytes()).hexdigest()
        stage("runtime_gate")
        result["runtime"] = _runtime(contract)
        import numpy as np
        import geometry
        import reference
        import analysis
        region_c, support_c = contract["region"], contract["support"]
        require(geometry.CENTER == region_c["center"] and geometry.HALF_WIDTH == region_c["half_width"]
                and geometry.LOAD_CUTOFF == region_c["high_load"] and geometry.KNEE == region_c["knee_not_center"]
                and geometry.ROW_CAP == support_c["max_original_neighbors"]
                and geometry.REFERENCE_QUANTILE == support_c["reference_quantile"]
                and geometry.MIN_REFERENCE_ROWS == support_c["min_reference_rows_with_full_pool"]
                and geometry.TIME_BIN == support_c["time_bin_width"]
                and list(geometry.LOAD_DEPTH_EDGES) == contract["global_coverage"]["load_depth_edges"]
                and geometry.CONTROL_SEED == contract["controls"]["random_seed"]
                and geometry.CALIBRATION_SEED == contract["calibration"]["original_time_seed"]
                and geometry.CALIBRATION_REPLACEMENT_SEED == contract["calibration"]["replacement_seed"],
                "Geometry constants differ from the frozen contract")
        ledger = reference.new_ledger(contract["caps"]["total_simulator_transitions"],
                                      contract["caps"]["actor_forward_rows_including_loader"])
        ledger.update(critic_rows_attempted=0, critic_rows_completed=0,
                      width_rows_attempted=0, width_rows_completed=0,
                      source_episodes_decoded=0, source_transition_rows_decoded=0,
                      selection_or_final_calibration_transition_reads=0)
        stage("source_identity")
        require(Path(os.environ.get("BCA_TRANSFER_BUNDLE_ROOT", str(BUNDLE))).resolve() == BUNDLE,
                "Conflicting portable transfer root")
        os.environ["BCA_TRANSFER_BUNDLE_ROOT"] = str(BUNDLE)
        checkpoint = _module("_design_gate_checkpoint_identity", PAYLOAD/"unifloral_operator_alignment_v1/checkpoint_io.py")
        result["checkpoint_identity"] = checkpoint.verify()
        require(result["checkpoint_identity"]["checkpoint_files"]["state.msgpack"] == contract["reference"]["checkpoint_state_sha256"],
                "Checkpoint is not the frozen Titan terminal state")
        core, result["core_files"] = _core()
        result["core_identity"] = reference.authenticate_core(core)
        _json(output/"identities_before.json", {k: result[k] for k in
              ("prepared_before", "contract_sha256", "runtime", "checkpoint_identity", "core_files", "core_identity")})

        stage("role_safe_logged_decode")
        logged = _module("_design_gate_logged", HERE.parent/"bca_one_sided_v1/logged_data.py")
        p = logged.provenance()
        # Zero-slab gate: the last 64 development validation groups. The first 64 were
        # the anchors of the shelved stall-knee gate (attempt 2).
        require(len(p["validation_indices"]) == 128, "Expected 128 development validation groups")
        indices = np.concatenate((p["train_indices"], p["calibration_indices"], p["validation_indices"][64:128]))
        codes = p["vectors"]["split_codes"]
        require(len(indices) == 960 and len(np.unique(indices)) == 960
                and np.all(codes[indices[:768]] == 0) and np.all(codes[indices[768:]] == 1), "Gate episode roles changed")
        arrays = {}
        ledger["source_episode_records_reserved"] = 960
        ledger["source_fields_completed"] = []
        with zipfile.ZipFile(logged.DATA) as archive:
            for name in logged.TRANSITION_NAMES:
                stage("decode_"+name)
                ledger["source_field_in_progress"] = name
                arrays[name] = logged.read_episodes(archive, name, indices, codes)
                ledger["source_fields_completed"].append(name)
                ledger["source_episodes_decoded"] = 960
                ledger["source_transition_rows_decoded"] = 96000
                ledger["source_field_in_progress"] = None
        ledger["source_episodes_decoded"] = 960
        ledger["source_transition_rows_decoded"] = 96000
        require(arrays["observations"].shape == (960,101,14) and arrays["observations"].dtype == np.float32
                and arrays["actions"].shape == (960,100,4) and arrays["actions"].dtype == np.float32,
                "Logged storage/shape mismatch")
        require(not np.any(arrays["truncated"]) and np.all(arrays["terminated"][:,-1])
                and not np.any(arrays["terminated"][:,:-1]), "Finite-horizon masks differ")
        groups = p["vectors"]["group_ids"][indices]
        trajectories = p["vectors"]["trajectory_ids"][indices]
        seeds = p["vectors"]["episode_seeds"][indices]
        times = np.tile(np.asarray(contract["data"]["probe_times_cycle"], np.int64), 16)
        anchor_obs = arrays["observations"][896+np.arange(64), times]
        train_obs = arrays["observations"][:768,:-1].reshape(-1,14)
        train_actions = arrays["actions"][:768].reshape(-1,4)
        row_episodes = np.repeat(groups[:768],100)
        row_times = np.tile(np.arange(100,dtype=np.int64),768)
        source_row_ids = np.repeat(trajectories[:768],100)*100+row_times
        result["data_identity"] = {"dataset_sha256": logged.EXPECTED_DATA_SHA,
            "bindings_sha256": logged.EXPECTED_BINDINGS_SHA, "source_episode_indices": indices.tolist(),
            "source_group_ids": groups.tolist(), "probe_episode_indices": indices[896:].tolist(),
            "probe_group_ids": groups[896:].tolist(), "probe_times": times.tolist(),
            "probe_seeds": seeds[896:].tolist(), "protected_transition_rows_decoded": 0}

        stage("geometry_and_calibration_plan")
        masks, control_receipt = geometry.control_masks(train_obs, train_actions, row_times, episode_ids=row_episodes)
        metric = geometry.fit_support_metric(train_obs)
        support = geometry.support_report(train_obs, train_actions, row_episodes, row_times,
            anchor_obs, groups[896:], times, masks, metric, source_row_ids=source_row_ids)
        calplan = geometry.calibration_plan(arrays["observations"][768:896,:-1].reshape(-1,14),
            arrays["actions"][768:896].reshape(-1,4), groups[768:896])
        result["calibration_plan"] = calplan
        result["geometry"] = analysis.analyze_geometry(train_obs, train_actions, row_episodes,
            row_times, masks, control_receipt, metric, support, calplan, anchor_obs, contract=contract)
        _json(output/"geometry.json", {"analysis": result["geometry"], "support": support,
              "metric": metric, "control_receipt": control_receipt, "calibration_plan": calplan,
              "data_identity": result["data_identity"]})
        require(calplan["status"] == "eligible_plan_only" and calplan["calibration_rows_planned"] == 128
                and calplan["hole_rows_used"] == 0, "Calibration eligibility integrity failure")

        stage("frozen_inference_load")
        inference = _module("_design_gate_inference", PAYLOAD/"unifloral_frozen_usefulness_v1_seedfix/inference.py")
        # This is the loader's existing provenance-only contract, not an invoked
        # evaluator protocol. The new gate never calls its old evaluator.
        load_protocol = json.loads((PAYLOAD/"unifloral_frozen_usefulness_v1_seedfix/protocol.json").read_text())
        _, _, pre = inference.read_contract(EXPORT, load_protocol)
        require(pre["file_hashes"]["parameters"] == contract["reference"]["export_parameters_sha256"],
                "Recovered parameter identity differs")
        # Uninstrumented loader checks have a conservative reservation of two
        # actor, two critic and two width rows. A partial loader failure has
        # uncertain actual forward counts, never a falsely exact zero.
        result["execution_counts_known"] = False
        ledger["loader_forward_rows_reserved"] = {"actor":2,"critic":2,"width":2}
        progress(force=True)
        policy, _scale_bank, result["inference_identity"] = inference.load(EXPORT, load_protocol)
        require(result["inference_identity"]["file_hashes"] == pre["file_hashes"], "Export changed during load")
        result["execution_counts_known"] = True
        ledger.update(actor_calls_attempted=1, actor_calls_completed=1, actor_rows_attempted=2, actor_rows_completed=2,
                      critic_rows_attempted=2, critic_rows_completed=2, width_rows_attempted=2, width_rows_completed=2)
        result["model_modules"] = {name:str(Path(value.__file__).resolve()) for name,value in sys.modules.items()
                                   if getattr(value,"__file__",None) and (name == "native_loader" or name.startswith("_unifloral"))}
        _json(output/"inference_identity.json", result["inference_identity"])

        stage("reconstruct_recorded_state_and_time")
        states = reference.reconstruct_states(core, seeds[896:], arrays["actions"][896:],
            arrays["observations"][896:], times, ledger, deadline=deadline, progress=progress)
        require(np.array_equal(reference.observation(states,times),anchor_obs), "Query reconstruction changed actor observations")
        stage("canonical_time_parity")
        result["parity"] = reference.runtime_parity(core, states[0], ledger, deadline=deadline, progress=progress)
        stage("fixed_candidate_actions_and_q1")
        banks = reference.candidate_banks(policy, anchor_obs, ledger, deadline=deadline, progress=progress)
        repeated_obs = np.repeat(anchor_obs,6,axis=0)
        ledger["critic_rows_attempted"] += 384
        require(ledger["critic_rows_attempted"] <= contract["caps"]["critic_forward_rows_including_loader"], "Critic row cap exceeded")
        progress(force=True)
        q = np.asarray(policy.q(repeated_obs,banks.reshape(384,4)))
        ledger["critic_rows_completed"] += 384
        require(q.shape == (384,2) and np.all(np.isfinite(q)), "Native twin-Q output differs")
        policy.assert_unchanged()
        with (output/"prelabel_probes.npz").open("xb") as stream:
            np.savez_compressed(stream, states=states,times=times,observations=anchor_obs,actions=banks,
                                q=q.reshape(64,6,2),episode_ids=groups[896:],episode_seeds=seeds[896:])
        stage("full_qpi_references")
        values = reference.continuation_values(core,policy,states,times,banks,ledger,deadline=deadline,progress=progress)
        with (output/"candidate_values.npz").open("xb") as stream:
            np.savez_compressed(stream,values=values,q1=q[:,0].reshape(64,6),actions=banks,
                                states=states,times=times,episode_ids=groups[896:])
        stage("free_actor_exposure")
        free = reference.free_policy_rollouts(core,policy,seeds[896:912],ledger,deadline=deadline,progress=progress)
        with (output/"free_policy.npz").open("xb") as stream:
            np.savez_compressed(stream,**{k:v for k,v in free.items() if isinstance(v,np.ndarray)})
        stage("fixed_analysis")
        result["values"] = analysis.analyze_values(values,q[:,0].reshape(64,6),banks,anchor_obs,free,contract)
        require(ledger["transition_steps_attempted"] == ledger["transition_steps_completed"] == 28104
                and ledger["resets_completed"] == 81 and ledger["critic_rows_completed"] == 386
                and ledger["actor_rows_completed"] <= 30000 and ledger["width_rows_completed"] == 2,
                "Completed work does not match exact gate accounting")
        policy.assert_unchanged()
        result["frozen_policy_unchanged"] = True
        result["prepared_after"] = launch.verify_prepared()
        require(result["prepared_after"] == result["prepared_before"], "Prepared identity changed during gate")
        result["decision"] = ("passed_development_design_gate" if result["geometry"]["passed"]
                              and result["values"]["passed"] else "failed_development_design_gate")
        result["status"] = "completed_development_design_gate"
        result["scope"] = "Zero-update development design gate; missing-label causal effect and BCA efficacy remain untested"
    except BaseException as exc:
        result["status"] = "failed_or_interrupted_design_gate"
        result["error"] = {"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        result["failed_phase"] = phase[0]
        pairs = [("transition_steps_attempted", "transition_steps_completed"),
                 ("actor_rows_attempted", "actor_rows_completed"),
                 ("critic_rows_attempted", "critic_rows_completed"),
                 ("width_rows_attempted", "width_rows_completed")]
        if any(ledger.get(a, 0) != ledger.get(b, 0) for a,b in pairs) or ledger.get("source_field_in_progress"):
            result["execution_counts_known"] = False
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result["ledger"] = dict(ledger)
        try:
            progress(force=True)
        except BaseException as progress_error:
            result["progress_save_error"] = str(progress_error)
        result["report_sha256"] = _json(output/"report.json",result)
    return result
