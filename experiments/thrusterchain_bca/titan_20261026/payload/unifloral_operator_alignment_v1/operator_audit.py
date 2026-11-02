"""Pure arithmetic audit of authenticated frozen native-operator arrays.

No model, simulator, checkpoint or calibration imports/calls. Four declared
smoothing draws remain separate observations of a finite noise bank.
"""
import math

RELATIONSHIPS=("u_abs_residual","u_signed_error","u_positive_error","u_absolute_error",
               "abs_residual_signed_error","abs_residual_positive_error","R_with_E","H_with_E","G_with_E")


def _np():
    import numpy as np
    return np


def _plain(value):
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    if hasattr(value,"tolist"):return _plain(value.tolist())
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,float) and math.isfinite(value):return value
    raise ValueError("Nonfinite/unsupported report value")


def _finite(value,name,shape):
    np=_np();array=np.asarray(value,dtype=np.float64)
    if array.shape!=shape or not np.all(np.isfinite(array)):raise ValueError(name+" shape/finite check failed")
    return array


def pearson(x,y):
    np=_np();x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if x.ndim!=1 or y.shape!=x.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Correlation inputs must be aligned finite vectors")
    reason="fewer_than_two_rows" if len(x)<2 else "constant_predictor" if np.all(x==x[0]) else "constant_outcome" if np.all(y==y[0]) else None
    if reason:return {"defined":False,"value":None,"reason":reason,"rows":len(x)}
    a=x-x.mean();b=y-y.mean();a/=np.max(np.abs(a));b/=np.max(np.abs(b))
    value=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    return {"defined":True,"value":float(np.clip(value,-1.,1.)),"reason":None,"rows":len(x)}


def _magnitude(x):
    np=_np();x=np.asarray(x,dtype=np.float64)
    return {"mean":float(x.mean()),"median":float(np.median(x)),"min":float(x.min()),"max":float(x.max()),
            "rms":float(np.sqrt(np.mean(x*x)))}


def _draw_spread(values):
    np=_np();defined=[value for value in values if value is not None]
    return {"per_draw":values,"defined_draws":len(defined),"mean":None if not defined else float(np.mean(defined)),
            "min":None if not defined else min(defined),"max":None if not defined else max(defined),
            "interpretation":"Descriptive finite four-draw summary, not an expected operator or independent-episode expansion"}


def _compare(actual,expected,name,protocol):
    np=_np();difference=np.abs(actual-expected)
    allowed=protocol["replay_atol"]+protocol["replay_rtol"]*np.abs(expected)
    if not np.all(difference<=allowed):raise ValueError(name+" replay tolerance failed")
    return {"passed":True,"max_abs_difference":float(difference.max()),"rtol":protocol["replay_rtol"],"atol":protocol["replay_atol"]}


def validate_bank(bank,protocol,development):
    np=_np();n=protocol["development_rows"] if development else protocol["fitting_rows"];k=protocol["smoothing_draws"]
    arrays={}
    for key,shape in {"observations":(n,14),"actions":(n,4),"q_current":(n,2),"u":(n,),
                      "rewards":(n,),"done":(n,),"next_obs":(n,14),"target_actor_next":(n,4),
                      "raw_noise":(k,n,4),"clipped_noise":(k,n,4),"next_actions":(k,n,4),
                      "target_q":(k,n,2),"native_y":(k,n)}.items():arrays[key]=_finite(bank[key],key,shape)
    if np.any(arrays["u"]<0):raise ValueError("Negative residual radius")
    for key in ("u_usable","u_valid"):
        flag=np.asarray(bank[key])
        if flag.dtype.kind!="b" or flag.shape not in ((),(n,)) or not flag.size or not np.all(flag):
            raise ValueError("Unready/invalid frozen reference: "+key)
    snapshot=bank.get("snapshot_receipt",{})
    if snapshot.get("final_refresh_verified") is not True or snapshot.get("current_equals_reference") is not True:
        raise ValueError("Unsupported or unverified final-reference boundary")
    if np.any((arrays["done"]!=0)&(arrays["done"]!=1)):raise ValueError("Terminal mask must be binary")
    for key in ("actions","target_actor_next","next_actions"):
        if np.any(np.abs(arrays[key])>protocol["action_limit"]):raise ValueError("Action bound failed: "+key)
    clipped=np.clip(arrays["raw_noise"],-protocol["noise_clip"],protocol["noise_clip"])
    next_actions=np.clip(arrays["target_actor_next"][None,:,:]+arrays["clipped_noise"],-protocol["action_limit"],protocol["action_limit"])
    # Reconstruct the declared native float32 expression in source order. Only
    # subsequent same-scalar accounting uses float64; cancellation makes these
    # different numerical operations, not interchangeable implementations.
    reward32=arrays["rewards"].astype(np.float32)[None,:]
    done32=arrays["done"].astype(np.float32)[None,:]
    target_q32=arrays["target_q"].astype(np.float32).min(axis=-1)
    target=(reward32+(np.float32(1.)-done32)*np.float32(protocol["gamma"])*target_q32).astype(np.float64)
    replay={"noise_clip":_compare(arrays["clipped_noise"],clipped,"noise clipping",protocol),
            "next_action":_compare(arrays["next_actions"],next_actions,"next action",protocol),
            "native_y":_compare(arrays["native_y"],target,"native target",protocol)}
    metadata=bank["metadata"]
    if len(metadata)!=n:raise ValueError("Metadata row count differs")
    if development:
        arrays["saved_values"]=_finite(bank["saved_values"],"saved values",(n,))
        physics_rewards=_finite(bank["physics_rewards"],"physics rewards",(n,))
        if not np.array_equal(physics_rewards.astype(np.float32).astype(np.float64),arrays["rewards"]):
            raise ValueError("Native reward is not the declared float32 physics reward cast")
        expected=[(g,c) for g in range(protocol["development_states"]) for c in range(protocol["candidates_per_state"])]
        if [(row["group_id"],row["candidate_id"]) for row in metadata]!=expected:
            raise ValueError("Development state/candidate order changed")
        for group in range(protocol["development_states"]):
            group_meta=metadata[group*9:(group+1)*9]
            if len({(row["batch"],row["time"],row["episode_seed"]) for row in group_meta})!=1:
                raise ValueError("Candidates in one state have inconsistent identity")
        state_meta=metadata[::9]
        for batch in protocol["batch_names"]:
            for time in protocol["query_times"]:
                if sum(row["batch"]==batch and row["time"]==time for row in state_meta)!=16:
                    raise ValueError("Development batch/time cell does not contain16 states")
    else:
        if any("group_id" not in row or "row_index" not in row for row in metadata):
            raise ValueError("Fitting episode/row identity missing")
    return arrays,metadata,replay


def accounting(q_current,native_y,saved_values,protocol):
    np=_np();q1=q_current[:,0];qmin=q_current.min(axis=1)
    E=q1-saved_values;H=q1-qmin;R=qmin[None,:]-native_y;G=native_y-saved_values[None,:]
    difference=np.abs(E[None,:]-(R+H[None,:]+G))
    allowed=protocol["accounting_epsilon_multiplier"]*np.finfo(np.float64).eps*(1.+np.abs(E[None,:])+np.abs(R)+np.abs(H[None,:])+np.abs(G))
    if not np.all(difference<=allowed):raise ValueError("Tight same-scalar accounting identity failed")
    return {"E":E,"R":R,"H":H,"G":G},{"passed":True,"max_abs_difference":float(difference.max()),
            "max_fraction_of_allowed_roundoff":float(np.max(difference/allowed)),
            "epsilon_multiplier":protocol["accounting_epsilon_multiplier"],"evaluated_row_draw_pairs":int(difference.size),
            "distinct_from_float32_forward_replay":True}


def _summarize_states(states):
    np=_np();relationships={}
    for key in RELATIONSHIPS:
        entries=[row["relationships"][key] for row in states];values=[entry["value"] for entry in entries if entry["defined"]]
        reasons={}
        for entry in entries:
            if not entry["defined"]:reasons[entry["reason"]]=reasons.get(entry["reason"],0)+1
        relationships[key]={"defined_states":len(values),"undefined_states":len(states)-len(values),
                            "equal_state_mean":None if not values else float(np.mean(values)),"undefined_reasons":reasons}
    return {"states":len(states),"relationships":relationships,
            "mean_state_centered_action_rms":{key:float(np.mean([row["centered_action_rms"][key] for row in states])) for key in ("E","R","H","G")},
            "equal_state_radius_containment":float(np.mean([row["radius_containment_fraction"] for row in states])),
            "containment_guarantee_claimed":False,"independence_unit":"episode/state, not actions or noise draws"}


def development_audit(bank,protocol):
    np=_np();arrays,metadata,replay=validate_bank(bank,protocol,True)
    terms,identity=accounting(arrays["q_current"],arrays["native_y"],arrays["saved_values"],protocol)
    draws=[]
    for draw in range(protocol["smoothing_draws"]):
        states=[]
        for group in range(protocol["development_states"]):
            indices=slice(group*9,(group+1)*9);meta=metadata[group*9]
            U=arrays["u"][indices];E=terms["E"][indices];R=terms["R"][draw,indices];H=terms["H"][indices];G=terms["G"][draw,indices]
            positive=np.maximum(E,0.);absolute=np.abs(R)
            relations={"u_abs_residual":pearson(U,absolute),"u_signed_error":pearson(U,E),
                       "u_positive_error":pearson(U,positive),"u_absolute_error":pearson(U,np.abs(E)),
                       "abs_residual_signed_error":pearson(absolute,E),"abs_residual_positive_error":pearson(absolute,positive),
                       "R_with_E":pearson(R,E),"H_with_E":pearson(H,E),"G_with_E":pearson(G,E)}
            values={"E":E,"R":R,"H":H,"G":G}
            states.append({"group_id":group,"batch":meta["batch"],"time":meta["time"],"episode_seed":meta["episode_seed"],
                           "relationships":relations,"terms":values,"radius":U,"native_y":arrays["native_y"][draw,indices],
                           "centered_action_rms":{key:float(np.sqrt(np.mean((value-value.mean())**2))) for key,value in values.items()},
                           "radius_containment":absolute<=U,"radius_containment_fraction":float(np.mean(absolute<=U))})
        draws.append({"draw":draw,"noise_seed":protocol["development_noise_seeds"][draw],"states":states,
                      "overall":_summarize_states(states),
                      "by_batch":{batch:_summarize_states([row for row in states if row["batch"]==batch]) for batch in protocol["batch_names"]},
                      "by_time":{str(time):_summarize_states([row for row in states if row["time"]==time]) for time in protocol["query_times"]},
                      "by_batch_time":{batch:{str(time):_summarize_states([row for row in states if row["batch"]==batch and row["time"]==time])
                                               for time in protocol["query_times"]} for batch in protocol["batch_names"]}})
    spread={"relationships":{key:_draw_spread([row["overall"]["relationships"][key]["equal_state_mean"] for row in draws]) for key in RELATIONSHIPS},
            "centered_action_rms":{key:_draw_spread([row["overall"]["mean_state_centered_action_rms"][key] for row in draws]) for key in ("E","R","H","G")},
            "radius_containment":_draw_spread([row["overall"]["equal_state_radius_containment"] for row in draws])}
    return {"rows":protocol["development_rows"],"independent_states":protocol["development_states"],"draws":draws,
            "draw_summary":spread,"accounting_identity":identity,"operator_replay":replay,
            "reference_snapshot_term":{"verified_zero":True,"reason":"Final-refresh boundary/current-export equality verified by checkpoint layer; no separately stored stale native critic snapshot"},
            "residual_arithmetic":"Float64 difference of authenticated native Qmin and native float32 target scalars; not a replay claim for every calibration float32 subtraction",
            "radius_rule":"abs(R)<=U; selected live_width returns a radius, with no factor2",
            "coverage_guarantee_claimed":False}


def fitting_audit(bank,protocol):
    np=_np();arrays,metadata,replay=validate_bank(bank,protocol,False)
    groups={}
    for i,row in enumerate(metadata):groups.setdefault(row["group_id"],[]).append(i)
    qmin=arrays["q_current"].min(axis=1);U=arrays["u"];draws=[]
    for draw in range(protocol["smoothing_draws"]):
        residual=qmin-arrays["native_y"][draw];absolute=np.abs(residual);contained=absolute<=U
        group_rows=[{"group_id":group,"rows":len(indices),"containment_fraction":float(np.mean(contained[indices]))} for group,indices in groups.items()]
        draws.append({"draw":draw,"noise_seed":protocol["fitting_noise_seeds"][draw],"pearson_u_abs_residual":pearson(U,absolute),
                      "row_weighted_radius_containment":float(np.mean(contained)),
                      "equal_episode_group_radius_containment":float(np.mean([row["containment_fraction"] for row in group_rows])),
                      "groups":group_rows,"U_magnitude":_magnitude(U),"absolute_residual_magnitude":_magnitude(absolute),
                      "signed_residual_magnitude":_magnitude(residual),"residual":residual,"native_y":arrays["native_y"][draw],"radius_containment":contained})
    return {"rows":len(U),"unique_episode_groups":len(groups),"operator_replay":replay,"draws":draws,
            "draw_summary":{"pearson_u_abs_residual":_draw_spread([row["pearson_u_abs_residual"]["value"] for row in draws]),
                            "row_containment":_draw_spread([row["row_weighted_radius_containment"] for row in draws]),
                            "equal_group_containment":_draw_spread([row["equal_episode_group_radius_containment"] for row in draws])},
            "role":protocol["fitting_bank_role"],"heldout_coverage_claimed":False,
            "residual_arithmetic":"Float64 Qmin-native_y; descriptive numeric containment at U, not U/2"}


def analyze(development,fitting,protocol):
    expected={"phase":"frozen_development_operator_alignment_only","gamma":.98,"smoothing_draws":4,
              "development_states":128,"development_rows":1152,"candidates_per_state":9,"fitting_rows":256,
              "noise_std":.2,"noise_clip":.5,"action_limit":1.,"replay_rtol":2e-5,"replay_atol":2e-6,
              "accounting_epsilon_multiplier":64,"development_noise_seeds":[2026103041,2026103042,2026103043,2026103044],
              "fitting_noise_seeds":[2026103051,2026103052,2026103053,2026103054]}
    for key,value in expected.items():
        if protocol.get(key)!=value:raise ValueError("Frozen operator protocol changed: "+key)
    return _plain({"status":"pass","phase":protocol["phase"],"claim_scope":protocol["claim_scope"],
        "development":development_audit(development,protocol),"fitting":fitting_audit(fitting,protocol),
        "full_training_hold":True,"automatic_training_approval":False,"new_policy_training":False,
        "new_diagnostic_fits":0,"policy_optimizer_updates":0,"calibrator_updates":0,"confirmation_files_read":0,
        "method_or_strength_selected":False,"model_or_simulator_calls_by_audit_module":0,
        "expected_operator_claimed":False,"new_confidence_intervals":False,
        "limitations":[protocol["post_hoc_status"],protocol["noise_interpretation"],protocol["interpretation"],
            protocol["fitting_bank_role"],protocol["continuation_noise_note"],
            "Centered RMS terms can cancel; their magnitudes do not partition error variance or identify causal contributions.",
            "Development candidates are not an in-distribution calibration set. Descriptive containment is not a conformal coverage guarantee.",
            "Constant-vector correlations are undefined and omitted only with explicit counts; states and draws are never inflated into independent action samples.",
            "The math module relies on the caller's source/runtime/replay/transition and frozen-state proof; it performs no model or simulator calls."]})
