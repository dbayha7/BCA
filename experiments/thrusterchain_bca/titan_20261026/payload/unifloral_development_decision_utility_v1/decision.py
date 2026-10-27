"""No-fit decision utility of fixed saved held-out signed-error predictions.

Scoring is label-blind and sealed before looking up selected continuation
values. No model, checkpoint, simulator, fitting or confirmation imports.
"""
import hashlib
import json
import math

METHODS=("corrected_base","corrected_width","raw_q","bca_kappa_3","contrast_actor_distance_kappa_0.1")


def _plain(value):
    if isinstance(value,dict): return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [_plain(v) for v in value]
    if hasattr(value,"tolist"): return _plain(value.tolist())
    if value is None or isinstance(value,(str,bool,int)): return value
    if isinstance(value,float) and math.isfinite(value): return value
    raise ValueError("Unsupported or nonfinite report value")


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _finite(values,name,length=9):
    try:
        if any(isinstance(value,bool) for value in values): raise ValueError("Boolean numeric value")
        result=[float(value) for value in values]
    except (TypeError,ValueError) as error:
        raise ValueError(name+" requires a numeric vector") from error
    if len(result)!=length or not all(math.isfinite(value) for value in result):
        raise ValueError(name+" must have exactly the declared finite entries")
    return result


def selection(scores,atol,rtol):
    scores=_finite(scores,"candidate scores")
    maximum=max(scores)
    near=[i for i,value in enumerate(scores) if abs(value-maximum)<=atol+rtol*abs(maximum)]
    ordered=sorted(scores); index=near[0]
    return {"index":index,"exact_argmax_index":scores.index(maximum),"near_max_indices":near,
            "score_ambiguous":len(near)>1,"maximum_score":maximum,"selected_score":scores[index],
            "top_two_gap":ordered[-1]-ordered[-2]}


def score_methods(q1,u,distance,predictions,protocol):
    """Only prediction/input signals are accepted; never continuation values."""
    q1=_finite(q1,"Q1"); u=_finite(u,"U"); distance=_finite(distance,"actor distance")
    if min(u+distance)<0: raise ValueError("Width and squared distance must be nonnegative")
    base=_finite(predictions[protocol["base_prediction_key"]],"saved base predictions")
    width=_finite(predictions[protocol["width_prediction_key"]],"saved width predictions")
    scores={"corrected_base":[q-protocol["correction_coefficient"]*p for q,p in zip(q1,base)],
            "corrected_width":[q-protocol["correction_coefficient"]*p for q,p in zip(q1,width)],
            "raw_q":q1,"bca_kappa_3":[q-protocol["kappa_penalty_bca"]*v for q,v in zip(q1,u)],
            "contrast_actor_distance_kappa_0.1":[q-protocol["kappa_penalty_actor_distance"]*protocol["actor_distance_factor"]*v
                                                  for q,v in zip(q1,distance)]}
    return {name:{"scores":_finite(scores[name],"computed scores"),
                  "selection":selection(scores[name],protocol["tie_atol"],protocol["tie_rtol"])} for name in METHODS}


def _identity(row):
    return (row["group_id"],row["batch"],row["time"],row["episode_seed"])


def _index_queries(batches,protocol):
    if [batch["name"] for batch in batches]!=protocol["batch_names"]: raise ValueError("Development batch identities changed")
    queries={}
    for batch in batches:
        if len(batch["queries"])!=64: raise ValueError("Each batch requires64 episodes")
        for query in batch["queries"]:
            row={**query,"batch":batch["name"]}; group=row["group_id"]
            if isinstance(group,bool) or not isinstance(group,int) or group in queries: raise ValueError("Invalid/duplicate group ID")
            queries[group]=row
    if sorted(queries)!=list(range(128)): raise ValueError("Missing development group")
    return queries


def evaluate_saved_predictions(queries,prediction_rows,protocol,expected_count):
    """Freeze the complete selection table before touching any saved values."""
    if len(prediction_rows)!=expected_count: raise ValueError("Incomplete held-out prediction bank")
    scored=[]; seen=set()
    for pred in prediction_rows:
        group=pred["group_id"]
        if group in seen or group not in queries: raise ValueError("Invalid/duplicate prediction group")
        seen.add(group); query=queries[group]
        if _identity(pred)!=_identity(query): raise ValueError("Prediction/input episode identity mismatch")
        methods=score_methods(query["q1"],query["u"],query["actor_distance"],pred["predictions"],protocol)
        if "raw_q_index" in query and methods["raw_q"]["selection"]["index"]!=query["raw_q_index"]:
            raise ValueError("Raw-Q selection differs from the saved reviewed rule")
        scored.append({"group_id":group,"batch":query["batch"],"time":query["time"],"episode_seed":query["episode_seed"],
                       "fold":pred.get("fold"),"methods":methods,
                       "saved_predictions":{"base":_finite(pred["predictions"][protocol["base_prediction_key"]],"base prediction"),
                                            "width":_finite(pred["predictions"][protocol["width_prediction_key"]],"width prediction")}})
    seal=_digest(scored)
    rows=[]
    for record in scored:
        values=_finite(queries[record["group_id"]]["values"],"saved continuation values")
        maximum=max(values); methods={}
        raw_index=record["methods"]["raw_q"]["selection"]["index"]
        for name,method in record["methods"].items():
            index=method["selection"]["index"]
            methods[name]={**method,"selected_index":index,"selected_value":values[index],
                           "candidate_bank_regret":maximum-values[index],"changed_from_raw_q":index!=raw_index}
        base,width=methods["corrected_base"],methods["corrected_width"]
        gain=width["selected_value"]-base["selected_value"]
        band=protocol["tie_atol"]+protocol["tie_rtol"]*max(abs(width["selected_value"]),abs(base["selected_value"]))
        direction="near_equal" if abs(gain)<=band else "win" if gain>0 else "loss"
        rows.append({**record,"methods":methods,"candidate_values":values,"maximum_candidate_value":maximum,
                     "primary":{"gain":gain,"different_indices":width["selected_index"]!=base["selected_index"],
                                  "return_direction":direction,"value_numerical_band":band,
                                  "score_ambiguous":base["selection"]["score_ambiguous"] or width["selection"]["score_ambiguous"]}})
    if _digest(scored)!=seal: raise ValueError("Selection table changed during value lookup")
    return rows,{"prelabel_selection_sha256":seal,"episodes":len(scored),"selected_before_value_lookup":True,
                 "score_recentered":False,"predictions_refitted":False}


def summarize(rows):
    n=len(rows)
    if n==0: raise ValueError("Empty utility summary")
    mean=lambda values:math.fsum(v/len(values) for v in values)
    gains=[row["primary"]["gain"] for row in rows]
    changed=[row for row in rows if row["primary"]["different_indices"]]
    frequency=len(changed)/n
    conditional=None if not changed else mean([row["primary"]["gain"] for row in changed])
    overall=mean(gains); product=0. if conditional is None else frequency*conditional
    identity_error=abs(overall-product)
    roundoff_tolerance=16*math.ulp(max(1.,abs(overall),abs(product)))
    if identity_error>roundoff_tolerance: raise ValueError("Decision-disagreement identity failed")
    return {"episodes":n,"primary_mean_gain":overall,"decision_disagreement":{"different_index_episodes":len(changed),
                "prevalence":frequency,"conditional_mean_gain":conditional,"unconditional_mean_gain":overall,
                "prevalence_times_conditional_gain":product,"identity_absolute_error":identity_error,
                "identity_roundoff_tolerance":roundoff_tolerance,"identity_verified":True,
                "conditional_return_counts":{tag:sum(row["primary"]["return_direction"]==tag for row in changed)
                                            for tag in ("win","loss","near_equal")}},
            "primary_score_ambiguous_episodes":sum(row["primary"]["score_ambiguous"] for row in rows),
            "all_episodes_retained":True,
            "methods":{name:{"mean_selected_value":mean([row["methods"][name]["selected_value"] for row in rows]),
                             "mean_candidate_bank_regret":mean([row["methods"][name]["candidate_bank_regret"] for row in rows]),
                             "score_ambiguous_episodes":sum(row["methods"][name]["selection"]["score_ambiguous"] for row in rows),
                             "changed_from_raw_q_episodes":sum(row["methods"][name]["changed_from_raw_q"] for row in rows)} for name in METHODS},
            "corrected_width_minus_other_mean_values":{name:mean([row["methods"]["corrected_width"]["selected_value"]
                    -row["methods"][name]["selected_value"] for row in rows]) for name in METHODS if name!="corrected_width"}}


def bootstrap_primary(rows,protocol):
    import numpy as np
    if len(rows)!=128: raise ValueError("Primary bootstrap requires128 episodes")
    gains=np.asarray([row["primary"]["gain"] for row in rows],dtype=np.float64)
    rng=np.random.default_rng(protocol["bootstrap_seed"])
    indices=np.empty((protocol["bootstrap_replicates"],128),dtype=np.int64)
    cells=[]; covered=[]
    for batch in protocol["batch_names"]:
        for time in protocol["query_times"]:
            positions=np.asarray([i for i,row in enumerate(rows) if row["batch"]==batch and row["time"]==time],dtype=np.int64)
            if len(positions)!=16: raise ValueError("Each bootstrap cell requires16 episodes")
            draws=rng.integers(0,16,size=(protocol["bootstrap_replicates"],16))
            indices[:,positions]=positions[draws];covered.extend(positions.tolist())
            cells.append({"batch":batch,"time":time,"positions":positions.tolist(),"group_ids":[rows[i]["group_id"] for i in positions]})
    if sorted(covered)!=list(range(128)): raise ValueError("Bootstrap cells do not partition all episodes")
    lower,upper=np.quantile(gains[indices].mean(axis=1),[.025,.975],method="linear")
    return {"mean":float(gains.mean()),"conditional_percentile_ci95":[float(lower),float(upper)],
            "interval_contains_zero":bool(lower<=0<=upper),"seed":protocol["bootstrap_seed"],
            "replicates":protocol["bootstrap_replicates"],"resample_indices_sha256":_digest(indices),
            "cells":cells,"scope":protocol["bootstrap_scope"],"primary_only":True,"scientific_decision_assigned":False},indices


def _protocol(protocol):
    expected={"phase":"exploratory_saved_development_decision_utility_only", "method_names":list(METHODS),
              "base_prediction_key":"base","width_prediction_key":"base_plus_residual_width",
              "correction_coefficient":1.,"kappa_penalty_bca":3.,"kappa_penalty_actor_distance":.1,
              "actor_distance_factor":19.29619572569095,"tie_atol":1e-6,"tie_rtol":1e-6,
              "bootstrap_replicates":2000,"bootstrap_seed":2026103031,"candidates_per_episode":9,
              "query_times":[0,25,50,75],"new_diagnostic_fits":0}
    for key,value in expected.items():
        if protocol.get(key)!=value: raise ValueError("Locked decision diagnostic changed: "+key)


def analyze(data,protocol):
    _protocol(protocol)
    queries=_index_queries(data["batches"],protocol); prediction=data["prediction_receipt"]
    rows,seal=evaluate_saved_predictions(queries,prediction["cv"]["episodes"],protocol,128)
    primary,_=bootstrap_primary(rows,protocol)
    transfer=[]
    saved_transfers=prediction["cross_batch_transfer"]
    expected={(protocol["batch_names"][0],protocol["batch_names"][1]),(protocol["batch_names"][1],protocol["batch_names"][0])}
    if len(saved_transfers)!=2 or {(tr["training_batch"],tr["heldout_batch"]) for tr in saved_transfers}!=expected:
        raise ValueError("Expected both exact cross-batch transfer directions")
    for item in saved_transfers:
        if any(row["batch"]!=item["heldout_batch"] for row in item["episodes"]): raise ValueError("Transfer prediction batch mismatch")
        records,receipt=evaluate_saved_predictions(queries,item["episodes"],protocol,64)
        transfer.append({"training_batch":item["training_batch"],"heldout_batch":item["heldout_batch"],
                         "summary":summarize(records),"episodes":records,"selection_receipt":receipt,"interval_computed":False})
    return _plain({"status":"pass","phase":protocol["phase"],"claim_scope":protocol["claim_scope"],
        "cv":{"overall":summarize(rows),"primary":primary,"episodes":rows,"selection_receipt":seal,
              "by_batch":{name:summarize([row for row in rows if row["batch"]==name]) for name in protocol["batch_names"]},
              "by_time":{str(time):summarize([row for row in rows if row["time"]==time]) for time in protocol["query_times"]},
              "by_batch_time":{name:{str(time):summarize([row for row in rows if row["batch"]==name and row["time"]==time])
                                    for time in protocol["query_times"]} for name in protocol["batch_names"]}},
        "cross_batch_transfer":transfer,"new_diagnostic_fits":0,"new_model_fitted":False,"new_policy_training":False,
        "new_simulator_calls":0,"new_gpu_work":0,"policy_optimizer_updates":0,"calibrator_updates":0,
        "confirmation_files_read":0,"full_training_hold":True,"automatic_training_approval":False,
        "method_or_strength_selected":False,"actionselector_diagnostic_only":True,"production_method_approved":False,
        "bootstrap_performed":True,"limitations":[protocol["post_hoc_status"],protocol["bootstrap_scope"],
            protocol["method_identity"],protocol["target_caveats"],
            "Numerical first-near-max selection is offset-sensitive; saved predictions/scores were not recentered and outcomes never break score ties.",
            "All ambiguous episodes and all unthresholded gains remain in the paired primary mean; win/loss labels are descriptive only.",
            "Only two corrected models have matched privileged-label fitting. Comparisons with original BCA and distance scores are descriptive, not deployable algorithm superiority.",
            "This no-fit calculation does not alter the earlier inconclusive confirmation or authorize production correction, normalization, new data or full training."]})
