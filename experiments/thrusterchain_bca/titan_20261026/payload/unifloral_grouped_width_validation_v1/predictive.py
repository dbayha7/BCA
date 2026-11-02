"""Fixed grouped predictive diagnostics, not actor/critic/calibrator training.

All fitting is inside analyze/fit_models. This module reads no files and calls
no policy, simulator, GPU, optimizer or confirmation interface.
"""
import hashlib
import json
import math
import random

MODELS = ("base", "base_plus_residual_width", "base_plus_residual_permuted_width")


def _np():
    import numpy as np
    return np


def _plain(value):
    if isinstance(value, dict): return {str(key): _plain(item) for key,item in value.items()}
    if isinstance(value, (tuple,list)): return [_plain(item) for item in value]
    if hasattr(value,"tolist"): return _plain(value.tolist())
    if value is None or isinstance(value,(str,bool,int)): return value
    if isinstance(value,float) and math.isfinite(value): return value
    raise ValueError("Nonfinite or unsupported report value")


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _finite(value, name, ndim=None):
    np=_np(); array=np.asarray(value,dtype=np.float64)
    if not array.size or (ndim is not None and array.ndim!=ndim) or not np.all(np.isfinite(array)):
        raise ValueError(name+" must be finite, nonempty and correctly ranked")
    return array


def center_states(array):
    array=_finite(array,"state array")
    if array.ndim not in (2,3): raise ValueError("Expected states x actions [x features]")
    return array-array.mean(axis=1,keepdims=True)


def _standardizer(train):
    np=_np(); train=_finite(train,"training features",2)
    mean=train.mean(axis=0)
    std=np.sqrt(np.mean((train-mean)**2,axis=0))
    if not np.all(np.isfinite(std)): raise ValueError("Feature scale overflow")
    return {"mean":mean,"std":std,"active":std>0}


def _standardize(values, stats):
    np=_np(); values=_finite(values,"feature values",2)
    transformed=np.zeros_like(values)
    active=stats["active"]
    transformed[:,active]=(values[:,active]-stats["mean"][active])/stats["std"][active]
    return _finite(transformed,"standardized features",2)


def ridge_solve(x,y,alpha=1.0):
    np=_np(); x=_finite(x,"ridge design",2); y=_finite(y,"ridge target",1)
    if len(x)!=len(y) or alpha!=1.0: raise ValueError("Aligned rows and locked alpha1 required")
    normal=x.T@x/len(x)+alpha*np.eye(x.shape[1],dtype=np.float64)
    beta=np.linalg.solve(normal,x.T@y/len(x))
    return _finite(beta,"ridge coefficients",1)


def fit_models(train_base,train_width,train_permuted_width,train_target,
               test_base,test_width,test_permuted_width,protocol):
    """Train-only preprocessing/projection and three fixed prediction specs.

    Caller supplies within-state-centered inputs/targets. No held-out targets
    are accepted by this interface. An inactive added feature copies the exact
    base prediction, rather than relying on a nearly equivalent solve.
    """
    np=_np(); tb=_finite(train_base,"train base",3); xb=_finite(test_base,"test base",3)
    ty=_finite(train_target,"training target",2)
    if tb.shape[2]!=2 or xb.shape[2]!=2 or tb.shape[:2]!=ty.shape or tb.shape[1]!=xb.shape[1]:
        raise ValueError("Base and training target shapes do not align")
    train_widths=[_finite(train_width,"train width",2),_finite(train_permuted_width,"train permuted width",2)]
    test_widths=[_finite(test_width,"test width",2),_finite(test_permuted_width,"test permuted width",2)]
    if any(w.shape!=tb.shape[:2] for w in train_widths) or any(w.shape!=xb.shape[:2] for w in test_widths):
        raise ValueError("Width features do not align with states/actions")
    xtrain=tb.reshape(-1,2); xtest=xb.reshape(-1,2); y=ty.reshape(-1)
    stats=_standardizer(xtrain)
    ztrain=_standardize(xtrain,stats); ztest=_standardize(xtest,stats)
    beta=ridge_solve(ztrain,y,protocol["ridge_alpha"])
    base_prediction=center_states((ztest@beta).reshape(xb.shape[:2]))
    predictions={MODELS[0]:base_prediction}
    fits={MODELS[0]:{"coefficients":beta,"feature_mean":stats["mean"],"feature_std":stats["std"],
                     "feature_active":stats["active"],"training_episodes":len(tb),"training_action_rows":len(y),
                     "ridge_alpha":protocol["ridge_alpha"],"intercept":False}}
    counts={"predictor_specifications":3,"ridge_system_solves":1,"width_projection_lstsq_fits":0,"inactive_width_collapses":0}
    design=np.column_stack((np.ones(len(ztrain)),ztrain))
    test_design=np.column_stack((np.ones(len(ztest)),ztest))
    for name,tw,xw in zip(MODELS[1:],train_widths,test_widths):
        wtrain=tw.reshape(-1); wtest=xw.reshape(-1)
        projection,_,rank,singular_values=np.linalg.lstsq(design,wtrain,rcond=protocol["lstsq_rcond"])
        counts["width_projection_lstsq_fits"]+=1
        residual=wtrain-design@projection
        residual_test=wtest-test_design@projection
        width_norm=float(np.linalg.norm(wtrain)); residual_norm=float(np.linalg.norm(residual))
        residual_mean=float(residual.mean()); residual_std=float(np.sqrt(np.mean((residual-residual_mean)**2)))
        ratio=None if width_norm==0 else residual_norm/width_norm
        inactive=(width_norm==0 or residual_norm<=protocol["residual_relative_l2_floor"]*width_norm or residual_std==0)
        if not np.all(np.isfinite([width_norm,residual_norm,residual_mean,residual_std])):
            raise ValueError("Width projection or scale overflow")
        column_norm=np.linalg.norm(design,axis=0)
        correlations=[None if residual_norm==0 or scale==0 else float(abs(np.dot(design[:,i],residual))/(scale*residual_norm))
                      for i,scale in enumerate(column_norm)]
        receipt={"projection_coefficients":projection,"projection_rank":int(rank),"projection_singular_values":singular_values,
                 "projection_rcond":protocol["lstsq_rcond"],"training_width_norm":width_norm,
                 "training_residual_norm":residual_norm,"residual_relative_norm":ratio,
                 "residual_feature_mean":residual_mean,"residual_feature_std":residual_std,
                 "residual_feature_active":not inactive,"training_residual_normalized_dots":correlations,
                 "base_feature_mean":stats["mean"],"base_feature_std":stats["std"],
                 "training_episodes":len(tb),"training_action_rows":len(y),"intercept":False}
        if inactive:
            predictions[name]=base_prediction.copy()
            receipt.update(coefficients=np.concatenate((beta,[0.])),inactive_reason=("zero_centered_training_width" if width_norm==0
                else "residual_at_or_below_relative_floor" if residual_norm<=protocol["residual_relative_l2_floor"]*width_norm else "zero_residual_std"),
                reused_exact_base_prediction=True,base_coefficient_max_abs_difference=0.)
            counts["inactive_width_collapses"]+=1
        else:
            residual_z=(residual-residual_mean)/residual_std
            residual_test_z=(residual_test-residual_mean)/residual_std
            augmented=np.column_stack((ztrain,residual_z)); test_augmented=np.column_stack((ztest,residual_test_z))
            added_beta=ridge_solve(augmented,y,protocol["ridge_alpha"])
            counts["ridge_system_solves"]+=1
            predictions[name]=center_states((test_augmented@added_beta).reshape(xb.shape[:2]))
            receipt.update(coefficients=added_beta,inactive_reason=None,reused_exact_base_prediction=False,
                           base_coefficient_max_abs_difference=float(np.max(np.abs(added_beta[:2]-beta))))
        fits[name]=receipt
    counts["total_linear_fitting_operations"]=counts["ridge_system_solves"]+counts["width_projection_lstsq_fits"]
    return {"predictions":predictions,"fits":fits,"fit_counts":counts}


def assign_folds(metadata,protocol):
    folds=[None]*len(metadata); cells=[]
    for b,batch in enumerate(protocol["batch_names"]):
        for t,time in enumerate(protocol["query_times"]):
            cell=[i for i,row in enumerate(metadata) if row["batch"]==batch and row["time"]==time]
            if len(cell)!=16: raise ValueError("Each batch/time cell must contain16 distinct episodes")
            ordered=sorted(cell,key=lambda i:metadata[i]["episode_seed"])
            random.Random(protocol["fold_seed"]+b*4+t).shuffle(ordered)
            for position,index in enumerate(ordered): folds[index]=position%protocol["fold_count"]
            cells.append({"batch":batch,"time":time,"indices":cell})
    if any(fold is None for fold in folds): raise ValueError("Unassigned episode")
    return folds,cells


def permute_width(width,metadata,seed):
    np=_np(); width=_finite(width,"centered width",2)
    if len(width)!=len(metadata): raise ValueError("Permutation metadata does not align")
    permutations=[]
    for row in metadata:
        order=list(range(width.shape[1])); random.Random(seed+row["group_id"]).shuffle(order); permutations.append(order)
    return np.asarray([row[order] for row,order in zip(width,permutations)],dtype=np.float64),permutations


def ranking_concordance(prediction,target,atol,rtol):
    prediction=_finite(prediction,"ranking prediction",1); target=_finite(target,"ranking target",1)
    if len(prediction)!=len(target): raise ValueError("Ranking vectors do not align")
    comparable=correct=wrong=predicted_ties=target_ties=0
    for i in range(len(target)):
        for j in range(i+1,len(target)):
            delta_y=target[i]-target[j]
            if abs(delta_y)<=atol+rtol*max(abs(target[i]),abs(target[j])):
                target_ties+=1; continue
            comparable+=1; delta_p=prediction[i]-prediction[j]
            if abs(delta_p)<=atol+rtol*max(abs(prediction[i]),abs(prediction[j])): predicted_ties+=1
            elif (delta_p>0)==(delta_y>0): correct+=1
            else: wrong+=1
    return {"defined":comparable>0,"score":None if not comparable else (correct+.5*predicted_ties)/comparable,
            "comparable_pairs":comparable,"correct_pairs":correct,"wrong_pairs":wrong,
            "predicted_tied_pairs":predicted_ties,"target_tied_pairs":target_ties,
            "reason":None if comparable else "all_target_pairs_tied"}


def _prepare(batches,protocol):
    np=_np()
    if [batch["name"] for batch in batches]!=protocol["batch_names"] or len(batches)!=2:
        raise ValueError("Expected the two fixed development batches")
    metadata=[]; base=[]; width=[]; target=[]
    for batch in batches:
        if len(batch["queries"])!=64: raise ValueError("Expected64 episodes per batch")
        for row in batch["queries"]:
            q=_finite(row["q1"],"Q1",1); d=_finite(row["actor_distance"],"distance",1)
            u=_finite(row["u"],"width",1); values=_finite(row["values"],"continuation values",1)
            if any(array.shape!=(9,) for array in (q,d,u,values)): raise ValueError("Expected nine aligned candidates")
            if np.any(d<0) or np.any(u<0): raise ValueError("Distance and width must be nonnegative")
            base.append(np.column_stack((q,d))); width.append(u); target.append(q-values)
            metadata.append({"batch":batch["name"],"group_id":row["group_id"],"time":row["time"],"episode_seed":row["episode_seed"]})
    if sorted(row["group_id"] for row in metadata)!=list(range(128)) or len({row["episode_seed"] for row in metadata})!=128:
        raise ValueError("Expected128 distinct state/episode identities")
    # Feature centering is label-free. Target centering is evaluation/training
    # response construction only and is never passed into a feature fit.
    return {"metadata":metadata,"base":center_states(base),"width":center_states(width),"target":center_states(target)}


def _score_rows(predictions,target,metadata,folds,protocol):
    np=_np(); rows=[]
    for i,row in enumerate(metadata):
        losses={name:float(np.mean((predictions[name][i]-target[i])**2)) for name in MODELS}
        if not all(math.isfinite(value) for value in losses.values()): raise ValueError("Nonfinite held-out loss")
        ranking={name:ranking_concordance(predictions[name][i],target[i],protocol["tie_atol"],protocol["tie_rtol"]) for name in MODELS}
        rows.append({**row,"fold":None if folds is None else folds[i],"mse":losses,"ranking":ranking,
                     "base_minus_width_mse":losses[MODELS[0]]-losses[MODELS[1]],
                     "base_minus_permuted_width_mse":losses[MODELS[0]]-losses[MODELS[2]],
                     "predictions":{name:predictions[name][i] for name in MODELS}})
    return rows


def _summary(rows):
    np=_np(); n=len(rows)
    if n==0: raise ValueError("Empty predictive summary")
    ranking={}
    for name in MODELS:
        defined=[row["ranking"][name]["score"] for row in rows if row["ranking"][name]["defined"]]
        ranking[name]={"defined_episodes":len(defined),"undefined_episodes":n-len(defined),
                       "equal_episode_mean":None if not defined else float(np.mean(defined))}
    differences={}
    for name in MODELS[1:]:
        common=[row for row in rows if row["ranking"][MODELS[0]]["defined"] and row["ranking"][name]["defined"]]
        differences[name+"_minus_base"]={"common_defined_episodes":len(common),"equal_episode_mean":None if not common else
            float(np.mean([row["ranking"][name]["score"]-row["ranking"][MODELS[0]]["score"] for row in common]))}
    return {"episodes":n,"model_mse":{name:float(np.mean([row["mse"][name] for row in rows])) for name in MODELS},
            "base_minus_width_mse":float(np.mean([row["base_minus_width_mse"] for row in rows])),
            "base_minus_permuted_width_mse":float(np.mean([row["base_minus_permuted_width_mse"] for row in rows])),
            "ranking":ranking,"paired_ranking_differences":differences}


def bootstrap_primary(differences,cells,protocol):
    np=_np(); differences=_finite(differences,"episode MSE differences",1)
    if len(differences)!=128 or len(cells)!=8: raise ValueError("Expected128 episodes in8 bootstrap cells")
    rng=np.random.default_rng(protocol["bootstrap_seed"])
    indices=np.empty((protocol["bootstrap_replicates"],128),dtype=np.int64)
    coverage=[]
    for cell in cells:
        positions=np.asarray(cell["indices"],dtype=np.int64)
        if len(positions)!=16: raise ValueError("Each bootstrap cell requires16 episodes")
        draws=rng.integers(0,16,size=(protocol["bootstrap_replicates"],16))
        indices[:,positions]=positions[draws]; coverage.extend(positions.tolist())
    if sorted(coverage)!=list(range(128)): raise ValueError("Bootstrap cells must partition episodes exactly")
    means=differences[indices].mean(axis=1)
    bounds=np.quantile(means,[.025,.975],method="linear")
    return {"mean":float(differences.mean()),"conditional_percentile_ci95":bounds,
            "interval_contains_zero":bool(bounds[0]<=0<=bounds[1]),"replicates":protocol["bootstrap_replicates"],
            "seed":protocol["bootstrap_seed"],"resample_indices_sha256":_digest(indices),
            "cells":cells,"scope":protocol["bootstrap_scope"],"primary_only":True,
            "refitting_uncertainty_included":False},indices


def _protocol(protocol):
    expected={"phase":"exploratory_grouped_saved_development_only","ridge_alpha":1.,"fold_count":8,
              "fold_seed":2026103021,"permutation_seed":2026103023,"bootstrap_seed":2026103022,
              "bootstrap_replicates":2000,"residual_relative_l2_floor":1e-12,"lstsq_rcond":1e-12,
              "tie_atol":1e-6,"tie_rtol":1e-6,"expected_diagnostic_fits":30,"candidates_per_episode":9,
              "model_names":list(MODELS),"query_times":[0,25,50,75]}
    for key,value in expected.items():
        if protocol.get(key)!=value: raise ValueError("Fixed grouped protocol changed: "+key)


def analyze(batches,protocol):
    np=_np(); _protocol(protocol)
    data=_prepare(batches,protocol); meta=data["metadata"]
    folds,cells=assign_folds(meta,protocol)
    permuted,permutations=permute_width(data["width"],meta,protocol["permutation_seed"])
    predictions={name:np.full_like(data["target"],np.nan) for name in MODELS}
    fit_receipts=[]; counts={key:0 for key in ("predictor_specifications","ridge_system_solves","width_projection_lstsq_fits","inactive_width_collapses","total_linear_fitting_operations")}
    def fit(train,test,label):
        out=fit_models(data["base"][train],data["width"][train],permuted[train],data["target"][train],
                       data["base"][test],data["width"][test],permuted[test],protocol)
        for key,value in out["fit_counts"].items(): counts[key]+=value
        fit_receipts.append({"split":label,"training_group_ids":[meta[i]["group_id"] for i in train],
                             "heldout_group_ids":[meta[i]["group_id"] for i in test],"models":out["fits"],"fit_counts":out["fit_counts"]})
        return out["predictions"]
    for fold in range(8):
        test=np.asarray([i for i,value in enumerate(folds) if value==fold],dtype=np.int64)
        train=np.asarray([i for i,value in enumerate(folds) if value!=fold],dtype=np.int64)
        if len(test)!=16 or len(train)!=112 or set(train)&set(test): raise ValueError("Grouped CV split is invalid")
        out=fit(train,test,"cv_fold_"+str(fold))
        for name in MODELS: predictions[name][test]=out[name]
    cv_rows=_score_rows(predictions,data["target"],meta,folds,protocol)
    primary,_=bootstrap_primary([row["base_minus_width_mse"] for row in cv_rows],cells,protocol)
    transfers=[]
    for train_name,test_name in (protocol["batch_names"],protocol["batch_names"][::-1]):
        train=np.asarray([i for i,row in enumerate(meta) if row["batch"]==train_name],dtype=np.int64)
        test=np.asarray([i for i,row in enumerate(meta) if row["batch"]==test_name],dtype=np.int64)
        if len(train)!=64 or len(test)!=64 or set(train)&set(test): raise ValueError("Cross-batch split invalid")
        out=fit(train,test,"transfer_"+train_name+"_to_"+test_name)
        rows=_score_rows(out,data["target"][test],[meta[i] for i in test],None,protocol)
        transfers.append({"training_batch":train_name,"heldout_batch":test_name,"summary":_summary(rows),"episodes":rows,"interval_computed":False})
    if counts["predictor_specifications"]!=protocol["expected_diagnostic_fits"]: raise ValueError("Unexpected diagnostic predictor count")
    cv={"overall":_summary(cv_rows),"primary":primary,"episodes":cv_rows,
        "by_batch":{name:_summary([row for row in cv_rows if row["batch"]==name]) for name in protocol["batch_names"]},
        "by_time":{str(time):_summary([row for row in cv_rows if row["time"]==time]) for time in protocol["query_times"]},
        "by_batch_time":{name:{str(time):_summary([row for row in cv_rows if row["batch"]==name and row["time"]==time])
                         for time in protocol["query_times"]} for name in protocol["batch_names"]}}
    return _plain({"status":"pass","claim_scope":protocol["claim_scope"],"phase":protocol["phase"],
        "cv":cv,"cross_batch_transfer":transfers,"fit_receipts":fit_receipts,"fit_counts":counts,
        "diagnostic_regression_fits":counts["predictor_specifications"],
        "fit_count_note":"30 predictor specifications; actual ridge solves and width-projection least-squares operations counted separately. No iterative optimizer or RL updates.",
        "fold_assignment":folds,"fold_assignment_sha256":_digest(folds),"width_permutations":permutations,
        "width_permutations_sha256":_digest(permutations),"fixed_permutation_realizations":1,
        "full_training_hold":True,"automatic_training_approval":False,"method_or_strength_selected":False,
        "action_policy_changed":False,"new_policy_training":False,"bootstrap_performed":True,"confirmation_files_read":0,
        "new_simulator_calls":0,"new_gpu_work":0,"optimizer_updates":0,"calibrator_updates":0,
        "limitations":[protocol["post_hoc_status"],protocol["bootstrap_scope"],
            "Residual width tests additional linear predictive value beyond the training linear span of Q1 and distance; nonlinear redundancy can still help, so this is not information-complete independence or causality.",
            "A single fixed within-state permutation is a negative-control realization, not a permutation test or p-value.",
            "Near-threshold active residuals are standardized to unit variance; numerical activity alone does not imply scientific importance.",
            "All outcomes concern candidate-bank discrepancy prediction and ranking, not a new action-selection rule or policy-learning benefit.",
            "Cross-batch transfer reuses these development batches and the same trained checkpoint; it is not confirmation or training-seed replication.",
            protocol["target_caveats"]]})
