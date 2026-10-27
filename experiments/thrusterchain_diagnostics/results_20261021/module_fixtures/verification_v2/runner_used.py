"""Small deterministic fixtures around the real BCA/IWBCA implementation.

Run with the repository's existing Python; writes only beside this script.
No learning, rollouts, checkpoint changes, network access, or repository edits.
"""
import argparse
import ast
import ctypes
import hashlib
import inspect
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "JAX_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
             "TF_NUM_INTEROP_THREADS"):
    os.environ[name] = "1"
os.environ.update(JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu",
                  JAX_ENABLE_COMPILATION_CACHE="false", PYTHONDONTWRITEBYTECODE="1",
                  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_force_host_platform_device_count=1")


def constrain_process():
    if os.name != "nt":
        raise RuntimeError("This recorded command is for the authorized Windows Zeus environment only")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    handle = kernel.GetCurrentProcess()
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.GetPriorityClass.argtypes = [ctypes.c_void_p]
    kernel.GetPriorityClass.restype = ctypes.c_uint32
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    if not kernel.SetProcessAffinityMask(handle, 0xF0):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.SetPriorityClass(handle, 0x40):
        raise ctypes.WinError(ctypes.get_last_error())
    current, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle, ctypes.byref(current), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"cpu_indices": [4, 5, 6, 7], "affinity_mask": int(current.value),
            "priority_class": int(kernel.GetPriorityClass(handle)), "priority": "IDLE",
            "thread_environment": {k: os.environ[k] for k in os.environ if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}}


RESOURCE = constrain_process()
import numpy as np
from scipy import integrate, stats


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return "inf" if np.isposinf(value) else "-inf" if np.isneginf(value) else None if np.isnan(value) else float(value)
    return value


def source_ref(obj):
    lines, first = inspect.getsourcelines(obj)
    path = Path(inspect.getsourcefile(obj)).resolve()
    return {"path": path.relative_to(REPO).as_posix(), "first_line": first,
            "last_line": first + len(lines) - 1,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


CASES = []


def record(case_id, stage, premise, inputs, expected, observed, passed, interpretation, refs,
           contract_passed=None):
    contract = passed if contract_passed is None else contract_passed
    CASES.append(dict(id=case_id, stage=stage, premise=premise, inputs=clean(inputs),
                      expected=clean(expected), observed=clean(observed),
                      status="PASS" if passed and contract else "FAIL",
                      harness_status="PASS" if passed else "FAIL",
                      contract_status="PASS" if contract else "FAIL", interpretation=interpretation,
                      source_refs=[ref if isinstance(ref,dict) else source_ref(ref) for ref in refs]))


def independent_calibration(scores, weights, query, exponentials, alpha, beta):
    """Independent specification: enumerate candidates and test tail risk directly.

    Unlike native cumulative crossing, this sums losses of scores strictly ABOVE
    each candidate plus the separate worst-case query atom. No native helper is
    called. The exponentials are sorted-score aligned by the caller.
    """
    scores, weights = np.asarray(scores), np.asarray(weights)
    order = np.argsort(scores, kind="stable")
    s, w = scores[order], weights[order]
    candidates = np.unique(s)
    empirical = next(float(t) for t in candidates if sum(Fraction.from_float(float(v)) for v in w[s > t])
                     <= Fraction(str(alpha)) * sum(Fraction.from_float(float(v)) for v in w))
    crossing = []
    for draw in exponentials:
        observed_mass = draw[:-1] * w
        unseen = float(draw[-1] * query)
        total = float(observed_mass.sum() + unseen)
        feasible = [float(t) for t in candidates
                    if float(observed_mass[s > t].sum() + unseen) <= alpha * total + 1e-13]
        crossing.append(min(feasible) if feasible else math.inf)
    crossing = np.sort(crossing)
    posterior = float(crossing[math.ceil(Fraction(str(beta)) * len(crossing)) - 1])
    return dict(lambda_hat=empirical, lambda_hpd=posterior,
                threshold=max(empirical, posterior), crossings=crossing,
                n_eff=float(weights.sum() ** 2 / np.dot(weights, weights)))


def snapshot(cal):
    return dict(threshold=cal.threshold, lambda_hat=cal.lambda_hat, lambda_hpd=cal.lambda_hpd,
                n_eff=cal.n_eff, finite_draw_fraction=float(np.isfinite(cal.crossings).mean()),
                certified_finite=cal.certified)


def fixture_cal(scores, weights=None, query=None, seed=20261021, draws=1000):
    return WB.calibrate(np.asarray(scores), np.random.default_rng(seed), weights, query, draws=draws)


def frozen(R=1., unit=1., ready=True):
    return REF.FrozenReference({}, jnp.asarray(unit, jnp.float32), jnp.asarray(R, jnp.float32),
                               jnp.asarray(R, jnp.float32), jnp.asarray(R, jnp.float32),
                               jnp.asarray(128., jnp.float32), jnp.asarray(ready))


def run_cases():
    # W1: ratios independently calculated from normalized densities.
    c = np.array([.1, .35, .6, 1.1])
    expected = np.exp(4 * (c - .6))
    observed = WORLD.ratio(c, "strong")
    m = lambda lam: (math.exp(1.1 * lam) - math.exp(.1 * lam)) / lam if lam else 1.
    q2_over_p, _ = integrate.quad(lambda x: (math.exp(2*x)/m(2))**2 / (math.exp(-2*x)/m(-2)), .1, 1.1)
    record("known_population_ratio", "W1", "Known truncated-exponential source and target laws; no fitted ratio.",
           dict(c=c, source_lambda=-2, target_lambda=2),
           dict(ratios=expected, target_mean_ratio=q2_over_p),
           dict(ratios=observed, target_mean_ratio=WORLD.full_test_mass("strong")),
           np.allclose(expected, observed, atol=1e-13) and abs(q2_over_p-WORLD.full_test_mass("strong")) < 1e-12,
           "Exact ratio arithmetic passes for supplied laws; this does not certify a learned ratio.", [WORLD.ratio, WORLD.full_test_mass])

    # Exact sample-level IID classifier optimum with unequal class sizes.
    x = np.concatenate([np.eye(13), -np.eye(13)])
    x[:, 10] = .6 + .25 * x[:, 10]
    clf = CLASSIFIER.fit(x, np.repeat(x, 2, axis=0))
    w = clf.weight(x)
    record("iid_classifier_class_size_correction", "W1", "Both classes have the identical empirical covariate law; target class has twice as many rows.",
           dict(source_rows=26, target_rows=52, construction="plus/minus identity design, every target row repeated twice"),
           dict(log_odds=math.log(2), weights=np.ones(26), converged=True),
           dict(log_odds=clf.log_odds(x), weights=w, converged=clf.converged, max_score=clf.max_score),
           np.allclose(w, 1, atol=1e-12) and clf.converged,
           "Class-size correction is necessary: raw odds would be 2 despite IID covariates. This exact finite design does not test estimator generalization.",
           [CLASSIFIER.fit, CLASSIFIER.Classifier.log_weight, CLASSIFIER.Classifier.weight])

    # N5: hand-computed atoms and independent posterior selection.
    s, w, q = np.array([1., 3., 9.]), np.array([1., 2., 1.]), 1.
    E = np.array([[1, 1, 1, 1], [1, 1, 1, 4], [2, 1, 1, .5], [1, 1, 1, 0]], dtype=float)
    observed = WB.crossings(s, w, q, E, .4)
    expected = np.array([3., math.inf, 3., 3.])
    record("hand_query_atom_crossings", "N5", "Candidate is feasible when mass of scores above it plus the unseen atom is at most alpha of all mass.",
           dict(scores=s, weights=w, query_mass=q, alpha=.4, exponentials=E), expected, observed,
           np.array_equal(expected, observed), "A sufficiently large query atom gives infinity even though all observed scores are finite.", [WB.crossings])
    scores = np.arange(1., 49.) / 10
    weights = np.tile([1., 2., .5], 16)
    seed, draws = 20261021, 256
    exponentials = np.random.default_rng(seed).standard_exponential((draws, len(scores)+1))
    independent = independent_calibration(scores, weights, .75, exponentials, .1, .95)
    actual = fixture_cal(scores, weights, .75, seed, draws)
    record("independent_tail_risk_posterior", "N5", "Enumerate tail-risk inequalities and credible selection independently of native cumulative sums.",
           dict(scores=scores, weights=weights, query_mass=.75, alpha=.1, beta=.95, draws=draws, seed=seed),
           independent, dict(**snapshot(actual), crossings=actual.crossings),
           np.array_equal(independent['crossings'], actual.crossings) and independent['threshold'] == actual.threshold
           and independent['lambda_hat'] == actual.lambda_hat and abs(independent['n_eff']-actual.n_eff) < 1e-10,
           "Checks arithmetic conditional on the recorded exponential draw realization. This is not a coverage theorem.", [WB.calibrate])

    # W2: exact control and a deliberately incorrect one-sided normalization.
    scores = np.linspace(.1, 2, 128)
    unweighted = fixture_cal(scores)
    unit = fixture_cal(scores, np.ones(128), 1.)
    record("all_one_weights_reduce_to_native", "W2", "Same scores, test mass 1 and same posterior random stream.",
           dict(n=128, seed=20261021, weights="all one", query_mass=1), snapshot(unweighted), snapshot(unit),
           np.array_equal(unweighted.crossings, unit.crossings) and unweighted.threshold == unit.threshold,
           "Finite-threshold arithmetic is identical. Refusal-policy equivalence is a separate fixture.", [WB.calibrate, IW.calibrate])
    w = np.tile([.5, 1, 2, 4], 32)
    first = fixture_cal(scores, w, .75)
    scaled = fixture_cal(scores, 8*w, 8*.75)
    record("common_rescaling_invariance", "W2", "A common positive factor multiplies every bank weight and the query mass.",
           dict(scores=scores, weights=w, query_mass=.75, factor=8, seed=20261021), snapshot(first), snapshot(scaled),
           np.array_equal(first.crossings, scaled.crossings) and first.n_eff == scaled.n_eff,
           "Only common rescaling is invariant; normalizing bank weights alone changes the query's relative mass.", [WB.calibrate])
    normalizations = []
    for seed in (7, 19, 20261021):
        correct = fixture_cal(np.ones(48), np.full(48, 2.), 2., seed)
        incorrect = fixture_cal(np.ones(48), np.ones(48), 2., seed)
        normalizations.append(dict(seed=seed, common_scale=snapshot(correct), bank_only=snapshot(incorrect)))
    record("bank_only_normalization_exception", "W2", "Intentionally wrong preprocessing divides only bank weights by two.",
           dict(n=48, scores=1., original_bank_weights=2., original_query_mass=2., draws=1000, seeds=[7,19,20261021]),
           "Common-scale thresholds finite at 1; bank-only normalization refuses at infinity for these fixed seeds.", normalizations,
           all(r['common_scale']['threshold'] == 1 and math.isinf(r['bank_only']['threshold']) for r in normalizations),
           "Reproducible exceptional failure mechanism, not a defect in the native common-scale normalization.", [WB.calibrate])

    # W3: actual zero-support ratio vs finite-bank posterior refusal.
    gap = SimpleNamespace(contains=lambda actions: actions[:, 0] == 0)
    ratios = WORLD.exact_ratio(np.array([.4, .4]), np.array([[0.,0.,0.,0.], [.5,0.,0.,0.]]), gap, "none", .8)
    rejected = False
    try:
        fixture_cal(np.ones(64), np.ones(64), math.inf)
    except ValueError:
        rejected = True
    record("zero_support_is_refused_not_epsilon_repaired", "W3", "The removed action set has zero source support; target-only actions have infinite density ratio.",
           dict(current=.4, Z=.8, tilt="none", removed_action_first_coordinate=0),
           dict(ratios=[math.inf,.8], infinite_test_mass_rejected=True),
           dict(ratios=ratios, infinite_test_mass_rejected=rejected),
           np.array_equal(ratios, [math.inf,.8]) and rejected,
           "Structural support refusal is upstream of calibration; finite positive query mass can separately cause a posterior threshold infinity.",
           [WORLD.exact_ratio, WB.calibrate])
    grid = np.array([.1, 1., 5., 1000.])
    tab = IW.threshold_table(np.linspace(.1, 2, 64), [20261021, 1], np.ones(64), grid)
    looked, indices = IW.lookup(tab, np.array([.1,.5,2.,1000.]))
    record("query_mass_table_and_conservative_lookup", "W6", "Coupled posterior draws; query masses round up to the next grid point.",
           dict(n=64, scores="linspace(.1,2,64)", weights=1, stream=[20261021,1], grid=grid, query=[.1,.5,2,1000]),
           dict(monotone=True, running_max_applied=False, indices=[0,1,2,3], last_threshold="inf"),
           dict(table=tab.summary(), lookup=looked, indices=indices),
           tab.monotone and not tab.running_max_applied and np.array_equal(indices,[0,1,2,3]) and math.isinf(looked[-1]),
           "The fixed sample demonstrates monotonic arithmetic and conservative lookup, not target-population coverage.", [IW.threshold_table, IW.lookup])

    # W4/W5: ESS and distinction between actual no-clipping and a sensitivity control.
    w = np.array([.01,1.,100.])
    expected_ess = sum(w)**2 / sum(v*v for v in w)
    raw = fixture_cal(np.array([1.,2.,3.]), w, 2.)
    clipped_w = np.minimum(w,10.)
    sensitivity = fixture_cal(np.array([1.,2.,3.]), clipped_w, 2.)
    record("concentration_ess_and_no_native_ratio_clipping", "W4", "The native engine rescales but does not clip supplied ratios; clipping below is an explicit input sensitivity control.",
           dict(raw_weights=w, sensitivity_only_cap=10, query_mass=2),
           dict(raw_ess=expected_ess, uniform_ess=3., clipping_is_native=False),
           dict(raw_ess=raw.n_eff, iw_kish=IW.kish(w), uniform_ess=IW.kish(np.ones(3)), manually_capped_ess=sensitivity.n_eff),
           abs(raw.n_eff-expected_ess) < 1e-12 and raw.n_eff < 1.1 and IW.kish(np.ones(3)) == 3.,
           "Concentration reduces ESS; raising ESS by manually clipping changes the target weighting. No native clipping stage is claimed.", [WB.calibrate, IW.kish, CLASSIFIER.Classifier.weight])

    # N4/N6/N7: freeze scores and scalar units explicitly, without fitting a network.
    config = REF.WBCPConfig(draws=1000)
    cases = []
    for residual, unit_value in ((.1,1.),(10.,1.),(10.,100.)):
        ref, valid, diag = REF.freeze_reference({}, unit_value, np.ones(128), np.full(128,residual),
                                                 jnp.array([20261021,3],jnp.uint32), config)
        dose = DOSE.frozen_level_dose(ref, jnp.ones(3), .5)
        cases.append(dict(residual=residual,unit=unit_value,valid=valid,threshold=float(ref.threshold),
                          width=np.asarray(dose.width),dose=np.asarray(dose.dose),support=np.asarray(dose.support_mask)))
    expected = [1+.5*.1/1.1, 1+.5*10/11, 1+.5*.1/1.1]
    record("iid_same_target_low_high_residual", "N4", "IID/unit-weight calibration, fixed eta=1; residual and frozen unit are controlled separately.",
           dict(n=128, eta=1, cases=[[.1,1],[10,1],[10,100]], key=[20261021,3], blend=.5),
           dict(doses=expected, widths=[.1,10,10]), cases,
           all(row['valid'] and np.allclose(row['dose'],want,atol=2e-7) and
               np.allclose(row['width'],width,atol=2e-6) and
               np.isclose(row['threshold'],threshold,atol=2e-7)
               for row,want,width,threshold in zip(cases,expected,[.1,10,10],[.1,10,.1])),
           "IID does not imply small normalized error or low penalty. High absolute residual with proportionally larger unit can have the same dose as the low-residual case.",
           [REF.freeze_reference, REF.positive_scale, DOSE.frozen_level_dose])
    invalid, valid, _ = REF.freeze_reference({}, 0., np.ones(128), np.ones(128),jnp.array([1,2],jnp.uint32),config)
    record("invalid_reference_does_not_become_ready", "N9", "A zero residual unit is outside the positive-scale contract.",
           dict(unit=0, n=128), dict(valid=False,ready=False), dict(valid=valid,ready=bool(invalid.ready)),
           not valid and not bool(invalid.ready), "The helper returns an invalid/not-ready reference; caller preservation of a previous state is source-audited separately.", [REF.freeze_reference])
    widths = jnp.array([0.,1.,4.,jnp.inf])
    dose = DOSE.level_critic_dose(widths,jnp.ones(4,dtype=bool),1.,.5)
    expected = np.array([1.,1.25,1.4,1.])
    record("native_dose_bound_direction_and_infinite_fallback", "N7", "Finite width increases the multiplier monotonically; native unsupported infinity returns neutral dose 1.",
           dict(widths=np.asarray(widths),unit=1.,blend=.5), dict(dose=expected,support=[True,True,True,False]),
           dict(dose=np.asarray(dose.dose),support=np.asarray(dose.support_mask)),
           np.allclose(np.asarray(dose.dose),expected,atol=1e-7) and np.array_equal(dose.support_mask,[True,True,True,False]),
           "Infinity is a refusal branch, not the finite-width limit 1+blend. Plot it as refusal, not as ordinary high uncertainty.", [DOSE.level_critic_dose])

    # W7: actual IW policies, including the all-ones refusal confound.
    out0, refused0 = IW.per_row_dose(frozen(),jnp.ones(2),jnp.array([1.,jnp.inf]),.5,False)
    out1, refused1 = IW.per_row_dose(frozen(),jnp.ones(2),jnp.array([1.,jnp.inf]),.5,True)
    record("identical_thresholds_different_refusal_policy", "W7", "Hold eta, unit and row thresholds fixed; change only the actual most_caution flag.",
           dict(thresholds=[1,math.inf],eta=1,unit=1,blend=.5),
           dict(native_rule=[1.25,1.],most_caution=[1.25,1.5],refused=[False,True]),
           dict(native_rule=np.asarray(out0.dose),most_caution=np.asarray(out1.dose),refused=np.asarray(refused1)),
           np.allclose(out0.dose,[1.25,1.]) and np.allclose(out1.dose,[1.25,1.5]) and np.array_equal(refused0,refused1),
           "A penalty increase at refusal can be caused entirely by the refusal rule, even when all importance weights are one. Do not attribute it to ratio weighting.", [IW.per_row_dose])
    native_iql, _ = IW.per_row_iql_weights(frozen(),jnp.ones(3),jnp.full(3,jnp.inf),jnp.array([1.,0.,-1.]),3.,100.,False)
    cautious_iql, _ = IW.per_row_iql_weights(frozen(),jnp.ones(3),jnp.full(3,jnp.inf),jnp.array([1.,0.,-1.]),3.,100.,True)
    record("iql_refusal_is_host_specific", "W7", "The IQL action weight is not a TD3/CQL/ReBRAC penalty multiplier.",
           dict(advantages=[1,0,-1],thresholds="inf",beta=3,cap=100),
           dict(native_rule=[0,0,0],most_caution=[1,1,math.exp(-3)]),
           dict(native_rule=np.asarray(native_iql.weights),most_caution=np.asarray(cautious_iql.weights),
                native_has_support=bool(native_iql.has_support),cautious_has_support=bool(cautious_iql.has_support)),
           np.allclose(native_iql.weights,0) and np.allclose(cautious_iql.weights,[1,1,math.exp(-3)],atol=1e-7),
           "Native IQL refuses unsupported rows; most-caution IQL uses cloning weight 1 for positive advantage and the native lower weight otherwise.", [IW.per_row_iql_weights])

    # Controlled finite IW moves in both directions, no target-policy training.
    movements = []
    for direction,scores,w in (
        ("increase",np.r_[np.ones(124),np.full(4,10.)],np.r_[np.ones(124),np.full(4,16.)]),
        ("decrease",np.r_[np.ones(64),np.full(64,10.)],np.r_[np.full(64,8.),np.full(64,.125)])):
        for seed in (7,19,20261021):
            plain = fixture_cal(scores,seed=seed)
            weighted = fixture_cal(scores,w,1.,seed)
            expected_pair = [1.,10.] if direction == "increase" else [10.,1.]
            movements.append(dict(direction=direction,seed=seed,plain=snapshot(plain),weighted=snapshot(weighted),
                                  plain_dose=1+.5*plain.threshold/(plain.threshold+1),
                                  weighted_dose=1+.5*weighted.threshold/(weighted.threshold+1),expected_thresholds=expected_pair))
    record("controlled_iw_increase_and_decrease", "N5", "Fixed scores and random streams; only relative calibration weights change, with query mass 1.",
           dict(seeds=[7,19,20261021],draws=1000,
                increase="124 scores=1 weights=1; 4 scores=10 weights=16",
                decrease="64 scores=1 weights=8; 64 scores=10 weights=.125",query_mass=1),
           "Increase: R 1 to 10 and dose 1.25 to 1.454545. Decrease: reversed; both finite for all three seeds.", movements,
           all([row['plain']['threshold'],row['weighted']['threshold']] == row['expected_thresholds'] for row in movements),
           "Correct IW need not always raise caution: it emphasizes whichever residual region the supplied target weights favor. These are controlled fixtures, not deployment results.", [WB.calibrate])

    # N8: detached multiplier and oracle alignment limitation.
    gradient = np.asarray(jax.grad(lambda x: DOSE.level_critic_dose(x,jnp.ones(3,dtype=bool),1.,.5).dose.sum())(jnp.array([.1,1.,10.])))
    record("dose_is_detached_from_width_gradient", "N8", "The multiplier is a frozen coefficient; this probes its actual autodiff behavior only.",
           dict(widths=[.1,1,10],blend=.5,unit=1), [0,0,0], gradient,
           np.array_equal(gradient,np.zeros(3)), "Checks stop-gradient ownership at the dose boundary. This does not execute any host optimizer update.", [DOSE.level_critic_dose])
    action = np.array([-1.,0.,1.]); truth = -action**2; prediction = np.zeros(3)
    error, harm = abs(prediction-truth), truth.max()-truth
    alignments = []
    for name,eta in (("aligned",[1,.01,1]),("reversed",[.01,1,.01])):
        read = DOSE.frozen_level_dose(frozen(),jnp.array(eta),.5)
        d = np.asarray(read.dose)
        alignments.append(dict(case=name,dose=d,half_width=np.asarray(read.width),
                               error_rank_correlation=stats.spearmanr(d,error).statistic,
                               action_harm_rank_correlation=stats.spearmanr(d,harm).statistic))
    record("known_oracle_alignment_and_counterexample", "N8", "Analytic one-step oracle Q(a)=-a^2, a in {-1,0,1}; predictor Qhat=0. Eta is deliberately specified, not fitted.",
           dict(actions=action,true_Q=truth,predicted_Q=prediction,absolute_error=error,action_harm=harm,R=1,unit=1),
           dict(aligned_correlation=1,reversed_correlation=-1),alignments,
           alignments[0]['action_harm_rank_correlation'] > .999 and alignments[1]['action_harm_rank_correlation'] < -.999,
           "Correct width-to-dose arithmetic alone does not guarantee useful error or action-harm ranking. The reversed fixture is an intentional counterexample, not an observed trained-agent failure.", [DOSE.frozen_level_dose])


def run_exploratory_counterexamples():
    """Execute exact source AST slices, never the exploratory driver/imports/constructors."""
    path=REPO/'outputs/shift_iw/explore/channel/command_band_check.py'
    tree=ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    names={'recorder_mean','normal_logdensity','recorder_logdensity','spread_logdensity','Rows','Thresholds'}
    assignments={'TILT','SPREAD','CLIP','FLOOR','REC','LOG_PHI0'}
    selected=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id in assignments for t in node.targets for n in ast.walk(t)):
            selected.append(node)
    namespace={'__name__':'fixture_source_slice','np':np,'math':math,'W':WORLD,'RC':WORLD.RC,'CV':CV}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    src_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    refs=[dict(path=path.relative_to(REPO).as_posix(),first_line=lo,last_line=hi,sha256=src_hash)
          for lo,hi in [(52,85),(130,136),(160,172),(261,262)]]
    rows=object.__new__(namespace['Rows'])
    rows.x=np.zeros((1,13));rows.x[:,10]=.6
    rows.c=np.array([.6]);rows.eps=np.ones(1);rows.hard=np.zeros(1);rows.Z=np.ones(1)
    threshold=object.__new__(namespace['Thresholds'])
    threshold.floor=1e-4;threshold.cap=1.
    threshold.table=CV.Table(np.array([1e-4,1.,1000.]),np.ones(3),np.ones(3),1.,64.,True,False)
    for kind,centre,action,cutoff in [
        ('noise_clip_atom',0.,.5,2.5),('action_clip_atom',.8,1.,1.)]:
        a=np.array([[action,0.,0.,0.]])
        mu=np.array([[centre,0.,0.,0.]])
        lb=namespace['recorder_logdensity'](a,rows.x,rows.eps,rows.hard)
        ls=namespace['spread_logdensity'](a,mu)
        lw=rows.log_ratio(np.array([0]),a,mu)
        R,refused=threshold.read(lw)
        behavior_reproduced=(np.isclose(lb[0],-math.log(16)) and not refused[0] and R[0]==1.
                              and (np.isneginf(lw[0]) if kind=='noise_clip_atom' else np.isfinite(lw[0])))
        mass=float(stats.norm.sf(cutoff))
        record('exploratory_'+kind,'E1',
               'The clipped target has positive mass on a coordinate hyperplane; the eps=1 continuous source law has zero mass there.',
               dict(centre=mu,action=a,source_random_share=1.,current=.6,Z=1.,spread=.2,noise_clip=.5,
                    action_clip=[-1,1],test_table_grid=threshold.table.grid,test_table_thresholds=threshold.table.thresholds,
                    extraction='Exact AST function/class bodies and constants; bypass constructors and entire experiment driver.'),
               dict(continuous_source_event_mass=0.,target_coordinate_event_mass=mass,
                    proper_continuous_law_ratio='singular: no finite Radon-Nikodym density',proper_refusal=True,
                    reproduced_legacy_log_ratio='-inf' if kind=='noise_clip_atom' else 'finite',reproduced_legacy_refusal=False),
               dict(source_logdensity=lb,spread_logdensity=ls,legacy_log_ratio=lw,returned_threshold=R,refused=refused,
                    source_file_sha256=src_hash),behavior_reproduced,
               'HARNESS PASS means the defective behavior was reproduced; SCIENTIFIC CONTRACT FAIL means the formula does not represent the clipped target law. '
               'The first case floors a -inf log mass; the second uses an ordinary finite density on an action-boundary atom. '
               'Source-zero statements concern the continuous law represented by the formulas. Executable float32 laws require consistent lattice probabilities, which are not supplied by this density formula. '
               'Exploratory command-weighted experiment only; no claim that native/core current-law IW or frozen30 is affected.',
               refs,contract_passed=False)


def write_stage_map():
    definitions = [
        ("W1","Estimate or supply density ratios","Source and target laws are named, with common conditional outcome law.","covariates and class labels / explicit laws","bank ratios and query ratios","Separate ratio arithmetic from fitted-ratio correctness.","IID law gives ratio 1 after class-prior correction.",
         ["w(x) = p_target(x) / p_source(x)","Estimated w(x) = exp(classifier_log_odds(x)) * n_source/n_target"],[WORLD.ratio,CLASSIFIER.Classifier.log_weight]),
        ("W2","Normalize bank and query masses together","Only a common positive scale factor is irrelevant.","bank w and query t","scaled w/max(w), t/max(w)","Preserve the relative unseen-query mass.","Scaling BOTH inputs leaves crossings, threshold and ESS unchanged.",
         ["c = max_i w_i", "w_i <- w_i/c; t <- t/c"],[WB.calibrate]),
        ("W3","Check support and choose query mass","Zero support cannot be repaired by an epsilon denominator.","source support, bank weights, query point/population","finite query mass or upstream refusal","Separate structural refusal from finite-bank posterior refusal.","Target-only support is infinite ratio, never a fabricated finite weight.",
         ["if query action has zero source support: refuse before calibration", "scalar mode uses target mean weight; per-row mode uses that row's own weight"],[WORLD.exact_ratio,WORLD.full_test_mass,IW.calibrate]),
        ("W4","Report weight concentration (ESS)","Kish ESS is a weight-concentration summary, not a coverage certificate.","nonnegative bank weights","ESS", "Make concentrated weight uncertainty visible.","ESS=(sum w)^2/sum(w^2); uniform ESS=n and concentrated ESS approaches 1.",
         ["ESS = (sum_i w_i)^2 / sum_i w_i^2"],[IW.kish,WB.calibrate]),
        ("W5","State clipping status explicitly","Native calibration and this IW ratio path do not clip supplied weights.","supplied ratio vector","same ratios up to common scale","Avoid attributing a manually clipped sensitivity input to production code.","Rescaling is not clipping; no ratio cap is introduced by these functions.",
         ["No importance-weight clipping operation is present in this path", "Manual cap experiment is a separate changed-input sensitivity control"],[WB.calibrate,CLASSIFIER.Classifier.weight]),
        ("N4","Freeze residual scale and held-out scores","A frozen eta and positive residual unit are available; no scale network is trained in these fixtures.","residual y-q, eta, u, random key","s_i=abs(y_i-q_i)/(max(eta_i,1e-6)*u)","Check scale/score bookkeeping separately from learning.","IID restricts the distribution ratio, not the magnitude of these scores.",
         ["sigma_i = max(eta_i, 1e-6) * u", "s_i = abs(residual_i) / sigma_i", "Freeze scale parameters and unit at refresh"],[REF.freeze_reference,REF.positive_scale]),
        ("N5","Select empirical and posterior thresholds","Scores and weights frozen; posterior draw stream recorded.","scores, w, t, alpha, beta, exponentials","lambda_hat, lambda_hpd, R, finite/refused","Verify threshold arithmetic against direct tail-risk enumeration.","R=max(lambda_hat,lambda_hpd); each posterior draw includes the query atom at worst-case loss.",
         ["lambda_hat = smallest observed score with weighted empirical tail <= alpha", "draw mass_i = w_i E_i; unseen_mass = t E_query", "crossing = smallest score with (tail_mass+unseen_mass)/total_mass <= alpha; else infinity", "lambda_hpd = beta order statistic of crossings; R=max(lambda_hat,lambda_hpd)"],[WB.crossings,WB.calibrate]),
        ("W6","Look up each row's threshold","Every grid point uses the same posterior draw stream.","bank scores/weights, query-mass grid and row masses","per-row R_i","Keep query mass separate from calibration weight normalization.","R(t) does not decrease; lookup rounds t upward and refuses values beyond the grid.",
         ["tabulate R(t) on ascending mass grid with coupled random draws", "choose first grid mass >= query mass; take its threshold"],[IW.threshold_table,IW.lookup]),
        ("N6","Convert threshold to half-width","The reference has frozen positive unit and predicted scale.","R, eta, u","U=R*max(eta,1e-6)*u","Separate normalized threshold from response units.","U is a half-width, not the full interval width 2U.",
         ["U_i = R_i * max(eta_i,1e-6) * u"],[DOSE.frozen_level_dose]),
        ("N7","Convert finite width to bounded dose","Blend is fixed in [0,1]; width and residual unit are valid.","U,u,blend,usable","dose and support mask","Check penalty direction independently of coverage and host placement.","Finite dose=1+blend*U/(U+u), lies in [1,1+blend]; native unsupported row uses dose 1.",
         ["supported = valid and usable and finite(U)", "dose = 1 + blend*U/(U+u) on supported rows; native fallback=1", "stop_gradient(dose)"],[DOSE.level_critic_dose]),
        ("W7","Apply explicit refusal policy","A ready reference and infinite row threshold indicate refusal.","R_i, host type, most_caution flag","host-specific fallback weight/dose","Expose a policy difference that can remain even with unit ratios.","TD3/ReBRAC/CQL: native=1, most-caution=1.5 at blend=.5; IQL has its own rule.",
         ["if refused and most_caution: TD3/ReBRAC/CQL dose=1+blend", "IQL native refusal weight=0; most-caution positive-advantage weight=1"],[IW.per_row_dose,IW.per_row_iql_weights]),
        ("N8","Attach detached multiplier and assess alignment","Host placement is source-audited; these fixtures do not run a host update.","dose/actor weight and known diagnostic oracle","gradient check; error/harm ranking","Distinguish arithmetic correctness from useful caution.","Dose boundary has zero gradient; correct arithmetic alone need not rank actual error or harm.",
         ["Use detached coefficient in the host location documented by the source audit", "Compare dose/width to known absolute Q error and action disadvantage independently"],[DOSE.level_critic_dose,IW.per_row_iql_weights]),
        ("N9","Reject invalid reference inputs","Positive finite unit and valid scale/residual arrays are required.","candidate refresh inputs","valid flag / not-ready reference","Keep malformed arithmetic out of deployment.","Invalid helper refresh cannot become ready; upstream state-preservation is outside this fixture.",
         ["if scale/unit/residual inputs invalid: return valid=False and not-ready reference"],[REF.freeze_reference]),
    ]
    stages=[]
    for sid,title,premise,inp,out,purpose,invariant,pseudo,refs in definitions:
        shared={'W5':['concentration_ess_and_no_native_ratio_clipping'],
                'N5':['iid_same_target_low_high_residual'],
                'N6':['iid_same_target_low_high_residual','known_oracle_alignment_and_counterexample'],
                'N7':['iid_same_target_low_high_residual'],
                'W3':['hand_query_atom_crossings','query_mass_table_and_conservative_lookup']}
        selected=[c for c in CASES if c['stage']==sid or c['id'] in shared.get(sid,[])]
        stages.append(dict(id=sid,title=title,premise=premise,input=inp,output=out,purpose=purpose,invariant=invariant,
                           status='PASS' if selected and all(c['status']=='PASS' for c in selected) else 'SOURCE_ONLY' if not selected else 'FAIL',
                           pseudocode=[dict(line=i+1,text=t,source_ref=source_ref(refs[min(i,len(refs)-1)])) for i,t in enumerate(pseudo)],
                           source_refs=[source_ref(r) for r in refs],fixture_ids=[c['id'] for c in selected],
                           takeaway=selected[0]['interpretation'] if selected else 'Source-mapped; not a separate dynamic fixture.'))
    for sid,title in [('N1','Episode reservation'),('N2','Host Bellman target'),('N3','Scale-network learning')]:
        stages.append(dict(id=sid,title=title,status='UNTESTED',premise='Handled in the source audit and frozen experiment evidence.',
                           input='Not executed by this fixture runner.',output='No dynamic fixture result.',purpose='Preserve scope and avoid claiming unexecuted verification.',
                           invariant='No claim of dynamic verification from this module.',pseudocode=[],source_refs=[],fixture_ids=[],
                           takeaway='Explicitly untested here; no fresh training or host optimizer updates.'))
    negatives=[c for c in CASES if c['stage']=='E1']
    stages.append(dict(id='E1',title='Exploratory clipped-command mixed-measure contract',status='FAIL',
                       premise='The target clips Gaussian noise and clips actions; its continuous-law distribution includes singular boundary mass.',
                       input='Target spread, clipping boundaries, continuous source mixture, query action',output='Support/refusal decision',
                       purpose='Distinguish a reproducible code path from an exact target-density claim.',
                       invariant='A continuous-density ratio must account for every target measure component or explicitly refuse target-only mass.',
                       pseudocode=[dict(line=i+1,text=t,source_ref=negatives[0]['source_refs'][min(i,3)]) for i,t in enumerate([
                           'Source eps=1 is uniform continuous density 1/16 on four-dimensional action box.',
                           'Target sampler clips noise to +/-0.5 then action to +/-1, creating positive boundary mass.',
                           'Legacy spread formula uses only Gaussian interior density; -inf mass is replaced by a small floor.',
                           'Fixture reproduces both non-refusal branches; exact continuous-law contract fails.'])],
                       source_refs=negatives[0]['source_refs'],fixture_ids=[c['id'] for c in negatives],
                       takeaway='Two harness passes expose two scientific contract failures, isolated to the exploratory command-weighted density path.'))
    (OUTPUT/'stage_map.json').write_text(json.dumps(clean({'stages':stages}),indent=2,allow_nan=False),encoding='utf-8')
    refs={}
    for s in stages:
        for r in s['source_refs']:
            refs[(r['path'],r['first_line'],r['last_line'])]=r
    excerpts=['# Actual source excerpts\n\nSource files are read from the recorded repository; line numbers and SHA-256 are recorded.\n']
    for (path,first,last),r in refs.items():
        lines=(REPO/path).read_text(encoding='utf-8').splitlines()
        excerpts += [f"\n## {path}:{first}-{last}\n\nSHA-256: `{r['sha256']}`\n\n```python\n",
                     '\n'.join(f'{i:4d}: {lines[i-1]}' for i in range(first,last+1)), '\n```\n']
    (OUTPUT/'source_excerpts.md').write_text(''.join(excerpts),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default=r'C:\Users\David\dev\BCA')
    parser.add_argument('--output',help='New output directory; refuses to overwrite prior results. Default: timestamped subdirectory beside runner.')
    args=parser.parse_args()
    REPO=Path(args.repo).resolve()
    OUTPUT=Path(args.output).resolve() if args.output else HERE/('rerun_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
    OUTPUT.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(REPO))
    start=time.perf_counter()
    import jax
    import jax.numpy as jnp
    from calibration import wbcp as WB, reference as REF, dose as DOSE
    from experiments.shift_hosts import iw as IW
    from experiments.shift_iw import world as WORLD, classifier as CLASSIFIER, coverage as CV
    run_cases()
    run_exploratory_counterexamples()
    write_stage_map()
    files=sorted({r['path'] for c in CASES for r in c['source_refs']})
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,text=True,capture_output=True)
    head=git.stdout.strip() if git.returncode==0 else None
    closure={}
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path:
            path=Path(path).resolve()
            if path.is_file() and path.is_relative_to(REPO) and '.venv' not in path.parts:
                closure[path.relative_to(REPO).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    closure.update({f:hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in files})
    versions={}
    for name in ['numpy','scipy','jax','jaxlib','flax','optax','chex','gymnasium','orbax-checkpoint']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    manifest=dict(repository=str(REPO),repository_head=head,files=dict(sorted(closure.items())),
                  runtime_versions=versions,python=sys.version,source_loading='Normal imports plus exact AST slices from exploratory command_band_check.py',
                  BCA_SHIFT_DEPS=os.environ.get('BCA_SHIFT_DEPS'),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUTPUT/'source_closure.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    report=dict(schema_version=2,run_kind='isolated_native_function_fixtures',repository=str(REPO),
                repository_head=head,repository_head_status='read from git' if head else 'unavailable; source hashes are authoritative',
                executed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),elapsed_seconds=time.perf_counter()-start,
                environment=dict(python=sys.version,numpy=np.__version__,jax=jax.__version__,host=platform.node(),
                                 executable=sys.executable,devices=[str(d) for d in jax.devices()],resources=RESOURCE),
                sources={f:hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in files},
                coverage_claim='No population coverage certified. Arithmetic tests condition on fixed inputs and posterior draws; oracle alignment is an intentional analytic example.',
                cases=CASES,summary=dict(passed=sum(c['status']=='PASS' for c in CASES),failed=sum(c['status']=='FAIL' for c in CASES),total=len(CASES),
                harness_passed=sum(c['harness_status']=='PASS' for c in CASES),harness_failed=sum(c['harness_status']=='FAIL' for c in CASES),
                scientific_contract_failed=sum(c['contract_status']=='FAIL' for c in CASES)),
                untested=['Episode reservation','Bellman target construction','Scale-network fitting','Host optimizer update','Population coverage or learned-ratio correctness'])
    (OUTPUT/'fixture_results.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(clean(dict(summary=report['summary'],elapsed_seconds=report['elapsed_seconds'],output=str(OUTPUT/'fixture_results.json')))))
    raise SystemExit(1 if report['summary']['failed'] else 0)
