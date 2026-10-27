"""Frozen fresh-bank replication; pure statistics import no model or simulator."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "direct_q_diagnostic_20261021_v2"
SEED = 2026102123
COEFFICIENT = 0.519506430786272
METHODS = ("raw_q", "live_width", "matched_distance", "shuffled_width", "constant_offset")
CONTEXTS = ("stall_low", "stall_high", "monotone_low", "monotone_high")
EXPECTED_V2_SOURCE = "6c0ed9257b54e10c3c44850bb05274cdb8289b9659524e26f089d3472f8b2fc9"
EXPECTED_FIXTURES = "a6bd8c0cc5246070f7dfa8b325cdf07ac499c30dcaf1bf196d7805d4611d783a"


def require(condition, message):
    if not condition: raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_safe(value):
    import math
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["status"])
        writer.writeheader(); writer.writerows(rows)


def rng(stream):
    import numpy as np
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, stream])))


def source_hashes():
    return {name: digest(ROOT/name) for name in ("replication.py", "run_replication.py", "protocol.json")}


def authenticate_prior():
    """Read-only authentication; does not import the engine or inspect prior Q labels."""
    freeze, manifest = read_json(V2/"freeze.json"), read_json(V2/"prepared/preparation_manifest.json")
    status, fixture = read_json(V2/"results/status.json"), read_json(V2/"results/fixture_results.json")
    require(digest(V2/"diagnostic.py") == EXPECTED_V2_SOURCE == freeze["source_sha256"]["diagnostic.py"], "v2 deployment source changed")
    require(digest(V2/"prepared/preparation_manifest.json") == freeze["preparation_manifest_sha256"], "v2 prepared identity changed")
    for name, value in freeze["source_sha256"].items():
        require(digest(V2/name) == value, "v2 source changed: "+name)
    require(digest(V2/"results/fixture_results.json") == EXPECTED_FIXTURES == status["output_sha256"]["fixture_results.json"], "prior replay receipt changed")
    require(status["status"] == "complete" and fixture["status"] == "passed" and fixture["plant_transitions"] == 608,
            "prior canonical replay did not complete")
    require(status["parameters_before"] == status["parameters_after"] == manifest["parameters_before"] == manifest["parameters_after"], "prior frozen model mismatch")
    engine_path = ROOT.parent/"action_only_20261021/engine.py"
    engine_freeze = ROOT.parent/"action_only_20261021/freeze.json"
    require(digest(engine_path) == manifest["actual_engine_sha256"] and digest(engine_freeze) == manifest["original_engine_freeze_sha256"], "actual engine/source identity changed since v2 validation")
    coefficient = read_json(V2/"prepared/coefficient_receipt.json")
    require(digest(V2/"prepared/coefficient_receipt.json") == manifest["artifact_sha256"]["coefficient_receipt.json"], "fixed coefficient receipt changed")
    require(coefficient["result"]["coefficient"] == COEFFICIENT, "fixed coefficient differs")
    paths = ["diagnostic.py", "run_diagnostic.py", "protocol.json", "freeze.json", "prepared/preparation_manifest.json",
             "prepared/coefficient_receipt.json", "results/status.json", "results/fixture_results.json"]
    return dict(file_sha256={name: digest(V2/name) for name in paths}, parameters=status["parameters_after"],
                validated_fixture=fixture, coefficient=COEFFICIENT,
                actual_engine_sha256=digest(engine_path), actual_engine_freeze_sha256=digest(engine_freeze),
                additional_validation_transitions=0, prior_validation_transitions_reused=608)


def load_engine():
    prior = authenticate_prior()
    spec = importlib.util.spec_from_file_location("replication_authenticated_v2", V2/"diagnostic.py")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    engine = module.load_engine()
    require(module.parameter_hashes(engine) == prior["parameters"], "loaded frozen parameters differ from prior validation")
    return module, engine, prior


def generate_states():
    import numpy as np
    random = rng(1)
    context = np.repeat(np.arange(4, dtype=np.int64), 8)
    times = np.tile(np.resize(np.array([0, 50, 99], dtype=np.int64), 8), 4)
    states = np.empty((32, 13), np.float64)
    states[:, :10] = random.uniform(-.5, .5, (32, 10))
    states[:, 10] = np.where(context % 2 == 0, .3, 1.1)
    phase = random.uniform(-np.pi, np.pi, 32)
    states[:, 11:] = .2*np.column_stack((np.cos(phase), np.sin(phase)))
    regimes = context//2
    return dict(states=states, regimes=regimes, context=context, time=times,
                observation=np.column_stack((states, regimes, (100-times)/100)).astype(np.float32))


def read_native_outputs(engine, bank):
    """V2's native readout arithmetic, preserving invalid rows for state-level refusal."""
    import numpy as np
    import jax.numpy as jnp
    from calibration.dose import frozen_level_dose
    obs = np.repeat(bank["observation"], 12, axis=0)
    actions = bank["actions"].reshape(-1, 4)
    heads = np.asarray(engine.result.critic(obs, actions), np.float64)
    require(heads.shape == (384, 2), "invalid critic output shape")
    normalized = (obs-engine.result.mean)/engine.result.std
    eta_jax = engine.result.models[2].apply(engine.result.state.posterior.cal_params,
        jnp.asarray(normalized), jnp.asarray(actions))
    reference = engine.result.state.posterior
    native = frozen_level_dose(reference, eta_jax, .5)
    eta, width = np.asarray(eta_jax, np.float64), np.asarray(native.width, np.float64)
    formula = float(reference.threshold)*float(reference.residual_scale)*np.maximum(eta, 1e-6)
    require(np.allclose(width, formula, atol=2e-6, rtol=2e-6, equal_nan=True), "native width/formula mismatch")
    usable = np.asarray(native.support_mask, bool) & bool(reference.ready) & bool(native.inputs_valid)
    shape = (32, 12)
    return dict(q1=heads[:, 0].reshape(shape), q2=heads[:, 1].reshape(shape), eta=eta.reshape(shape),
        width=width.reshape(shape), width_formula=formula.reshape(shape), width_usable=usable.reshape(shape),
        width_refusal=(~usable | ~np.isfinite(width)).reshape(shape),
        native_inputs_valid=np.full(shape, bool(native.inputs_valid)))


def prepare_bank(module, engine):
    import numpy as np
    bank = generate_states()
    indices, _ = engine.D.behavior_parameters(bank["observation"], engine.C.Objective())
    mode = engine.atoms[indices].astype(np.float32)
    bank.update(module.build_candidates(bank["observation"], engine.direct_policy(bank["observation"]), mode, engine.atoms, rng(2)))
    bank.update(read_native_outputs(engine, bank))
    random = rng(6)
    bank["permutation"] = np.stack([random.permutation(12) for _ in range(32)]).astype(np.int64)
    bank["shuffled_width"] = np.take_along_axis(bank["width"], bank["permutation"], axis=1)
    bank["shuffled_width_usable"] = np.take_along_axis(bank["width_usable"], bank["permutation"], axis=1)
    return bank


def score_arrays(q1, width, distance, usable, permutation):
    import numpy as np
    q1, width, distance = map(lambda x: np.asarray(x, float), (q1, width, distance))
    usable, permutation = np.asarray(usable, bool), np.asarray(permutation, int)
    require(q1.ndim == 2 and q1.shape == width.shape == distance.shape == usable.shape == permutation.shape, "unaligned scoring arrays")
    require(np.all(np.sort(permutation, axis=1) == np.arange(q1.shape[1])), "each shuffled width must be a permutation of indexed candidates")
    allowed = usable & np.isfinite(width) & (width >= 0)
    shuffled, shuffled_ok = np.take_along_axis(width, permutation, axis=1), np.take_along_axis(allowed, permutation, axis=1)
    count = allowed.sum(axis=1, keepdims=True)
    offset = np.divide(np.where(allowed, width, 0).sum(axis=1, keepdims=True), count,
                       out=np.zeros_like(count, dtype=float), where=count > 0)
    result = dict(raw_q=q1.copy(), live_width=q1-np.where(allowed, width, 0),
        matched_distance=q1-COEFFICIENT*distance, shuffled_width=q1-np.where(shuffled_ok, shuffled, 0),
        constant_offset=q1-offset)
    for i in np.flatnonzero(np.isfinite(q1).all(axis=1) & np.isfinite(result["constant_offset"]).all(axis=1)):
        require(np.array_equal(np.argsort(q1[i], kind="stable"), np.argsort(result["constant_offset"][i], kind="stable")), "constant-offset arithmetic changed ranking")
    return result


def invalid_state_reasons(bank, truth=None):
    import numpy as np
    n = len(bank["q1"]); reasons = [[] for _ in range(n)]
    for key in ("states", "observation", "actions", "behavior_mode", "actor_actions", "q1", "q2", "eta", "width", "distance", "beta"):
        if key not in bank: continue
        value = np.asarray(bank[key]); bad = ~np.isfinite(value).reshape(n, -1).all(axis=1)
        for i in np.flatnonzero(bad): reasons[i].append("nonfinite_"+key)
    for i in np.flatnonzero(np.any(np.asarray(bank["width"]) < 0, axis=1)): reasons[i].append("negative_width")
    if truth is not None:
        for i in np.flatnonzero(~np.isfinite(truth).all(axis=1)): reasons[i].append("nonfinite_or_missing_truth")
    for key in ("rollout_completed", "rollout_finite"):
        if key in bank:
            for i in np.flatnonzero(~np.asarray(bank[key], bool).all(axis=1)): reasons[i].append(key+"_false")
    if "rollout_steps" in bank:
        expected = (100-np.asarray(bank["time"]))[:, None]
        for i in np.flatnonzero(np.any(bank["rollout_steps"] != expected, axis=1)): reasons[i].append("incomplete_rollout_steps")
    return reasons


def valid_state_mask(bank, truth=None):
    import numpy as np
    return np.array([not r for r in invalid_state_reasons(bank, truth)], bool)


def pearson(x, y):
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    require(x.ndim == y.ndim == 1 and x.shape == y.shape, "Pearson expects aligned vectors")
    if not (np.isfinite(x).all() and np.isfinite(y).all()) or len(x) == 0: return None
    x, y = x-x.mean(), y-y.mean()
    denominator = np.linalg.norm(x)*np.linalg.norm(y)
    if denominator == 0 or not np.isfinite(denominator): return None
    return float(np.clip(np.dot(x, y)/denominator, -1, 1))


def pairwise_pearson(x, y):
    """Normalized dot product of i<j differences, NOT re-centered differences."""
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    require(x.ndim == y.ndim == 1 and x.shape == y.shape, "pairwise correlation expects aligned vectors")
    if not (np.isfinite(x).all() and np.isfinite(y).all()): return None
    i, j = np.triu_indices(len(x), 1); dx, dy = x[i]-x[j], y[i]-y[j]
    denominator = np.linalg.norm(dx)*np.linalg.norm(dy)
    if denominator == 0 or not np.isfinite(denominator): return None
    return float(np.clip(np.dot(dx, dy)/denominator, -1, 1))


def concordance(scores, truth):
    import numpy as np
    scores, truth = np.asarray(scores, float), np.asarray(truth, float)
    require(scores.shape == truth.shape and scores.ndim == 1, "unaligned ranking arrays")
    if not (np.isfinite(scores).all() and np.isfinite(truth).all()):
        return dict(value=None, pairs=0, true_ties=0, predicted_ties_among_comparable=0)
    i, j = np.triu_indices(len(truth), 1)
    td, pd = truth[i]-truth[j], scores[i]-scores[j]
    valid, tied = np.abs(td) > 1e-6, np.abs(pd) <= 1e-6
    credit = np.where(tied, .5, np.sign(td) == np.sign(pd))
    return dict(value=float(credit[valid].mean()) if valid.any() else None,
        pairs=int(valid.sum()), true_ties=int((~valid).sum()), predicted_ties_among_comparable=int(np.sum(tied & valid)))


def state_metrics(bank, truth, scores):
    import numpy as np
    truth = np.asarray(truth, float); n, m = truth.shape
    require(bank["q1"].shape == truth.shape and m == 12 and all(np.asarray(scores[k]).shape == truth.shape for k in METHODS), "state metric shape mismatch")
    reasons = invalid_state_reasons(bank, truth)
    rows = []
    for i in range(n):
        for method in METHODS:
            if not np.isfinite(scores[method][i]).all(): reasons[i].append("nonfinite_score_"+method)
        row = dict(state_id=i, valid=not reasons[i], invalid_reasons=reasons[i],
            rho_width=None, rho_distance=None, rho_shuffled=None,
            rho_undefined_reasons={key: ["state_invalid"] if reasons[i] else [] for key in ("width", "distance", "shuffled")}, methods={})
        if reasons[i]: rows.append(row); continue
        error = bank["q1"][i]-truth[i]
        shuffled = np.asarray(bank.get("shuffled_width", np.take_along_axis(bank["width"], bank["permutation"], axis=1)))[i]
        for name, value in (("width", bank["width"][i]), ("distance", bank["distance"][i]), ("shuffled", shuffled)):
            row["rho_"+name] = pearson(value, error)
            if np.linalg.norm(value-value.mean()) == 0: row["rho_undefined_reasons"][name].append("zero_predictor_centered_norm")
            if np.linalg.norm(error-error.mean()) == 0: row["rho_undefined_reasons"][name].append("zero_signed_error_centered_norm")
            if row["rho_"+name] is None and not row["rho_undefined_reasons"][name]: row["rho_undefined_reasons"][name].append("nonfinite_correlation_arithmetic")
            pair = pairwise_pearson(value, error)
            row["pairwise_rho_"+name] = pair
            require((pair is None) == (row["rho_"+name] is None), "correlation equivalence undefined mismatch")
            if pair is not None: require(abs(pair-row["rho_"+name]) <= 1e-12, "pairwise/centered correlation mismatch")
        base = int(np.argmax(scores["raw_q"][i])); actor = truth[i, 6]
        raw_actor = bool(np.array_equal(bank["actions"][i, base], bank["actions"][i, 6]))
        row.update(actor_true_q=float(actor), raw_chosen_index=base)
        for method in METHODS:
            choice = int(np.argmax(scores[method][i])); con = concordance(scores[method][i], truth[i])
            is_actor = bool(np.array_equal(bank["actions"][i, choice], bank["actions"][i, 6]))
            row["methods"][method] = dict(concordance=con["value"], comparable_pairs=con["pairs"],
                concordance_undefined_reason="no_comparable_true_pairs" if con["value"] is None else None,
                true_near_ties=con["true_ties"], predicted_near_ties=con["predicted_ties_among_comparable"],
                chosen_index=choice, chosen_true_q=float(truth[i, choice]), delta_vs_raw=float(truth[i, choice]-truth[i, base]),
                advantage_vs_actor=float(truth[i, choice]-actor), changed_index=choice != base,
                changed_command=not np.array_equal(bank["actions"][i, choice], bank["actions"][i, base]),
                selected_actor_command=is_actor, actor_restoration=not raw_actor and is_actor)
        rows.append(row)
    return rows


def bootstrap_plan(context, times):
    import numpy as np
    context, times = np.asarray(context), np.asarray(times)
    require(context.shape == times.shape == (32,), "bootstrap preserves original32-state design")
    strata = context*100+times; random = rng(7)
    draws = np.empty((10000, 32), np.int64)
    for label in np.unique(strata):
        ids = np.flatnonzero(strata == label)
        draws[:, ids] = random.choice(ids, size=(10000, len(ids)), replace=True)
    return draws


def bootstrap_statistic(values, eligible, draws):
    import numpy as np
    values, eligible, draws = np.asarray(values, float), np.asarray(eligible, bool), np.asarray(draws, int)
    require(values.ndim == 1 and values.shape == eligible.shape and draws.ndim == 2, "unaligned bootstrap arrays")
    eligible = eligible & np.isfinite(values)
    available = eligible[draws]; count = available.sum(axis=1)
    sums = np.where(available, values[draws], 0).sum(axis=1)
    means = np.divide(sums, count, out=np.full(len(draws), np.nan), where=count > 0)
    undefined = int(np.sum(~np.isfinite(means)))
    return dict(mean=float(values[eligible].mean()) if eligible.any() else None, defined_states=int(eligible.sum()),
        planned_states=len(values), undefined_bootstrap_draws=undefined,
        bootstrap_defined_denominator_min=int(count.min()), bootstrap_defined_denominator_max=int(count.max()),
        ci95=np.quantile(means, [.025, .975], method="linear").tolist() if undefined == 0 else None,
        interval_status="available" if undefined == 0 else "unavailable_undefined_draws"), means, count


def analyze(bank, truth, scores, draws):
    import numpy as np
    rows = state_metrics(bank, truth, scores); n = len(rows)
    valid = np.array([r["valid"] for r in rows]); nan = lambda x: np.nan if x is None else x
    rhos = {key: np.array([nan(r["rho_"+key]) for r in rows]) for key in ("width", "distance", "shuffled")}
    rho_masks = {key: valid & np.isfinite(value) for key, value in rhos.items()}
    rho_pair_masks = {key: rho_masks["width"] & rho_masks[key] for key in ("distance", "shuffled")}
    con = {key: np.array([nan(r["methods"].get(key, {}).get("concordance")) for r in rows]) for key in METHODS}
    common_con = valid & np.isfinite(np.column_stack(list(con.values()))).all(axis=1)
    metrics, boot = {}, {}
    def add(name, values, eligible):
        record, means, denominators = bootstrap_statistic(values, eligible, draws)
        metrics[name] = record; boot[name+"__mean"] = means; boot[name+"__denominator"] = denominators
    for name, values in rhos.items(): add("primary_rho_"+name, values, rho_masks[name])
    for name in ("distance", "shuffled"):
        add("primary_rho_width_minus_"+name, rhos["width"]-rhos[name], rho_pair_masks[name])
        add("paired_rho_width_for_"+name, rhos["width"], rho_pair_masks[name])
        add("paired_rho_"+name+"_for_width", rhos[name], rho_pair_masks[name])
    for method in METHODS:
        add("concordance_"+method, con[method], common_con)
        add("primary_concordance_change_"+method, con[method]-con["raw_q"], common_con)
        for field in ("chosen_true_q", "delta_vs_raw", "advantage_vs_actor"):
            values = np.array([nan(r["methods"].get(method, {}).get(field)) for r in rows])
            add("secondary_"+field+"_"+method, values, valid)
    add("secondary_original_actor_true_q", np.array([nan(r.get("actor_true_q")) for r in rows]), valid)
    for control in ("matched_distance", "shuffled_width"):
        values = np.array([nan(r["methods"].get("live_width", {}).get("chosen_true_q"))-nan(r["methods"].get(control, {}).get("chosen_true_q")) for r in rows])
        add("secondary_selected_q_live_width_minus_"+control, values, valid)
    summary = dict(descriptive_only=True, training_replications=1, planned_states=n, valid_states=int(valid.sum()),
        invalid_states=int((~valid).sum()), pair_common_defined_rho_states={key: int(mask.sum()) for key, mask in rho_pair_masks.items()},
        individually_defined_rho_states={key: int(np.sum(valid & np.isfinite(value))) for key, value in rhos.items()},
        common_defined_concordance_states=int(common_con.sum()), metrics=metrics,
        counts={method: {field: sum(bool(r["methods"].get(method, {}).get(field, False)) for r in rows)
                        for field in ("changed_index", "changed_command", "selected_actor_command", "actor_restoration")} for method in METHODS},
        invalid_reasons={str(r["state_id"]): r["invalid_reasons"] for r in rows if not r["valid"]},
        interpretation="All intervals use one common original32-state context-time bootstrap plan; endpoint contributions use stated common masks; any undefined draw makes its interval unavailable. No p-values, winner rule or training-seed inference.")
    return summary, rows, boot
