"""Frozen fitting-factor selectors and paired development contrasts.

The authenticated original metrics module supplies unchanged model queries and
analysis. No factor is fitted here, and score_candidates has no label argument.
Scientific dependencies are imported only when these functions are invoked.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import types


LEGACY_METRICS_SHA256 = "be997b8f6780b45557097f17d732697c7ddfb2964569bba29eb61909be6430b7"
FIXED_FACTORS = {"bca": 31.372587203979492,
                 "disagreement": 4.926129267651156,
                 "actor_distance": 19.29619572569095}
FAMILIES = ("bca", "contrast_bca", "contrast_disagreement", "contrast_actor_distance")
KAPPAS = (0, .03, .1, .3, 1, 3, 10)
ZERO_ALIASES = tuple(family + "_kappa_0" for family in FAMILIES)


def _load_source(path, name, expected_sha256=None):
    """Execute precisely the checked source buffer, only at runtime."""
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Pinned metrics source identity changed")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


@lru_cache(maxsize=1)
def _legacy():
    path = Path(__file__).resolve().parent.parent / "unifloral_frozen_usefulness_v1_seedfix" / "metrics.py"
    return _load_source(path, "_frozen_scale_original_metrics", LEGACY_METRICS_SHA256)


@lru_cache(maxsize=1)
def _contract():
    # The supervisor authenticates this local file in its prepared-source receipt.
    return _load_source(Path(__file__).resolve().with_name("score_contract.py"), "_frozen_scale_score_contract")


def _protocol(protocol):
    checked = _legacy()._protocol(protocol)
    if checked.get("contrast_factors") != FIXED_FACTORS:
        raise ValueError("Only the preregistered fitting-derived contrast factors are allowed")
    return checked


def _expected_methods():
    return {"raw_q", "actor", "twin_min"} | {
        family + "_kappa_" + format(kappa, "g") for family in FAMILIES for kappa in KAPPAS}


def _method_checks(methods):
    if len(methods) != 31 or set(methods) != _expected_methods():
        raise ValueError("Expected exactly the 31 preregistered selectors")
    if not all(entry.get("available") is True for entry in methods.values()):
        raise ValueError("Invalid or unavailable signals cannot receive a fallback selector")
    raw = methods["raw_q"]
    for name in ZERO_ALIASES:
        entry = methods[name]
        if entry["scores"] != raw["scores"] or entry["selected_indices"] != raw["selected_indices"]:
            raise ValueError("A zero-penalty selector failed exact raw-Q equivalence")


def score_candidates(policy, observations_list, banks, scale_bank, protocol):
    """Query authentic frozen models, then seal all 31 selectors before labels."""
    legacy = _legacy()
    protocol = _protocol(protocol)
    scored = legacy.score_candidates(policy, observations_list, banks, scale_bank, protocol)
    original_seal = {key: value for key, value in scored.items() if key != "prelabel_signature"}
    if legacy._digest(original_seal) != scored["prelabel_signature"]:
        raise ValueError("Original model-query receipt did not seal correctly")
    queries_sha256 = legacy._digest(scored["queries"])
    methods = _contract().make_methods(scored["queries"], protocol)
    _method_checks(methods)
    if legacy._digest(scored["queries"]) != queries_sha256:
        raise ValueError("Selector construction altered the authentic model-query table")

    original_scale = scored["scale_bank"]
    scored["scale_bank"] = {
        "rows": original_scale["rows"], "role": original_scale["role"],
        "bank_sha256": original_scale["bank_sha256"],
        "membership_verification": original_scale["membership_verification"],
        "legacy_unused_rms_descriptives": original_scale,
        "legacy_rms_matching_applied": False,
        "fixed_contrast_factors": dict(FIXED_FACTORS),
        "contrast_factors_refitted": False,
        "factor_source": "Saved fitting-candidate centered contrasts; fixed in the protocol before development labels",
        "note": "The original scorer recomputes RMS descriptives, but none is consumed by these selectors."}
    scored["methods"] = methods
    scored["comparison_design"] = {
        "schema": "unifloral_frozen_scale_comparison.v1", "selector_count": 31,
        "families": list(FAMILIES), "fixed_contrast_factors": dict(FIXED_FACTORS),
        "factor_arithmetic": "Fixed scalar factors applied by the standard-library score contract to model-output values",
        "raw_uncertainty_changed": False, "development_factor_fit": False,
        "zero_penalty_aliases_of_raw_q": list(ZERO_ALIASES),
        "zero_penalty_aliases_are_independent_evidence": False,
        "primary_endpoint": "Paired first-action continuation-value gain over raw Q1 selection",
        "secondary_endpoints": "Candidate-bank regret and paired comparisons with fixed contrast controls",
        "all_kappas_reported": True, "winner_selected": False,
        "legacy_metrics_sha256": LEGACY_METRICS_SHA256,
        "authentic_model_query_sha256": queries_sha256}
    scored.pop("prelabel_signature", None)
    scored = legacy._plain(scored)
    scored["prelabel_signature"] = legacy._digest(scored)
    return scored


def analyze(prelabeldata, truevalues, protocol):
    """Preserve original analysis and append paired, common-resample contrasts."""
    import numpy as np

    legacy = _legacy()
    protocol = _protocol(protocol)
    _method_checks(prelabeldata["methods"])
    if prelabeldata.get("comparison_design", {}).get("fixed_contrast_factors") != FIXED_FACTORS:
        raise ValueError("Missing frozen-factor comparison receipt")
    # This unchanged function verifies the complete prelabel signature/protocol,
    # validates deterministic values, and analyzes every frozen selector.
    result = legacy.analyze(prelabeldata, truevalues, protocol)
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    indices = np.empty((protocol["bootstrap_draws"], 64), dtype=np.int64)
    for stratum, _ in enumerate(protocol["query_times"]):
        group_indices = np.arange(stratum, 64, 4, dtype=np.int64)
        draws = rng.integers(0, len(group_indices), size=(protocol["bootstrap_draws"], len(group_indices)))
        indices[:, group_indices] = group_indices[draws]
    indices_sha256 = legacy._digest(indices)
    if indices_sha256 != result["bootstrap"]["index_sha256"]:
        raise ValueError("Paired comparisons did not reproduce the original stratified bootstrap draws")

    def interval(per_episode):
        values = legacy._finite(per_episode, "paired per-episode difference", (64,))
        replicates = values[indices].mean(axis=1)
        lo, hi = np.quantile(replicates, [.025, .975], method="linear")
        return {"mean": float(values.mean()), "pointwise_ci95": [float(lo), float(hi)],
                "pointwise_interval_contains_zero": bool(lo <= 0.0 <= hi)}

    reports = result["methods"]
    raw_values = np.asarray(reports["raw_q"]["selected_values"], dtype=np.float64)
    reproduced = []
    # Check all 31 raw-Q gain intervals, not only the vacuous raw-Q-minus-itself.
    # Equal arithmetic/order and shared resamples should give exact JSON values.
    for name, entry in reports.items():
        values = np.asarray(entry["selected_values"], dtype=np.float64)
        if interval(values - raw_values) != entry["gain_vs_raw_q"]:
            raise ValueError("Raw-Q gain interval reproduction failed for " + name)
        source_entry = prelabeldata["methods"][name]
        for field in ("penalty_factor", "factor_kind", "rms_match_factor", "actual_u_scale_zero"):
            entry[field] = source_entry[field]
        reproduced.append(name)

    paired = {}
    for kappa in KAPPAS:
        tag = format(kappa, "g")
        lhs_name = "contrast_bca_kappa_" + tag
        lhs = reports[lhs_name]
        lhs_values = np.asarray(lhs["selected_values"], dtype=np.float64)
        for comparator in ("contrast_disagreement", "contrast_actor_distance", "bca"):
            rhs_name = comparator + "_kappa_" + tag
            rhs = reports[rhs_name]
            rhs_values = np.asarray(rhs["selected_values"], dtype=np.float64)
            difference = lhs_values - rhs_values
            tolerance = protocol["tie_atol"] + protocol["tie_rtol"] * np.abs(rhs_values)
            uncertain = np.abs(difference) <= tolerance
            paired[lhs_name + "_minus_" + rhs_name] = {
                "lhs_method": lhs_name, "rhs_method": rhs_name, "kappa_penalty": kappa,
                "orientation": "Positive means higher deterministic first-action continuation value for contrast BCA",
                "gain": interval(difference), "per_episode_gain": difference.tolist(),
                "numerically_inconclusive_flags": uncertain.tolist(),
                "improved_count": int(np.sum(difference > tolerance)),
                "harmed_count": int(np.sum(difference < -tolerance)),
                "same_selected_action_count": sum(a == b for a, b in zip(lhs["selected_indices"], rhs["selected_indices"])),
                "lhs_score_near_tie_count": lhs["score_near_tie_count"],
                "rhs_score_near_tie_count": rhs["score_near_tie_count"],
                "zero_penalty_duplicate": kappa == 0,
                "multiple_comparison_corrected": False, "usefulness_claim_allowed": False}

    result["comparison_design"] = prelabeldata["comparison_design"]
    result["paired_fixed_contrast_comparisons"] = paired
    result["paired_bootstrap_consistency"] = {
        "passed": True, "index_sha256": indices_sha256,
        "reproduced_raw_q_gain_intervals": reproduced, "reproduced_count": len(reproduced),
        "comparison": "Exact equality of mean, percentile bounds and zero-inclusion flag",
        "common_resamples_for_all_pairs": True}
    result["legacy_analysis_notes"] = {
        "original_analysis_source_unchanged": True,
        "rms_matching_limitation_applies_to_current_methods": False,
        "explanation": "Any inherited RMS-matching limitation describes the unused legacy method; current factors are frozen fitting-candidate contrast ratios."}
    result["limitations"].extend([
        "All 31 selectors and all 21 paired contrasts are reported; pointwise intervals are not multiplicity-corrected.",
        "The four zero-penalty family entries duplicate raw Q1 selection and are not independent evidence.",
        "Contrast matching equalizes a saved fitting-candidate magnitude, not gradient strength, information quality or development-state magnitude.",
        "Fixed-factor contrast BCA is algebraically a prespecified effective-kappa rescaling at this checkpoint, not evidence that normalization is necessary.",
        "No factor, kappa or winner is selected from these development labels; confirmation remains untouched."])
    result["best_kappa_selected"] = False
    return legacy._plain(result)
