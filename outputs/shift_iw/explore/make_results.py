"""EXPLORATORY results table for the BCA arms run by explore_arms.py and scored by explore_eval.py.

EXPLORATORY: outside the pre-registered claims; the version-5 gates did not pass.

Reads only: the explore evaluation and training records under runs/shift_iw_v5_explore (each must be finished, match its
sidecar and be labelled exploratory), and, from runs/shift_iw_v5, the reference controllers' evaluation records
(hosts.json: plain TD3+BC; room.json: the room check's no-caution actor; room_lower_edge.json: the lower edge with no
penalty and with the perfect penalty) and each quality's report (marks R1, R2, L1, L2: the pre-registered predictions).
Writes RESULTS.json next to this file (pass --json to change it). Nothing under runs/ is written.

Every difference is paired by seed. For each it gives the per-seed values, the mean, the one-sided 95% lower bound and
the two-sided 95% interval (t with n - 1 = 4 degrees of freedom: coverage.across_seeds, as the report uses), and the
one-sided p-value that the difference is above 0 (run.p_one_sided), uncorrected and for orientation only.

Run from /root/bca-shift with PYTHONPATH=/root/bca-shift:
  python /mnt/c/Users/David/dev/bca-shift-work/iw_explore/make_results.py
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import experiments.shift  # noqa: F401  (gymnasium path and version checks)

from experiments.shift_iw import run as R

import explore_arms as XA
import explore_eval as XE

LAWS = ("none", "mild", "strong")
HERE = Path(__file__).resolve().parent

ARMS = {  # name in the evaluation records -> (where it is, plain-language name)
    "bca_unweighted": ("explore:bca", "capped pull, unweighted"),
    "bca_exact_strong": ("explore:bca", "capped pull, weighted per step (strong mix)"),
    "bca_exact_strong_single": ("explore:bca", "capped pull, weighted single threshold (strong mix)"),
    "le_bca_unweighted": ("explore:bca_lower_edge", "lower edge, unweighted"),
    "le_bca_exact_strong": ("explore:bca_lower_edge", "lower edge, weighted per step (strong mix)"),
    "le_bca_exact_mild": ("explore:bca_lower_edge", "lower edge, weighted per step (mild mix; shown only)"),
    "plain_gap": ("v5:hosts", "plain TD3+BC on the gapped data"),
    "none": ("v5:room", "room check's actor with no extra caution"),
    "le_none": ("v5:room_lower_edge", "lower edge with no penalty"),
    "le_oracle": ("v5:room_lower_edge", "lower edge with the perfect penalty"),
}

CAPPED = ("bca_unweighted", "bca_exact_strong", "bca_exact_strong_single")
LOWER = ("le_bca_unweighted", "le_bca_exact_strong", "le_bca_exact_mild")

COMPARISONS = (  # (key, use of the band, a, b): a minus b
    [("capped:per_step_minus_unweighted", "capped", "bca_exact_strong", "bca_unweighted"),
     ("capped:single_minus_unweighted", "capped", "bca_exact_strong_single", "bca_unweighted"),
     ("capped:per_step_minus_single", "capped", "bca_exact_strong", "bca_exact_strong_single")]
    + [(f"capped:{a}_minus_{b}", "capped", a, b) for a in CAPPED for b in ("plain_gap", "none")]
    + [("lower:per_step_minus_unweighted", "lower", "le_bca_exact_strong", "le_bca_unweighted"),
       ("lower:mild_minus_unweighted", "lower", "le_bca_exact_mild", "le_bca_unweighted")]
    + [(f"lower:{a}_minus_{b}", "lower", a, b) for a in LOWER for b in ("plain_gap", "none", "le_none")]
    + [("lower:le_bca_unweighted_minus_le_oracle", "lower", "le_bca_unweighted", "le_oracle")])


def summary(values):
    """per-seed values, mean, sd, two-sided 95% interval, one-sided 95% lower bound, one-sided p (uncorrected)."""
    s = R._summary(values)
    if s is None:
        return None
    s = {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()}
    s["p_one_sided_uncorrected"] = R.p_one_sided(values)
    return s


def read_quality(quality, spec, used):
    ctx = R.Context(root=XA.V5_ROOT, quality=quality)
    vals = {s: {} for s in R.SEEDS}
    for s in R.SEEDS:
        for group in XE.GROUPS:
            path = XE.eval_path(XA.EXPLORE_ROOT, quality, s, group)
            rec = XE.explore_record(path, spec)
            if rec is None:
                raise R.Refused(f"refused: {path} is missing or unfinished")
            if rec.get("smoke") or rec.get("missing_arms"):
                raise R.Refused(f"refused: {path} is a smoke run or lists missing arms {rec.get('missing_arms')}")
            used[str(path)] = R.LI.sha256_file(path)
            vals[s].update(rec["evaluations"])
        for group in ("hosts", "room", "room_lower_edge"):
            path = R.evaluate_path(ctx, s, group)
            rec = R.finished(path, spec)
            if rec is None:
                raise R.Refused(f"refused: {path} is missing or unfinished")
            used[str(path)] = R.LI.sha256_file(path)
            for name, (where, _) in ARMS.items():
                if where == f"v5:{group}":
                    vals[s][name] = rec["evaluations"][name]

    def per_seed(name, law, what):
        v = [R.num(R._get(vals[s], name, law, what)) for s in R.SEEDS]
        if any(x is None for x in v):
            raise R.Refused(f"refused: {quality} {name} {law} {what} is missing on a seed")
        return v

    arms = {n: {law: {w: summary(per_seed(n, law, w)) for w in ("J", "points")} for law in LAWS} for n in ARMS}
    comps = {}
    for key, use, a, b in COMPARISONS:
        comps[key] = dict(use=use, a=a, b=b, a_name=ARMS[a][1], b_name=ARMS[b][1], laws={
            law: {w: summary([x - y for x, y in zip(per_seed(a, law, w), per_seed(b, law, w))])
                  for w in ("J", "points")} for law in LAWS})

    # lower-edge arms' own checks (recorded, do not stop the arms; as in the official stage)
    checks = {}
    for arm in XA.LOWER_EDGE_ARMS:
        recs = [R.finished(XA.lower_edge_dir(XA.EXPLORE_ROOT, quality, arm, s) / "record.json", spec) for s in R.SEEDS]
        checks[f"le_bca_{arm}"] = dict(
            band_penalty_shape_check_passed=[R._get(r, "result", "arm", "check_e", "passed") for r in recs],
            failed_parts=[R._get(r, "result", "arm", "check_e", "failed") for r in recs],
            refused_rows=[R._get(r, "result", "arm", "refused_rows") for r in recs])
    for arm in XA.CAPPED_ARMS:
        recs = [R.finished(XA.capped_dir(XA.EXPLORE_ROOT, quality, arm, s) / "record.json", spec) for s in R.SEEDS]
        checks[f"bca_{arm}"] = dict(failed=[R._get(r, "result", "failed") for r in recs],
                                    final_update=[R._get(r, "result", "final_update") for r in recs],
                                    uncertified_refreshes=[R._get(r, "result", "uncertified_refreshes") for r in recs])

    # the pre-registered predictions and room checks, from the frozen report
    rep = R.finished(R.report_path(ctx), spec)
    if rep is None:
        raise R.Refused(f"refused: {R.report_path(ctx)} is missing or unfinished")
    used[str(R.report_path(ctx))] = R.LI.sha256_file(R.report_path(ctx))
    m = rep["marks"]
    g = R._get
    mild_change = g(m, "L2", "mild_scale_change", "mean")
    slope_k = g(m, "L2", "return_per_unit_k", "mean")
    pre = dict(
        bca_allowed=rep.get("bca_allowed"), bca_lower_edge_allowed=rep.get("bca_lower_edge_allowed"),
        capped=dict(
            room_check=dict(status=g(m, "R1", "status"), placed_minus_none=g(m, "R1", "placed_minus_none"),
                            placed_minus_none_points=g(m, "R1", "placed_minus_none_points")),
            prediction=dict(status=g(m, "R2", "status"), predicted_return=g(m, "R2", "predicted_return"),
                            predicted_points=g(m, "R2", "predicted_points"),
                            dose_change=g(m, "R2", "dose_change", "mean"),
                            return_per_unit_dose=g(m, "R2", "return_per_unit_dose", "mean"),
                            single_threshold_predicted_return=g(m, "R2", "single_threshold", "predicted_return"))),
        lower=dict(
            room_check=dict(status=g(m, "L1", "status"), oracle_minus_none=g(m, "L1", "oracle_minus_none"),
                            oracle_minus_none_points=g(m, "L1", "oracle_minus_none_points")),
            prediction=dict(status=g(m, "L2", "status"), predicted_return=g(m, "L2", "predicted_return"),
                            predicted_points=g(m, "L2", "predicted_points"),
                            scale_change=g(m, "L2", "scale_change", "mean"),
                            return_per_unit_k=slope_k,
                            band_over_perfect=g(m, "L2", "proxy_check", "band_over_oracle"),
                            band_penalty_shape_check_passed=g(m, "L2", "proxy_check", "band_check_e_passed"),
                            mild_derived_not_preregistered=None if mild_change is None or slope_k is None else
                            float(mild_change) * float(slope_k))))

    def versus(key, predicted):
        obs = comps[key]["laws"]["strong"]["J"]
        lo, hi = obs["interval"]
        return dict(predicted=predicted, observed_mean=obs["mean"], observed_interval=[lo, hi],
                    prediction_inside_interval=None if predicted is None else bool(lo <= predicted <= hi),
                    observed_minus_predicted=None if predicted is None else obs["mean"] - predicted)

    match = dict(
        capped_per_step=versus("capped:per_step_minus_unweighted", pre["capped"]["prediction"]["predicted_return"]),
        capped_single=versus("capped:single_minus_unweighted",
                             pre["capped"]["prediction"]["single_threshold_predicted_return"]),
        lower_per_step=versus("lower:per_step_minus_unweighted", pre["lower"]["prediction"]["predicted_return"]),
        lower_mild_derived=versus("lower:mild_minus_unweighted",
                                  pre["lower"]["prediction"]["mild_derived_not_preregistered"]))
    return dict(quality=quality, arms=arms, comparisons=comps, pre_registered=pre, prediction_vs_observed=match,
                checks=checks)


def main(argv=None):
    p = argparse.ArgumentParser(description=f"{XA.LABEL}. Results table of the exploratory BCA arms.")
    p.add_argument("--json", default=str(HERE / "RESULTS.json"))
    a = p.parse_args(argv)
    try:
        spec = R.spec_digest(R.Context(root=XA.V5_ROOT))
        used = {}
        qs = [read_quality(q, spec, used) for q in R.QUALITIES]
        out = dict(kind="shift-iw-explore-results", label=XA.LABEL, exploratory=True, smoke=False,
                   spec_sha256=spec, seeds=list(R.SEEDS), made_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   how=("paired by seed; mean, one-sided 95% lower bound and two-sided 95% interval with t on 4 "
                        "degrees of freedom; p one-sided, uncorrected, for orientation only; J = exact 100-step "
                        "return (return units), points = 0 random commands to 100 the stall-aware controller"),
                   arm_names={k: v[1] for k, v in ARMS.items()},
                   inputs_sha256=used, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   qualities=qs)
        Path(a.json).write_text(json.dumps(R.plain(out), indent=1, allow_nan=False) + "\n", encoding="utf8")
        print(f"wrote {a.json}")
    except R.Refused as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
