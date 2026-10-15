"""EXPLORATORY summary of the BCA arms run by explore_arms.py and scored by explore_eval.py, across the five qualities.

EXPLORATORY: outside the pre-registered claims; the version-5 gates did not pass.

Reads (read only): the explore evaluation and training records under --out, and, for reference controllers, the
pre-registered run's evaluation records (runs/shift_iw_v5/<quality>/evaluate/seed<k>/hosts.json, room.json,
room_lower_edge.json). Writes <out>/summary/summary.json (+ .sha256) and summary.txt (plain language), and prints the
text. Differences are paired by seed; their one-sided p-values (run.p_one_sided, no correction for the many
comparisons) are for orientation only: no claim is judged here.
"""

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np

import experiments.shift  # noqa: F401  (gymnasium path and version checks)

from experiments.shift_iw import run as R

import explore_arms as XA
import explore_eval as XE

LAWS = ("none", "mild", "strong")
CAPPED_NAMES = {"bca_unweighted": "unweighted", "bca_exact_strong": "weighted, per step (strong mix)",
                "bca_exact_strong_single": "weighted, single threshold (strong mix)"}
LE_NAMES = {"le_bca_unweighted": "unweighted", "le_bca_exact_strong": "weighted, per step (strong mix)",
            "le_bca_exact_mild": "weighted, per step (mild mix; shown only)"}
REFERENCES = {"plain_gap": ("hosts", "plain TD3+BC on the gapped data (reference)"),
              "none": ("room", "cloned actor, no extra caution (room check; reference)"),
              "le_none": ("room_lower_edge", "lower edge with no penalty (reference)"),
              "le_oracle": ("room_lower_edge", "lower edge with the perfect penalty (reference)")}
COMPARISONS = [
    ("capped pull: weighted per step (strong) minus unweighted", "bca", "bca_exact_strong", "bca_unweighted"),
    ("capped pull: weighted single threshold (strong) minus unweighted", "bca", "bca_exact_strong_single",
     "bca_unweighted"),
    ("capped pull: unweighted BCA minus plain TD3+BC (gapped)", "bca", "bca_unweighted", "plain_gap"),
    ("lower edge: weighted per step (strong) minus unweighted", "bca_lower_edge", "le_bca_exact_strong",
     "le_bca_unweighted"),
    ("lower edge: weighted per step (mild) minus unweighted", "bca_lower_edge", "le_bca_exact_mild",
     "le_bca_unweighted"),
    ("lower edge: unweighted BCA minus no penalty", "bca_lower_edge", "le_bca_unweighted", "le_none"),
    ("lower edge: unweighted BCA minus the perfect penalty", "bca_lower_edge", "le_bca_unweighted", "le_oracle"),
]


def stat(values):
    """Mean, standard error over seeds and n of the finite values; None when there are none."""
    v = np.asarray([x for x in values if x is not None], np.float64)
    if not len(v):
        return None
    if not np.all(np.isfinite(v)):
        return dict(n=int(len(v)), mean=None, se=None, nonfinite=int((~np.isfinite(v)).sum()), per_seed=v.tolist())
    se = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None
    return dict(n=int(len(v)), mean=float(v.mean()), se=se, per_seed=v.tolist())


def explore_eval(out, quality, seed, group, spec):
    rec = R.finished(XE.eval_path(out, quality, seed, group), spec)
    if rec is not None and rec.get("label") != XA.LABEL:
        raise R.Refused(f"refused: {XE.eval_path(out, quality, seed, group)} is not labelled exploratory")
    return rec


def quality_summary(out, quality, spec):
    ctx = R.Context(root=XA.V5_ROOT, quality=quality)
    per_seed = {}          # seed -> {name: {law: {J, points}}}
    missing, smoke = [], False
    for s in R.SEEDS:
        vals = {}
        for group in XE.GROUPS:
            rec = explore_eval(out, quality, s, group, spec)
            if rec is None:
                missing.append(f"seed {s} {group}: not evaluated")
                continue
            smoke = smoke or bool(rec.get("smoke"))
            missing += [f"seed {s} {group}: {m}" for m in rec.get("missing_arms") or []]
            vals.update(rec["evaluations"])
        for name, (group, _) in REFERENCES.items():   # only on seeds the explore arms were scored on (paired)
            rec = R.finished(R.evaluate_path(ctx, s, group), spec) if vals else None
            if rec is not None and name in rec["evaluations"]:
                vals[name] = rec["evaluations"][name]
        per_seed[s] = vals
    names = list(CAPPED_NAMES) + list(LE_NAMES) + list(REFERENCES)

    def val(s, n, law, what):          # records store a non-finite number as a string ('nan', 'inf')
        return R.num(R._get(per_seed[s], n, law, what))

    arms = {n: {law: {what: stat([val(s, n, law, what) for s in R.SEEDS]) for what in ("J", "points")}
                for law in LAWS} for n in names}
    comps = []
    for label, group, a, b in COMPARISONS:
        laws = {}
        for law in LAWS:
            entry = {}
            for what in ("J", "points"):
                d = [val(s, a, law, what) - val(s, b, law, what) for s in R.SEEDS
                     if val(s, a, law, what) is not None and val(s, b, law, what) is not None]
                entry[what] = stat(d)
                if what == "J":
                    entry["p_one_sided_uncorrected"] = R.p_one_sided(d) if len(d) > 1 else None
            laws[law] = entry
        comps.append(dict(comparison=label, group=group, a=a, b=b, laws=laws))
    diag = {"capped": {}, "lower_edge": {}}
    for arm in XA.CAPPED_ARMS:
        recs = [R.finished(XA.capped_dir(out, quality, arm, s) / "record.json", spec) for s in R.SEEDS]
        diag["capped"][arm] = dict(
            seeds_finished=sum(r is not None for r in recs),
            failed=[s for s, r in zip(R.SEEDS, recs) if r is not None and r["result"]["failed"]],
            uncertified_refreshes=[R._get(r, "result", "uncertified_refreshes") for r in recs],
            uncertified_weighted_refreshes=[R._get(r, "result", "uncertified_weighted_refreshes") for r in recs],
            refused_rows_max=[R._get(r, "result", "refused_rows_max") for r in recs],
            minutes=[None if r is None else round(r["seconds"] / 60, 1) for r in recs])
    for arm in XA.LOWER_EDGE_ARMS:
        recs = [R.finished(XA.lower_edge_dir(out, quality, arm, s) / "record.json", spec) for s in R.SEEDS]
        diag["lower_edge"][arm] = dict(
            seeds_finished=sum(r is not None for r in recs),
            refused_rows=[R._get(r, "result", "arm", "refused_rows") for r in recs],
            R_mean=[R._get(r, "result", "arm", "R_mean") for r in recs],
            u_logged_mean=[R._get(r, "result", "arm", "u_logged_mean") for r in recs],
            check_e_passed=[R._get(r, "result", "arm", "check_e", "passed") for r in recs],
            finite=[R._get(r, "result", "training", "finite", arm) for r in recs])
    rep = R.finished(R.report_path(ctx), spec)
    return dict(quality=quality, smoke=smoke, missing=missing, arms=arms, comparisons=comps, diagnostics=diag,
                pre_registered=dict(bca_allowed=R._get(rep, "bca_allowed"),
                                    bca_lower_edge_allowed=R._get(rep, "bca_lower_edge_allowed")))


# ------------------------------------------------------------------------------------------------ text

def f(x, d=1, sign=False):
    if x is None:
        return "-"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def cell(st, sign=False):
    if st is None:
        return "-"
    if st.get("mean") is None:
        return f"not finite ({st.get('nonfinite')} seeds)"
    return f"{f(st['mean'], 1, sign)} ± {f(st['se'], 1)}" if st.get("se") is not None else f(st["mean"], 1, sign)


def text(summary):
    L = [XA.LABEL, ""]
    if summary["smoke"]:
        L += [XA.SMOKE_LABEL, ""]
    L += ["BCA training arms of version 5, run at the owner's request although no quality passed the pre-registered",
          "checks that allow them. Nothing here is a pre-registered result.",
          "",
          "How to read it:",
          "  Return: the exact 100-step return from fixed starting states (all negative; closer to 0 is better).",
          "  Points: the same on a 0-100 scale, 0 = random commands, 100 = the stall-aware controller.",
          "  Mixes of starting states: none = like the training data; mild and strong = shifted toward strong water",
          "  currents (the conditions the weighting is meant for).",
          "  Capped pull: BCA's own use of its band: a stronger pull toward the recorded commands where the band is",
          "  wide (co-trained with TD3+BC for 100,000 updates). Lower edge: a penalty on commands where the band is",
          "  wider than at the recorded command (actor-only training against the frozen critic).",
          "  Weighted: the band's threshold is set for the test mix (per step: each state's own threshold).",
          "  Numbers are mean ± standard error over the seeds; differences are paired by seed. p: one-sided t-test",
          "  that the difference is above 0, not corrected for the many comparisons, for orientation only.",
          ""]
    for q in summary["qualities"]:
        L.append(f"=== {q['quality']} (pre-registered checks allow: capped pull {q['pre_registered']['bca_allowed']}, "
                 f"lower edge {q['pre_registered']['bca_lower_edge_allowed']})")
        for title, names in (("Capped pull", CAPPED_NAMES), ("Lower edge", LE_NAMES)):
            L.append(f"  {title}: return on each mix [points]")
            refs = ["plain_gap"] if title == "Capped pull" else ["le_none", "le_oracle"]
            for n in list(names) + refs:
                label = names.get(n) or REFERENCES[n][1]
                a = q["arms"][n]
                L.append(f"    {label:48s}" + "  ".join(
                    f"{law}: {cell(a[law]['J'])} [{f(R._get(a[law], 'points', 'mean'))}]" for law in LAWS))
        L.append("  Differences (return; points in brackets)")
        for c in q["comparisons"]:
            L.append(f"    {c['comparison']}")
            L.append("      " + "  ".join(
                f"{law}: {cell(c['laws'][law]['J'], True)} [{cell(c['laws'][law]['points'], True)}]"
                f" p {f(c['laws'][law]['p_one_sided_uncorrected'], 2)}" for law in LAWS))
        dc, dl = q["diagnostics"]["capped"], q["diagnostics"]["lower_edge"]
        L.append("  Checks")
        for arm, d in dc.items():
            L.append(f"    capped {arm}: seeds {d['seeds_finished']}, failed {d['failed'] or 'none'}, refreshes the "
                     f"band could not certify {d['uncertified_refreshes']}, weighted {d['uncertified_weighted_refreshes']}"
                     f", most training rows refused at a refresh {d['refused_rows_max']}")
        for arm, d in dl.items():
            L.append(f"    lower edge {arm}: seeds {d['seeds_finished']}, rows without a finite threshold "
                     f"{d['refused_rows']}, finite {d['finite']}, check (e) on the band's penalty passed "
                     f"{d['check_e_passed']}")
        if q["missing"]:
            L.append(f"  Missing: {q['missing']}")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=f"{XA.LABEL}. Summarise the exploratory BCA arms.")
    p.add_argument("--out", default=str(XA.EXPLORE_ROOT))
    p.add_argument("--smoke", action="store_true", help="only to allow a scratch --out")
    p.add_argument("--qualities", nargs="*", choices=R.QUALITIES, default=list(R.QUALITIES))
    a = p.parse_args(argv)
    try:
        out = XA.check_out(a.out, a.smoke)
        spec = R.spec_digest(R.Context(root=XA.V5_ROOT))
        qs = [quality_summary(out, q, spec) for q in R.QUALITIES if q in a.qualities]
        summary = dict(kind="shift-iw-explore-summary", label=XA.LABEL, exploratory=True,
                       smoke=any(q["smoke"] for q in qs), spec_sha256=spec, seeds=list(R.SEEDS),
                       made_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), qualities=qs,
                       explore_code_sha256={Path(m).name: hashlib.sha256(Path(m).read_bytes()).hexdigest()
                                            for m in (__file__, XA.__file__, XE.__file__)})
        body = text(summary)
        R.write_record(out / "summary" / "summary.json", summary)
        (out / "summary" / "summary.txt").write_text(body + "\n")
        print(body)
    except (R.Refused,) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
