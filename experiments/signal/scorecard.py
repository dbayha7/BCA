"""Score the signal study (steps 1-3) against its pre-registered expectations.

Reads runs/wbcp_signal (expectations.md, SHA-256 53c13f11...): the LQ harness results (lq/results.json), the
frozen-critic evaluations (frozen/<pool>/evaluation_b4000_*.json), the pool verification and the step-1 reruns.
It writes scorecard.md and scorecard.json, with one row per expectation: the measured values, the verdict and the
rule that produced it.

Verdict rules, fixed before any real result was read:
- A range expectation on a rate measured over banks is **met** when the 95% interval of the measured rate overlaps
  the stated range. It is **missed** otherwise, with the direction.
- A comparison "within two standard errors" uses the paired difference's own standard error over replicates.
- [derived] items are checked exactly, up to float rounding. A miss there means a bug, not a finding.
- Healthy pools are hopper, walker2d, halfcheetah, maze2d and pen-expert. pen-cloned and pen-human are reported but
  excluded from verdicts (critic not converged; DEPENDENCE.md, Change 12).

python experiments/signal/scorecard.py [--root runs/wbcp_signal]
"""

import argparse
import glob
import json
import math
import os

HEALTHY = ("hopper-medium-v2", "walker2d-medium-replay-v2", "halfcheetah-medium-expert-v2", "maze2d-large-v1",
           "pen-expert-v1")
SHORT = {"hopper-medium-v2": "hopper", "walker2d-medium-replay-v2": "walker2d",
         "halfcheetah-medium-expert-v2": "halfcheetah", "maze2d-large-v1": "maze2d", "pen-expert-v1": "pen-expert",
         "pen-cloned-v1": "pen-cloned", "pen-human-v1": "pen-human"}
ALIGNMENT_STUDY = {"hopper": 37.4, "walker2d": 70.2, "halfcheetah": 14.4, "maze2d": 10.7, "pen-expert": 8.2}
SIGNALS = ("min", "q1", "maxabs")


def overlaps(ci, lo, hi):
    return ci[0] <= hi and ci[1] >= lo


def verdict(ok, why=""):
    return ("met" if ok else "missed") + (f" ({why})" if why and not ok else "")


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


# ---- step 2 ----

def load_frozen(root):
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "frozen", "*", "evaluation_b4000_*.json"))):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        out[data["dataset"]] = data
    return out


def design(data, name):
    return next(d for d in data["designs"] if d["design"]["name"] == name)


def step2(frozen):
    rows, table = [], []
    healthy = [d for d in HEALTHY if d in frozen]
    for ds, data in frozen.items():
        for dname in [d["design"]["name"] for d in data["designs"]]:
            des = design(data, dname)
            for s in SIGNALS:
                f = des["signals"][s]["failure"]
                table.append(dict(dataset=SHORT.get(ds, ds), design=dname, signal=s,
                                  **{f"fail_{r}": f[r]["fail"] for r in f}, **{f"ci_{r}": f[r]["ci"] for r in f},
                                  threshold_over_lambda_star=des["signals"][s]["threshold_over_lambda_star"],
                                  half_width=des["signals"][s]["half_width_mean"]))
    # 1. own validity
    iid_ok, iid_vals = True, []
    for ds in healthy:
        des = design(frozen[ds], "iid")
        for s in SIGNALS:
            f = des["signals"][s]["failure"][s]
            iid_vals.append(f"{SHORT[ds]} {s} {pct(f['fail'])}")
            iid_ok &= overlaps(f["ci"], 0.035, 0.065)
    rows.append(dict(item="2.1a", text="each signal fails its own score in 3.5-6.5% of independent-row banks",
                     label="derived", values="; ".join(iid_vals), verdict=verdict(iid_ok, "some outside")))
    ranges = {"hopper": (0.035, 0.065), "halfcheetah": (0.035, 0.065), "maze2d": (0.05, 0.07),
              "pen-expert": (0.05, 0.07), "walker2d": (0.15, 0.20)}
    conf_ok, conf_vals = True, []
    for ds in healthy:
        names = [d["design"]["name"] for d in frozen[ds]["designs"] if d["design"]["name"] != "iid"]
        if not names:
            continue
        des = design(frozen[ds], names[0])
        lo, hi = ranges[SHORT[ds]]
        for s in SIGNALS:
            f = des["signals"][s]["failure"][s]
            conf_vals.append(f"{SHORT[ds]} {s} {pct(f['fail'])}")
            conf_ok &= overlaps(f["ci"], lo, hi)
    rows.append(dict(item="2.1b", text="configured bank: own failure about 5% (hopper, halfcheetah), 5-7% (maze2d, "
                     "pen-expert), 15-20% (walker2d)", label="replication", values="; ".join(conf_vals),
                     verdict=verdict(conf_ok, "some outside")))
    # 2. min band misses Q1
    q1_under_min = {SHORT[ds]: design(frozen[ds], "iid")["signals"]["min"]["failure"]["q1"] for ds in healthy}
    above8 = sum(v["fail"] > 0.08 for v in q1_under_min.values())
    order = sorted(q1_under_min, key=lambda k: -q1_under_min[k]["fail"])
    within10 = all(abs(100 * v["fail"] - ALIGNMENT_STUDY[k]) <= 10 for k, v in q1_under_min.items())
    expected_order = ["walker2d", "hopper", "halfcheetah"]
    rows.append(dict(item="2.2", text="min band fails for Q1's residual in >8% of banks on >=4 of 5 pools; ranking "
                     "walker2d > hopper > halfcheetah > maze2d ~ pen-expert; each within ±10 points of the alignment study",
                     label="replication",
                     values="; ".join(f"{k} {pct(v['fail'])} (study {ALIGNMENT_STUDY[k]}%)" for k, v in q1_under_min.items())
                     + f"; order {' > '.join(order)}",
                     verdict=verdict(above8 >= 4 and order[:3] == expected_order and within10,
                                     f">8% on {above8}/5; order {order}; within ±10: {within10}")))
    # 3. q1 band covers min
    min_under_q1 = {SHORT[ds]: design(frozen[ds], "iid")["signals"]["q1"]["failure"]["min"] for ds in healthy}
    rows.append(dict(item="2.3", text="q1 band fails for the min residual in at most 3% of banks", label="replication, low confidence",
                     values="; ".join(f"{k} {pct(v['fail'])}" for k, v in min_under_q1.items()),
                     verdict=verdict(all(v["ci"][0] <= 0.03 for v in min_under_q1.values()), "above 3%")))
    # 4. maxabs covers both
    d4, r4 = True, True
    vals4 = []
    for ds in healthy:
        sig = design(frozen[ds], "iid")["signals"]["maxabs"]["failure"]
        own = sig["maxabs"]["fail"]
        for r in ("q1", "min"):
            d4 &= sig[r]["fail"] <= own + 1e-12
            r4 &= sig[r]["ci"][0] <= 0.01
            vals4.append(f"{SHORT[ds]} {r} {pct(sig[r]['fail'])} (own {pct(own)})")
    rows.append(dict(item="2.4a", text="maxabs band's failure for Q1 and min residuals <= its own failure", label="derived",
                     values="; ".join(vals4), verdict=verdict(d4, "exceeds own")))
    rows.append(dict(item="2.4b", text="... and at most 1%", label="replication", values="(same)", verdict=verdict(r4, "above 1%")))
    # 5. width
    w = {SHORT[ds]: design(frozen[ds], "iid")["width_ratio"] for ds in healthy}
    rows.append(dict(item="2.5", text="maxabs band 3-15% wider than q1's; q1 and min within ±10%", label="forecast",
                     values="; ".join(f"{k} maxabs/q1 {v['maxabs/q1']:.3f}, q1/min {v['q1/min']:.3f}" for k, v in w.items()),
                     verdict=verdict(all(1.03 <= v["maxabs/q1"] <= 1.15 and 0.9 <= v["q1/min"] <= 1.1 for v in w.values()),
                                     "some ratios outside")))
    # 6. dose
    dose_vals, dose_ok = [], True
    for ds in healthy:
        # the dose at each calibrated threshold, averaged over the independent-row banks (evaluator: doses[s].banks.iid)
        doses = {s: frozen[ds].get("doses", {}).get(s, {}).get("banks", {}).get("iid") for s in SIGNALS}
        if not all(doses.values()):
            dose_ok = False
            dose_vals.append(f"{SHORT[ds]} dose missing")
            continue
        means = {s: doses[s]["mean"] for s in SIGNALS}
        sds = {s: doses[s]["sd"] for s in SIGNALS}
        dose_ok &= all(1.25 <= m <= 1.33 for m in means.values()) and all(v <= 0.06 for v in sds.values()) \
            and max(means.values()) - min(means.values()) <= 0.03
        dose_vals.append(f"{SHORT[ds]} " + ", ".join(f"{s} {means[s]:.3f}±{sds[s]:.3f}" for s in SIGNALS))
    rows.append(dict(item="2.6", text="dose mean 1.25-1.33, row SD <= 0.06, means differ by <= 0.03 across signals",
                     label="derived/forecast", values="; ".join(dose_vals), verdict=verdict(dose_ok, "see values")))
    return rows, table


# ---- step 3 ----

def lq_results(root):
    with open(os.path.join(root, "lq", "results.json"), encoding="utf-8") as handle:
        return json.load(handle)


def cells(res, **match):
    return [r for r in res["results"] if all(r.get(k) == v for k, v in match.items())]


def m(r, name):
    x = r["metrics"].get(name)
    return (None, None) if x is None else (x["mean"], x["se"])


def step3(res):
    rows = []
    checks = res["builtin_checks_summary"]
    rows.append(dict(item="3.checks", text="built-in checks 1-5 (identities)", label="derived",
                     values=", ".join(f"{k}={v}" for k, v in checks.items()), verdict=verdict(all(checks.values()), "a check failed")))
    # E1 dose
    means = [m(r, "dose_mean")[0] for r in res["results"]]
    maxes = [m(r, "dose_max")[0] for r in res["results"]]
    exact = [m(r, "dose_mean")[0] for r in res["results"] if r["case"] in ("clean", "noisy_reward")]
    rows.append(dict(item="3.E1", text="mean dose 1.2-1.4 in every case, never reaches 1.5; ~1.3 with exact critics",
                     label="derived", values=f"mean range {min(means):.3f}-{max(means):.3f}; max dose {max(maxes):.4f}; "
                     f"exact-critic means {min(exact):.3f}-{max(exact):.3f}",
                     verdict=verdict(min(means) >= 1.2 and max(means) <= 1.4 and max(maxes) < 1.5, "out of range")))
    # E2 q1_optimistic logged
    vals, own_ok, min_grows, q1_bound = [], True, True, True
    for beh in res["meta"]["behaviors"]:
        for off in res["meta"]["pi_offsets"]:
            seq = []
            for kap in res["meta"]["kappas"]:
                for r in cells(res, case="q1_optimistic", behavior=beh, pi_offset=off, kappa=kap):
                    own = m(r, "miss_own_logged")[0]
                    own_ok &= 0.05 <= own <= 0.15
                    miss = m(r, "miss_q1err_logged")[0]
                    if r["signal"] == "min":
                        seq.append(miss)
                    else:
                        q1_bound &= miss <= 0.20
                    vals.append(f"{beh}/off{off}/k{kap}/{r['signal']}: own {pct(own)}, Q1-err miss {pct(miss)}")
            min_grows &= all(b >= a - 1e-9 for a, b in zip(seq, seq[1:]))
    rows.append(dict(item="3.E2", text="q1_optimistic: own coverage ~90%; min misses true Q1 error more as kappa grows; "
                     "q1/maxabs miss it on <= ~20% of rows (probably <= 10%)", label="derived (+forecast for 10%)",
                     values=" | ".join(vals[:12]) + (" | ..." if len(vals) > 12 else ""),
                     verdict=verdict(own_ok and min_grows and q1_bound,
                                     f"own~90%: {own_ok}; min grows with kappa: {min_grows}; q1/maxabs <=20%: {q1_bound}")))
    # E3 proposed actions
    worse, grows, n = 0, True, 0
    for case in res["meta"]["cases"]:
        for beh in res["meta"]["behaviors"]:
            for s in res["meta"]["signals"]:
                seq = []
                for off in res["meta"]["pi_offsets"]:
                    rs = cells(res, case=case, behavior=beh, signal=s, pi_offset=off)
                    rs = [r for r in rs if r["kappa"] in (None, max(res["meta"]["kappas"]))]
                    for r in rs[:1]:
                        a, b = m(r, "miss_q1err_logged")[0], m(r, "miss_q1err_pi")[0]
                        if a is None or b is None:
                            continue
                        n += 1
                        worse += b > a
                        seq.append(b)
                grows &= all(y >= x - 0.02 for x, y in zip(seq, seq[1:]))
    rows.append(dict(item="3.E3", text="at pi(s), every signal misses the true Q1 error more than at logged actions, "
                     "more so with distance", label="forecast", values=f"pi worse than logged in {worse}/{n} cells; "
                     f"monotone in offset (tolerance 2 points): {grows}", verdict=verdict(worse == n and grows, "not everywhere")))
    # E5 targeting vs strength
    within, bigger, total = 0, 0, 0
    for r in res["results"]:
        g_c, se_c = m(r, "J_gain_bca_minus_constant")
        g_s, se_s = m(r, "J_gain_bca_minus_shuffled")
        g_n, se_n = m(r, "J_gain_bca_minus_none")
        if None in (g_c, se_c, g_s, se_s, g_n):
            continue
        total += 1
        within += abs(g_c) <= 2 * se_c and abs(g_s) <= 2 * se_s
        bigger += abs(g_n) > max(abs(g_c), abs(g_s))
    rows.append(dict(item="3.E5", text="BCA's J change within 2 SE of constant-dose and shuffled controls in every case, "
                     "while the gap to no BCA is larger", label="forecast, moderate",
                     values=f"within 2 SE of both: {within}/{total}; gap to none larger: {bigger}/{total}",
                     verdict=verdict(within == total and bigger == total, f"{within}/{total} and {bigger}/{total}")))
    # E6 exact critics: none >= bca
    ok6, n6 = 0, 0
    for r in res["results"]:
        if r["case"] not in ("clean", "noisy_reward"):
            continue
        g, se = m(r, "J_gain_bca_minus_none")
        if g is None:
            continue
        n6 += 1
        ok6 += g <= 2 * (se or 0)
    rows.append(dict(item="3.E6", text="exact critics: J improves at least as much without BCA (gap larger with poor data)",
                     label="forecast", values=f"no-BCA >= BCA (within 2 SE) in {ok6}/{n6} cells",
                     verdict=verdict(ok6 == n6, f"{ok6}/{n6}")))
    # E7 shared bias
    vals7 = []
    for beh in res["meta"]["behaviors"]:
        for r in cells(res, case="shared_bias", behavior=beh, pi_offset=1.0):
            vals7.append(f"{beh}/k{r['kappa']}/{r['signal']}: own miss {pct(m(r, 'miss_own_logged')[0])}, "
                         f"Q1-err miss logged {pct(m(r, 'miss_q1err_logged')[0])}, at pi {pct(m(r, 'miss_q1err_pi')[0])}")
    rows.append(dict(item="3.E7", text="shared_bias: signals identical; residual coverage near nominal while true-error "
                     "coverage can fail badly", label="derived", values=" | ".join(vals7[:9]), verdict="see values"))
    # A1 independent errors
    vals_a, a2 = [], True
    for beh in res["meta"]["behaviors"]:
        for off in res["meta"]["pi_offsets"]:
            rs = {r["signal"]: r for r in cells(res, case="independent_errors", behavior=beh, pi_offset=off)}
            if len(rs) < 3:
                continue
            mm, mq = m(rs["min"], "miss_q1err_logged")[0], m(rs["q1"], "miss_q1err_logged")[0]
            a2 &= mm > mq
            vals_a.append(f"{beh}/off{off}: Q1-err miss min {pct(mm)} vs q1 {pct(mq)} vs maxabs "
                          f"{pct(m(rs['maxabs'], 'miss_q1err_logged')[0])}")
    rows.append(dict(item="A1.2", text="independent_errors: min misses the true Q1 error more often than q1",
                     label="forecast", values=" | ".join(vals_a), verdict=verdict(a2, "not in every cell")))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="runs/wbcp_signal")
    args = parser.parse_args(argv)
    out = dict(step2=None, step3=None)
    lines = ["# Signal study scorecard", "", "Verdict rules: see experiments/signal/scorecard.py.", ""]
    frozen = load_frozen(args.root)
    if frozen:
        rows, table = step2(frozen)
        out["step2"] = dict(rows=rows, table=table, pools=sorted(frozen))
        lines += ["## Step 2 (frozen D4RL critics)", "", "| Item | Expectation | Label | Measured | Verdict |", "|---|---|---|---|---|"]
        lines += [f"| {r['item']} | {r['text']} | {r['label']} | {r['values']} | {r['verdict']} |" for r in rows]
        lines.append("")
    if os.path.exists(os.path.join(args.root, "lq", "results.json")):
        rows = step3(lq_results(args.root))
        out["step3"] = dict(rows=rows)
        lines += ["## Step 3 (linear-quadratic harness)", "", "| Item | Expectation | Label | Measured | Verdict |", "|---|---|---|---|---|"]
        lines += [f"| {r['item']} | {r['text']} | {r['label']} | {r['values']} | {r['verdict']} |" for r in rows]
    with open(os.path.join(args.root, "scorecard.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
    with open(os.path.join(args.root, "scorecard.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
