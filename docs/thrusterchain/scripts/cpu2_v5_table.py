"""CPU 2: formatted v5 numbers per mark and quality, for the weighting-test draft (read only)."""
import json
QS = ["random", "medium-replay", "medium", "medium-expert", "expert"]
ROOT = "/root/bca-shift/runs/shift_iw_v5"
R = {q: json.load(open(f"{ROOT}/{q}/report/report.json")) for q in QS}
S = json.load(open(f"{ROOT}/summary/summary.json"))


def m(x):
    return x["mean"] if isinstance(x, dict) and "mean" in x else x


def lo(x):
    return x.get("lower_one_sided")


def ps(x):
    return x.get("per_seed")


def pct(v, d=1):
    return f"{100 * v:.{d}f}"


def row(name, f):
    print(f"{name:42s}", " | ".join(f"{q}: {f(q)}" for q in QS))


def d1(q, k, d=1, scale=100):
    vals = [p["checks"][k]["value"] for p in R[q]["marks"]["D1"]["per_seed"]]
    return f"{scale * min(vals):.{d}f}-{scale * max(vals):.{d}f}"


print("summary finished_at", S["finished_at"], "spec", S["spec_sha256"], "code_is_approved", S["code_is_approved"])
row("D1 holds removed % (min-max seeds)", lambda q: d1(q, "holds_removed", 2))
row("D1 Z", lambda q: d1(q, "Z", 3, 1))
row("D1 in-ball mild %", lambda q: d1(q, "in_ball_mild"))
row("D1 in-ball strong %", lambda q: d1(q, "in_ball_strong"))
row("M1 pooled % (lowest seed)", lambda q: f'{pct(m(R[q]["marks"]["M1"]["coverage"]), 2)} ({pct(min(ps(R[q]["marks"]["M1"]["coverage"])))})')
row("M2 within deciles (lower)", lambda q: f'{m(R[q]["marks"]["M2"]["within_deciles"]):.3f} ({lo(R[q]["marks"]["M2"]["within_deciles"]):.3f})')
row("M2 given float radius (lower)", lambda q: f'{m(R[q]["marks"]["M2"]["given_float_radius_within_deciles"]):.3f} ({lo(R[q]["marks"]["M2"]["given_float_radius_within_deciles"]):.3f})')
row("M2 pooled (lower)", lambda q: f'{m(R[q]["marks"]["M2"]["pooled"]):.3f} ({lo(R[q]["marks"]["M2"]["pooled"]):.3f})')
row("M3 max drop / inversions", lambda q: f'{max(p["max_drop"] for p in R[q]["marks"]["M3"]["per_seed"]):.2e} / {min(p["inversions"] for p in R[q]["marks"]["M3"]["per_seed"])}-{max(p["inversions"] for p in R[q]["marks"]["M3"]["per_seed"])}')
row("W0 status", lambda q: R[q]["marks"]["W0"]["status"])
row("W1 mild % (lowest seed)", lambda q: f'{pct(m(R[q]["marks"]["W1"]["coverage"]["mild"]), 2)} ({pct(min(ps(R[q]["marks"]["W1"]["coverage"]["mild"])))})')
row("W1 strong % (lowest seed)", lambda q: f'{pct(m(R[q]["marks"]["W1"]["coverage"]["strong"]), 2)} ({pct(min(ps(R[q]["marks"]["W1"]["coverage"]["strong"])))})')
row("W1 single strong %", lambda q: f'{pct(m(R[q]["marks"]["W1"]["single_threshold_coverage"]["strong"]), 2)}')
row("W2 unweighted strong % (highest seed)", lambda q: f'{pct(m(R[q]["marks"]["W2"]["coverage"]), 2)} ({pct(max(ps(R[q]["marks"]["W2"]["coverage"])))})')
row("W1strong - W2 (points)", lambda q: f'{100 * (m(R[q]["marks"]["W1"]["coverage"]["strong"]) - m(R[q]["marks"]["W2"]["coverage"])):.2f}')


def w3(q, t):
    u = R[q]["marks"]["W3"]["tilts"][t]["unweighted"]
    i, o = u.get("inside"), u.get("outside")
    return (f'in {pct(m(i))} out {pct(m(o))} judged={u.get("judged")} lower_inside={u.get("lower_inside")} '
            f'steps {int(min(u["in_ball_steps"]))}-{int(max(u["in_ball_steps"]))}')


for t in ["none", "mild", "strong"]:
    row(f"W3 unweighted {t}", lambda q, t=t: w3(q, t))


def refs(q):
    out = []
    for t in ["none", "mild", "strong"]:
        tt = R[q]["marks"]["W3"]["tilts"][t]
        out.append(t + ":" + ",".join(
            f'{a}:{min(tt[a]["refused_inside"])}-{max(tt[a]["refused_inside"])}/{min(tt[a]["refused_outside"])}-{max(tt[a]["refused_outside"])}'
            for a in ["exact", "estimated"]))
    return " ".join(out)


row("W3 refused inside/outside", refs)
row("W3 status", lambda q: R[q]["marks"]["W3"]["status"])
row("W4 strong est-exact (pts)", lambda q: f'{100 * R[q]["marks"]["W4"]["strong_estimated_minus_exact"]:+.2f}')
row("W4 none est-unw (pts)", lambda q: f'{100 * R[q]["marks"]["W4"]["none_estimated_minus_unweighted"]:+.2f}')
row("W4 slope strong fitted (exact 4)", lambda q: f'{m(R[q]["marks"]["W4"]["classifier_slope_on_c"]["strong"]["fitted"]):.2f}')
row("W4 slope mild fitted (exact 2)", lambda q: f'{m(R[q]["marks"]["W4"]["classifier_slope_on_c"]["mild"]["fitted"]):.2f}')
row("R1 placed-none units (lower) [pts]", lambda q: f'{m(R[q]["marks"]["R1"]["placed_minus_none"]):+.1f} ({lo(R[q]["marks"]["R1"]["placed_minus_none"]):+.1f}) [{m(R[q]["marks"]["R1"]["placed_minus_none_points"]):+.2f}]')
row("R1 placed-matched even units", lambda q: "missing" if R[q]["marks"]["R1"].get("placed_minus_matched_even") is None else f'{m(R[q]["marks"]["R1"]["placed_minus_matched_even"]):+.1f} (lower {lo(R[q]["marks"]["R1"]["placed_minus_matched_even"]):+.1f})')
row("R2 dose change / ret per dose / pred units [pts]", lambda q: f'{m(R[q]["marks"]["R2"]["dose_change"]):.3f} / {m(R[q]["marks"]["R2"]["return_per_unit_dose"]):.1f} / {R[q]["marks"]["R2"]["predicted_return"]:+.2f} [{R[q]["marks"]["R2"]["predicted_points"]:+.3f}]')
row("R2 high-current dose change", lambda q: f'{m(R[q]["marks"]["R2"]["high_current_dose_change"]):.3f}')
row("L1 oracle-none units (lower) [pts]", lambda q: f'{m(R[q]["marks"]["L1"]["oracle_minus_none"]):+.1f} ({lo(R[q]["marks"]["L1"]["oracle_minus_none"]):+.1f}) [{m(R[q]["marks"]["L1"]["oracle_minus_none_points"]):+.2f}]')
row("L1 oracle-matched even units", lambda q: f'{m(R[q]["marks"]["L1"]["oracle_minus_matched_even"]):+.1f}')
row("L1 check_e passed", lambda q: str([c["passed"] for c in R[q]["marks"]["L1"]["check_e"]]))
row("L2 scale change / ret per k / pred units [pts]", lambda q: f'{m(R[q]["marks"]["L2"]["scale_change"]):.3f} / {m(R[q]["marks"]["L2"]["return_per_unit_k"]):+.2f} / {R[q]["marks"]["L2"]["predicted_return"]:+.2f} [{R[q]["marks"]["L2"]["predicted_points"]:+.3f}]')
row("L2 band/oracle (1/x) pearson spearman", lambda q: f'{m(R[q]["marks"]["L2"]["proxy_check"]["band_over_oracle"]):.4f} (1/{1 / m(R[q]["marks"]["L2"]["proxy_check"]["band_over_oracle"]):.0f}) {m(R[q]["marks"]["L2"]["proxy_check"]["pearson"]):+.2f} {m(R[q]["marks"]["L2"]["proxy_check"]["spearman"]):+.2f}')
row("L2 band check e passed", lambda q: str(R[q]["marks"]["L2"]["proxy_check"]["band_check_e_passed"]))


def SH(q):
    return R[q]["shown"]


row("LE pts strong none/oracle/oracle1.5", lambda q: f'{m(SH(q)["lower_edge"]["room_points"]["le_none"]["strong"]):.2f}/{m(SH(q)["lower_edge"]["room_points"]["le_oracle"]["strong"]):.2f}/{m(SH(q)["lower_edge"]["room_points"]["le_oracle_x1.5"]["strong"]):.2f}')
row("LE units strong oracle/oracle1.5", lambda q: f'{m(SH(q)["lower_edge"]["room_return"]["le_oracle"]["strong"]):.1f}/{m(SH(q)["lower_edge"]["room_return"]["le_oracle_x1.5"]["strong"]):.1f}')
row("room pts strong none/placed", lambda q: f'{m(SH(q)["room_points"]["none"]["strong"]):.2f}/{m(SH(q)["room_points"]["placed"]["strong"]):.2f}')
row("plain TD3BC gap pts none/mild/strong", lambda q: "/".join(f'{m(SH(q)["plain_td3bc_points"]["gap"][t]):.1f}' for t in ["none", "mild", "strong"]))
row("plain TD3BC full pts strong", lambda q: f'{m(SH(q)["plain_td3bc_points"]["full"]["strong"]):.1f}')
row("full-gap strong pts", lambda q: f'{m(SH(q)["full_minus_gapped_plain_td3bc_strong"]):+.2f}')
row("data pts with hold none/mild/strong", lambda q: "/".join(f'{S["rows"][q]["data_points"][t]["with_hold"]:.1f}' for t in ["none", "mild", "strong"]))
row("in-ball share each test log", lambda q: json.dumps(SH(q).get("in_ball_share_of_each_test_log"))[:200])
row("refused by own threshold outside", lambda q: json.dumps(SH(q).get("refused_by_own_threshold_outside_ball"))[:200])
row("R_w/R_u", lambda q: json.dumps(SH(q).get("R_weighted_over_R_unweighted"))[:400])
row("claims", lambda q: str(S["rows"][q]["claims"]))
