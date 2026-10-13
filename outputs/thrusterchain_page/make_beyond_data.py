"""Write beyond_data.json: the numbers of the page's "After the stop" subsection (EXPLORATORY, outside the gate).

Reads only committed or saved result files, read only, and checks each against its SHA-256 sidecar where one exists:
  - Stage 1, the room test of the value-penalty route (no BCA):
      outputs/shift/beyond-the-data/stage1/spec.json, results/pooled.json, results/futility.json,
      results/seed{0..4}/values.json, .../no_stall/values.json, .../replication/values.json,
      checks_seed0_e.json and checks_e_seed{1..4}.json (check (e), the penalty's shape, before any training),
      results/seed{0..4}/lock.json (the strength matching, written before any return was computed)
  - the deployment-time diagnostic from saved actors (no training, no BCA): runs/shift_explore/fallback/summary.json
  - the anatomy of the harm (no training, no BCA): runs/shift_explore/anatomy/summary.json, curves.json,
    gate_readout.json
  - status only (no result is read or stated): runs/shift_explore/stage2/prereg.json (the flags claim's wording, its
    version, and whether its pins still match the files they name, compared here the way approve.py compares them),
    runs/shift_explore/trace/spec.json (the trace test bed's design), whether Stage 2 is committed in the folder the
    user approves from and whether an approval exists, and whether each running piece has written its final file
    (only existence and the file's kind are read), whether the stand-ins' launcher has logged its wait or its start
    and whether the user's choice of their mark forms exists (and which forms it names, when), the commit and
    fingerprint of the committed Stage 2 pre-registration, the commits of the user's approval and of its import, and
    whether any Stage 2 run output exists (read-only git queries and directory listings). The script stops when this
    state differs from PAGE_STATE, the state the page's status text describes (the stand-ins box included).
    rebuild.sh runs it before every build.
Every value is a mean over seeds with the lowest and highest seed ("lo", "hi") unless its name says otherwise.
Writes nothing else. build.py loads the output as D["beyond"].
"""

import hashlib
import json
import os
import re
import subprocess

BCA = "/home/user/bca"
S1 = f"{BCA}/outputs/shift/beyond-the-data/stage1"
RES = f"{S1}/results"
FB = f"{BCA}/runs/shift_explore/fallback"
AN = f"{BCA}/runs/shift_explore/anatomy"
ST2 = f"{BCA}/runs/shift_explore/stage2"
TR = f"{BCA}/runs/shift_explore/trace"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beyond_data.json")

SOURCES = {}


def load(path):
    raw = open(path, "rb").read()
    h = hashlib.sha256(raw).hexdigest()
    side = path + ".sha256"
    if os.path.exists(side):
        want = open(side).read().split()[0]
        assert want == h, f"{path}: SHA-256 {h} does not match its sidecar {want}"
    SOURCES[os.path.relpath(path, BCA)] = h
    return json.loads(raw)


def stats(values):
    v = [float(x) for x in values]
    assert len(v) == 5, values
    return dict(mean=sum(v) / len(v), lo=min(v), hi=max(v), values=v)


def pooled_stat(d):
    """anatomy summaries already hold {mean, lo, hi, n}"""
    assert d["n"] == 5, d
    return dict(mean=d["mean"], lo=d["lo"], hi=d["hi"])


def num(pattern, text, cast=float):
    m = re.search(pattern, text)
    assert m, (pattern, text)
    return cast(m.group(1))


def git_short(path):
    return subprocess.check_output(["git", "-C", BCA, "log", "-1", "--format=%h", "--", path], text=True).strip()


SIT = ["recorded", "strong_current", "big_waves", "big_kicks"]

# ------------------------------------------------------------------------------------------------ Stage 1: spec

spec = load(f"{S1}/spec.json")
assert spec["label"].startswith("EXPLORATORY"), spec["label"]
seeds = spec["seeds"]
assert seeds == [0, 1, 2, 3, 4]
marks = spec["marks"]
a1 = [c for c in spec["changes_before_any_run"] if c["id"].startswith("A1")]
assert len(a1) == 1
main_arm = [a for a in spec["arms"] if a["name"] == "perfect_band"][0]
assert main_arm["W"] == "[delta*(s_i, a_i) - delta*(s_i, a)]_+" and main_arm["k"] == 1.0
lit_arm = [a for a in spec["arms"] if a["name"] == "perfect_band_literal"][0]
assert lit_arm["W"] == "|delta*(s, a)|"
half_arm = [a for a in spec["arms"] if a["name"] == "perfect_band_half"][0]
hinge_arm = [a for a in spec["arms"] if a["name"] == "range_hinge"][0]
even_arm = [a for a in spec["arms"] if a["name"] == "even_pull"][0]
setup = spec["setup"]

design = dict(
    seeds_lo=min(seeds), seeds_hi=max(seeds), n_seeds=len(seeds),
    futility_seeds=" and ".join(str(s) for s in spec["futility_seeds"]),
    actor_steps=setup["actor_steps"], alpha=setup["alpha"], batch=setup["batch_size"],
    k_main=main_arm["k"], k_half=half_arm["k"],
    gamma=spec["oracle"]["gamma"],
    hinge_weight=num(r"^(\d+) \*", hinge_arm["W"], int),
    range_lo_q=num(r"the ([\d.]+)% and", hinge_arm["range"]) / 100,
    range_hi_q=num(r"and ([\d.]+)% quantiles", hinge_arm["range"]) / 100,
    harm_rollout_steps=num(r"(\d+)-step rollout", [a for a in spec["arms"] if a["name"] == "long_run_harm"][0]["W"], int),
    strength_states=int(re.search(r"the ([\d,]+) evaluation-split", spec["strength"]["definition"]).group(1).replace(",", "")),
    check_e_states=int(re.search(r"On ([\d,]+) full-throttle-region", spec["checks"]["e"]).group(1).replace(",", "")),
    random_surface_hidden=num(r"Dense (\d+) - ReLU", [a for a in spec["arms"] if a["name"] == "random_1"][0]["W"], int),
    even_ladder_top=max(even_arm["ladder_nu"]),
    spec_sha=SOURCES[os.path.relpath(f"{S1}/spec.json", BCA)],
)
design["spec_sha_short"] = design["spec_sha"][:8]
design["spec_commit"] = git_short("outputs/shift/beyond-the-data/stage1/spec.json")
design["results_commit"] = git_short("outputs/shift/beyond-the-data/stage1/results/pooled.json")
design["range_lo_pct"] = design["range_lo_q"]
design["range_share"] = design["range_hi_q"] - design["range_lo_q"]

mk = dict(
    room_margin=num(r"by at least (\d+) points", marks["room"]),
    room_positive=num(r"positive in at least (\d) of", marks["room"], int),
    room_of=num(r"at least \d of (\d) seeds", marks["room"], int),
    recovery_threshold=num(r"at least ([\d.]+) points above no caution", marks["recovery"]),
    recovery_harm=num(r"half of check 2's ([\d.]+)-point harm", marks["recovery"]),
    keep_recorded_vs_none=num(r"no caution - (\d+) and", marks["keeps_improvement"]),
    keep_recorded_vs_clone=num(r"cloned actor \+ (\d+)", marks["keeps_improvement"]),
    keep_shifted_vs_none=num(r"big kicks: perfect band >= no caution - (\d+)", marks["keeps_improvement"]),
    futility_margin=num(r"by less than (\d+) points", marks["futility"]),
)
assert abs(mk["recovery_threshold"] - mk["recovery_harm"] / 2) < 0.06  # 49.5 is half of 98.9, rounded up

# check (e): the penalty's shape on full-throttle-region training states, before any training (every seed)
FORM = {"main": "main (perfect_band)", "literal": "abs_lit (perfect_band_literal)", "centred": "abs_rel (perfect_band_centred)"}
ce = {k: [] for k in ("delta_at_recorded", "q1_best_past80", "main_fall", "main_best_past80", "main_rise",
                      "main_zero_both", "lit_fall", "lit_best_past80", "centred_fall")}
for sd in seeds:
    f = f"{S1}/checks_seed0_e.json" if sd == 0 else f"{S1}/checks_e_seed{sd}.json"
    e = load(f)["e"]
    assert e["passed"] is True
    st = e["stall"]
    fm = st["forms"]
    assert fm[FORM["main"]]["passed"] is True and fm[FORM["literal"]]["passed"] is False
    ce["delta_at_recorded"].append(st["delta_at_recorded_mean"])
    ce["q1_best_past80"].append(st["q1_alone_best_past80_share"])
    ce["main_fall"].append(fm[FORM["main"]]["fall_share_80_to_81"])
    ce["main_best_past80"].append(fm[FORM["main"]]["best_past80_share"])
    ce["main_rise"].append(fm[FORM["main"]]["mean_W_full_minus_80"])
    ce["main_zero_both"].append(fm[FORM["main"]]["unchanged_share_80_to_81"])
    ce["lit_fall"].append(fm[FORM["literal"]]["fall_share_80_to_81"])
    ce["lit_best_past80"].append(fm[FORM["literal"]]["best_past80_share"])
    ce["centred_fall"].append(fm[FORM["centred"]]["fall_share_80_to_81"])
check_e = {k: stats(v) for k, v in ce.items()}

# ------------------------------------------------------------------------------------------------ Stage 1: results

pooled = load(f"{RES}/pooled.json")
fut = load(f"{RES}/futility.json")
for d in (pooled, fut):
    assert d["label"].startswith("EXPLORATORY") and d["spec_sha256"] == design["spec_sha"]
assert pooled["complete"] is True and pooled["n"] == 5 and fut["stop"] is False

STALL_ARMS = ["clone", "none", "perfect_band", "perfect_band_half", "perfect_band_centred", "perfect_band_literal",
              "long_run_harm", "state_shuffled_band", "underrating_half", "even_pull", "push_less", "random_1",
              "random_2", "range_hinge", "overrating_half_literal"]
NS_ARMS = ["clone", "none", "perfect_band", "perfect_band_rising", "even_pull_locked", "range_hinge"]
REP_ARMS = ["none", "perfect_band", "even_pull", "harm_placed", "strength_matched_even", "cap2"]

st_pts = {a: {s: [] for s in SIT} for a in STALL_ARMS}
st_p80 = {a: {s: [] for s in SIT} for a in STALL_ARMS}
ns_pts = {a: {s: [] for s in SIT} for a in NS_ARMS}
ns_p80 = {a: {s: [] for s in SIT} for a in NS_ARMS}
rep_pts = {a: {s: [] for s in SIT} for a in REP_ARMS}
rep_p80 = {a: {s: [] for s in SIT} for a in ("perfect_band", "even_pull")}
matched = {k: [] for k in ("even_pull", "push_less", "random_1", "random_2", "rep_even_pull")}
matched_names = {}

for sd in seeds:
    v = load(f"{RES}/seed{sd}/values.json")
    assert v["label"].startswith("EXPLORATORY") and v["world"] == "stall" and v["spec_sha256"] == design["spec_sha"]
    name = dict(v["matched_names"])
    matched_names[sd] = name
    for k in ("even_pull", "push_less", "random_1", "random_2"):
        matched[k].append(float(name[k].split("@")[1]))
    for a in STALL_ARMS:
        ev = v["evaluations"][name.get(a, a)]
        for s in SIT:
            assert ev[s]["finite"] is True
            st_pts[a][s].append(ev[s]["points"])
            st_p80[a][s].append(ev[s]["unrecorded_throttle_share"])
    n = load(f"{RES}/seed{sd}/no_stall/values.json")
    assert n["label"].startswith("EXPLORATORY") and n["world"] == "no_stall"
    for a in NS_ARMS:
        for s in SIT:
            ns_pts[a][s].append(n["evaluations"][a][s]["points"])
            ns_p80[a][s].append(n["evaluations"][a][s]["unrecorded_throttle_share"])
    r = load(f"{RES}/seed{sd}/replication/values.json")
    assert r["label"].startswith("EXPLORATORY")
    ep = [k for k in r["evaluations"] if k.startswith("even_pull@")]
    assert len(ep) == 1
    matched["rep_even_pull"].append(float(ep[0].split("@")[1]))
    for s in SIT:
        pts = r["situations"][s]["points"]
        rep_pts["none"][s].append(pts["check3_none"])
        rep_pts["perfect_band"][s].append(pts["perfect_band"])
        rep_pts["even_pull"][s].append(pts[ep[0]])
        rep_pts["harm_placed"][s].append(pts["check3_harm_placed"])
        rep_pts["strength_matched_even"][s].append(pts["check3_strength_matched_even"])
        rep_pts["cap2"][s].append(pts["check3_cap2"])
        assert abs(pts["perfect_band"] - r["evaluations"]["perfect_band"][s]["points"]) < 1e-9
        rep_p80["perfect_band"][s].append(r["evaluations"]["perfect_band"][s]["unrecorded_throttle_share"])
        rep_p80["even_pull"][s].append(r["evaluations"][ep[0]][s]["unrecorded_throttle_share"])


# each seed's lock, written before any return was computed: how close each matched control came to the oracle's
# strength S*, and how far below S* the state-shuffled oracle (not matched in strength) sits
match_dev, shuf_below = [], []
for sd in seeds:
    lk = load(f"{RES}/seed{sd}/lock.json")
    assert lk["spec_sha256"] == design["spec_sha"] and lk["label"].startswith("EXPLORATORY")
    t = lk["target_strength"]
    for c in ("even_pull", "push_less", "random_1", "random_2"):
        m = lk["matches"][c]
        assert m["matched"] is True and m["arm"] == matched_names[sd][c], (sd, c, m["arm"])
        match_dev.append(abs(lk["strengths"][m["arm"]] / t - 1))
    shuf_below.append(1 - lk["strengths"]["state_shuffled_band"] / t)


def table_of(pts):
    return {a: {s: stats(pts[a][s]) for s in SIT} for a in pts}


stall = table_of(st_pts)
stall_past80 = table_of(st_p80)
no_stall = table_of(ns_pts)
no_stall_past80 = table_of(ns_p80)
replication = table_of(rep_pts)
replication_past80 = table_of(rep_p80)
# the cost of caution in the no-stall world: no extra caution minus the arm, paired by seed
no_stall_cost = {a: {s: stats([x - y for x, y in zip(ns_pts["none"][s], ns_pts[a][s])]) for s in SIT}
                 for a in NS_ARMS if a not in ("none",)}


def paired(a, b, s, src=st_pts):
    return stats([x - y for x, y in zip(src[a][s], src[b][s])])


# cross-check against the pooled look (the marks are computed there, independently)
rec = paired("perfect_band", "none", "strong_current")
assert abs(rec["mean"] - pooled["recovery"]["mean"]) < 1e-9
for c in ("even_pull", "push_less", "random_1", "random_2"):
    assert abs(paired("perfect_band", c, "strong_current")["mean"] - pooled["room"]["controls"][c]["mean"]) < 1e-9
assert abs(paired("perfect_band", "none", "recorded")["mean"] - pooled["keeps_improvement"]["recorded_vs_none"]["mean"]) < 1e-9
assert abs(paired("perfect_band", "clone", "recorded")["mean"] - pooled["keeps_improvement"]["recorded_vs_clone"]["mean"]) < 1e-9

room = {c: dict(stats(pooled["room"]["controls"][c]["values"]), positive=pooled["room"]["controls"][c]["positive_seeds"],
                passed=pooled["room"]["controls"][c]["passed"]) for c in ("even_pull", "push_less", "random_1", "random_2")}
ki = pooled["keeps_improvement"]
pooled_marks = dict(
    room=dict(passed=pooled["room"]["passed"], controls=room,
              lo=min(r["mean"] for r in room.values()), hi=max(r["mean"] for r in room.values()),
              positive_min=min(r["positive"] for r in room.values())),
    recovery=dict(stats(pooled["recovery"]["values"]), passed=pooled["recovery"]["passed"],
                  share_of_harm=pooled["recovery"]["mean"] / mk["recovery_harm"]),
    keeps=dict(passed=ki["passed"], **{k: stats(ki[k]["values"]) for k in
                                       ("recorded_vs_none", "recorded_vs_clone", "big_waves_vs_none", "big_kicks_vs_none")}),
    futility=dict(stop=fut["stop"], decision=fut["decision"],
                  **{f"seed{k}": v["perfect_band_minus_matched_even_pull"] for k, v in fut["seeds"].items()}),
    all_passed=bool(pooled["room"]["passed"] and pooled["recovery"]["passed"] and ki["passed"]),
)
assert pooled_marks["all_passed"]

untargeted = ["even_pull", "push_less", "random_1", "random_2"]
oracle_forms = ["perfect_band", "perfect_band_half", "perfect_band_centred", "perfect_band_literal", "long_run_harm"]
derived = dict(
    untargeted_strong_lo=min(stall[a]["strong_current"]["mean"] for a in untargeted),
    untargeted_strong_hi=max(stall[a]["strong_current"]["mean"] for a in untargeted),
    oracle_forms_strong_lo=min(stall[a]["strong_current"]["mean"] for a in oracle_forms),
    oracle_forms_strong_hi=max(stall[a]["strong_current"]["mean"] for a in oracle_forms),
    oracle_gain_other_lo=min(paired("perfect_band", "none", s)["mean"] for s in ("recorded", "big_waves", "big_kicks")),
    oracle_gain_other_hi=max(paired("perfect_band", "none", s)["mean"] for s in ("recorded", "big_waves", "big_kicks")),
    hinge_minus_oracle_strong=paired("range_hinge", "perfect_band", "strong_current"),
    hinge_minus_none_strong=paired("range_hinge", "none", "strong_current"),
    rep_oracle_minus_even_strong=stats([x - y for x, y in zip(rep_pts["perfect_band"]["strong_current"],
                                                               rep_pts["even_pull"]["strong_current"])]),
    rep_oracle_minus_none_strong=stats([x - y for x, y in zip(rep_pts["perfect_band"]["strong_current"],
                                                               rep_pts["none"]["strong_current"])]),
    ns_oracle_cost_other_lo=min(no_stall_cost["perfect_band"][s]["mean"] for s in ("recorded", "big_waves", "big_kicks")),
    ns_oracle_cost_other_hi=max(no_stall_cost["perfect_band"][s]["mean"] for s in ("recorded", "big_waves", "big_kicks")),
    ns_hinge_cost_other_lo=min(no_stall_cost["range_hinge"][s]["mean"] for s in ("recorded", "big_waves", "big_kicks")),
    ns_hinge_cost_other_hi=max(no_stall_cost["range_hinge"][s]["mean"] for s in ("recorded", "big_waves", "big_kicks")),
    even_nu=stats(matched["even_pull"]),
    rep_even_nu=stats(matched["rep_even_pull"]),
)
derived["ns_hinge_over_oracle_cost"] = (no_stall_cost["range_hinge"]["strong_current"]["mean"]
                                        / no_stall_cost["perfect_band"]["strong_current"]["mean"])
# what untargeted caution of the oracle's strength recovered over no extra caution in strong current, paired by seed
derived["untargeted_gain_strong"] = {a: paired(a, "none", "strong_current") for a in untargeted}
derived["untargeted_gain_strong_hi"] = max(v["mean"] for v in derived["untargeted_gain_strong"].values())
derived["untargeted_gain_strong_lo"] = min(v["mean"] for v in derived["untargeted_gain_strong"].values())
# the matched controls' strength: the largest gap to the oracle's strength over the four controls and five seeds
derived["match_dev_max"] = max(match_dev)
# the oracle read at another training state: not matched in strength, so its gain is compared, not called "the same"
derived["shuffled_below"] = stats(shuf_below)
derived["shuffled_minus_none_strong"] = paired("state_shuffled_band", "none", "strong_current")
derived["oracle_minus_shuffled_strong"] = paired("perfect_band", "state_shuffled_band", "strong_current")
assert derived["shuffled_below"]["lo"] > 0.1, derived["shuffled_below"]  # the page says "weaker"
# the page says the clip is ahead of the oracle in strong current "on every seed"
assert derived["hinge_minus_oracle_strong"]["lo"] > 0, derived["hinge_minus_oracle_strong"]
# the page says untargeted caution recovered "at most" this much of the harm: far below the oracle's recovery
assert derived["untargeted_gain_strong_hi"] < 0.1 * mk["recovery_harm"], derived["untargeted_gain_strong_hi"]

# ------------------------------------------------------------------------------------------------ deployment check

fb = load(f"{FB}/summary.json")
G, NS, FULL = fb["stall/gapped"], fb["no_stall/no_stall_gapped"], fb["stall/full"]
assert G["seeds"] == seeds and NS["seeds"] == seeds


def rule(block, s, r, field="points"):
    m, se, n = block["situations"][s]["rules"][r][field]
    assert n == 5
    return dict(mean=m, se=se)


def diff(block, s, key):
    m, se, n = block["situations"][s]["differences"][key]
    assert n == 5
    return dict(mean=m, se=se)


deploy = dict(
    n_starts=G["n_starts"][0], bank=1000,
    harm_on_starts=FULL["situations"]["strong_current"]["rules"]["none:None"]["points"][0]
    - G["situations"]["strong_current"]["rules"]["none:None"]["points"][0],
    alone={s: rule(G, s, "none:None") for s in SIT},
    check={s: rule(G, s, "sa95:clip99") for s in SIT},
    check_gain={s: diff(G, s, "sa95:clip99 minus none") for s in SIT},
    check_over_random={s: diff(G, s, "sa95:clip99 minus random95:clip99") for s in SIT},
    random_gain={s: diff(G, s, "random95:clip99 minus none") for s in SIT},
    clip_all={s: rule(G, s, "always:clip99") for s in SIT},
    clip_all_gain={s: diff(G, s, "always:clip99 minus none") for s in SIT},
    perfect_gain={s: diff(G, s, "unrec:clip99 minus none") for s in SIT},
    exact_gain={s: diff(G, s, "oracle:clip99 minus none") for s in SIT},
    clone_fallback_gain={s: diff(G, s, "sa95:clone minus none") for s in SIT},
    changed={s: rule(G, s, "sa95:clip99", "changed_share") for s in SIT},
    recall_strong=rule(G, "strong_current", "sa95:clip99", "recall_past80"),
    precision_strong=rule(G, "strong_current", "sa95:clip99", "precision_past80"),
    no_stall=dict(check_cost=diff(NS, "strong_current", "sa95:clip99 minus none"),
                  check_cost_over_random=diff(NS, "strong_current", "sa95:clip99 minus random95:clip99"),
                  clip_all_cost=diff(NS, "strong_current", "always:clip99 minus none")),
    level=95,
    neighbours=10,
)
# costs as positive numbers (the actor alone minus the rule), for the sentences that say "costs"
deploy["no_stall"]["cost"] = {k: -v["mean"] for k, v in deploy["no_stall"].items()}
deploy["clone_fallback_cost"] = {s: -deploy["clone_fallback_gain"][s]["mean"] for s in SIT}
deploy["check_share_of_harm"] =deploy["check_gain"]["strong_current"]["mean"] / deploy["harm_on_starts"]
deploy["targeting_share_of_gain"] = (deploy["check_over_random"]["strong_current"]["mean"]
                                     / deploy["check_gain"]["strong_current"]["mean"])
rows80 = G["reference"]["training_rows_past_80"]
deploy["rows_past_80_lo"], deploy["rows_past_80_hi"] = min(rows80), max(rows80)

# ------------------------------------------------------------------------------------------------ anatomy

an = load(f"{AN}/summary.json")
cv = load(f"{AN}/curves.json")
gr = load(f"{AN}/gate_readout.json")
for d in (an, cv):
    assert d["label"].startswith("EXPLORATORY")
grid = cv["v_grid"]
reg = cv["curves"]["train_region"]
mech = an["part4_mechanism"]
beh = mech["behaviour_at_training_states"]
slp = mech["slopes"]["gapped"]["throttle1_0.80-1.00"]
vd = mech["value_differences"]["recorded_vs_stall_aware"]
h2 = mech["h2_points"]["strong_current"]
gapped_sc = gr["actors"]["harm_return:gapped"]["strong_current"]
anatomy = dict(
    critic_090=reg["controller|gapped"][grid.index(0.9)], truth_090=reg["controller|truth"][grid.index(0.9)],
    critic_100=reg["controller|gapped"][grid.index(1.0)], truth_100=reg["controller|truth"][grid.index(1.0)],
    full_critic_100=reg["controller|full"][grid.index(1.0)],
    slope_gapped=pooled_stat(slp["dQ_dthrottle1_gapped"]), slope_true=pooled_stat(slp["dQ_dthrottle1_true"]),
    slope_share_rows=pooled_stat(slp["share_rows"]),
    past80_region_gapped=pooled_stat(beh["gapped"]["share_past80_region"]),
    past80_region_full=pooled_stat(beh["full"]["share_past80_region"]),
    throttle1_region_recorded=pooled_stat(beh["recorded"]["throttle1_mean_region"]),
    throttle1_region_gapped_actor=pooled_stat(beh["gapped"]["throttle1_mean_region"]),
    recorded_abs_error_gapped=pooled_stat(vd["gapped_error_abs_mean"]),
    recorded_abs_error_full=pooled_stat(vd["full_error_abs_mean"]),
    strong_gapped=pooled_stat(h2["gapped"]), strong_full=pooled_stat(h2["full"]),
    strong_gapped_capped=gapped_sc["capped_points"],
    strong_past80_gapped=gapped_sc["unrecorded_throttle_share"],
)
anatomy["harm"] = anatomy["strong_full"]["mean"] - anatomy["strong_gapped"]["mean"]
assert abs(anatomy["harm"] - mk["recovery_harm"]) < 0.05, (anatomy["harm"], mk["recovery_harm"])
assert abs(anatomy["strong_gapped"]["mean"] - stall["none"]["strong_current"]["mean"]) < 1e-9

# ------------------------------------------------------------------------------------------------ status only

pre = load(f"{ST2}/prereg.json")
trace = load(f"{TR}/spec.json")
assert trace["label"].startswith("EXPLORATORY")
status = dict(
    stage2_label=re.sub(r"(\d)-(\d)", "\\1\u2013\\2", pre["label"]),  # seeds 0-4 shown with an en dash
    stage2_version=pre["version"],
    stage2_explore_seeds=re.search(r"exploratory on seeds (\d+-\d+)", pre["label"]).group(1).replace("-", "–"),
    stage2_confirm_seeds=re.search(r"confirmatory on fresh seeds (\d+-\d+)", pre["label"]).group(1).replace("-", "–"),
    stage2_approved=os.path.exists(f"{ST2}/approval.json"),
    flags_claim=pre["claims"]["flags_claim"],
    trace_share_lo=min(trace["shares"]), trace_share_hi=max(trace["shares"]),
    trace_seeds_lo=min(trace["seeds"]), trace_seeds_hi=max(trace["seeds"]),
    trace_harm_mark=num(r"is at least (\d+) \(every seed", trace["questions"]["Q1"]),
    trace_selectivity_mark=num(r"selectivity of at least ([\d.]+)", trace["questions"]["Q2"]),
)


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


def stage2_pin_differences(pre):
    """Compare the pins prereg.json records with the files as they are now, the way approve.py's check
    (prereg_payload.check_pins) does, without importing or running any Stage 2 code. Returns the pins that differ."""
    pins, bad = pre["pins"], []
    if pins["room"]["manifest_sha256"] != sha_file(f"{BCA}/runs/shift_explore/room/code_manifest.json"):
        bad.append("room manifest")
    if pins["room"]["spec_sha256"] != sha_file(f"{BCA}/runs/shift_explore/room/spec.json"):
        bad.append("room spec")
    if pins["room"]["harness_sha256"] != sha_file(pins["room"]["harness_copy"]):
        bad.append("room harness")
    if pins["fallback_py"]["sha256"] != sha_file(pins["fallback_py"]["path"]):
        bad.append("fallback.py")
    sm = f"{BCA}/runs/shift_explore/standins/code_manifest.json"
    if pins["standins_manifest_sha256"] != sha_file(sm):
        bad.append("stand-ins manifest")
    else:
        man = json.load(open(sm))
        if pins["standins_marks_spec_sha256"] != man.get("spec_sha256"):
            bad.append("stand-ins spec")
        if not (pins["standins_py"]["sha256"] == man["files"].get("standins.py") == sha_file(pins["standins_py"]["path"])):
            bad.append("standins.py")
    dm = f"{BCA}/runs/shift_explore/detect/code_manifest.json"
    if pins["detect_manifest_sha256"] != sha_file(dm):
        bad.append("detection manifest")
    else:
        man = json.load(open(dm))
        mod = pins["detect_module"]
        if not (mod["sha256"] == man["files"].get(mod["module"]) == sha_file(mod["path"])):
            bad.append("detection script")
    if pre["code"]["code_manifest_sha256"] != sha_file(f"{ST2}/code_manifest.json"):
        bad.append("Stage 2 code manifest")
    return bad


def tracked(rel):
    """True when git tracks this repository path (read-only query)."""
    out = subprocess.run(["git", "-C", BCA, "ls-files", "--error-unmatch", rel], capture_output=True, text=True)
    return out.returncode == 0


def last_commit(rel):
    """The short id of the last commit that changed this repository path (read-only query); '' when none."""
    out = subprocess.run(["git", "-C", BCA, "log", "-1", "--format=%h", "--", rel], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def pushed(commit):
    """True when the commit is on a remote-tracking branch (read-only query)."""
    out = subprocess.run(["git", "-C", BCA, "branch", "-r", "--contains", commit], capture_output=True, text=True)
    return out.returncode == 0 and bool(out.stdout.strip())


def finished(path, kind):
    """A component's final pooled or summary file exists and matches its sidecar. Only its kind is read: no value."""
    if not os.path.exists(path):
        return False
    side = path + ".sha256"
    assert os.path.exists(side) and open(side).read().split()[0] == sha_file(path), f"{path}: sidecar mismatch"
    got = json.load(open(path)).get("kind")
    assert got == kind, (path, got)
    return True


# the method the page's "What must change in BCA" shows: the paired penalty is the main version, the plan's unpaired
# lower edge a declared secondary arm, and "cannot certify" acts per command only in the weighted version
pm = pre["method"]
assert pm["primary_penalty"].startswith("P(s_i, a) = [ U(s_i, a) - U(s_i, a_i) ]_+"), pm["primary_penalty"]
assert pm["unpaired_secondary_arm"].startswith("Q1(s_i, pi(s_i)) - k * U(s_i, pi(s_i))"), pm["unpaired_secondary_arm"]
assert "only in the weighted version" in " ".join(pm["what_changes"]), pm["what_changes"]
assert "inert" in pm["cannot_certify"]["uniform"], pm["cannot_certify"]
status["stage2_k"] = pm["k"]["primary"]
status["stage2_k_secondary"] = pm["k"]["declared_secondary"]  # the declared secondary arm

REPO_ST2 = "outputs/shift/beyond-the-data/stage2"
# what Stage 2's run folder holds before any run: the pre-registration, the code and its manifest, their histories,
# and the imported approval
ST2_BEFORE_RUNS = {"prereg.json", "prereg.json.sha256", "prereg_history", "code", "code_manifest.json",
                   "code_manifest.json.sha256", "code_manifest_history", "approval.json", "approval.json.sha256"}
st2_bad = stage2_pin_differences(pre)
st2_repo_prereg = f"{BCA}/{REPO_ST2}/prereg.json"
SI = f"{BCA}/runs/shift_explore/standins"
facts = dict(
    # Stage 2 (the flags claim): approvable only once its pins verify, it is committed in the folder the user
    # approves from (prereg.json approval.local_terminal), and the user's approval file arrives there
    stage2_approvable=bool(pre.get("approvable")) and not pre.get("missing_pins"),
    stage2_pins_verify=not st2_bad,
    stage2_committed=(os.path.exists(st2_repo_prereg) and tracked(f"{REPO_ST2}/prereg.json")
                      and sha_file(st2_repo_prereg) == sha_file(f"{ST2}/prereg.json")),
    stage2_approved=(os.path.exists(f"{ST2}/approval.json")
                     or (os.path.isdir(f"{BCA}/{REPO_ST2}")
                         and any(f.startswith("approval_") for f in os.listdir(f"{BCA}/{REPO_ST2}")))),
    # the three exploratory pieces ahead of Stage 2, and the trace test bed: finished or not (no value is read)
    detect_finished=finished(f"{BCA}/runs/shift_explore/detect/pooled.json", "detect-pooled"),
    standins_finished=os.path.exists(f"{BCA}/runs/shift_explore/standins/marks.json"),
    deploy_finished=os.path.exists(f"{BCA}/runs/shift_explore/deploy/summary.json"),
    trace_finished=os.path.exists(f"{TR}/pooled.json"),
    # the data-only stand-ins: launched (their launcher has logged that it waits for the CPUs it needs), started (it
    # has logged starting its lanes), and whether the user has made the choice of mark forms their marks need
    standins_launched=os.path.exists(f"{SI}/logs/launch_all.log"),
    standins_started=(os.path.exists(f"{SI}/logs/launch_all.log")
                      and "starting lanes" in open(f"{SI}/logs/launch_all.log").read()),
    standins_choice_made=os.path.exists(f"{SI}/approval.json"),
    # the launcher logs "lanes done" once all five seeds have trained and been scored; with the user's choice present
    # it then runs the marks look itself (launch_all.sh)
    standins_lanes_done=(os.path.exists(f"{SI}/logs/launch_all.log")
                         and "lanes done" in open(f"{SI}/logs/launch_all.log").read()),
    standins_choice_forms=(json.load(open(f"{SI}/approval.json"))["mark_forms"] if os.path.exists(f"{SI}/approval.json")
                           else None),
    # Stage 2 after the approval: the import record (written here by 'approve.py import' and committed) and whether
    # any Stage 2 run has written anything (every name in its run folder other than those written before any run)
    stage2_imported=(os.path.exists(f"{ST2}/approval.json") and os.path.isdir(f"{BCA}/{REPO_ST2}")
                     and any(f.startswith("import_") and f.endswith(".json") and tracked(f"{REPO_ST2}/{f}")
                             for f in os.listdir(f"{BCA}/{REPO_ST2}"))),
    stage2_run_started=any(n not in ST2_BEFORE_RUNS for n in os.listdir(ST2)),
)
status.update(facts, stage2_pins_differ=st2_bad)
# the committed Stage 2 pre-registration: the commit that holds it and its fingerprint (the SHA-256 of prereg.json)
if facts["stage2_committed"]:
    status["stage2_commit"] = last_commit(f"{REPO_ST2}/prereg.json")
    status["stage2_sha256"] = sha_file(st2_repo_prereg)
    assert status["stage2_commit"] and pushed(status["stage2_commit"]), "the Stage 2 commit is not on the remote"
    assert open(st2_repo_prereg + ".sha256").read().split()[0] == status["stage2_sha256"]


def utc_words(stamp):
    """'2026-10-13T08:32:23Z' or '2026-10-13T08:36:28+0000' -> '13 October 2026 at 08:32 UTC'"""
    m = re.fullmatch(r"(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):\d\d(?:Z|\+0000|\+00:00)", stamp)
    assert m, stamp
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    return f"{int(m.group(3))} {months[int(m.group(2)) - 1]} {m.group(1)} at {m.group(4)}:{m.group(5)} UTC"


# the user's approval of Stage 2 and its import: when and in which commits (status only; read-only git queries)
if facts["stage2_imported"]:
    pfx = status["stage2_sha256"][:16]
    ap_rel, im_rel = f"{REPO_ST2}/approval_{pfx}.json", f"{REPO_ST2}/import_{pfx}.json"
    ap, im, rs = load(f"{BCA}/{ap_rel}"), load(f"{BCA}/{im_rel}"), load(f"{ST2}/approval.json")
    assert ap["prereg_sha256"] == im["prereg_sha256"] == rs["prereg_sha256"] == status["stage2_sha256"]
    assert ap["prereg_version"] == im["prereg_version"] == status["stage2_version"]
    assert ap["mode"] == rs["mode"] == "local-terminal" and ap["typed"] == pfx
    status["stage2_approval_commit"] = last_commit(ap_rel)
    status["stage2_import_commit"] = last_commit(im_rel)
    assert im["commit"].startswith(status["stage2_approval_commit"]), (im["commit"], status["stage2_approval_commit"])
    assert pushed(status["stage2_approval_commit"]) and pushed(status["stage2_import_commit"])
    status["stage2_approved_at"] = utc_words(ap["approved_at_utc"])
    status["stage2_imported_at"] = utc_words(im["imported_at"])
# the data-only stand-ins: when their lanes started and on which seeds and CPUs (the launcher's own log), and the
# user's choice of their mark forms (status only: no run output is read)
if facts["standins_started"]:
    lg = open(f"{SI}/logs/launch_all.log").read()
    m = re.findall(r"^(\S+Z) starting lanes A \(seeds ([\d ]+), cpu (\d+)\) and B \(seeds ([\d ]+), cpu (\d+)\)$", lg, re.M)
    assert len(m) == 1, m  # one start only: a second launch would be a duplicate run
    t0, sa, ca, sb, cb = m[0]
    status["standins_started_at"] = utc_words(t0)
    status["standins_started_time"] = t0[11:19]
    status["standins_lane_a"] = dict(seeds=", ".join(sa.split()), cpu=int(ca))
    status["standins_lane_b"] = dict(seeds=", ".join(sb.split()), cpu=int(cb))
    assert sorted(int(x) for x in (sa + " " + sb).split()) == seeds
# the stand-ins' design, for the stand-ins box (status only: the spec, its fingerprints and its marks; no run output)
si_spec = load(f"{SI}/spec.json")
assert si_spec["label"].startswith("EXPLORATORY") and si_spec["seeds"] == seeds
si_cap = si_spec["marks"]["captures_the_room"]
standins = dict(
    version=si_spec["version"], spec_sha256=SOURCES[os.path.relpath(f"{SI}/spec.json", BCA)],
    manifest_sha256=sha_file(f"{SI}/code_manifest.json"),
    previous_spec_sha256=sha_file(f"{SI}/{si_spec['previous_spec']['archived']}"),
    previous_manifest_sha256=sha_file(f"{SI}/superseded/v2/code_manifest.json"),
    standins_py_sha256=pre["pins"]["standins_py"]["sha256"],
    n_penalties=len(si_spec["standins"]), seeds_lo=min(si_spec["seeds"]), seeds_hi=max(si_spec["seeds"]),
    share_of_oracle=num(r"at least (\d+)% of the perfect band's gain", si_cap) / 100,
    recorded_margin=num(r"at least no caution - (\d+) in recorded", si_cap, int),
    even_margin=num(r"beats the matched even pull by at least (\d+) points", si_cap, int),
)
assert open(f"{SI}/code_manifest.json.sha256").read().split()[0] == standins["manifest_sha256"]
assert si_spec["previous_spec"]["archived"] == "spec_history/spec_v2.json" and standins["version"] == 3
assert pre["pins"]["standins_marks_spec_sha256"] == standins["spec_sha256"]
assert pre["pins"]["standins_manifest_sha256"] == standins["manifest_sha256"]
standins.update(spec_prefix=standins["spec_sha256"][:16], spec_short=standins["spec_sha256"][:8],
                manifest_short=standins["manifest_sha256"][:8], previous_spec_short=standins["previous_spec_sha256"][:8],
                previous_manifest_short=standins["previous_manifest_sha256"][:8],
                standins_py_short=standins["standins_py_sha256"][:8])
status["standins"] = standins
if facts["standins_choice_made"]:
    ch = load(f"{SI}/approval.json")
    assert ch["kind"] == "standins-approval" and ch["marks_look_run"] is False
    words = {"re_referenced": "approve F1-F3", "literal": "decline F1-F3"}[ch["mark_forms"]]  # approve.py's phrases
    assert words in ch["message_verbatim"] and ch["message_verbatim"].count("F1-F3") == 1, ch["message_verbatim"]
    status["standins_choice_at"] = utc_words(ch["approved_at"])
    status["standins_choice_words"] = words
    status["standins_choice_mode"] = ch["mode"]
    # the box says no run output existed when the choice was recorded
    assert ch["run_outputs_at_approval"] == [], ch["run_outputs_at_approval"]
    assert ch["spec_sha256"] == standins["spec_sha256"] and ch["code_manifest_sha256"] == standins["manifest_sha256"]
    assert standins["spec_prefix"] in ch["message_verbatim"] and "accept chat mode" in ch["message_verbatim"]
# The page's status text (the top of the page, the options list, the claim cards, the Version History and the
# stand-ins box, id "standins-ready") is written for exactly this state. Any change stops the build until that text
# is rewritten and this line updated with it.
PAGE_STATE = dict(stage2_approvable=True, stage2_pins_verify=True, stage2_committed=True, stage2_approved=True,
                  stage2_imported=True, stage2_run_started=False,
                  detect_finished=True, standins_finished=False, deploy_finished=True, trace_finished=True,
                  standins_launched=True, standins_started=True, standins_lanes_done=False, standins_choice_made=True,
                  standins_choice_forms="re_referenced")
changed = {k: (PAGE_STATE[k], facts[k]) for k in PAGE_STATE if facts[k] != PAGE_STATE[k]}
if changed:
    raise SystemExit("the state behind the page's status text changed (fact: (page text, now)): "
                     f"{changed}; pins that differ now: {st2_bad}. Rewrite the status text, then PAGE_STATE.")

out = dict(
    label="EXPLORATORY: outside the pre-registered gate; no BCA band, calibrator, threshold, dose or BCA update was run",
    design=design, marks=mk, check_e=check_e, pooled=pooled_marks, derived=derived,
    stall=stall, stall_past80=stall_past80, no_stall=no_stall, no_stall_past80=no_stall_past80,
    no_stall_cost=no_stall_cost, replication=replication, replication_past80=replication_past80,
    deploy=deploy, anatomy=anatomy, status=status, sources=SOURCES,
)
json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)
print(f"wrote {OUT}: {len(SOURCES)} source files, all sidecars matched")
print(f"  strong current, oracle {stall['perfect_band']['strong_current']['mean']:.1f} "
      f"({stall['perfect_band']['strong_current']['lo']:.1f} to {stall['perfect_band']['strong_current']['hi']:.1f}), "
      f"none {stall['none']['strong_current']['mean']:.1f}, clip {stall['range_hinge']['strong_current']['mean']:.1f}")
