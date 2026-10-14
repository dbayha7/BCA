"""Pool the read-only consumer outputs over seeds 0-4 (no simulation here; reads this folder's JSON and the saved
Stage 2 values.json files). Writes summary.json."""
import json
from pathlib import Path

import numpy as np

OUT = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/restructure/consumer")
R2 = Path("/home/user/bca/runs/shift_explore/stage2")
SEEDS = range(5)
SITS = ("strong_current", "recorded", "big_waves", "big_kicks")


def load(name):
    return [json.loads((OUT / name.format(k=k)).read_text()) for k in SEEDS]


def m(v):
    return float(np.mean(v))


def main():
    S = {}
    dep = load("deploy_seed{k}.json")
    ext = load("deploy_extra_seed{k}.json") if all((OUT / f"deploy_extra_seed{k}.json").exists() for k in SEEDS) else None
    cap = load("capsweep_seed{k}.json") if all((OUT / f"capsweep_seed{k}.json").exists() for k in SEEDS) else None
    ext2 = load("deploy_extra2_seed{k}.json") if all((OUT / f"deploy_extra2_seed{k}.json").exists() for k in SEEDS) \
        else None

    def pts(src, world, actor, sit, rule):
        return [d["worlds"][world]["actors"][actor][sit][rule]["points"] for d in src]

    def field(src, world, actor, sit, rule, f):
        return [d["worlds"][world]["actors"][actor][sit][rule][f] for d in src]

    rules = list(dep[0]["worlds"]["stall"]["actors"]["none"]["strong_current"].keys())
    rules = [r for r in rules if isinstance(dep[0]["worlds"]["stall"]["actors"]["none"]["strong_current"][r], dict)
             and "points" in dep[0]["worlds"]["stall"]["actors"]["none"]["strong_current"][r]]
    table = {}
    for actor in ("none", "bca_k1"):
        base_s = pts(dep, "stall", actor, "strong_current", "never")
        base_n = pts(dep, "no_stall", actor, "strong_current", "never")
        base_r = pts(dep, "stall", actor, "recorded", "never")
        rows = {}
        srcs = [(dep, rules)]
        if ext:
            srcs.append((ext, list(ext[0]["worlds"]["stall"]["actors"]["none"]["strong_current"].keys())))
        if ext2:
            srcs.append((ext2, list(ext2[0]["worlds"]["stall"]["actors"]["none"]["strong_current"].keys())))
        if cap:
            srcs.append((cap, list(cap[0]["worlds"]["stall"]["actors"]["none"]["strong_current"].keys())))
        for src, rl in srcs:
            for rule in rl:
                s_sc = pts(src, "stall", actor, "strong_current", rule)
                n_sc = pts(src, "no_stall", actor, "strong_current", rule)
                row = dict(stall_sc=m(s_sc), stall_sc_seeds=[round(x, 1) for x in s_sc],
                           gain_sc=m(np.subtract(s_sc, base_s)),
                           nostall_sc=m(n_sc), nostall_cost_sc=m(np.subtract(base_n, n_sc)),
                           net=m(np.subtract(s_sc, base_s)) - m(np.subtract(base_n, n_sc)))
                for sit in ("recorded", "big_waves", "big_kicks"):
                    row[f"stall_{sit}"] = m(pts(src, "stall", actor, sit, rule))
                    row[f"nostall_{sit}"] = m(pts(src, "no_stall", actor, sit, rule))
                if src is not cap:
                    row["fire_stall_sc"] = m(field(src, "stall", actor, "strong_current", rule, "fire_share"))
                    row["fire_nostall_sc"] = m(field(src, "no_stall", actor, "strong_current", rule, "fire_share"))
                    row["fire_stall_rec"] = m(field(src, "stall", actor, "recorded", rule, "fire_share"))
                    if "recall_past80" in src[0]["worlds"]["stall"]["actors"][actor]["strong_current"][rule]:
                        row["recall80_stall_sc"] = m(field(src, "stall", actor, "strong_current", rule, "recall_past80"))
                        row["ff_stall_sc"] = m(field(src, "stall", actor, "strong_current", rule,
                                                     "flag_rate_not_past80"))
                    if "ge95_applied" in src[0]["worlds"]["stall"]["actors"][actor]["strong_current"][rule]:
                        row["ge95_applied_stall_sc"] = m(field(src, "stall", actor, "strong_current", rule,
                                                               "ge95_applied"))
                        row["past80_applied_stall_sc"] = m(field(src, "stall", actor, "strong_current", rule,
                                                                 "past80_applied"))
                rows[rule] = row
        # minus random at own rate (strong current, stall), per seed
        for rule, rnd, src in (("project_clamped", "random:project_clamped", dep),
                               ("refuse_clamped", "random:project_clamped", dep),
                               ("tau_raw", None, dep)):
            if rnd:
                d = np.subtract(pts(src, "stall", actor, "strong_current", rule),
                                pts(src, "stall", actor, "strong_current", rnd))
                rows[rule]["minus_random_stall_sc"] = m(d)
                rows[rule]["minus_random_stall_sc_seeds"] = [round(x, 1) for x in d]
                rows[rule]["minus_random_nostall_sc"] = m(np.subtract(
                    pts(src, "no_stall", actor, "strong_current", rule),
                    pts(src, "no_stall", actor, "strong_current", rnd)))
            rows[rule]["recorded_change"] = m(np.subtract(pts(src, "stall", actor, "recorded", rule), base_r))
        for src_x, rl_x in ((ext, ("tau_clamped", "refuse_raw", "uniform_tau_clamped")),
                            (ext2, ("plain_refuse_clamped", "ratio95_clamped"))):
            if not src_x:
                continue
            for rule in rl_x:
                if rule in rows:
                    d = np.subtract(pts(src_x, "stall", actor, "strong_current", rule),
                                    pts(src_x, "stall", actor, "strong_current", "random:" + rule))
                    rows[rule]["minus_random_stall_sc"] = m(d)
                    rows[rule]["minus_random_stall_sc_seeds"] = [round(x, 1) for x in d]
                    rows[rule]["recorded_change"] = m(np.subtract(pts(src_x, "stall", actor, "recorded", rule), base_r))
                    dn = np.subtract(pts(src_x, "no_stall", actor, "strong_current", rule),
                                     pts(src_x, "no_stall", actor, "strong_current", "random:" + rule))
                    rows[rule]["minus_random_nostall_sc"] = m(dn)
        table[actor] = rows
    S["deploy_table"] = table
    # saved Stage 2 references (policy test values.json)
    ref = {}
    for k in SEEDS:
        v = json.loads((R2 / "policy" / f"seed{k}" / "values.json").read_text())
        ev = v["evaluations"]
        st = v["standin"]["arm"]
        vn = json.loads((R2 / "policy" / f"seed{k}" / "no_stall" / "values.json").read_text())["evaluations"]
        ref[k] = dict(matched_hinge_sc=ev[st]["strong_current"]["points"],
                      hand_hinge_sc=ev["range_hinge"]["strong_current"]["points"],
                      bca_k1_sc=ev["bca_k1"]["strong_current"]["points"], none_sc=ev["none"]["strong_current"]["points"],
                      perfect_sc=ev["perfect_band"]["strong_current"]["points"],
                      none_capped_sc=ev["none"]["strong_current"]["capped_points"],
                      ns_none_sc=vn["none"]["strong_current"]["points"],
                      ns_matched_sc=vn["standin_locked"]["strong_current"]["points"],
                      ns_hand_sc=vn["range_hinge"]["strong_current"]["points"],
                      ns_bca_sc=vn["bca_k1"]["strong_current"]["points"])
    S["policy_refs_per_seed"] = ref
    S["policy_refs_pooled"] = {f: m([ref[k][f] for k in SEEDS]) for f in ref[0]}
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    keys = ("stall_sc", "gain_sc", "nostall_cost_sc", "net", "recorded_change", "stall_big_kicks", "nostall_big_kicks",
            "fire_stall_sc", "minus_random_stall_sc", "minus_random_nostall_sc", "recall80_stall_sc", "ff_stall_sc", "ge95_applied_stall_sc")
    for actor, rows in table.items():
        print(f"== actor {actor}")
        print("rule".ljust(24) + " ".join(k[:12].rjust(12) for k in keys))
        for rule, row in rows.items():
            print(rule.ljust(24) + " ".join(
                (f"{row[k]:12.1f}" if k in row and not k.startswith(("fire", "recall", "ff", "ge95")) else
                 f"{row[k]:12.3f}" if k in row else " " * 12) for k in keys))
    print("policy refs pooled:", {k: round(v, 1) for k, v in S["policy_refs_pooled"].items()})


if __name__ == "__main__":
    main()
