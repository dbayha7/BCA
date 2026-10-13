"""EXPLORATORY (plan section 4.3): pool the deploy jobs over gate seeds 0-4 and read the mark fixed in the spec.

Refuses until every job of the spec (5 seeds x 5 actor-world jobs) has a labelled values file for the current spec and
code manifest, or a stop file. Means are over seeds of differences paired by seed (and, within a seed, computed on the
same 200 starts); "se" is the standard error over seeds (ddof 1); "positive" counts seeds with a difference above 0.

Mark, "the channel is open" (spec marks.channel_open): on the gapped critic's actor in the stalling world with the clip
fallback, (i) the perfect detector beats random switching at its own rate by at least 15 points (pooled mean) in strong
current or in big kicks, and (ii) its cost in recorded conditions (never switching minus the perfect detector) is at
most 3 points. Every seed must have its values. The same reading is reported for plain TD3+BC and for every trigger.

Usage: python summarise.py   (writes runs/shift_explore/deploy/summary.json and tables.md, with sidecars)
"""

import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
SITUATIONS = ("recorded", "strong_current", "big_waves", "big_kicks")
SIT_LABEL = dict(recorded="Recorded conditions", strong_current="Strong current", big_waves="Big waves",
                 big_kicks="Big kicks")
JOBS = (("stall", "gapped"), ("stall", "td3bc_gapped"), ("no_stall", "no_stall_gapped"),
        ("no_stall", "td3bc_gapped"), ("no_stall", "gapped"))
TRIGGERS = ("unrec", "onestep", "density", "ratio", "heads", "onestep_lit", "onestep_clone", "ratio_refuse", "sa95")
FALLBACKS = ("clip99", "clone")
TRIG_LABEL = dict(unrec="perfect detector of past-80% proposals", onestep="one-step oracle (re-referenced)",
                  density="behaviour density", ratio="classifier ratio", heads="two-head disagreement",
                  onestep_lit="one-step oracle, plan's literal |delta*| (reported)",
                  onestep_clone="one-step oracle re-referenced to the cloned actor (reported)",
                  ratio_refuse="classifier ratio at the refusal limit l* (reported)",
                  sa95="distance check (carried over, reported)")
FB_LABEL = dict(clip99="clip to recorded range", clone="cloned actor")
JOB_LABEL = {("stall", "gapped"): "gapped critic's actor, stalling world",
             ("stall", "td3bc_gapped"): "plain TD3+BC, stalling world",
             ("no_stall", "no_stall_gapped"): "gapped critic's actor of the no-stall world, no-stall world",
             ("no_stall", "td3bc_gapped"): "plain TD3+BC (stalling-world actor), no-stall world",
             ("no_stall", "gapped"): "gapped critic's actor (stalling-world actor), no-stall world"}
PAIRS = {"gapped critic's actor, each world's own": (("stall", "gapped"), ("no_stall", "no_stall_gapped")),
         "gapped critic's actor, the same actor in both worlds": (("stall", "gapped"), ("no_stall", "gapped")),
         "plain TD3+BC, the same actor in both worlds": (("stall", "td3bc_gapped"), ("no_stall", "td3bc_gapped"))}
MARK = dict(job=("stall", "gapped"), trigger="unrec", fallback="clip99", min_gain_over_random=15.0,
            situations_any=("strong_current", "big_kicks"), max_recorded_cost=3.0)
STAGE2_BAR = dict(min_gain_over_random=10.0, max_recorded_cost=2.0)


def load(spec_sha, man_sha):
    jobs, status = {}, {}
    for rep in SEEDS:
        for world, actor in JOBS:
            name = f"seed{rep}_{world}_{actor}"
            v = C.read_json(C.OUT / f"{name}.json")
            stop = C.read_json(C.OUT / f"stop_{name}.json")
            if v is not None:
                if (v.get("kind") != "deploy-values" or not str(v.get("label", "")).startswith("EXPLORATORY")
                        or v.get("spec_sha256") != spec_sha or v.get("code_manifest_sha256") != man_sha
                        or v.get("n_starts") != 200 or sorted(v["situations"]) != sorted(SITUATIONS)):
                    raise SystemExit(f"refused: {name}.json is not a complete values file of this spec and manifest")
                jobs[(rep, world, actor)] = v
                status[name] = "values"
            elif stop is not None:
                if stop.get("spec_sha256") != spec_sha:
                    raise SystemExit(f"refused: stop_{name}.json names another spec")
                jobs[(rep, world, actor)] = None
                status[name] = "stopped"
            else:
                raise SystemExit(f"refused: {name} has neither values nor a stop file yet")
    return jobs, status


def rule(v, s, k):
    return None if v is None else v["situations"][s]["rules"][k]


def pts(v, s, k):
    r = rule(v, s, k)
    return None if r is None else r["points"]


def diff(a, b):
    return None if a is None or b is None else a - b


def plus(a, b):
    return None if a is None or b is None else a + b


def per_seed(jobs, world, actor, f):
    return [f(jobs[(rep, world, actor)]) for rep in SEEDS]


def trigger_table(jobs, world, actor, trig, fb):
    k, kr = f"{trig}:{fb}", f"random:{trig}:{fb}"
    out = {}
    for s in SITUATIONS:
        out[s] = dict(
            points=C.mean_se(per_seed(jobs, world, actor, lambda v: pts(v, s, k))),
            change_vs_none=C.mean_se(per_seed(jobs, world, actor, lambda v: diff(pts(v, s, k), pts(v, s, "none:None")))),
            random_change_vs_none=C.mean_se(per_seed(jobs, world, actor,
                                                     lambda v: diff(pts(v, s, kr), pts(v, s, "none:None")))),
            minus_random=C.mean_se(per_seed(jobs, world, actor, lambda v: diff(pts(v, s, k), pts(v, s, kr)))),
            fire_share=C.mean_se(per_seed(jobs, world, actor, lambda v: None if v is None else rule(v, s, k)["fire_share"])),
            changed_share=C.mean_se(per_seed(jobs, world, actor,
                                             lambda v: None if v is None else rule(v, s, k)["changed_share"])),
            recall_past80=C.mean_se(per_seed(jobs, world, actor,
                                             lambda v: None if v is None else rule(v, s, k)["recall_past80"])),
            precision_past80=C.mean_se(per_seed(jobs, world, actor,
                                                lambda v: None if v is None else rule(v, s, k)["precision_past80"])),
            flag_rate_not_past80=C.mean_se(per_seed(jobs, world, actor,
                                                    lambda v: None if v is None else rule(v, s, k)["flag_rate_not_past80"])))
    cost = C.mean_se(per_seed(jobs, world, actor, lambda v: diff(pts(v, "recorded", "none:None"), pts(v, "recorded", k))))
    gains = {s: out[s]["minus_random"]["mean"] for s in MARK["situations_any"]}
    complete = all(jobs[(rep, world, actor)] is not None for rep in SEEDS)

    def bar(b):
        return bool(complete and cost["mean"] is not None and cost["mean"] <= b["max_recorded_cost"]
                    and any(g is not None and g >= b["min_gain_over_random"] for g in gains.values()))
    return dict(situations=out, recorded_cost=cost, complete=complete,
                channel_open_bar=bar(MARK), stage2_bar=bar(STAGE2_BAR))


def selectivity(jobs, pair, trig, fb):
    (ws, as_), (wn, an) = pair
    k, kr = f"{trig}:{fb}", f"random:{trig}:{fb}"
    out = {}
    for s in SITUATIONS:
        def both(f):
            return [plus(f(jobs[(rep, ws, as_)]), f(jobs[(rep, wn, an)])) for rep in SEEDS]
        out[s] = dict(
            stall_gain_minus_no_stall_loss=C.mean_se(both(lambda v: diff(pts(v, s, k), pts(v, s, "none:None")))),
            targeting_sum_over_worlds=C.mean_se(both(lambda v: diff(pts(v, s, k), pts(v, s, kr)))),
            clip_every_step_reference=C.mean_se(both(lambda v: diff(pts(v, s, f"always:{fb}"), pts(v, s, "none:None")))),
            catch_rate_stall_minus_no_stall=C.mean_se(
                [diff(None if jobs[(rep, ws, as_)] is None else rule(jobs[(rep, ws, as_)], s, k)["recall_past80"],
                      None if jobs[(rep, wn, an)] is None else rule(jobs[(rep, wn, an)], s, k)["recall_past80"])
                 for rep in SEEDS]))
    return out


def f1(x, se=None):
    if x is None:
        return "-"
    return f"{x:+.1f}" if se is None else f"{x:+.1f} ± {se:.1f}"


def tables(summary):
    md = ["# Deployment-check channel (plan 4.3): pooled tables", "",
          "EXPLORATORY. No training and no BCA: every trigger is a stand-in or an oracle. Saved check-2 actors, exact "
          "simulation, the first 200 starts of each evaluation bank, gate seeds 0-4. Differences are paired by seed; "
          "± is the standard error over seeds.", ""]
    m = summary["mark"]
    md += ["## The mark: is the channel open?", "",
           f"Perfect detector minus random switching at its own rate (gapped critic's actor, stalling world, clip "
           f"fallback): strong current {f1(m['gain_over_random']['strong_current']['mean'], m['gain_over_random']['strong_current']['se'])}, "
           f"big kicks {f1(m['gain_over_random']['big_kicks']['mean'], m['gain_over_random']['big_kicks']['se'])} "
           f"(need at least +15 in one). Cost in recorded conditions: "
           f"{f1(m['recorded_cost']['mean'], m['recorded_cost']['se'])} (need at most 3). "
           f"**The channel is {'open' if m['passed'] else 'not shown to be open'}.**", ""]
    for (world, actor) in JOBS:
        jl = JOB_LABEL[(world, actor)]
        for fb in FALLBACKS:
            md += [f"## {jl}: fallback {FB_LABEL[fb]}", "",
                   "Each cell: the trigger minus random switching at its own rate (points); in brackets the change from "
                   "the actor alone. Last columns: cost in recorded conditions (actor alone minus trigger), share of "
                   "past-80% proposals caught in strong current, share of steps switched in strong current.", "",
                   "| Trigger | " + " | ".join(SIT_LABEL[s] for s in SITUATIONS) + " | Recorded cost | Caught (strong) | "
                   "Switched (strong) |", "|---" * (len(SITUATIONS) + 4) + "|"]
            for trig in TRIGGERS:
                t = summary["triggers"][f"{world}/{actor}"].get(f"{trig}:{fb}")
                if t is None:
                    continue
                cells = []
                for s in SITUATIONS:
                    c = t["situations"][s]
                    cells.append(f"{f1(c['minus_random']['mean'], c['minus_random']['se'])} "
                                 f"({f1(c['change_vs_none']['mean'])})")
                sc = t["situations"]["strong_current"]
                rc = sc["recall_past80"]["mean"]
                fs = sc["fire_share"]["mean"]
                md.append(f"| {TRIG_LABEL[trig]} | " + " | ".join(cells) + f" | {f1(t['recorded_cost']['mean'])} | "
                          f"{'-' if rc is None else f'{rc:.2f}'} | {'-' if fs is None else f'{fs:.2f}'} |")
            md.append("")
    md += ["## Selectivity between the worlds", "",
           "Stalling-world gain minus no-stall loss: (trigger minus actor alone) summed over the two worlds, strong "
           "current, clip fallback. 'Targeting' sums trigger minus random. 'Clip every step' is the same sum for the "
           "plain range limit. Catch-rate selectivity: share of past-80% proposals caught, stalling minus no-stall.", "",
           "| Pair | Trigger | Gain minus loss | Targeting | Clip every step | Catch-rate selectivity |", "|---|---|---|---|---|---|"]
    for pname, per in summary["selectivity"].items():
        for trig in TRIGGERS:
            v = per.get(f"{trig}:clip99")
            if v is None:
                continue
            c = v["strong_current"]
            cr = c["catch_rate_stall_minus_no_stall"]["mean"]
            md.append(f"| {pname} | {TRIG_LABEL[trig]} | {f1(c['stall_gain_minus_no_stall_loss']['mean'])} | "
                      f"{f1(c['targeting_sum_over_worlds']['mean'])} | {f1(c['clip_every_step_reference']['mean'])} | "
                      f"{'-' if cr is None else f'{cr:+.2f}'} |")
    md.append("")
    return "\n".join(md)


def main():
    spec, spec_sha = C.load_spec()
    C.check_pins(spec)
    man, man_sha = C.check_code(spec_sha)
    jobs, status = load(spec_sha, man_sha)
    trig = {}
    for world, actor in JOBS:
        d = {}
        for t in TRIGGERS:
            for fb in (("clip99",) if t == "sa95" else FALLBACKS):
                d[f"{t}:{fb}"] = trigger_table(jobs, world, actor, t, fb)
        for fb in FALLBACKS:
            d[f"always:{fb}"] = dict(situations={s: dict(change_vs_none=C.mean_se(per_seed(
                jobs, world, actor, lambda v: diff(pts(v, s, f"always:{fb}"), pts(v, s, "none:None"))))) for s in SITUATIONS})
        d["none:None"] = dict(situations={s: dict(points=C.mean_se(per_seed(jobs, world, actor,
                                                                            lambda v: pts(v, s, "none:None"))))
                                          for s in SITUATIONS})
        trig[f"{world}/{actor}"] = d
    sel = {p: {f"{t}:{fb}": selectivity(jobs, pair, t, fb) for t in TRIGGERS
               for fb in (("clip99",) if t == "sa95" else FALLBACKS)} for p, pair in PAIRS.items()}
    mt = trig["{}/{}".format(*MARK["job"])][f"{MARK['trigger']}:{MARK['fallback']}"]
    gains = {s: mt["situations"][s]["minus_random"] for s in MARK["situations_any"]}
    mark = dict(rule=spec["marks"]["channel_open"], gain_over_random=gains, recorded_cost=mt["recorded_cost"],
                complete=mt["complete"], passed=mt["channel_open_bar"],
                same_reading_td3bc=trig["stall/td3bc_gapped"]["unrec:clip99"]["channel_open_bar"])
    out = dict(label=C.LABEL, kind="deploy-summary", spec_sha256=spec_sha, code_manifest_sha256=man_sha,
               seeds=list(SEEDS), job_status=status, mark=mark, triggers=trig, selectivity=sel,
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    C.write_json(C.OUT / "summary.json", out)
    text = tables(out)
    p = C.guard_out(C.OUT / "tables.md")
    p.write_text(text, encoding="utf8")
    p.with_name("tables.md.sha256").write_text(f"{C.sha256_bytes(text.encode('utf8'))}  tables.md\n", encoding="utf8")
    C.log(f"channel open: {mark['passed']}; wrote summary.json and tables.md")


if __name__ == "__main__":
    main()
