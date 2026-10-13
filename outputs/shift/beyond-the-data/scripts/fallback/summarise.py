"""EXPLORATORY: pool the per-seed fallback outputs into tables (means over gate seeds, standard errors over seeds).

Usage: python summarise.py  (reads /home/user/bca/runs/shift_explore/fallback/seed*_*.json, writes summary.json and
tables.md next to them)."""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path("/home/user/bca/runs/shift_explore/fallback")
SIT = ("recorded", "strong_current", "big_waves", "big_kicks")
SIT_LABEL = {"recorded": "Recorded conditions", "strong_current": "Strong current", "big_waves": "Big waves",
             "big_kicks": "Big kicks"}
FB_LABEL = {None: "", "clone": "cloned actor", "clip99": "clip to recorded range", "project": "nearest recorded command",
            "clipmm": "clip to full recorded range (min to max)"}
TRIG_LABEL = {"none": "never switch (the actor alone)", "always": "every step",
              "state95": "new state, 95% level", "state99": "new state, 99% level",
              "sa95": "new (state, command) pair, 95% level", "sa99": "new (state, command) pair, 99% level",
              "local95": "command outside the commands recorded at nearby states, 95% level",
              "local99": "command outside the commands recorded at nearby states, 99% level",
              "unrec": "perfect detector: the actor proposes past 80% (reference)",
              "oracle": "exact one-step check (reference)",
              "random95": "random, at the rate of the 95% (state, command) check",
              "random99": "random, at the rate of the 99% (state, command) check",
              "rlocal95": "random, at the rate of the 95% nearby-commands check"}
ORDER = ("none:None", "always:clone", "always:clip99", "always:project", "always:clipmm",
         "state95:clone", "state95:clip99", "state95:project", "state99:clone", "state99:clip99",
         "sa95:clone", "sa95:clip99", "sa95:project", "sa99:clone", "sa99:clip99",
         "random95:clone", "random95:clip99", "random95:project", "random99:clone", "random99:clip99",
         "local95:clone", "local95:clip99", "local99:clone", "local99:clip99", "rlocal95:clone", "rlocal95:clip99",
         "unrec:clone", "unrec:clip99", "oracle:clone", "oracle:clip99")
PAIRS = (("sa95", "random95"), ("sa99", "random99"), ("local95", "rlocal95"))


def load():
    files = sorted(p for p in OUT.glob("seed*_*.json"))
    rows = {}
    for p in files:
        d = json.loads(p.read_text())
        rows[(d["world"], d["actor"], d["rep"])] = d
    return rows


def mse(v):
    v = np.asarray([x for x in v if x is not None], np.float64)
    if len(v) == 0:
        return None, None, 0
    return float(v.mean()), (float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None), int(len(v))


def label(key):
    t, f = key.split(":")
    f = None if f == "None" else f
    return TRIG_LABEL.get(t, t) + ("" if f is None else f" -> {FB_LABEL[f]}")


def fmt(m, se=None, d=1):
    if m is None:
        return "-"
    return f"{m:.{d}f}" if se is None else f"{m:.{d}f} ± {se:.{d}f}"


HEADLINE = ("none:None", "always:clone", "always:clip99", "always:clipmm", "always:project", "state95:clip99",
            "sa95:clone", "sa95:clip99", "sa95:project", "sa99:clip99", "random95:clone", "random95:clip99",
            "random99:clip99", "local95:clip99", "rlocal95:clip99", "unrec:clip99", "oracle:clone", "oracle:clip99")


def headline(summary):
    """Compact tables: change from the actor alone per situation; strong current also as a share of the harm (the
    full-log critic's actor minus the gapped critic's actor, both alone, on the same starts and seeds)."""
    md = []
    full = summary.get("stall/full")
    for key in ("stall/gapped", "stall/td3bc_gapped", "stall/full", "no_stall/no_stall_gapped"):
        v = summary.get(key)
        if v is None:
            continue
        md.append(f"## Headline: {key} (seeds {v['seeds']})\n")
        md.append("Points of the rule, and in brackets the change from the actor alone (paired by seed); strong "
                  "current also gives the share of the harm recovered; last two columns: share of steps switched in "
                  "recorded conditions / strong current.\n")
        md.append("| Rule | Recorded conditions | Strong current | Harm recovered | Big waves | Big kicks | "
                  "switched (recorded / strong) |")
        md.append("|---|---|---|---|---|---|---|")
        harm = None
        if (full is not None and key.startswith("stall/")
                and "none:None" in full["situations"]["strong_current"]["rules"]):
            g = v["situations"]["strong_current"]["rules"]["none:None"]["per_seed_points"]
            f = full["situations"]["strong_current"]["rules"]["none:None"]["per_seed_points"]
            fs = dict(zip(full["situations"]["strong_current"]["seeds"], f))
            gs = dict(zip(v["situations"]["strong_current"]["seeds"], g))
            common = sorted(set(fs) & set(gs))
            if common and key != "stall/full":
                harm = (common, float(np.mean([fs[r] - gs[r] for r in common])))
        for k in HEADLINE:
            cells = []
            r0 = v["situations"]["recorded"]["rules"].get(k)
            if r0 is None:
                continue
            for s in SIT:
                r = v["situations"][s]["rules"].get(k)
                dk = v["situations"][s]["differences"].get(f"{k} minus none")
                if r is None:
                    cells.append("-")
                else:
                    cells.append(f"{r['points'][0]:.1f} ({dk[0]:+.1f})" if dk and k != "none:None"
                                 else f"{r['points'][0]:.1f}")
                if s == "strong_current":
                    if harm and r is not None and k != "none:None":
                        seeds = v["situations"][s]["seeds"]
                        d = dict(zip(seeds, [None if (a is None or b is None) else a - b for a, b in
                                             zip(r["per_seed_points"], v["situations"][s]["rules"]["none:None"]
                                                 ["per_seed_points"])]))
                        rec = np.mean([d[x] for x in harm[0] if d.get(x) is not None])
                        cells.append(f"{100 * rec / harm[1]:.0f}%")
                    else:
                        cells.append("-")
            rs = v["situations"]["strong_current"]["rules"].get(k)
            cells.append(f"{r0['changed_share'][0]:.2f} / {rs['changed_share'][0]:.2f}" if rs else "-")
            md.append(f"| {label(k)} | " + " | ".join(cells) + " |")
        if harm:
            md.append(f"\nHarm on these starts (full-log critic's actor minus this actor, both alone, seeds "
                      f"{harm[0]}): {harm[1]:.1f} points.\n")
        md.append("")
    return md


def main():
    rows = load()
    summary = {}
    worlds_actors = sorted({(w, a) for w, a, _ in rows})
    for w, a in worlds_actors:
        seeds = sorted(r for ww, aa, r in rows if (ww, aa) == (w, a))
        S = {}
        for s in SIT:
            per = {}
            keys = set()
            for r in seeds:
                sit = rows[(w, a, r)]["situations"].get(s)
                if sit:
                    keys |= set(sit["rules"])
            for k in keys:
                vals = {m: [] for m in ("points", "fire_share", "changed_share", "past80_applied", "past80_proposed",
                                        "recall_past80", "precision_past80")}
                for r in seeds:
                    sit = rows[(w, a, r)]["situations"].get(s)
                    rr = sit["rules"].get(k) if sit else None
                    for m in vals:
                        vals[m].append(None if rr is None else rr[m])
                per[k] = {m: mse(v) for m, v in vals.items()}
                per[k]["per_seed_points"] = vals["points"]
            checks = [rows[(w, a, r)]["situations"][s]["gate_check"] for r in seeds
                      if s in rows[(w, a, r)]["situations"]]
            S[s] = dict(rules=per, seeds=[r for r in seeds if s in rows[(w, a, r)]["situations"]],
                        gate_check_max_abs_diff=max((c["max_abs_diff"] for c in checks), default=None),
                        full_bank_points=mse([c["gate_points"] for c in checks]),
                        full_bank_capped=mse([c["gate_capped"] for c in checks]))
            # paired differences by seed: targeted minus random at the same rate, rule minus actor alone
            diffs = {}
            for k in per:
                t, f = k.split(":")
                base = per.get("none:None")
                if base:
                    diffs[f"{k} minus none"] = mse([None if (x is None or y is None) else x - y for x, y in
                                                    zip(per[k]["per_seed_points"], base["per_seed_points"])])
            for t, rnd in PAIRS:
                for f in ("clone", "clip99", "project"):
                    k1, k2 = f"{t}:{f}", f"{rnd}:{f}"
                    if k1 in per and k2 in per:
                        diffs[f"{k1} minus {k2}"] = mse([None if (x is None or y is None) else x - y for x, y in
                                                         zip(per[k1]["per_seed_points"], per[k2]["per_seed_points"])])
            S[s]["differences"] = diffs
        refs = [rows[(w, a, r)]["reference"] for r in seeds]
        summary[f"{w}/{a}"] = dict(world=w, actor=a, seeds=seeds, situations=S,
                                   n_starts=sorted({rows[(w, a, r)]["n_starts"] for r in seeds}),
                                   reference=dict(
                                       state_threshold={lv: mse([x["state"]["threshold"][lv] for x in refs])
                                                        for lv in ("95", "99")},
                                       state_command_threshold={lv: mse([x["state_command"]["threshold"][lv]
                                                                         for x in refs]) for lv in ("95", "99")},
                                       local_threshold={lv: mse([x["local_command"]["threshold"][lv] for x in refs])
                                                        for lv in ("95", "99")},
                                       state_false_alarm={lv: mse([x["state"]["evaluation_split_rate_above"][lv]
                                                                   for x in refs]) for lv in ("95", "99")},
                                       state_command_false_alarm={
                                           lv: mse([x["state_command"]["evaluation_split_rate_above"][lv]
                                                    for x in refs]) for lv in ("95", "99")},
                                       local_false_alarm={lv: mse([x["local_command"]["evaluation_split_rate_above"]
                                                                   [lv] for x in refs]) for lv in ("95", "99")},
                                       clip99_low=np.mean([x["clip99"]["low"] for x in refs], 0).tolist(),
                                       clip99_high=np.mean([x["clip99"]["high"] for x in refs], 0).tolist(),
                                       clipmm_low=np.mean([x["clipmm"]["low"] for x in refs], 0).tolist(),
                                       training_rows_past_80=[x["training_rows_past_80"] for x in refs],
                                       heldout_rows=[x["heldout_rows"] for x in refs]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    md = []
    for key, v in summary.items():
        md.append(f"## {key}  (seeds {v['seeds']}, starts per bank {v['n_starts']})\n")
        md.append("Points: mean ± standard error over seeds; in brackets the share of steps the trigger fired.\n")
        md.append("| Rule | " + " | ".join(SIT_LABEL[s] for s in SIT) + " |")
        md.append("|---|" + "---|" * len(SIT))
        for k in ORDER:
            cells = []
            present = False
            for s in SIT:
                r = v["situations"][s]["rules"].get(k)
                if r is None or r["points"][0] is None:
                    cells.append("-")
                    continue
                present = True
                cells.append(f"{fmt(r['points'][0], r['points'][1])} [{r['fire_share'][0]:.2f}]")
            if present:
                md.append(f"| {label(k)} | " + " | ".join(cells) + " |")
        md.append("")
        md.append("Paired differences by seed (points, mean ± standard error):\n")
        md.append("| Difference | " + " | ".join(SIT_LABEL[s] for s in SIT) + " |")
        md.append("|---|" + "---|" * len(SIT))
        found = set()
        for s in SIT:
            found |= set(v["situations"][s]["differences"])
        dkeys = [f"{t}:{f} minus {r}:{f}" for t, r in PAIRS for f in ("clone", "clip99", "project")]
        dkeys += [f"{k} minus none" for k in ORDER if k != "none:None"]
        dkeys = [k for k in dkeys if k in found]
        for dk in dkeys:
            cells = [fmt(*v["situations"][s]["differences"].get(dk, (None, None, 0))[:2]) for s in SIT]
            md.append(f"| {dk} | " + " | ".join(cells) + " |")
        md.append("")
        md.append("Strong current: share of steps the actor proposes past 80%, share applied past 80%, and the "
                  "share of past-80% proposals the trigger caught (recall) / share of firings on past-80% proposals "
                  "(precision):\n")
        md.append("| Rule | proposed past 80% | applied past 80% | recall | precision |")
        md.append("|---|---|---|---|---|")
        for k in ORDER:
            r = v["situations"]["strong_current"]["rules"].get(k)
            if r is None or r["points"][0] is None:
                continue
            md.append(f"| {label(k)} | {r['past80_proposed'][0]:.2f} | {r['past80_applied'][0]:.2f} | "
                      f"{r['recall_past80'][0]:.2f} | {r['precision_past80'][0]:.2f} |")
        md.append("")
        md.append("Check against the gate (whole 1,000-start bank, no switching and capped at 80%): largest absolute "
                  "difference " + ", ".join(f"{s} {v['situations'][s]['gate_check_max_abs_diff']:.1e}"
                                            for s in SIT if v['situations'][s]['gate_check_max_abs_diff'] is not None)
                  + "; whole-bank points of the actor alone: "
                  + ", ".join(f"{s} {fmt(*v['situations'][s]['full_bank_points'][:2])}" for s in SIT) + "\n")
        ref = v["reference"]
        md.append(f"Thresholds (mean over seeds): state {fmt(*ref['state_threshold']['95'][:2], d=2)} / "
                  f"{fmt(*ref['state_threshold']['99'][:2], d=2)}; (state, command) "
                  f"{fmt(*ref['state_command_threshold']['95'][:2], d=2)} / "
                  f"{fmt(*ref['state_command_threshold']['99'][:2], d=2)}; nearby commands "
                  f"{fmt(*ref['local_threshold']['95'][:2], d=3)} / {fmt(*ref['local_threshold']['99'][:2], d=3)}. "
                  f"Rate above threshold on held-out evaluation-split rows: state "
                  f"{ref['state_false_alarm']['95'][0]:.3f} / {ref['state_false_alarm']['99'][0]:.3f}; (state, "
                  f"command) {ref['state_command_false_alarm']['95'][0]:.3f} / "
                  f"{ref['state_command_false_alarm']['99'][0]:.3f}; nearby commands "
                  f"{ref['local_false_alarm']['95'][0]:.3f} / {ref['local_false_alarm']['99'][0]:.3f}. Clip range "
                  f"(99% of recorded commands): low {np.round(ref['clip99_low'], 2).tolist()}, high "
                  f"{np.round(ref['clip99_high'], 2).tolist()}; full-range low {np.round(ref['clipmm_low'], 2).tolist()}"
                  f"; training rows past 80%: {ref['training_rows_past_80']}; held-out rows {ref['heldout_rows']}.\n")
    md = headline(summary) + md
    (OUT / "tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
