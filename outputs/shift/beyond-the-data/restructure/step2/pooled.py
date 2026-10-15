"""Step 2: the pooled look over seeds 0-4 (results.json + .sha256 and results.md, write-once). EXPLORATORY; see
PRESPEC.md.

Reads every seed-world's values.json (or stop.json), each checked against its sidecar, its lock (SHA-256 and record
checks) and the frozen prespec.json and code manifest. Builds the points of every logical arm (the same arm in both
worlds: e.g. the even pull matched in the stalling world and locked in the no-stall world), then computes NET for
every arm, the go/no-go G1-G3, the attribution A1-A3 and version 6's marks restated for obca_k1 (reported), exactly
as step2lib defines them. A stopped seed-world, a failed arm or an unmatched control leaves its values missing, and a
missing value fails every condition that needs it.

Scratch mode (--out-root, outside the repository; --seeds a subset): the same code on a smoke run; labelled SCRATCH.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402


sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2lib as S  # noqa: E402

ALL_IDS = ("clone",) + S.RAMP_ARMS + ("standin", "even_pull", "push_less", "random_1", "random_2", "range_hinge",
                                      "perfect_band")


def gather(root, seeds, spec, spec_sha, man_sha, official):
    pts = {w: {s: {} for s in seeds} for w in S.WORLDS}
    status, values, locks = {}, {}, {}
    for s in seeds:
        for w in S.WORLDS:
            d = S.seed_dir(root, s, w)
            v = S.read_json_checked(d / "values.json", required=False)
            st = S.read_json_checked(d / "stop.json", required=False)
            key = f"{w}/seed{s}"
            if v is not None and st is not None:
                raise SystemExit(f"refused: {d} has both values.json and stop.json")
            if v is None and st is None:
                raise SystemExit(f"refused: {key} has neither values.json nor stop.json (the pooled look needs every "
                                 f"seed-world)")
            if st is not None:
                if st.get("prespec_sha256") != spec_sha or (official and st.get("mode") != "official"):
                    raise SystemExit(f"refused: {d}/stop.json is not this specification's official output")
                status[key] = dict(status="stopped", reason=st.get("reason"))
                continue
            lock = S.read_json_checked(d / "lock.json")
            problems = S.lock_problems(lock, spec_sha, man_sha, s, w, official, spec["protocol"]["actor_steps"])
            if v.get("kind") != "step2-policy-values" or v.get("lock_sha256") != S.file_sha256(d / "lock.json"):
                problems.append("values.json is not the evaluation of this lock")
            if v.get("prespec_sha256") != spec_sha or v.get("code_manifest_sha256") != man_sha:
                problems.append("values.json was computed under another specification or code")
            if official and v.get("mode") != "official":
                problems.append("values.json is not official")
            if problems:
                raise SystemExit(f"refused: {key}: {problems}")
            for arm, name in lock["arm_ids"].items():
                pts[w][s][arm] = None if name is None else {
                    sit: v["situations"][sit]["points"].get(name) for sit in S.SITUATIONS}
            status[key] = dict(status="values")
            values[key], locks[key] = v, lock
    return pts, status, values, locks


def tables(pts, values, locks, seeds):
    """Reported tables: points per arm, world and situation; where the actors go; the built-in checks; matching;
    resources per phase."""
    points = {}
    for w in S.WORLDS:
        points[w] = {}
        ids = ("none", "full_none") + ALL_IDS if w == "stall" else ("none",) + ALL_IDS
        for arm in ids:
            row = {}
            for sit in S.SITUATIONS:
                per = [S.pts_of(pts, w, s, arm, sit) for s in seeds]
                row[sit] = dict(per_seed=per, mean=S.mean_or_none(per))
            points[w][arm] = row
    visits, checks, matching, resources = {}, {}, {}, {}
    for key, lock in locks.items():
        v = values[key]
        w = lock["world"]
        visits[key] = {arm: (None if name is None else {sit: v["situations"][sit]["past_t_lo_share"].get(name)
                                                        for sit in S.SITUATIONS})
                       for arm, name in lock["arm_ids"].items()}
        checks[key] = lock["checks"]
        if w == "stall":
            matching[key] = dict(target_strength=lock["target_strength"], matched_names=lock["matched_names"],
                                 standin=dict(arm=lock["standin"].get("arm"),
                                              coefficient=lock["standin"].get("coefficient"),
                                              error=lock["standin"].get("error")),
                                 strengths={n: lock["strengths"].get(n) for n in lock["scored_names"]})
        else:
            matching[key] = dict(locked_controls=lock.get("locked_controls"),
                                 standin=dict(arm=lock["standin"].get("arm"), error=lock["standin"].get("error")),
                                 strengths={n: lock["strengths"].get(n) for n in lock["scored_names"]})
        resources[key] = dict(train_seconds=lock["seconds"]["total"], train_vmhwm_mb=lock.get("vmhwm_mb"),
                              eval_seconds=v["seconds"]["total"], eval_vmhwm_mb=v.get("vmhwm_mb"),
                              processor=lock.get("processor"), steps=lock.get("steps"))
    return dict(points=points, past_t_lo_share=visits, checks=checks, matching=matching, resources=resources)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=None, help="scratch mode: the scratch folder (outside the repository)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="scratch only: a subset of seeds 0-4")
    a = ap.parse_args(argv)
    t_start = time.perf_counter()
    official = a.out_root is None
    if official and a.seeds is not None:
        raise SystemExit("refused: --seeds is for scratch runs only; the official look pools seeds 0-4")
    spec, spec_sha = S.load_prespec()
    man, man_sha = S.verify_manifest()
    seeds = [S.check_seed(s) for s in (a.seeds if a.seeds is not None else spec["seeds"])]
    root = S.OFFICIAL_OUT if official else Path(a.out_root)
    out = S.guard_out(root / "results.json", official)
    if out.exists():
        raise SystemExit(f"refused: {out} exists (write-once)")
    pts, status, values, locks = gather(root, seeds, spec, spec_sha, man_sha, official)
    rule = spec["go_no_go"]["rule"]
    verdict = S.go_no_go(pts, seeds, rule)
    attr = S.attribution(pts, seeds, rule)
    marks = S.analogues(pts, seeds, rule)
    nets = {arm: S.net(pts, arm, seeds) for arm in ALL_IDS}
    nets_recorded = {arm: S.net(pts, arm, seeds, "recorded") for arm in ALL_IDS}
    results = dict(label=S.LABEL, mode="official" if official else "SCRATCH (does not count)", seeds=seeds,
                   prespec_sha256=spec_sha, code_manifest_sha256=man_sha,
                   step1_results_sha256=spec["inputs"]["step1_results"]["sha256"], status=status, go_no_go=verdict,
                   attribution=attr, v6_marks_restated=marks, net_strong_current=nets,
                   net_recorded_conditions_information_only=nets_recorded, **tables(pts, values, locks, seeds),
                   seconds=time.perf_counter() - t_start, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    digest = S.write_json_once(out, results, official)
    try:
        md = results_md(results, spec, digest)
    except Exception as e:      # the report must never cost the verdict (results.json is already written)
        md = (f"# Step 2 results: {verdict['verdict']}\n\nThe full report could not be formatted ({e!r}); read "
              f"results.json (SHA-256 {digest}).\n")
    S.guard_out(root / "results.md", official).write_text(md, encoding="utf8")
    print(f"verdict {verdict['verdict']}: G1 {verdict['G1']}, G2 {verdict['G2']}, G3 {verdict['G3']}; NET obca_k1 "
          f"{nets['obca_k1']['net']}; results {out}", flush=True)
    return results


def f(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, str):
        return x
    if not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    return f"{x:+.{nd}f}"


def results_md(r, spec, digest):
    v, nets = r["go_no_go"], r["net_strong_current"]
    g1 = v["G1_detail"]
    scratch = "" if r["mode"] == "official" else " (SCRATCH: does not count)"
    L_ = [f"# Step 2 results{scratch}: {v['verdict']}", "", r["label"], "",
          f"Mode: {r['mode']}. Seeds {r['seeds']}. results.json SHA-256 {digest}. prespec.json {r['prespec_sha256']}.",
          "", "## Go / no-go (fixed in prespec.json before any step-2 actor was trained)", "",
          f"- G1 NET of obca_k1 {f(g1['net_primary']['net'])} against the best world-blind rival "
          f"({g1['best_world_blind']}) {f(g1['best_world_blind_net'])} + 10: {f(v['G1'])}",
          f"- G2 keeps the improvement in the stalling world (obca_k1 - none >= -3): recorded "
          f"{f(v['G2_detail']['recorded']['mean'])}, big waves {f(v['G2_detail']['big_waves']['mean'])}, big kicks "
          f"{f(v['G2_detail']['big_kicks']['mean'])}: {f(v['G2'])}",
          f"- G3 stalling world, strong current, obca_k1 - matched even pull {f(v['G3_detail']['mean'])} >= 15: "
          f"{f(v['G3'])}", "",
          "## NET across worlds, strong current (mean over seeds of [stall gain over none] - [no-stall loss "
          "against none])", "", "| arm | NET | stalling gain | no-stall loss |", "|---|---|---|---|"]
    for arm in ALL_IDS:
        n = nets[arm]
        L_.append(f"| {arm}{' (baseline)' if arm == 'clone' else ''} | {f(n['net'])} | {f(n['gain_stall'])} | "
                  f"{f(n['loss_no_stall'])} |")
    L_ += ["", "## Attribution (reported; decides only the wording of the claim)", ""]
    for tag, x in r["attribution"].items():
        L_.append(f"- {tag} NET({x['arm']}) {f(x['net_arm'])} >= NET(obca_k1) {f(x['net_primary'])} - 5: "
                  f"{f(x['holds'])}" + (f" ({x['meaning_if_holds']})" if x["holds"] else ""))
    m = r["v6_marks_restated"]
    L_ += ["", "## Version 6's marks restated for obca_k1 (reported)", "",
           f"- M1 analogue: obca_k1 - matched even pull {f(m['M1_even_pull']['mean'])} (positive in "
           f"{m['M1_even_pull']['positive']} of {len(r['seeds'])}): {f(m['M1_even_pull']['passed'])}",
           f"- M2 analogue: random surface 1 {f(m['M2_random_surfaces']['random_1']['mean'])}, 2 "
           f"{f(m['M2_random_surfaces']['random_2']['mean'])}: {f(m['M2_random_surfaces']['passed'])}",
           f"- M3 analogue: recorded vs none {f(m['M3_keeps_improvement']['recorded_vs_none']['mean'])}, vs clone "
           f"{f(m['M3_keeps_improvement']['recorded_vs_clone']['mean'])}: {f(m['M3_keeps_improvement']['passed'])}",
           f"- M4 analogue: obca_k1 - matched stand-in {f(m['M4_best_standin']['mean'])} >= -5: "
           f"{f(m['M4_best_standin']['passed'])}", "",
           "## Points, strong current (mean over seeds)", "", "| arm | stalling world | no-stall world |", "|---|---|---|"]
    for arm in ("none", "clone") + tuple(a for a in ALL_IDS if a != "clone"):
        s_ = r["points"]["stall"].get(arm, {}).get("strong_current", {}).get("mean")
        n_ = r["points"]["no_stall"].get(arm, {}).get("strong_current", {}).get("mean")
        L_.append(f"| {arm} | {f(s_)} | {f(n_)} |")
    L_ += ["", "## Meaning", "", spec["go_no_go"]["meaning_go"] if v["verdict"] == "GO" else
           spec["go_no_go"]["meaning_no_go"], ""]
    stopped = {k: x for k, x in r["status"].items() if x["status"] != "values"}
    if stopped:
        L_ += ["Stopped seed-worlds (they fail every condition that needs them): " +
               "; ".join(f"{k}: {x['reason']}" for k, x in stopped.items()), ""]
    return "\n".join(L_)


if __name__ == "__main__":
    main()
