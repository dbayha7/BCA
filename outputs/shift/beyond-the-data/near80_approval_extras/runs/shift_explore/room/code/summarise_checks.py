"""EXPLORATORY: merge the seed-0 checks (checks_seed0_{a,b,c,d}.json) into checks_seed0.json with pass flags, and
project the wall-clock of one seed from the measured timings (timing.json). No return is computed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import run_seed as R  # noqa: E402

# arms per seed (stalling world, before matching): kind -> count
PLAN_ARMS = {"plain": 6, "band": 6, "band_shuffled": 1, "harm": 1, "hinge": 1, "push_less": 5, "random": 10}
SCORED_STALL = 15           # none, clone, 9 fixed arms, 4 matched controls
MATCH_ASSUMED = dict(typical_rounds=3, worst_rounds=5, arms_per_round={"plain": 1, "push_less": 1, "random": 2})


def main():
    spec, spec_sha = H.load_spec()
    H.check_gate_files(spec)
    man, man_sha = H.check_code(spec_sha)
    parts = {k: H.read_json(H.OUT / f"checks_seed0_{k}.json") for k in "abcde"}
    missing = [k for k, v in parts.items() if v is None]
    if missing:
        raise SystemExit(f"missing checks {missing}")
    stale = [k for k, v in parts.items() if v["spec_sha256"] != spec_sha or v["code_manifest_sha256"] != man_sha]
    if stale:
        raise SystemExit(f"checks {stale} ran under another spec or code manifest")
    a, b, c, d, e = (parts[k][k] for k in "abcde")
    other_e = {}
    for rep in H.SEEDS[1:]:
        x = H.read_json(H.OUT / f"checks_e_seed{rep}.json")
        if x is not None and x["spec_sha256"] == spec_sha and x["code_manifest_sha256"] == man_sha:
            other_e[rep] = x["e"]
    passed = dict(
        a=bool(a["passed"]),
        b=bool(b["passed"]),
        c=bool(c["plumbing_reproduces_readout"] and c["precision_passed"] and c["gradient_finite_both_sides"]),
        d=bool(d["all_finite"]),
        e=bool(e["passed"]))
    # timing: per 1,000 steps per arm, from the 400-step runs (compiled second call) and the 20,000-step plain group
    t = {k: v["seconds_per_1000_steps_per_arm"] for k, v in d["configs"]["stall"]["timing"].items()}
    t["plain_20000_step_group"] = b["plain_seconds_per_1000_steps_per_arm"]
    per_arm_20k = {k: t[k] * H.STEPS / 1000 for k in PLAN_ARMS}
    per_arm_20k["plain"] = t["plain_20000_step_group"] * H.STEPS / 1000
    train_fixed = sum(PLAN_ARMS[k] * per_arm_20k[k] for k in PLAN_ARMS)
    round_cost = sum(n * per_arm_20k[k] for k, n in MATCH_ASSUMED["arms_per_round"].items())
    eval_actor = d.get("evaluation_cost", {}).get("estimated_seconds_per_actor") or parts["d"]["evaluation_cost"][
        "estimated_seconds_per_actor"]
    overhead = 120.0  # loading, shared arrays, slopes, strengths, compiles
    e_stall, e_repl = e["stall"]["seconds"], e["replication"]["seconds"]
    stall_typical = a["seconds"] + e_stall + train_fixed + MATCH_ASSUMED["typical_rounds"] * round_cost + overhead
    stall_worst = (a["seconds"] + e_stall + train_fixed + MATCH_ASSUMED["worst_rounds"] * round_cost + 3 * round_cost
                   + overhead)
    eval_stall = SCORED_STALL * eval_actor + 30
    no_stall = (a["seconds"] * 2 / 3 + 2 * e_stall + 3 * per_arm_20k["plain"] + 2 * per_arm_20k["band"] + overhead +
                6 * eval_actor)
    replication = (e_repl + 6 * per_arm_20k["plain"] + per_arm_20k["band"] + MATCH_ASSUMED["typical_rounds"] *
                   per_arm_20k["plain"] + overhead + 2 * eval_actor)
    proj = dict(
        seconds_per_1000_steps_per_arm=t, seconds_per_arm_20000_steps=per_arm_20k,
        stall_train_phase_minutes=dict(typical=stall_typical / 60, worst=stall_worst / 60,
                                       of_which_long_run_harm=per_arm_20k["harm"] / 60,
                                       of_which_check_a=a["seconds"] / 60),
        stall_eval_phase_minutes=eval_stall / 60,
        stall_seed_total_minutes=dict(typical=(stall_typical + eval_stall) / 60,
                                      worst=(stall_worst + eval_stall) / 60),
        no_stall_minutes_per_seed=no_stall / 60, replication_minutes_per_seed=replication / 60,
        all_runs_per_seed_minutes=(stall_typical + eval_stall + no_stall + replication) / 60,
        assumptions=dict(matching=MATCH_ASSUMED, evaluation_seconds_per_actor=eval_actor,
                         overhead_seconds=overhead, scored_actors_stall=SCORED_STALL,
                         note=("one pinned CPU per seed process; evaluation cost estimated from the policy's cost "
                               "alone (no return computed in the checks)")))
    def e_main(x):
        return {w: R.summary_e(x[w]) | dict(passed=x[w]["passed"]) for w in ("stall", "replication")}

    out = dict(label=H.LABEL + " (seed-0 checks; no return, points or score computed)", kind="room-checks-seed0",
               spec_sha256=spec_sha, code_manifest_sha256=man_sha, passed=passed, all_passed=bool(all(passed.values())),
               parts_sha256={k: H.file_sha256(H.OUT / f"checks_seed0_{k}.json") for k in "abcde"},
               what_c_checks=("c passes on the plumbing (the readout's Q1 form reproduces the readout), the float32 "
                              "bound and finite gradients. The trained residual (min over heads) does not reproduce "
                              "the plan's 7.3(b) numbers; that is reported under c.trained_form_vs_readout, not as a "
                              "pass (spec change A3). The literal |delta*| arm keeps an offset of about +0.15 at the "
                              "data; the amended main oracle is zero there."),
               summary=dict(
                   a={n: dict(equal=v["equal"], max_abs_diff=v["max_abs_diff"]) for n, v in a["arms"].items()},
                   b=dict(run_configuration=b["run_configuration"], alone=b["alone"],
                          rows_sha256_equals_check2=b["rows_sha256_equals_check2"]),
                   c=dict(plumbing_reproduces_readout=c["plumbing_reproduces_readout"],
                          trained_form_vs_readout=c["trained_form_vs_readout"],
                          readout_rows=c["readout_rows"], readout=dict(mean_past80=c["readout_mean_past80"],
                                                                       median_abs_other=c["readout_median_abs_other"]),
                          all_rows=c["all_rows"], all_rows_main_oracle=c["all_rows_main_oracle"],
                          thruster1_gradient=c["thruster1_gradient"],
                          thruster1_gradient_main_penalty=c["thruster1_gradient_main_penalty"],
                          harm_value_gapped_actor=c["harm_value_gapped_actor"]),
                   d={k: dict(all_finite=v["all_finite"], timing=v["timing"]) for k, v in d["configs"].items()},
                   e=dict(seed0=e_main(e), other_seeds_reported={rep: e_main(x) for rep, x in other_e.items()},
                          all_forms_seed0_stall={k: dict(fall=v["fall_share_80_to_81"], rise=v["rise_share_80_to_81"],
                                                         best_past80=v["best_past80_share"],
                                                         slope_805=v["mean_dW_dpush"]["0.805"]["mean"],
                                                         passed=v["passed"])
                                                 for k, v in e["stall"]["forms"].items()})),
               projection=proj)
    H.write_json(H.OUT / "checks_seed0.json", out)
    H.write_json(H.OUT / "timing.json", dict(label=H.LABEL, kind="room-timing", spec_sha256=spec_sha,
                                             code_manifest_sha256=man_sha, **proj))
    print(H.A.canonical(dict(passed=passed, projection=proj)))


if __name__ == "__main__":
    main()
