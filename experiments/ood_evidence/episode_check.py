"""Episode check: do played episodes match the exact formulas? Runs marks E1 to E4 of EPISODE_CHECK.md.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/ood_evidence/episode_check.py
    ... episode_check.py --smoke     wiring and timing only: another seed, few episodes, nothing written

The plan, the policies, the sample sizes and the pass marks are fixed in EPISODE_CHECK.md, which was
written before this script was run. The result file is written once and never overwritten.
Trains nothing; imports no learner code.
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import episode_sampler as S  # noqa: E402
import saved_policies as SP  # noqa: E402

env, cenv = S.env, S.cenv
import collect  # noqa: E402  (gridworld_ood_v1, on the path through episode_sampler)
import evaluation as ev  # noqa: E402

SEED, POLICY_SEED = 2026111501, 2026111502
N_DESIGNED, N_TRAINED, N_SAME = 200_000, 40_000, 200
ALPHA, Z_VALUE, VALUE_SLACK = 0.001, 5.0, 0.001
RESULT = HERE / "episode_check_result.json"
CHECKED_METHODS = ("baseline", "support_blind_width")


# ---------------------------------------------------------------- the cases fixed in EPISODE_CHECK.md, section 3
def inhouse_designed(task):
    best, _ = env.best_policy(task)
    withheld = collect.targeted_moves(task)
    demonstrator, _ = collect.demonstrator_commands(task, withheld)
    into = best.copy()
    for cell, (direction,) in withheld.items():
        into[cell] = env.CORNERS[ev.target_corner(direction)]
    return {"best policy": best,
            "demonstrator of the withheld-move data": demonstrator,
            "best policy, takes the withheld move": into,
            "best policy at half strength": 0.5 * best,
            "random commands": np.random.default_rng(POLICY_SEED).uniform(-1, 1, (S.CELLS, 4))}


def published_designed(task, full):
    moves, _ = task.best_policy()
    best = cenv.policy_from_moves(moves)
    out = {"best policy": best, "best policy at half strength": 0.5 * best}
    if full:
        lava = best.copy()
        lava[cenv.SECOND_CRITICAL] = cenv.CORNERS[cenv.UP]
        out["best policy, commands up into lava at cell 83"] = lava
        out["random preferences"] = np.random.default_rng(POLICY_SEED).uniform(-1, 1, (S.CELLS, cenv.N_MOVES))
    return out


def trained(records):
    return {f"{SP.METHOD_NAMES[r['method']]}, dataset {r['data_seed']}, training seed {r['train_seed']}": r["commands"]
            for r in records if r["method"] in CHECKED_METHODS}


def cases():
    """Yield (environment label, world, kind, {policy label: policy}, start cells)."""
    honeypot = env.honeypot()
    starts = (24, 33, 35, 44, env.A_START)
    yield "Honeypot", S.InHouseWorld(honeypot), "designed", inhouse_designed(honeypot), starts
    yield "Honeypot", S.InHouseWorld(honeypot), "trained", trained(SP.inhouse("honeypot")), starts
    for reward in env.LOW_EXIT_REWARDS:
        task = env.bottleneck_r2(reward)
        label, starts = f"Bottleneck, low exit {reward}", (env.B, env.C, env.A_START)
        yield label, S.InHouseWorld(task), "designed", inhouse_designed(task), starts
        yield label, S.InHouseWorld(task), "trained", trained(SP.inhouse(task.name)), starts
    starts = (cenv.START, cenv.ABOVE, cenv.SECOND_CRITICAL)
    for noise in (0.10, 0.0, 0.30):
        task = cenv.published(noise)
        label = f"Published gridworld, noise {round(100 * noise)}%"
        yield label, S.PublishedWorld(task), "designed", published_designed(task, full=noise == 0.10), starts
        if noise == 0.10:
            yield label, S.PublishedWorld(task), "trained", trained(SP.published("optimal")), starts


# ---------------------------------------------------------------- the exact numbers and the played counts
def exact(world, policy):
    """({ending: chance per start cell}, value per start cell) from the environment's formulas."""
    task = world.task
    out = task.outcomes(policy)
    if isinstance(world, S.InHouseWorld):
        ref = {"goal": out["goal"], "hazard": out["hazard"], "time-out": out["timeout"]}
        if task.low_exit is not None:
            ref["low exit"] = out["low_exit"]
        if task.name == "honeypot":
            ref[f"withheld hazard {env.UNSEEN_HAZARD}"] = ev.unseen_hazard_probability(task, policy)
    else:
        ref = {"goal": out["goal"], "lava": out["lava"], "time-out": out["timeout"]}
        for cell, chance in out["lava_by_cell"].items():
            ref[f"lava cell {cell}"] = chance
    return ref, task.value(policy)


def played_count(name, result):
    ending, last = result["ending"], result["last_cell"]
    if name == "goal":
        return int((ending == S.GOAL_END).sum())
    if name in ("hazard", "lava"):
        return int((ending == S.BAD_END).sum())
    if name == "low exit":
        return int((ending == S.LOW_EXIT_END).sum())
    if name == "time-out":
        return int((ending == S.TIMEOUT).sum())
    cell = int(name.split()[-1])                                   # "withheld hazard 34", "lava cell 66"
    return int(((ending == S.BAD_END) & (last == cell)).sum())


# ---------------------------------------------------------------- the four marks, for one policy and start
def same_draws(world, policy, start, rng, n):
    """E1: episodes that differ between the one-step-at-a-time play and the batch play."""
    draws = rng.random((world.horizon, n, world.draws))
    batch = S.play(world, policy, start, n, draws=draws, keep_paths=True)
    different = 0
    for i in range(n):
        path, rewards, ending = S.play_one(world, policy, start, draws[:, i])
        moves = len(rewards)
        same = (batch["ending"][i] == ending and batch["length"][i] == moves
                and np.array_equal(batch["paths"][i, :moves + 1], path)
                and np.all(batch["paths"][i, moves + 1:] == -1)
                and np.array_equal(batch["rewards"][i, :moves], rewards))
        different += not same
    return different


def impossible_steps(world, policy, result):
    """E2: steps the one-move rule does not allow, plus broken episode bookkeeping."""
    violations = int(result["transitions"][~world.allowed(policy)].sum())
    timed_out = result["ending"] == S.TIMEOUT
    violations += int((result["length"] > world.horizon).sum())
    violations += int((result["length"][timed_out] != world.horizon).sum())
    violations += int((world.ending_codes[result["last_cell"][~timed_out]] != result["ending"][~timed_out]).sum())
    return violations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="wiring and timing only; nothing is judged or written")
    args = parser.parse_args()
    if not args.smoke and RESULT.exists():
        sys.exit(f"{RESULT.name} already exists. The check is run once; it is not overwritten.")
    seed = SEED + 7 if args.smoke else SEED
    n_of = {"designed": 2_000 if args.smoke else N_DESIGNED, "trained": 2_000 if args.smoke else N_TRAINED}
    n_same = 20 if args.smoke else N_SAME

    rows, e1, e2, started = [], {}, {}, time.time()
    for case_index, (label, world, kind, policies, starts) in enumerate(cases()):
        tick = time.time()
        for policy_index, (policy_label, policy) in enumerate(policies.items()):
            reference, value = exact(world, policy)
            for start in starts:
                rng = np.random.default_rng([seed, case_index, policy_index, int(start)])
                e1[label] = e1.get(label, 0) + same_draws(world, policy, start, rng, n_same)
                n = n_of[kind]
                result = S.play(world, policy, start, n, rng=rng)
                e2[label] = e2.get(label, 0) + impossible_steps(world, policy, result)
                base = {"environment": label, "kind": kind, "policy": policy_label, "start": int(start), "n": n}
                for name, chance in reference.items():
                    p = float(np.clip(chance[start], 0.0, 1.0))
                    count = played_count(name, result)
                    rows.append({**base, "mark": "E3", "quantity": name, "count": count, "exact": p,
                                 "p_value": float(stats.binomtest(count, n, p).pvalue)})
                mean, spread = float(result["ret"].mean()), float(result["ret"].std(ddof=1))
                error = spread / np.sqrt(n)
                rows.append({**base, "mark": "E4", "quantity": "value", "played": mean, "exact": float(value[start]),
                             "standard_error": error,
                             "passes": bool(abs(mean - value[start]) <= Z_VALUE * error + VALUE_SLACK)})
        print(f"  {label:36s} {kind:9s} {len(policies):4d} policies  {time.time() - tick:6.1f} s", flush=True)

    endings = [r for r in rows if r["mark"] == "E3"]
    values = [r for r in rows if r["mark"] == "E4"]
    threshold = ALPHA / len(endings)
    for r in endings:
        r["passes"] = bool(r["p_value"] >= threshold)
    labels = sorted({r["environment"] for r in rows}, key=[r["environment"] for r in rows].index)
    by_environment = {}
    for label in labels:
        mine_e, mine_v = [r for r in endings if r["environment"] == label], [r for r in values if r["environment"] == label]
        by_environment[label] = {
            "policy_start_pairs": len(mine_v), "ending_comparisons": len(mine_e),
            "E1_episodes_that_differ": int(e1[label]), "E2_impossible_steps": int(e2[label]),
            "E3_failures": sum(not r["passes"] for r in mine_e), "E3_smallest_p": min(r["p_value"] for r in mine_e),
            "E3_largest_gap_points": 100 * max(abs(r["count"] / r["n"] - r["exact"]) for r in mine_e),
            "E4_failures": sum(not r["passes"] for r in mine_v),
            "E4_largest_gap": max(abs(r["played"] - r["exact"]) for r in mine_v)}
    marks = {"E1 same draws, same episode": sum(e1.values()) == 0,
             "E2 no impossible step": sum(e2.values()) == 0,
             "E3 endings match": all(r["passes"] for r in endings),
             "E4 value matches": all(r["passes"] for r in values)}
    verdict = "PASS" if all(marks.values()) else "FAIL"

    print()
    print(f"{'environment':36s} {'pairs':>5s} {'E1 differ':>9s} {'E2 imposs.':>10s} {'E3 fail':>7s} {'smallest p':>10s} "
          f"{'gap pts':>7s} {'E4 fail':>7s} {'value gap':>9s}")
    for label, s in by_environment.items():
        print(f"{label:36s} {s['policy_start_pairs']:5d} {s['E1_episodes_that_differ']:9d} {s['E2_impossible_steps']:10d} "
              f"{s['E3_failures']:7d} {s['E3_smallest_p']:10.2e} {s['E3_largest_gap_points']:7.3f} "
              f"{s['E4_failures']:7d} {s['E4_largest_gap']:9.4f}")
    below = float(np.mean([r["p_value"] < 0.05 for r in endings]))
    print(f"\nending comparisons K = {len(endings)}; a comparison fails below p = {threshold:.2e}")
    print(f"smallest p-value {min(r['p_value'] for r in endings):.2e}; share of p-values under 0.05: {100 * below:.1f}%")
    for name, ok in marks.items():
        print(f"  {name:32s} {'pass' if ok else 'FAIL'}")
    print(f"{'SMOKE, not judged' if args.smoke else 'VERDICT'}: {verdict}   ({time.time() - started:.0f} s)")

    if not args.smoke:
        RESULT.write_text(json.dumps({
            "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "plan": "EPISODE_CHECK.md", "seed": SEED, "policy_seed": POLICY_SEED,
            "episodes": {"designed": N_DESIGNED, "trained": N_TRAINED, "same_draws": N_SAME},
            "ending_comparisons": len(endings), "p_value_threshold": threshold,
            "share_of_p_values_under_0.05": below, "marks": marks, "verdict": verdict,
            "by_environment": by_environment, "rows": rows}, indent=1))
        print("wrote", RESULT.name)


if __name__ == "__main__":
    main()
