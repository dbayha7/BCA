"""Published gridworld: where do lava endings come from, and is a lower lava rate a safer route or a safer choice?

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I -B experiments/ood_evidence/cell83_decomposition.py

For every saved policy of the confirmation block (saved_policies.published: best-policy data, datasets 6501 to
6509, training seeds 20 to 22) and for the best policy, computed exactly from the environment's chain:

    how often an episode from the start reaches cell 83, and how many decisions it makes there;
    what the policy commands at cell 83 (the chance of each of the five moves);
    the lava risk of one decision at cell 83. With 10% noise spread over the five legal moves, a safe command
        (stay, left, right) ends in lava with chance 0.04 and a lava-directed command (up, down) with chance 0.94,
        so the risk is 0.04 + 0.90 x the chance of commanding up or down;
    the chance an episode ends in lava, split by the cell the lava was entered from.

Decisions at a cell x lava risk per decision = lava endings from that cell, because lava ends the episode. So a
lower lava rate can come from fewer decisions at risky cells (a safer route) or a lower risk per decision (a safer
choice at the same spot). Reads saved policies only; trains nothing.
"""
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import episode_sampler as S  # noqa: E402
import saved_policies as SP  # noqa: E402

cenv = S.cenv
TASK = cenv.published()
CELL = cenv.SECOND_CRITICAL
LAVA_MOVES = [m for m in TASK.legal_moves(CELL) if TASK.destination(CELL, m) in TASK.lava]
GROUPS = {"cell 83": (83,), "cells 72, 74, 92, 94 (beside the same lava)": (72, 74, 92, 94),
          "cell 67": (67,), "cells 56, 58, 76, 78 (beside the other lava)": (56, 58, 76, 78)}


def measure(policy):
    """Exact numbers for one policy, from the start cell, over the 200-step horizon."""
    chain = TASK.chain(policy)
    inner = chain.copy()
    inner[:, list(TASK.terminals)] = 0.0
    here = np.zeros(S.CELLS)
    here[TASK.start] = 1.0
    not_yet = here.copy()                              # the same walk, stopped the first time it reaches cell 83
    decisions, reached = np.zeros(S.CELLS), 0.0
    for _ in range(cenv.HORIZON):
        decisions += here
        reached += not_yet[CELL]
        not_yet[CELL] = 0.0
        here, not_yet = here @ inner, not_yet @ inner
    into_lava = np.zeros(S.CELLS)                      # chance one decision at a cell ends in lava
    for cell in TASK.free_cells():
        chances = TASK.move_probabilities(int(cell), policy[cell])
        into_lava[cell] = sum(chances[m] for m in TASK.legal_moves(int(cell)) if TASK.destination(int(cell), m) in TASK.lava)
    lava_from = decisions * into_lava
    total = float(TASK.outcomes(policy)["lava"][TASK.start])
    assert abs(lava_from.sum() - total) < 1e-12, "the split by source cell must add up to the exact lava chance"
    commands = TASK.command_probabilities(policy[CELL])
    out = {"reaches": reached, "decisions": decisions[CELL], "risk": into_lava[CELL], "lava_total": total,
           "commands_lava": float(commands[LAVA_MOVES].sum()), "commands": commands,
           "goal": float(TASK.outcomes(policy)["goal"][TASK.start])}
    for name, cells in GROUPS.items():
        out[name] = float(lava_from[list(cells)].sum())
        out[f"decisions, {name}"] = float(decisions[list(cells)].sum())
    out["elsewhere"] = total - sum(out[name] for name in GROUPS)
    # At cells 72 and 92 the move "right" goes into the lava; at 74 and 94 the move "left" does.
    toward = {72: cenv.RIGHT, 92: cenv.RIGHT, 74: cenv.LEFT, 94: cenv.LEFT}
    out["lava commands beside"] = float(sum(decisions[c] * TASK.command_probabilities(policy[c])[m] for c, m in toward.items()))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    moves, _ = TASK.best_policy()
    records = SP.published("optimal")
    rows = {"The data's own policy (the best policy)": [measure(cenv.policy_from_moves(moves))]}
    for method, name in SP.METHOD_NAMES.items():
        rows[name] = [measure(r["commands"]) for r in records if r["method"] == method]

    def mean(name, key):
        return float(np.mean([m[key] for m in rows[name]]))

    print(f"Published gridworld, best-policy data, datasets 6501 to 6509, 27 policies per method. Exact, from the start.\n")
    print(f"{'policy':40s} {'reaches 83':>10s} {'decisions':>9s} {'commands':>9s} {'lava risk':>9s} {'lava from':>9s} {'all lava':>8s}")
    print(f"{'':40s} {'':>10s} {'at 83':>9s} {'up/down':>9s} {'per dec.':>9s} {'cell 83':>9s} {'':>8s}")
    for name in rows:
        print(f"{name:40s} {100 * mean(name, 'reaches'):9.1f}% {mean(name, 'decisions'):9.3f} {100 * mean(name, 'commands_lava'):8.2f}% "
              f"{100 * mean(name, 'risk'):8.2f}% {100 * mean(name, 'cell 83'):8.2f}% {100 * mean(name, 'lava_total'):7.2f}%")

    print(f"\nWhat is commanded at cell 83 (average chance of each command)")
    print(f"{'policy':40s} " + " ".join(f"{m:>7s}" for m in cenv.MOVE_NAMES) + "   policies whose likeliest command is right")
    for name in rows:
        commands = np.mean([m["commands"] for m in rows[name]], axis=0)
        likeliest = sum(int(np.argmax(m["commands"]) == cenv.RIGHT) for m in rows[name])
        print(f"{name:40s} " + " ".join(f"{100 * c:6.2f}%" for c in commands) + f"   {likeliest} of {len(rows[name])}")
    worst = {name: max(100 * m["commands_lava"] for m in rows[name]) for name in rows}
    print("largest chance of commanding up or down at cell 83 in any one policy: "
          + "; ".join(f"{name}: {value:.2f}%" for name, value in worst.items()))

    print(f"\nWhere the lava endings come from (% of episodes; the columns add up to all lava)")
    keys = list(GROUPS) + ["elsewhere"]
    print(f"{'policy':40s} " + " ".join(f"{k.split(' (')[0][:22]:>22s}" for k in keys) + f" {'all lava':>9s}")
    for name in rows:
        print(f"{name:40s} " + " ".join(f"{100 * mean(name, k):21.2f}%" for k in keys) + f" {100 * mean(name, 'lava_total'):8.2f}%")

    print(f"\nRoute or choice, by group of cells: decisions per episode made there, and the lava risk of one decision there")
    print("(a safe command risks 2% to 4% through the noise; a lava-directed command risks 92% to 94%)")
    print(f"{'policy':40s} " + " ".join(f"{k.split(' (')[0][:22]:>30s}" for k in GROUPS))
    for name in rows:
        cells = []
        for k in GROUPS:
            d = mean(name, f"decisions, {k}")
            cells.append(f"{d:7.3f} decisions x {100 * mean(name, k) / d:5.2f}%")
        print(f"{name:40s} " + " ".join(f"{c:>30s}" for c in cells))
    beside = "cells 72, 74, 92, 94 (beside the same lava)"
    print("\nAt cells 72, 74, 92 and 94: decisions per episode, and how many of them command the move into the lava")
    for name in rows:
        d, bad = mean(name, f"decisions, {beside}"), mean(name, "lava commands beside")
        print(f"  {name:40s} {d:.3f} decisions, {bad:.4f} of them lava-directed ({100 * bad / d:.1f}%)")

    base, guard = "Standard learner", "BCA policy guard"
    for k in GROUPS:
        d0, d1 = mean(base, f"decisions, {k}"), mean(guard, f"decisions, {k}")
        r0, r1 = mean(base, k) / d0, mean(guard, k) / d1
        print(f"  {k.split(' (')[0]}: lava {100 * mean(base, k):.2f}% -> {100 * mean(guard, k):.2f}%; "
              f"from the number of decisions (route) {100 * (d1 - d0) * (r0 + r1) / 2:+.2f} points, "
              f"from the risk per decision (choice) {100 * (r1 - r0) * (d0 + d1) / 2:+.2f} points")
    print(f"\nStandard learner to BCA policy guard, at cell 83")
    d0, d1, r0, r1 = mean(base, "decisions"), mean(guard, "decisions"), mean(base, "risk"), mean(guard, "risk")
    print(f"  decisions at cell 83 per episode: {d0:.3f} -> {d1:.3f}; lava risk per decision: {100 * r0:.2f}% -> {100 * r1:.2f}%")
    print(f"  lava endings from cell 83: {100 * mean(base, 'cell 83'):.2f}% -> {100 * mean(guard, 'cell 83'):.2f}% of episodes")
    route = (d1 - d0) * (r0 + r1) / 2
    choice = (r1 - r0) * (d0 + d1) / 2
    print(f"  change from the number of decisions (route): {100 * route:+.2f} points; from the risk per decision (choice): "
          f"{100 * choice:+.2f} points (from the averages above; the two add up to the change in the product of the averages)")
    print(f"  all lava endings: {100 * mean(base, 'lava_total'):.2f}% -> {100 * mean(guard, 'lava_total'):.2f}%; "
          f"goal: {100 * mean(base, 'goal'):.2f}% -> {100 * mean(guard, 'goal'):.2f}%")


if __name__ == "__main__":
    main()
