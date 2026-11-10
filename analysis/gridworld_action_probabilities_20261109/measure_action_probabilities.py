"""Measure the movement probabilities the 108 archived GridWorld policies actually execute.

Read-only on the preserved stress package. It reads three saved files
(model_manifest.json, state_tables.json, exact_metrics.json), recomputes each
policy's softmax movement probabilities at the decision cells, and writes new
files under --out. Nothing is trained, and no checkpoint is unpickled.

Usage:
    python -I -B measure_action_probabilities.py --evidence <stress folder> --out <new folder>
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gridworld_min as gw  # noqa: E402

BETAS = (1.0, 2.0, 3.0)
NOMINAL_BETA = 2.0
HORIZON = 200
ARMS = ("native_aux", "bca_direct")
CONDITIONS = {"honeypot": ("full", "dangerous_actions", "random_rows"),
              "bottleneck": ("full", "fragments", "random_rows")}
TARGETED = {"honeypot": "dangerous_actions", "bottleneck": "fragments"}
B, GATE, C = 44, 45, 46

# (label, cell, directions, harmful). harmful=True means "more of this is worse",
# False means "more of this is better", None means neither by itself.
BOTTLENECK_TARGETS = (
    ("B_enter_bridge", B, (gw.RIGHT,), False),
    ("B_hazard_up", B, (gw.UP,), True),
    ("B_retreat", B, (gw.DOWN, gw.LEFT), None),
    ("gate_exit_to_C", GATE, (gw.RIGHT,), False),
    ("gate_back_to_B", GATE, (gw.LEFT,), None),
    ("gate_wall_bump", GATE, (gw.UP, gw.DOWN), True),
    ("C_wall_bump_up", C, (gw.UP,), True),
    ("C_back_into_gate", C, (gw.LEFT,), None),
    ("C_onward", C, (gw.RIGHT, gw.DOWN), False),
)
BOTTLENECK_CONTEXT_CELLS = (43, 54, 47, 56)  # supported neighbors of the bridge


def judge_shift(native_p, bca_p, floor, ceiling, harmful):
    """Decide whether one policy's change in a movement probability is a real change.

    native_p, bca_p : probability of the same direction(s) at the same cell, at
                      the same execution setting, for the matched pair.
    floor, ceiling  : the smallest / largest probability a single direction can
                      ever receive from a bounded command (0.61% and 94.8% at
                      the original setting).
    harmful         : True if more probability is worse (hazard, wall bump),
                      False if more is better (entering the bridge).

    Return "improved", "worsened" or "unchanged". Returning None leaves the
    verdict column as "pending" and the report shows raw numbers only.
    """
    # TODO(human): choose the scale and the threshold for calling a shift real.
    return None


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hazard_frontier(scenario):
    """Free cells one executed move from a hazard, mapped to the direction that enters it."""
    frontier = {}
    for cell in gw.free_cells(scenario):
        entering = [d for d in range(4) if gw.direction_role(scenario, cell, d) == "hazard"]
        if entering:
            assert len(entering) == 1, (scenario, cell, entering)
            frontier[cell] = entering[0]
    return frontier


def load_models(evidence):
    manifest = json.loads((evidence / "evaluation_v1/model_manifest.json").read_text())
    tables = json.loads((evidence / "evaluation_v1/state_tables.json").read_text())
    models = {}
    for meta in manifest:
        commands = np.zeros((100, 4))
        for cell in range(100):
            record = tables[f"{meta['id']}:{cell}"]
            assert record["state"] == cell
            commands[cell] = record["action"]
        assert np.all(np.abs(commands) <= 1 + 1e-7)
        models[meta["id"]] = {**meta, "commands": commands}
    return models


def check_parity(models, exact):
    """My mechanics must reproduce the saved exact 200-step outcomes of every policy."""
    worst = {"goal": 0.0, "hazard": 0.0}
    rows = 0
    for model_id, model in models.items():
        cells = gw.free_cells(model["scenario"])
        for beta in BETAS:
            goal, hazard = gw.finite_outcomes(model["scenario"], model["commands"], beta, HORIZON)
            saved = exact[model_id][str(beta)][str(HORIZON)]
            assert sorted(int(k) for k in saved) == cells
            for cell in cells:
                worst["goal"] = max(worst["goal"], abs(goal[cell] - saved[str(cell)]["goal"]))
                worst["hazard"] = max(worst["hazard"], abs(hazard[cell] - saved[str(cell)]["hazard"]))
                rows += 1
    return {"rows_checked": rows, "max_abs_goal_difference": worst["goal"],
            "max_abs_hazard_difference": worst["hazard"]}


def log_normalizer(command, beta):
    logits = beta * np.asarray(command, dtype=float)
    peak = logits.max()
    return float(peak + np.log(np.exp(logits - peak).sum()))


def bridge_chain(commands, beta):
    """First visit to the bridgehead, treating everything outside {B, gate} as an exit.

    Returns the probability of stepping out onto C before anything else, and the
    probability of stepping up into the hazard above B before anything else.
    """
    pb = gw.movement_probabilities(commands[B], beta)
    pg = gw.movement_probabilities(commands[GATE], beta)
    gate_stay = pg[gw.UP] + pg[gw.DOWN]
    back = pg[gw.LEFT] / (1.0 - gate_stay)
    out = pg[gw.RIGHT] / (1.0 - gate_stay)
    loop = pb[gw.RIGHT] * back  # B -> gate -> B
    return pb[gw.RIGHT] * out / (1.0 - loop), pb[gw.UP] / (1.0 - loop)


def shift_row(pair, native, bca, label, cell, directions, harmful, beta, bounds):
    pn = float(gw.movement_probabilities(native["commands"][cell], beta)[list(directions)].sum())
    pb = float(gw.movement_probabilities(bca["commands"][cell], beta)[list(directions)].sum())
    row = {"pair": pair, "scenario": native["scenario"], "condition": native["condition"],
           "dataset": native["dataset"], "seed": native["seed"], "beta": beta,
           "target": label, "cell": cell,
           "directions": "+".join(gw.DIRECTION_NAMES[d] for d in directions),
           "harmful": harmful, "native_p": pn, "bca_p": pb, "delta": pb - pn,
           "log_ratio": float(np.log(pb / pn)),
           "own_preference_term": None, "competition_term": None}
    if len(directions) == 1:
        d = directions[0]
        # log p = beta*a_d - logZ, so the change splits exactly into two parts:
        # lowering/raising this direction's own preference, and the other
        # directions gaining/losing weight.
        row["own_preference_term"] = float(beta * (bca["commands"][cell][d] - native["commands"][cell][d]))
        row["competition_term"] = float(-(log_normalizer(bca["commands"][cell], beta)
                                          - log_normalizer(native["commands"][cell], beta)))
    floor, ceiling = bounds[beta]
    verdict = judge_shift(pn, pb, floor * len(directions), min(1.0, ceiling * len(directions)), harmful)
    row["verdict"] = "pending" if verdict is None else verdict
    return row


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"n": int(len(values)), "mean": float(values.mean()), "min": float(values.min()),
            "max": float(values.max()), "n_negative": int((values < 0).sum()),
            "n_positive": int((values > 0).sum())}


def summarize_shifts(rows):
    out = {"native_mean": float(np.mean([r["native_p"] for r in rows])),
           "bca_mean": float(np.mean([r["bca_p"] for r in rows])),
           "delta": stats([r["delta"] for r in rows]),
           "by_dataset_mean_delta": {},
           "per_pair": [{k: r[k] for k in ("dataset", "seed", "native_p", "bca_p", "delta",
                                           "own_preference_term", "competition_term", "verdict")}
                        for r in sorted(rows, key=lambda r: (r["dataset"], r["seed"]))]}
    for dataset in sorted({r["dataset"] for r in rows}):
        out["by_dataset_mean_delta"][str(dataset)] = float(
            np.mean([r["delta"] for r in rows if r["dataset"] == dataset]))
    verdicts = [r["verdict"] for r in rows]
    if set(verdicts) != {"pending"}:
        out["verdicts"] = {v: verdicts.count(v) for v in sorted(set(verdicts))}
    return out


def decisiveness(model):
    cells = gw.free_cells(model["scenario"])
    probs = gw.movement_probabilities(model["commands"][cells], NOMINAL_BETA)
    return {"mean_top_probability": float(probs.max(axis=1).mean()),
            "mean_entropy_nats": float(-(probs * np.log(probs)).sum(axis=1).mean()),
            "mean_abs_preference": float(np.abs(model["commands"][cells]).mean()),
            "share_of_preferences_at_bound": float((np.abs(model["commands"][cells]) > 0.95).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    inputs = {name: args.evidence / "evaluation_v1" / name
              for name in ("model_manifest.json", "state_tables.json", "exact_metrics.json")}
    models = load_models(args.evidence)
    exact = json.loads(inputs["exact_metrics.json"].read_text())
    parity = check_parity(models, exact)
    assert max(parity["max_abs_goal_difference"], parity["max_abs_hazard_difference"]) < 1e-9, parity
    bounds = {beta: gw.probability_bounds(beta) for beta in BETAS}

    pairs = defaultdict(dict)
    for model in models.values():
        pairs[model["pair"]][model["arm"]] = model
    assert len(pairs) == 54 and all(set(p) == set(ARMS) for p in pairs.values())

    frontier = {scenario: hazard_frontier(scenario) for scenario in CONDITIONS}

    # ---- per-model movement probabilities at the cells of interest -------------
    cell_rows = []
    for model in models.values():
        scenario = model["scenario"]
        cells = sorted(set(frontier[scenario]) |
                       ({B, GATE, C, *BOTTLENECK_CONTEXT_CELLS} if scenario == "bottleneck" else set()))
        for beta in BETAS:
            for cell in cells:
                probs = gw.movement_probabilities(model["commands"][cell], beta)
                row = {"model": model["id"], "pair": model["pair"], "scenario": scenario,
                       "condition": model["condition"], "dataset": model["dataset"],
                       "seed": model["seed"], "arm": model["arm"], "beta": beta, "cell": cell}
                for d, name in enumerate(gw.DIRECTION_NAMES):
                    row[f"preference_{name}"] = float(model["commands"][cell][d])
                    row[f"p_{name}"] = float(probs[d])
                    row[f"role_{name}"] = gw.direction_role(scenario, cell, d)
                saved = exact[model["id"]][str(beta)][str(HORIZON)][str(cell)]
                row.update(goal_200=saved["goal"], hazard_200=saved["hazard"],
                           timeout_200=saved["timeout"], crossing_200=saved["crossing"],
                           discounted_return_200=saved["discounted_return"])
                cell_rows.append(row)

    # ---- matched native -> BCA shifts -------------------------------------------
    shift_rows = []
    for pair, arms in pairs.items():
        native, bca = arms["native_aux"], arms["bca_direct"]
        scenario = native["scenario"]
        for beta in BETAS:
            for cell, direction in frontier[scenario].items():
                label = "dangerous_action_24_down" if (scenario, cell) == ("honeypot", 24) \
                    else f"hazard_from_{cell}"
                shift_rows.append(shift_row(pair, native, bca, label, cell, (direction,), True, beta, bounds))
            if scenario == "bottleneck":
                for label, cell, directions, harmful in BOTTLENECK_TARGETS:
                    shift_rows.append(shift_row(pair, native, bca, label, cell, directions, harmful, beta, bounds))

    def select(scenario, condition, target, beta=NOMINAL_BETA):
        return [r for r in shift_rows if r["scenario"] == scenario and r["condition"] == condition
                and r["target"] == target and r["beta"] == beta]

    summary = {"inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()},
               "parity_with_saved_exact_outcomes": parity,
               "probability_floor_and_ceiling": {str(beta): list(bounds[beta]) for beta in BETAS},
               "note": "All probabilities are per executed step at the stated beta. beta 2 is the "
                       "original setting. Deltas are BCA minus matched native."}

    # ---- Honeypot ------------------------------------------------------------------
    honeypot = {"dangerous_action_24_down": {c: summarize_shifts(select("honeypot", c, "dangerous_action_24_down"))
                                             for c in CONDITIONS["honeypot"]},
                "dangerous_action_24_down_by_beta": {
                    str(beta): {c: summarize_shifts(select("honeypot", c, "dangerous_action_24_down", beta))
                                for c in CONDITIONS["honeypot"]} for beta in BETAS}}

    # Is the drop specific to the hazard direction, or does the harmless backward
    # step (LEFT from 24 lands on ordinary cell 23) drop just as much?
    specificity = {}
    for condition in CONDITIONS["honeypot"]:
        rows = []
        for pair, arms in pairs.items():
            native, bca = arms["native_aux"], arms["bca_direct"]
            if native["scenario"] != "honeypot" or native["condition"] != condition:
                continue
            pn = gw.movement_probabilities(native["commands"][24], NOMINAL_BETA)
            pb = gw.movement_probabilities(bca["commands"][24], NOMINAL_BETA)
            rows.append({"dataset": native["dataset"], "seed": native["seed"],
                         "native_down": float(pn[gw.DOWN]), "bca_down": float(pb[gw.DOWN]),
                         "native_left": float(pn[gw.LEFT]), "bca_left": float(pb[gw.LEFT]),
                         "native_up_plus_right": float(pn[gw.UP] + pn[gw.RIGHT]),
                         "bca_up_plus_right": float(pb[gw.UP] + pb[gw.RIGHT]),
                         "native_down_preference": float(native["commands"][24][gw.DOWN]),
                         "bca_down_preference": float(bca["commands"][24][gw.DOWN]),
                         "native_down_over_left": float(pn[gw.DOWN] / pn[gw.LEFT]),
                         "bca_down_over_left": float(pb[gw.DOWN] / pb[gw.LEFT])})
        specificity[condition] = {
            "mean": {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("dataset", "seed")},
            "pairs_where_down_over_left_fell": int(sum(r["bca_down_over_left"] < r["native_down_over_left"] for r in rows)),
            "pairs": len(rows), "per_pair": sorted(rows, key=lambda r: (r["dataset"], r["seed"]))}
    honeypot["specificity_at_24"] = specificity

    # All 16 hazard-adjacent cells, grouped only by geometry: does the hazard lie
    # toward the goal corner (up or right of the cell) or away from it?
    side = {cell: ("toward_goal_corner" if d in (gw.UP, gw.RIGHT) else "away_from_goal_corner")
            for cell, d in frontier["honeypot"].items()}
    by_side = {}
    for condition in CONDITIONS["honeypot"]:
        by_side[condition] = {}
        for group in ("toward_goal_corner", "away_from_goal_corner"):
            rows = [r for r in shift_rows if r["scenario"] == "honeypot" and r["condition"] == condition
                    and r["beta"] == NOMINAL_BETA and side.get(r["cell"]) == group]
            by_cell = {}
            for cell in sorted({r["cell"] for r in rows}):
                sub = [r for r in rows if r["cell"] == cell]
                by_cell[str(cell)] = {"hazard_direction": gw.DIRECTION_NAMES[frontier["honeypot"][cell]],
                                      "native_mean": float(np.mean([r["native_p"] for r in sub])),
                                      "bca_mean": float(np.mean([r["bca_p"] for r in sub])),
                                      "delta": stats([r["delta"] for r in sub])}
            by_side[condition][group] = {"native_mean": float(np.mean([r["native_p"] for r in rows])),
                                         "bca_mean": float(np.mean([r["bca_p"] for r in rows])),
                                         "delta": stats([r["delta"] for r in rows]), "by_cell": by_cell}
    honeypot["hazard_direction_probability_all_16_frontier_cells"] = by_side
    summary["honeypot"] = honeypot

    # ---- Bottleneck ----------------------------------------------------------------
    bottleneck = {label: {c: summarize_shifts(select("bottleneck", c, label)) for c in CONDITIONS["bottleneck"]}
                  for label, *_ in BOTTLENECK_TARGETS}
    chain = {}
    for condition in CONDITIONS["bottleneck"]:
        rows = []
        for pair, arms in pairs.items():
            native, bca = arms["native_aux"], arms["bca_direct"]
            if native["scenario"] != "bottleneck" or native["condition"] != condition:
                continue
            cn, hn = bridge_chain(native["commands"], NOMINAL_BETA)
            cb, hb = bridge_chain(bca["commands"], NOMINAL_BETA)
            en = exact[native["id"]][str(NOMINAL_BETA)][str(HORIZON)][str(B)]
            eb = exact[bca["id"]][str(NOMINAL_BETA)][str(HORIZON)][str(B)]
            rows.append({"dataset": native["dataset"], "seed": native["seed"],
                         "native_first_visit_cross": float(cn), "bca_first_visit_cross": float(cb),
                         "native_first_visit_hazard": float(hn), "bca_first_visit_hazard": float(hb),
                         "native_eventual_crossing_200": en["crossing"], "bca_eventual_crossing_200": eb["crossing"],
                         "native_goal_200": en["goal"], "bca_goal_200": eb["goal"],
                         "native_hazard_200": en["hazard"], "bca_hazard_200": eb["hazard"]})
        chain[condition] = {"mean": {k: float(np.mean([r[k] for r in rows])) for k in rows[0]
                                     if k not in ("dataset", "seed")},
                            "per_pair": sorted(rows, key=lambda r: (r["dataset"], r["seed"]))}
    bottleneck["bridge_from_B"] = chain
    summary["bottleneck"] = bottleneck

    # ---- What the data deletion itself did, holding the method fixed -----------------
    data_effect = {}
    by_key = {(m["scenario"], m["condition"], m["dataset"], m["seed"], m["arm"]): m for m in models.values()}
    probes = {"honeypot": (("p_down_at_24", 24, (gw.DOWN,)),),
              "bottleneck": (("p_right_at_B", B, (gw.RIGHT,)), ("p_up_hazard_at_B", B, (gw.UP,)),
                             ("p_right_at_gate", GATE, (gw.RIGHT,)), ("p_up_wall_at_C", C, (gw.UP,)))}
    for scenario, items in probes.items():
        data_effect[scenario] = {}
        for label, cell, directions in items:
            table = {}
            for arm in ARMS:
                table[arm] = {}
                for condition in CONDITIONS[scenario]:
                    values = [float(gw.movement_probabilities(m["commands"][cell], NOMINAL_BETA)[list(directions)].sum())
                              for key, m in sorted(by_key.items())
                              if key[0] == scenario and key[1] == condition and key[4] == arm]
                    table[arm][condition] = {"mean": float(np.mean(values)), "min": float(np.min(values)),
                                             "max": float(np.max(values)), "values": values}
            data_effect[scenario][label] = table
    summary["policy_probability_by_data_condition"] = data_effect

    # ---- Does BCA make policies more decisive everywhere? ----------------------------
    decisive = {}
    for scenario in CONDITIONS:
        decisive[scenario] = {}
        for condition in CONDITIONS[scenario]:
            per_arm = {arm: [decisiveness(m) for key, m in sorted(by_key.items())
                             if key[0] == scenario and key[1] == condition and key[4] == arm] for arm in ARMS}
            decisive[scenario][condition] = {
                metric: {"native_mean": float(np.mean([d[metric] for d in per_arm["native_aux"]])),
                         "bca_mean": float(np.mean([d[metric] for d in per_arm["bca_direct"]])),
                         "pairs_where_bca_higher": int(sum(b[metric] > n[metric] for n, b in
                                                           zip(per_arm["native_aux"], per_arm["bca_direct"])))}
                for metric in per_arm["native_aux"][0]}
    summary["decisiveness_over_all_free_cells"] = decisive

    for name, rows in (("per_model_cells.csv", cell_rows), ("pair_shifts.csv", shift_rows)):
        with open(args.out / name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({"models": len(models), "pairs": len(pairs), "cell_rows": len(cell_rows),
                      "shift_rows": len(shift_rows), **parity}))


if __name__ == "__main__":
    main()
