"""Version A in both environments on nine fresh datasets: every claim of PROTOCOL.md Amendment 10.

    python -I -B confirm_both.py --label confirm_v5

Reads g0.json, gates.json and the dataset choices of one run. Trains nothing and changes no file of the run.
The Honeypot is judged by confirm_version_a.judge, unchanged (claims C0, R1, R2).
The Bottleneck claims are judged here, at each exit value: C0, U1 (cautious), U2 (no wider harm), U3 (brings
the bridge back). X is the chance that an episode from B crosses the unseen bridge.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirm_version_a as honeypot_rules  # noqa: E402

HERE = Path(__file__).resolve().parent
VERSION_A = honeypot_rules.VERSION_A
BOTTLENECK_GROUP = "bottleneck_r2_one_hot"
EXITS = (("4.5", "bottleneck_r2_exit45_one_hot"), ("3.5", "bottleneck_r2_exit35_one_hot"),
         ("2.5", "bottleneck_r2_exit25_one_hot"))
P_MARK = honeypot_rules.P_MARK


def judge_bottleneck(gates, selection=None, g0=None):
    """One exit value. `gates` is that setting's block of gates.json; `selection` its dataset choice, or None."""
    method = gates["methods"][VERSION_A]
    rows = method["G3_more_than_decisiveness"]["per_dataset"]      # carries X for the baseline and Version A
    count = len(rows)
    needed = honeypot_rules.datasets_needed(count)
    g1_passing = sum(row["passed"] for row in gates["G1_gap_misleads_the_baseline"]["per_dataset"])
    if selection is not None:
        competent = bool(selection["enough_valid_datasets"])
        extra = {"candidate_datasets": len(selection["candidates"]),
                 "dropped_not_competent": selection["dropped_not_competent"],
                 "valid_not_needed": selection["valid_not_needed"]}
    else:
        competent, extra = bool(g0["runs_passing"] == g0["runs"]), {}
    c0 = {"G1_datasets_passing": g1_passing, **extra, "passed": bool(competent and g1_passing >= needed)}

    lower_by = [row["baseline"] - row["method"] for row in rows]       # positive: Version A crosses less
    mean = sum(lower_by) / count
    caution_p = honeypot_rules.sign_flip_p(lower_by)
    recovery_p = honeypot_rules.sign_flip_p([-d for d in lower_by])
    u1 = {"mean_of_baseline_minus_version_a": mean, "p": caution_p,
          "version_a_lower_in": sum(d > 0 for d in lower_by), "ties": sum(d == 0 for d in lower_by),
          "version_a_higher_in": sum(d < 0 for d in lower_by), "passed": bool(mean > 0 and caution_p <= P_MARK)}
    u3 = {"mean_of_version_a_minus_baseline": -mean, "p": recovery_p,
          "passed": bool(-mean > 0 and recovery_p <= P_MARK)}
    harm = sum(row["passed"] for row in method["G5_no_wider_harm"]["per_dataset"])
    u2 = {"G5_datasets_passing": harm, "passed": bool(harm >= needed)}
    also = {name.split("_")[0]: sum(row["passed"] for row in method[name]["per_dataset"])
            for name in ("G2_width_responds", "G3_more_than_decisiveness", "G4_specific_to_the_gap")}
    return {"datasets": count, "datasets_needed": needed, "C0": c0, "U1_cautious": u1, "U2_no_wider_harm": u2,
            "U3_brings_the_bridge_back": u3, "other_gates_datasets_passing": also,
            "crossings_by_dataset": {"data_seeds": [row["data_seed"] for row in rows],
                                     "baseline": [row["baseline"] for row in rows],
                                     "version_a": [row["method"] for row in rows],
                                     "sharpened_baseline": [row["sharpened_baseline"] for row in rows]}}


def bottleneck_verdict(by_exit):
    """Each claim holds only if it holds at all three exit values and C0 holds there."""
    valid = all(block["C0"]["passed"] for block in by_exit.values())
    holds = {name: valid and all(block[name]["passed"] for block in by_exit.values())
             for name in ("U1_cautious", "U2_no_wider_harm", "U3_brings_the_bridge_back")}
    if not valid:
        text = "No claim: C0 failed at one or more exit values."
    elif holds["U3_brings_the_bridge_back"]:
        text = "U3 holds: Version A relieves the underestimation."
    elif holds["U1_cautious"]:
        text = "U3 fails and U1 holds: Version A is cautious about an unseen move whether the move is bad or good."
    else:
        text = "U3 fails and U1 is not shown at every exit value: Version A does not relieve the underestimation."
    if valid and not holds["U2_no_wider_harm"]:
        text += " U2 fails at one or more exit values: reported with its harm."
    return {"C0_at_every_exit": valid, **holds, "verdict": text}


def as_text(result):
    word = lambda ok: "pass" if ok else "FAIL"
    lines = ["## Honeypot", "", honeypot_rules.as_text(result["honeypot"]), "", "## Bottleneck", ""]
    for exit_value, block in result["bottleneck"]["by_exit"].items():
        n, c0, u1, u3 = block["datasets"], block["C0"], block["U1_cautious"], block["U3_brings_the_bridge_back"]
        by = block["crossings_by_dataset"]
        if "candidate_datasets" in c0:
            used = (f"{n} valid datasets used of {c0['candidate_datasets']} candidates "
                    f"(dropped: {c0['dropped_not_competent'] or 'none'})")
        else:
            used = f"{n} datasets"
        lines += [f"### Low exit pays {exit_value}", "",
                  f"C0: {used}, G1 {c0['G1_datasets_passing']} of {n}: {word(c0['passed'])}", "",
                  "| Claim | Result | Verdict |", "|---|---|---|",
                  f"| U1 cautious | baseline minus Version A {100 * u1['mean_of_baseline_minus_version_a']:+.2f} points; "
                  f"lower / tie / higher {u1['version_a_lower_in']} / {u1['ties']} / {u1['version_a_higher_in']}; "
                  f"p {u1['p']:.4f} | {word(u1['passed'])} |",
                  f"| U2 no wider harm | G5 passes in {block['U2_no_wider_harm']['G5_datasets_passing']} of {n} "
                  f"(mark {block['datasets_needed']}) | {word(block['U2_no_wider_harm']['passed'])} |",
                  f"| U3 brings the bridge back | Version A minus baseline {100 * u3['mean_of_version_a_minus_baseline']:+.2f} points; "
                  f"p {u3['p']:.4f} | {word(u3['passed'])} |", "",
                  f"Other gates, datasets passing of {n}: {block['other_gates_datasets_passing']}", "",
                  "| Dataset | Baseline | Version A | Sharpened baseline |", "|---|---|---|---|"]
        for i, seed in enumerate(by["data_seeds"]):
            lines.append(f"| {seed} | {100 * by['baseline'][i]:.2f}% | {100 * by['version_a'][i]:.2f}% | "
                         f"{100 * by['sharpened_baseline'][i]:.2f}% |")
        lines.append("")
    lines.append(result["bottleneck"]["overall"]["verdict"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    folder = HERE / "results" / args.label
    g0 = json.loads((folder / "g0.json").read_text())
    gates = json.loads((folder / "gates.json").read_text())

    def choice(group, setting):
        path = folder / f"selected_datasets_{group}.json"
        return json.loads(path.read_text())[setting] if path.exists() else None

    result = {"honeypot": honeypot_rules.judge([g0["honeypot"]], [gates["honeypot"]], [choice("honeypot", "honeypot")])}
    by_exit = {exit_value: judge_bottleneck(gates[setting], choice(BOTTLENECK_GROUP, setting), g0[setting])
               for exit_value, setting in EXITS}
    result["bottleneck"] = {"by_exit": by_exit, "overall": bottleneck_verdict(by_exit)}
    (folder / "both.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    text = as_text(result)
    (folder / "both.md").write_text("# Version A in both environments\n\nRule: PROTOCOL.md Amendment 10.\n\n" + text + "\n",
                                    encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
