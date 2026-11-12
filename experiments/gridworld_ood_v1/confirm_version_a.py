"""Version A on nine fresh Honeypot datasets: claims C0, R1 and R2 of PROTOCOL.md Amendments 8 and 9.

    python -I -B confirm_version_a.py --label confirm_v4

Reads g0.json and gates.json written by evaluate_study.py, and the dataset choice written by run_study.py
when competence is judged per dataset (Amendment 9). Trains nothing and changes no file of the run.
With several labels the datasets are pooled and only printed; that is a description, not a verdict.
"""
import argparse
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION_A = "support_blind_width"
SETTING = "honeypot"
GATES = (("G2 width responds", "G2_width_responds"), ("G3 beats both controls", "G3_more_than_decisiveness"),
         ("G4 specific to the gap", "G4_specific_to_the_gap"), ("G5 no wider harm", "G5_no_wider_harm"))
CONTROLS = (("shuffled control", "shuffled_control"), ("sharpened baseline", "sharpened_baseline"))
P_MARK = 0.05


def datasets_needed(count):
    """Two thirds of the datasets, rounded up: 2 of 3, 6 of 9."""
    return -(-2 * count // 3)


def sign_flip_p(differences):
    """Exact one-sided p: the share of all sign patterns whose mean is at least the observed mean."""
    observed = sum(differences)
    patterns = list(itertools.product((1.0, -1.0), repeat=len(differences)))
    as_large = sum(sum(s * d for s, d in zip(signs, differences)) >= observed - 1e-12 for signs in patterns)
    return as_large / len(patterns)


def judge(g0, gates, selections=()):
    """g0 and gates are lists with one entry per label: the Honeypot blocks of g0.json and gates.json.
    selections holds, per label, the dataset choice of Amendment 9, or None for a run without one."""
    methods = [g["methods"][VERSION_A] for g in gates]
    g1_rows = [row for g in gates for row in g["G1_gap_misleads_the_baseline"]["per_dataset"]]
    count = len(g1_rows)
    needed = datasets_needed(count)
    runs_passing, runs = sum(x["runs_passing"] for x in g0), sum(x["runs"] for x in g0)
    g1_passing = sum(row["passed"] for row in g1_rows)
    chosen = [s for s in selections if s is not None]
    if chosen:      # Amendment 9: competence is judged per dataset, and only valid datasets are used
        competent = all(s["enough_valid_datasets"] for s in chosen)
        extra = {"candidate_datasets": sum(len(s["candidates"]) for s in chosen),
                 "dropped_not_competent": [seed for s in chosen for seed in s["dropped_not_competent"]],
                 "valid_not_needed": [seed for s in chosen for seed in s["valid_not_needed"]]}
    else:           # Amendment 8: every baseline run with complete data must be competent
        competent, extra = runs_passing == runs, {}
    c0 = {"G0_runs_passing": runs_passing, "G0_runs": runs, "G1_datasets_passing": g1_passing, **extra,
          "passed": bool(competent and g1_passing >= needed)}

    r1 = {}
    for name, key in GATES:
        passing = sum(row["passed"] for m in methods for row in m[key]["per_dataset"])
        r1[name] = {"datasets_passing": passing, "passed": bool(passing >= needed)}
    r1_passed = all(v["passed"] for v in r1.values())

    g3_rows = [row for m in methods for row in m["G3_more_than_decisiveness"]["per_dataset"]]
    r2 = {}
    for name, key in CONTROLS:
        differences = [row[key] - row["method"] for row in g3_rows]
        mean = sum(differences) / len(differences)
        p = sign_flip_p(differences)
        r2[name] = {"differences_control_minus_version_a": differences, "mean_difference": mean, "p": p,
                    "version_a_lower_in": sum(d > 0 for d in differences),
                    "ties": sum(d == 0 for d in differences),
                    "control_lower_in": sum(d < 0 for d in differences),
                    "passed": bool(mean > 0 and p <= P_MARK)}
    r2_passed = all(v["passed"] for v in r2.values())

    if not c0["passed"]:
        verdict = "No claim: the environment did not pass C0 on these data."
    elif r1_passed:
        verdict = "R1 passes: Version A is replicated in the Honeypot."
    elif r2_passed:
        verdict = ("R1 fails and R2 passes: Version A has fewer hazard endings than both controls on average, "
                   "and the per-dataset gates did not pass.")
    else:
        verdict = ("R1 fails and R2 fails: Version A's protection is not shown to be more than "
                   "a more decisive baseline on this testbed.")
    return {"datasets": count, "datasets_needed": needed, "C0": c0, "R1": {"gates": r1, "passed": r1_passed},
            "R2": {"controls": r2, "passed": r2_passed},
            "hazard_endings_by_dataset": {"data_seeds": [row["data_seed"] for row in g3_rows],
                                          "baseline": [row["baseline"] for row in g3_rows],
                                          "version_a": [row["method"] for row in g3_rows],
                                          **{key: [row[key] for row in g3_rows] for _, key in CONTROLS}},
            "verdict": verdict}


def as_text(result):
    n, need = result["datasets"], result["datasets_needed"]
    word = lambda ok: "pass" if ok else "FAIL"
    c0, by = result["C0"], result["hazard_endings_by_dataset"]
    if "candidate_datasets" in c0:
        competence = (f"{n} valid datasets used of {c0['candidate_datasets']} candidates "
                      f"(dropped, baseline not competent: {c0['dropped_not_competent'] or 'none'}; "
                      f"valid and not needed: {c0['valid_not_needed'] or 'none'})")
    else:
        competence = f"G0 {c0['G0_runs_passing']} of {c0['G0_runs']} runs"
    lines = [f"Datasets: {n}. A gate passes with at least {need}.", "",
             f"C0: {competence}, G1 {c0['G1_datasets_passing']} of {n} datasets: "
             f"{word(c0['passed'])}", "", "| R1 gate | Datasets passing | Verdict |", "|---|---|---|"]
    for name, v in result["R1"]["gates"].items():
        lines.append(f"| {name} | {v['datasets_passing']} of {n} | {word(v['passed'])} |")
    lines += [f"\nR1: {word(result['R1']['passed'])}", "",
              "| R2 control | Mean difference in H | Version A lower / tie / control lower | p | Verdict |",
              "|---|---|---|---|---|"]
    for name, v in result["R2"]["controls"].items():
        lines.append(f"| {name} | {100 * v['mean_difference']:+.2f} points | {v['version_a_lower_in']} / {v['ties']} / "
                     f"{v['control_lower_in']} | {v['p']:.4f} | {word(v['passed'])} |")
    lines += [f"\nR2: {word(result['R2']['passed'])}", "", "| Dataset | Baseline | Version A | Shuffled control | "
              "Sharpened baseline |", "|---|---|---|---|---|"]
    for i, seed in enumerate(by["data_seeds"]):
        lines.append(f"| {seed} | {100 * by['baseline'][i]:.2f}% | {100 * by['version_a'][i]:.2f}% | "
                     f"{100 * by['shuffled_control'][i]:.2f}% | {100 * by['sharpened_baseline'][i]:.2f}% |")
    lines += ["", result["verdict"]]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, nargs="+")
    args = parser.parse_args()
    folders = [HERE / "results" / label for label in args.label]
    g0 = [json.loads((f / "g0.json").read_text())[SETTING] for f in folders]
    gates = [json.loads((f / "gates.json").read_text())[SETTING] for f in folders]
    choice_files = [f / f"selected_datasets_{SETTING}.json" for f in folders]
    selections = [json.loads(p.read_text())[SETTING] if p.exists() else None for p in choice_files]
    result = judge(g0, gates, selections)
    text = as_text(result)
    if len(folders) == 1:
        (folders[0] / "version_a.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
        (folders[0] / "version_a.md").write_text(
            "# Version A on fresh Honeypot datasets\n\nHazard endings H, targeted arm, dataset means. "
            "Rule: PROTOCOL.md Amendment 8.\n\n" + text + "\n", encoding="utf-8", newline="\n")
    else:
        print("Pooled over " + ", ".join(args.label) + ": a description, not a verdict.\n")
    print(text)


if __name__ == "__main__":
    main()
