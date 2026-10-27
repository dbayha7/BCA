"""Build the meeting brief and raw control table from a completed immutable run."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from thrusterchain.reporting import validate_control_records

def write(path, data):
    with Path(path).open("x",encoding="utf8") as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False)

def build(run, report_path):
    run=Path(run); report_path=Path(report_path)
    r=json.loads((run/"benchmark_results.json").read_text())
    d=json.loads((run/"diagnostic_results.json").read_text())
    review=json.loads((run.parent.parent/"audits"/"ten_pass_v2"/"ten_pass_review.json").read_text())
    rows=[]
    for method, record in r["baselines"].items():
        for family, cell in record["families"].items():
            for i, seed in enumerate(cell["episode_seeds"]):
                rows.append(dict(method=method,family=family,evaluation_seed=seed,
                    return_undiscounted=cell["returns"][i],return_discounted=cell["discounted_returns"][i],
                    stall_fraction=cell["stall_fractions"][i]))
    if (run/"control_episode_records.csv").exists():
        validate_control_records(r,run/"control_episode_records.csv")
    else:
        with (run/"control_episode_records.csv").open("x",newline="",encoding="utf8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    recipe=dict(profile="smoke",independent_training_seeds=1,
        conditional_evaluation_episodes_per_family=r["evaluation_episodes_per_family"],
        evaluation_pairing="same reset seeds across methods and families; families are correlated",
        return_undiscounted="sum t=0..99 reward[t]",
        return_discounted="sum t=0..99 (0.98**t)*reward[t]",
        means="arithmetic mean over listed evaluation seeds within method/family",
        sample_standard_deviation="numpy.std(returns, ddof=1); null for n=1",
        confidence_intervals="none claimed for this tiny smoke",
        training_seed_uncertainty="not estimated from one trained dataset seed",
        diagnostic_repetitions=len(d["cells"][0]["repetitions"]),
        diagnostic_recipe="See diagnostic_results.json statistical_recipe and per-repetition raw_records; empirical fractions retain refusal and infinite statuses separately.",
        no_pooled_safety_or_policy_improvement_claim=True)
    write(run/"statistical_recipe.json",recipe)
    labels={"behavior_cloning":"Offline BC","random":"Uniform random",
            "true_model_mpc":"True-model MPC","identified_model_mpc":"Train-only ID + MPC"}
    families=("recorded","strong_current","big_waves","big_kicks")
    table=["| Controller | Recorded | Strong current | Big waves | Big kicks |",
           "|---|---:|---:|---:|---:|"]
    for method in labels:
        table.append("| "+labels[method]+" | "+" | ".join(f"{r['baselines'][method]['families'][f]['mean_return']:.2f}" for f in families)+" |")
    source=json.loads((run/"source_manifest_after.json").read_text())["source_sha256"]
    lines=["# ThrusterChain: meeting brief — October 23, 2026","",
      "The benchmark now runs independently of BCA: a fixed continuous offline dataset, offline training and selection, a frozen-policy receipt, and live simulator evaluation. Controlled coverage and terminal-Q diagnostics are separate data products.",
      "",
      "**Evidence level: verified software integration and a small diagnostic smoke. The nine-cell neural-policy research study is not complete.**",
      "",
      "## What ran",
      "",
      f"- {r['training_episodes']} offline episodes, one dataset/training seed, four disjoint offline roles.",
      f"- Four controllers, four evaluation families, {r['evaluation_episodes_per_family']} paired reset seeds per family; all 100-step episodes completed.",
      f"- Bounded MPC horizon {r['planner_horizon']}; this is a receding-horizon reference, not a full-episode optimum claim.",
      "- Nine terminal diagnostic cells, three independent repetitions per cell, four matched weighting/fallback arms.",
      f"- Ten topical post-change acceptance passes: {'all passed' if review['all_passed'] else 'not all passed'}. Independent source reviews are stored separately.",
      "- Historical audit: all45 terminal checkpoints restored; 540/540 sampled action rows matched exactly;22,683 historical files rehashed with zero byte changes.",
      "",
      "## Raw mean return from this smoke",
      "",
      "Higher (less negative) is better. These two evaluation seeds per family describe this one frozen policy; they do not establish algorithm superiority or training-seed uncertainty.",
      "",
      *table,
      "",
      "The structured baseline uses only public float32 training observations and executed commands. It assumes the public actuator map and reward structure. The true-model controller is explicitly evaluator-privileged. Both use the same constrained solver; neither is clipped unconstrained LQR.",
      "",
      "## What was corrected",
      "",
      "The finite task now includes remaining time and terminates at100; explicit legacy replay retains upstream truncation. Requested continuous-action density is separate from executed float32 mass and unknown trajectory occupancy. Query support, missing target mass, infinite quantiles and ESS have distinct statuses. Independent review caught edge-case support cancellation, density underflow, duplicate calibration roots, reversible NumPy read-only state, and weighted-quantile endpoint issues; fixes and regressions are preserved in the audit trail.",
      "",
      "## Exact historical actor equation",
      "",
      "D = stop_gradient(mean(abs(Q1))); c =2.5/D; U = R_frozen * max(eta_frozen,1e-6) * u_frozen.",
      "",
      "L_actor = -c * mean(Q1-U) + mean_batch,action((a_policy-a_logged)^2). There is no denominator epsilon. Critic loss and target are unchanged. Width parameters are frozen while the action derivative through eta remains live. The study used unweighted calibration, not IWBCA. Native IQL/CQL/ReBRAC integrations have different objectives.",
      "",
      "See [the pinned source and normalization audit](docs/historical_actor_audit.md). This fresh audit verified the newer45-run path; it does not close the separate older Unifloral exporter issue.",
      "",
      "## Limits and next research gates",
      "",
      "- Terminal Q equals the reward only for the final finite-objective decision. The response applies the actual environment float32 action map; covariate density remains over requested commands.",
      "- Bootstrap spread is not a posterior over true Q. Fixed-query marginal calibration does not establish adaptive action-selection coverage or policy improvement.",
      "- Zero action is the explicitly privileged minimum-command-cost fallback at the terminal step. Its success is not evidence that a learned uncertainty signal improves control.",
      "- The shifted families are withheld from this run's training/selection, but are established development families. A scientifically untouched final-family study remains pending.",
      "- Five independent training/data seeds, neural-host fidelity, full budgets, larger final diagnostic banks, fresh-host installation and institutional/license review remain open.",
      "- The Library handoff ZIP could not be downloaded. The reviewed design was supplied as parent text; archive hash/embedded manifest verification remains blocked.",
      "",
      "## Open and reproduce",
      "",
      "[Quickstart](README.md) · [Environment contract](docs/environment.md) · [Coverage definitions](docs/coverage_and_calibration.md) · [Regular protocol](docs/regular_mode.md)",
      "",
      f"Completed run: {run.as_posix()}. Raw control records and exact statistical recipe are alongside the results. Physics playback is contracts/trajectory_bank.json; it is fixed canonical playback, not a claim of arbitrary browser simulation.",
      "",
      "Final source snapshot SHA-256: "+source,
      ""]
    with report_path.open("x",encoding="utf8") as f:f.write("\n".join(lines))
    print(json.dumps({"report":str(report_path.resolve()),"raw_control_rows":len(rows),
                      "source_sha256":source}))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",required=True,type=Path)
    p.add_argument("--report",required=True,type=Path);a=p.parse_args();build(a.run,a.report)

