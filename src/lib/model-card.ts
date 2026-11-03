// Model card generator: a plain-language, honest description of a trained policy — what data it
// learned from, how it was trained and evaluated, how well it did, and where it should not be trusted.

import type { ExperimentRecord } from "./experiments";
import { summarize, tailStability } from "./experiments";
import type { LicenseConfig } from "./licensing";
import { LICENSE_INFO } from "./licensing";

export interface ModelCardInput {
  run: ExperimentRecord;
  dataset: { name: string; description: string; tag: string; transitions: number; episodes: number; groundTruth: "exact" | "empirical" };
  updatesPerStep: number;
  anomalies: string[];
  license?: LicenseConfig;
  /** Fraction of data-visited states where the greedy action never appears in the data. */
  oodRate?: number;
}

const f = (v: number | null | undefined, d = 2) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(d));

export function buildModelCard(i: ModelCardInput): string {
  const { run, dataset } = i;
  const s = summarize(run);
  const st = tailStability(run.history);
  const share = s.trueQ !== null && run.optimal > 0 ? `${((s.trueQ / run.optimal) * 100).toFixed(0)}% of V*(start)` : "—";
  const truth =
    dataset.groundTruth === "exact"
      ? "exact policy evaluation on the environment's known dynamics"
      : "policy evaluation on the empirical MDP built from the dataset (most-common next state, mean reward); for unseen (state, action) pairs this model is itself an assumption";
  const lic = i.license ? LICENSE_INFO[i.license.kind] : null;
  const lines = [
    `# Model card: ${run.name || "untitled run"}`,
    "",
    `Generated ${new Date().toISOString()} by the Offline Learning Lab.`,
    "",
    "## Model",
    "",
    `- Type: tabular Q-function (${run.q.length / 4} states × 4 actions), greedy policy.`,
    `- Method: conservative Q-learning (TD error + α · (logsumexp Q(s,·) − Q(s, a_data))), trained offline — no environment interaction during training.`,
    `- Final hyperparameters: α ${run.hyper.alpha}, γ ${run.hyper.gamma}, learning rate ${run.hyper.lr}, batch ${run.hyper.batch}${run.hyperLog.length > 1 ? ` (changed ${run.hyperLog.length - 1} time${run.hyperLog.length > 2 ? "s" : ""} during training; see the audit log)` : ""}.`,
    `- Training: ${s.updates.toLocaleString()} updates (${run.history.length.toLocaleString()} steps × ${i.updatesPerStep}), minibatch seed ${run.seed}.`,
    "",
    "## Data",
    "",
    `- ${dataset.name} (${dataset.tag}): ${dataset.description}`,
    `- ${dataset.transitions.toLocaleString()} logged transitions from ${dataset.episodes.toLocaleString()} episodes.`,
    run.reward ? `- Rewards were relabeled by a custom function (“${run.reward.name}”); the policy optimizes that reward, not the logged one. The function's code is in the audit log.` : "",
    "",
    "## Evaluation",
    "",
    `- True return V^π (averaged over data-visited states): ${f(s.trueQ)}; best during training ${f(s.bestTrueQ)}; optimal V*(start) ${f(run.optimal)} (${share}).`,
    `- The critic's own estimate: ${f(s.estQ)} — overestimation gap ${s.gap === null ? "—" : `${s.gap >= 0 ? "+" : ""}${s.gap.toFixed(2)}`}.`,
    `- Greedy actions never seen in the data: ${i.oodRate === undefined ? (s.oodRate === null ? "—" : `${(s.oodRate * 100).toFixed(1)}%`) : `${(i.oodRate * 100).toFixed(1)}%`} of data-visited states.`,
    st ? `- Stability over the last ${st.window} steps: σ ${f(st.std)}, largest drawdown ${f(st.maxDrawdown)}.` : "- Stability: too few steps to judge.",
    `- Ground truth comes from ${truth}.`,
    "",
    "## Known issues during training",
    "",
    ...(i.anomalies.length || run.guardEvents.length
      ? [...run.guardEvents.map((g) => `- Guard at ${g.atUpdates.toLocaleString()} updates: ${g.message}`), ...i.anomalies.slice(0, 12).map((a) => `- ${a}`), ...(i.anomalies.length > 12 ? [`- …and ${i.anomalies.length - 12} more in the audit log.`] : [])]
      : ["- None flagged."]),
    "",
    "## Limitations and intended use",
    "",
    "- Learned from a fixed dataset: behaviour in states or actions the data does not cover is extrapolation, and the overestimation gap above is the lab's measure of how much the critic trusts it.",
    "- A tabular policy on a discretised grid; for ThrusterChain the grid is a 2-D projection of a 13-dimensional system, so the policy ignores everything outside that projection.",
    "- Evaluated in simulation only. Do not deploy to hardware without independent safety limits and testing.",
    lic ? `- License: ${lic.name}${i.license?.organization ? ` — ${i.license.organization}` : ""}.` : "",
    run.notes ? `\n## Notes from the researcher\n\n${run.notes}` : "",
    "",
  ];
  return lines.filter((l) => l !== "").join("\n").replace(/\n## /g, "\n\n## ").replace(/\n\n\n+/g, "\n\n") + "\n";
}
