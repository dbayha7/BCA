// Serialize a trained tabular policy + metadata into a structured, portable JSON file.

import { greedy, type Dataset, type EvalResult, type Hyper, type Metrics } from "@/lib/rl";

export interface PolicyExport {
  format: "offline-learning-lab/policy";
  version: 1;
  exportedAt: string;
  algorithm: {
    name: string;
    family: "tabular-cql";
    updatesPerStep: number;
  };
  dataset: {
    id: string;
    name: string;
    tag: string;
    description: string;
    groundTruth: "exact" | "empirical";
    width: number;
    height: number;
    transitions: number;
    episodes: number;
    rewardRange: [number, number];
    actionNames: string[];
    start: [number, number];
  };
  hyperparameters: Hyper & { seed: number };
  training: {
    iterations: number;
    totalUpdates: number;
  };
  finalMetrics: {
    estimatedQ: number;
    groundTruthQ: number;
    overestimationGap: number;
    loss: number;
    tdLoss: number;
    cqlGap: number;
    oodActionRate: number;
    policyDivergence: number;
  } | null;
  policy: {
    /** Greedy action index per state (row-major, -1 for blocked/terminal), 0=Up 1=Right 2=Down 3=Left. */
    greedyActions: number[];
    /** Q(s,a) weights, row-major states × 4 actions. */
    qValues: number[][];
  };
}

export function buildPolicyExport(
  Q: Float64Array,
  ds: Dataset,
  hyper: Hyper,
  history: Metrics[],
  evalRes: EvalResult | null,
  seed: number,
  updatesPerStep: number,
): PolicyExport {
  const { env } = ds;
  const S = env.width * env.height;
  const qValues: number[][] = [];
  const greedyActions: number[] = [];
  for (let s = 0; s < S; s++) {
    const b = s * 4;
    qValues.push([Q[b], Q[b + 1], Q[b + 2], Q[b + 3]].map((v) => Math.round(v * 1e6) / 1e6));
    greedyActions.push(env.blocked.has(s) || env.terminals.has(s) ? -1 : greedy(Q, s));
  }
  const last = history[history.length - 1] ?? null;
  return {
    format: "offline-learning-lab/policy",
    version: 1,
    exportedAt: new Date().toISOString(),
    algorithm: {
      name: "Tabular Conservative Q-Learning (CQL-style)",
      family: "tabular-cql",
      updatesPerStep,
    },
    dataset: {
      id: env.id,
      name: env.name,
      tag: env.tag,
      description: env.description,
      groundTruth: env.groundTruth,
      width: env.width,
      height: env.height,
      transitions: ds.transitions.length,
      episodes: ds.episodes,
      rewardRange: [ds.rewardMin, ds.rewardMax],
      actionNames: [...env.actionNames],
      start: [env.start % env.width, Math.floor(env.start / env.width)],
    },
    hyperparameters: { ...hyper, seed },
    training: {
      iterations: history.length,
      totalUpdates: last?.updates ?? 0,
    },
    finalMetrics:
      last && evalRes
        ? {
            estimatedQ: last.estQ,
            groundTruthQ: last.trueQ,
            overestimationGap: last.estQ - last.trueQ,
            loss: last.loss,
            tdLoss: last.tdLoss,
            cqlGap: last.cqlGap,
            oodActionRate: evalRes.oodRate,
            policyDivergence: evalRes.divergence,
          }
        : null,
    policy: { greedyActions, qValues },
  };
}

export function downloadPolicyExport(exp: PolicyExport) {
  const slug = exp.dataset.id.replace(/[^a-z0-9-]/gi, "-");
  const blob = new Blob([JSON.stringify(exp, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `policy-${slug}-${exp.training.totalUpdates}u.json`;
  a.click();
  URL.revokeObjectURL(url);
}
