// Serialize a trained tabular agent (Q-table + greedy policy) together with everything an
// external simulator needs to reconstruct and run it: state encoding, action decoding,
// environment rules / empirical MDP, hyperparameters, metrics and an integrity checksum.

import { greedy, type Dataset, type EvalResult, type Hyper, type Metrics } from "@/lib/rl";
import { THRUSTER_PROJECTION } from "@/lib/thrusterchain";

export type ExportTrigger =
  | { kind: "manual" }
  | { kind: "threshold-auto" | "threshold-manual"; metric: "trueReturn" | "estimatedQ"; threshold: number; reachedAtUpdates: number; valueAtExport: number };

type Cell = [number, number];

export interface PolicyExport {
  format: "offline-learning-lab/policy";
  version: 2;
  exportedAt: string;
  trigger: ExportTrigger;
  algorithm: {
    name: string;
    family: "tabular-cql";
    objective: string;
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
    start: Cell;
  };
  hyperparameters: Hyper & { seed: number };
  training: { iterations: number; totalUpdates: number };
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
  /** How to map observations → state index and action index → environment command. */
  interface: {
    stateSpace: {
      type: "discrete-grid";
      numStates: number;
      encoding: "row-major: s = y * width + x (x = column, y = row, origin top-left)";
      axisX: string;
      axisY: string;
      /** Present for ThrusterChain: continuous 13-dim state → grid cell projection. */
      continuousProjection?: typeof THRUSTER_PROJECTION;
    };
    actionSpace: {
      type: "discrete";
      numActions: 4;
      names: string[];
      /** Grid displacement per action for grid-based envs (dx, dy). */
      gridDeltas?: Cell[];
      /** Continuous command per action (ThrusterChain thruster vector). */
      commands?: number[][];
    };
  };
  /** Environment rules, so the agent can be dropped into a re-implementation of the env. */
  environment: {
    blocked: Cell[];
    terminals: { cell: Cell; kind: "goal" | "hazard" }[];
    dynamics: string;
    reward: string;
    gamma: number;
  };
  /** Empirical MDP estimated from the offline data: per (s, a) visit count, mean reward and modal next state. */
  empiricalModel: {
    layout: "states × 4 actions, row-major";
    counts: number[][];
    meanReward: number[][];
    modalNext: (Cell | null)[][];
  };
  policy: {
    layout: "qValues[s][a]; s row-major, a ∈ {0,1,2,3}";
    /** Greedy action per state (-1 for blocked/terminal). */
    greedyActions: number[];
    /** Q(s,a) weights, states × 4 actions. */
    qValues: number[][];
    /** max_a Q(s,a) per state. */
    stateValues: number[];
    /** Exact V^π of the greedy policy (policy evaluation on the ground-truth / empirical MDP). */
    trueValues: number[];
    /** true when the dataset contains the greedy (s, a) pair — false = out-of-distribution. */
    inSupport: boolean[];
  };
  reconstruction: {
    steps: string[];
    pythonSnippet: string;
    checksum: { algorithm: "fnv1a-32"; over: "policy.qValues JSON"; value: string };
  };
}

const r6 = (v: number) => Math.round(v * 1e6) / 1e6;

function fnv1a(str: string) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

function envRules(ds: Dataset): { dynamics: string; reward: string } {
  switch (ds.env.id) {
    case "gridworld-random":
      return {
        dynamics: "Deterministic 4-neighbour moves. Moving into a wall or off the grid leaves the agent in place. Goal/hazard cells are absorbing terminals.",
        reward: "+10 on entering the goal, −5 on entering a hazard, −0.1 per other step.",
      };
    case "halfcheetah-medium":
      return {
        dynamics: "x = forward velocity bin, y = body pitch bin (row 4 = level). Pitch up/down moves y by ∓1; Accelerate moves x+1 and also pitches up (y−1) when x ≥ 6; Brake moves x−1. Rows 0 and 8 are flip terminals.",
        reward: "0.05 + x/11 − 0.3·|y − 4|/4 per step; −4 on flipping (entering row 0 or 8).",
      };
    case "thrusterchain-stall":
      return {
        dynamics: "Exact 13-dim float64 linear system x' = A·x + B·stall(clip(cmd)) (ThrusterChain v4 spec). The tabular agent observes only the (p1, v1) projection described in interface.stateSpace.continuousProjection.",
        reward: "Spec reward computed on the full continuous state; tabular rewards are per-(cell, action) means of the logged transitions (see empiricalModel).",
      };
    default:
      return {
        dynamics: "Empirical MDP: each (s, a) transitions to its modal next state observed in the data; unseen pairs self-loop.",
        reward: "Mean logged reward for seen (s, a); unseen pairs receive the dataset minimum reward.",
      };
  }
}

export function buildPolicyExport(
  Q: Float64Array,
  ds: Dataset,
  hyper: Hyper,
  history: Metrics[],
  evalRes: EvalResult | null,
  seed: number,
  updatesPerStep: number,
  trigger: ExportTrigger = { kind: "manual" },
): PolicyExport {
  const { env } = ds;
  const W = env.width;
  const S = W * env.height;
  const cell = (s: number): Cell => [s % W, Math.floor(s / W)];
  const qValues: number[][] = [];
  const greedyActions: number[] = [];
  const stateValues: number[] = [];
  const trueValues: number[] = [];
  const inSupport: boolean[] = [];
  const counts: number[][] = [];
  const meanReward: number[][] = [];
  const modalNext: (Cell | null)[][] = [];
  for (let s = 0; s < S; s++) {
    const b = s * 4;
    const q = [Q[b], Q[b + 1], Q[b + 2], Q[b + 3]].map(r6);
    qValues.push(q);
    const inert = env.blocked.has(s) || env.terminals.has(s);
    const g = inert ? -1 : greedy(Q, s);
    greedyActions.push(g);
    stateValues.push(Math.max(...q));
    trueValues.push(evalRes ? r6(evalRes.vPi[s]) : 0);
    inSupport.push(g >= 0 && ds.counts[b + g] > 0);
    counts.push([0, 1, 2, 3].map((a) => ds.counts[b + a]));
    meanReward.push([0, 1, 2, 3].map((a) => r6(ds.rewardMean[b + a])));
    modalNext.push([0, 1, 2, 3].map((a) => (ds.modeNext[b + a] >= 0 ? cell(ds.modeNext[b + a]) : null)));
  }
  const last = history[history.length - 1] ?? null;
  const isThruster = env.id === "thrusterchain-stall";
  const isGrid = env.id === "gridworld-random" || env.id === "custom";
  const rules = envRules(ds);

  const steps = isThruster
    ? [
        "Load policy.qValues (shape [width*height][4]).",
        "At each step, project the 13-dim state x onto a cell with interface.stateSpace.continuousProjection.cellFormula.",
        "Pick a = argmax_a qValues[cell][a] (or use policy.greedyActions[cell]).",
        "Send interface.actionSpace.commands[a] to the simulator (clip to [-1,1], float32, then stall nonlinearity).",
      ]
    : [
        "Load policy.qValues (shape [width*height][4]).",
        "Encode the observation (x, y) as s = y * width + x.",
        "Pick a = argmax_a qValues[s][a] (or use policy.greedyActions[s]); -1 means terminal/blocked.",
        isGrid ? "Apply interface.actionSpace.gridDeltas[a] in your simulator (environment section documents walls, terminals and rewards)." : "Apply the action using the dynamics in the environment section.",
      ];

  const pythonSnippet = [
    "import json, numpy as np",
    "m = json.load(open('policy.json'))",
    "Q = np.array(m['policy']['qValues'])  # [S, 4]",
    "W = m['dataset']['width']",
    isThruster
      ? "p = m['interface']['stateSpace']['continuousProjection']\ndef cell(x):\n    xa, ya = p['xAxis'], p['yAxis']\n    xi = int(np.clip(np.floor((x[0]-xa['lo'])/(xa['hi']-xa['lo'])*xa['bins']), 0, xa['bins']-1))\n    yi = int(np.clip(np.floor((ya['hi']-x[1])/(ya['hi']-ya['lo'])*ya['bins']), 0, ya['bins']-1))\n    return yi*W + xi\ndef act(x): return m['interface']['actionSpace']['commands'][int(Q[cell(x)].argmax())]"
      : "def act(x, y): return int(Q[y*W + x].argmax())",
  ].join("\n");

  return {
    format: "offline-learning-lab/policy",
    version: 2,
    exportedAt: new Date().toISOString(),
    trigger,
    algorithm: {
      name: "Tabular Conservative Q-Learning (CQL-style)",
      family: "tabular-cql",
      objective: "TD error² + α · (logsumexp_a Q(s,a) − Q(s, a_data)), minibatch updates over logged transitions",
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
      start: cell(env.start),
    },
    hyperparameters: { ...hyper, seed },
    training: { iterations: history.length, totalUpdates: last?.updates ?? 0 },
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
    interface: {
      stateSpace: {
        type: "discrete-grid",
        numStates: S,
        encoding: "row-major: s = y * width + x (x = column, y = row, origin top-left)",
        axisX: env.axisX,
        axisY: env.axisY,
        ...(isThruster ? { continuousProjection: THRUSTER_PROJECTION } : {}),
      },
      actionSpace: {
        type: "discrete",
        numActions: 4,
        names: [...env.actionNames],
        ...(isGrid ? { gridDeltas: [[0, -1], [1, 0], [0, 1], [-1, 0]] as Cell[] } : {}),
        ...(isThruster ? { commands: THRUSTER_PROJECTION.actionCommands } : {}),
      },
    },
    environment: {
      blocked: [...env.blocked].map(cell),
      terminals: [...env.terminals].map(([s, kind]) => ({ cell: cell(s), kind })),
      dynamics: rules.dynamics,
      reward: rules.reward,
      gamma: hyper.gamma,
    },
    empiricalModel: { layout: "states × 4 actions, row-major", counts, meanReward, modalNext },
    policy: {
      layout: "qValues[s][a]; s row-major, a ∈ {0,1,2,3}",
      greedyActions,
      qValues,
      stateValues,
      trueValues,
      inSupport,
    },
    reconstruction: {
      steps,
      pythonSnippet,
      checksum: { algorithm: "fnv1a-32", over: "policy.qValues JSON", value: fnv1a(JSON.stringify(qValues)) },
    },
  };
}

export function downloadPolicyExport(exp: PolicyExport) {
  const slug = exp.dataset.id.replace(/[^a-z0-9-]/gi, "-");
  const tag = exp.trigger.kind === "manual" ? "" : "-threshold";
  const blob = new Blob([JSON.stringify(exp, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `policy-${slug}-${exp.training.totalUpdates}u${tag}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/** Approximate file size in KB, for display. */
export function estimateExportKB(ds: Dataset) {
  const S = ds.env.width * ds.env.height;
  return Math.max(4, Math.round((S * 230 + 6000) / 1024));
}
