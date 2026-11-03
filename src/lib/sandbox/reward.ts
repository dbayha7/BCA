// Custom reward functions: presets, the `ctx` object handed to user code, relabeling helpers and
// the per-state reward heatmap. `relabelRows` is the only function that calls user code and it is
// only ever invoked inside the sandbox worker; everything else here is trusted main-thread code.

import { buildDataset, type Dataset, type Env, type StepResult, type Transition } from "../rl";
import { UserCodeFailure, describeValue } from "./protocol";

export interface CodePreset {
  id: string;
  name: string;
  description: string;
  code: string;
}

export const REWARD_PRESETS: CodePreset[] = [
  {
    id: "original",
    name: "Original (logged)",
    description: "Returns the reward stored in the dataset — a no-op relabel.",
    code: `// The reward that was logged with the dataset, unchanged.
function reward(s, a, s2, ctx) {
  return ctx.original;
}
`,
  },
  {
    id: "sparse-goal",
    name: "Sparse goal",
    description: "+1 on entering the goal, 0 otherwise. No step cost, no hazard penalty.",
    code: `// +1 for entering the goal cell, 0 for everything else (no step cost, no hazard penalty).
// Datasets without a goal cell (HalfCheetah, ThrusterChain) become all-zero: nothing to learn.
function reward(s, a, s2, ctx) {
  return ctx.isGoal(s2) ? 1 : 0;
}
`,
  },
  {
    id: "distance-shaping",
    name: "Distance shaping",
    description: "Potential-based shaping toward ctx.target: denser signal, same optimal policy.",
    code: `// Potential-based shaping (Ng, Harada & Russell 1999): add γ·Φ(s′) − Φ(s) to the logged reward.
// Φ = −(Manhattan distance to ctx.target) pulls toward the target without changing which policy
// is optimal. ctx.target is the goal cell, or the env's natural target when there is none
// (HalfCheetah: fastest level cell; ThrusterChain: the rest cell p₁ = v₁ = 0).
const GAMMA = 0.95; // match the discount you train with
const SCALE = 0.2;

function phi(c, ctx) {
  return -SCALE * (Math.abs(ctx.x(c) - ctx.target.x) + Math.abs(ctx.y(c) - ctx.target.y));
}

function reward(s, a, s2, ctx) {
  const next = ctx.done ? 0 : phi(s2, ctx); // terminal states have zero potential
  return ctx.original + GAMMA * next - phi(s, ctx);
}
`,
  },
  {
    id: "hazard-averse",
    name: "Hazard-averse",
    description: "Hazards cost −20 and every step next to a hazard costs extra.",
    code: `// Make hazards far more costly and charge for standing next to one, so a learned policy keeps
// a safety margin. Datasets without hazard cells (ThrusterChain) are left unchanged.
const HAZARD = -20; // replaces the logged hazard penalty
const NEAR = 0.5; // cost per neighbouring hazard cell

function reward(s, a, s2, ctx) {
  if (ctx.isHazard(s2)) return HAZARD;
  let near = 0;
  const x = ctx.x(s2);
  const y = ctx.y(s2);
  for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
    const c = ctx.cell(x + dx, y + dy);
    if (c >= 0 && ctx.isHazard(c)) near++;
  }
  return ctx.original - NEAR * near;
}
`,
  },
];

/** Plain-data description of the grid, shipped to the worker to build `ctx`. */
export interface RewardEnvInfo {
  width: number;
  height: number;
  start: number;
  target: number;
  goals: number[];
  hazards: number[];
  blocked: number[];
  actionNames: string[];
}

export function rewardEnvInfo(env: Env): RewardEnvInfo {
  const goals: number[] = [];
  const hazards: number[] = [];
  env.terminals.forEach((kind, s) => (kind === "goal" ? goals : hazards).push(s));
  goals.sort((a, b) => a - b);
  hazards.sort((a, b) => a - b);
  const startY = Math.floor(env.start / env.width);
  const target = goals.length ? goals[0] : env.id === "halfcheetah-medium" ? startY * env.width + env.width - 1 : env.start;
  return {
    width: env.width,
    height: env.height,
    start: env.start,
    target,
    goals,
    hazards,
    blocked: [...env.blocked].sort((a, b) => a - b),
    actionNames: [...env.actionNames],
  };
}

export interface RewardCtx {
  readonly width: number;
  readonly height: number;
  x(s: number): number;
  y(s: number): number;
  /** Cell index of (x, y), or −1 when outside the grid. */
  cell(x: number, y: number): number;
  isGoal(s: number): boolean;
  isHazard(s: number): boolean;
  isBlocked(s: number): boolean;
  readonly goal: { x: number; y: number } | null;
  readonly target: { x: number; y: number };
  readonly actionNames: readonly string[];
  /** The logged reward of this transition. */
  readonly original: number;
  /** Whether this transition ended the episode. */
  readonly done: boolean;
}

export type RewardFn = (s: number, a: number, s2: number, ctx: RewardCtx) => unknown;

export function makeRewardCtxBase(info: RewardEnvInfo): Omit<RewardCtx, "original" | "done"> {
  const W = info.width;
  const H = info.height;
  const goals = new Set(info.goals);
  const hazards = new Set(info.hazards);
  const blocked = new Set(info.blocked);
  const xy = (c: number) => Object.freeze({ x: c % W, y: Math.floor(c / W) });
  return Object.freeze({
    width: W,
    height: H,
    x: (s: number) => s % W,
    y: (s: number) => Math.floor(s / W),
    cell: (x: number, y: number) => (Number.isInteger(x) && Number.isInteger(y) && x >= 0 && y >= 0 && x < W && y < H ? y * W + x : -1),
    isGoal: (s: number) => goals.has(s),
    isHazard: (s: number) => hazards.has(s),
    isBlocked: (s: number) => blocked.has(s),
    goal: info.goals.length ? xy(info.goals[0]) : null,
    target: xy(info.target),
    actionNames: Object.freeze([...info.actionNames]),
  });
}

/** Row layout for the worker: [s, a, s2, done(0|1)] per transition. */
export function packRows(rows: { s: number; a: number; s2: number; done: boolean }[]): Int32Array {
  const out = new Int32Array(rows.length * 4);
  rows.forEach((t, i) => {
    out[i * 4] = t.s;
    out[i * 4 + 1] = t.a;
    out[i * 4 + 2] = t.s2;
    out[i * 4 + 3] = t.done ? 1 : 0;
  });
  return out;
}

/**
 * Call the user's reward function on every row and validate each result is a finite number.
 * Runs INSIDE the sandbox worker. Throws UserCodeFailure with the offending row.
 */
export function relabelRows(rows: Int32Array, original: Float64Array, info: RewardEnvInfo, fn: RewardFn): Float64Array {
  const n = original.length;
  const base = makeRewardCtxBase(info);
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const s = rows[k * 4];
    const a = rows[k * 4 + 1];
    const s2 = rows[k * 4 + 2];
    const done = rows[k * 4 + 3] === 1;
    const ctx = Object.freeze({ ...base, original: original[k], done });
    const where = `transition #${k} (s=${s}, a=${a}, s2=${s2})`;
    let v: unknown;
    try {
      v = fn(s, a, s2, ctx);
    } catch (err) {
      throw new UserCodeFailure("runtime", `reward() threw at ${where}`, err);
    }
    if (typeof v !== "number" || !Number.isFinite(v)) {
      throw new UserCodeFailure("output", `reward() must return a finite number, got ${describeValue(v)} at ${where}.`);
    }
    out[k] = v;
  }
  return out;
}

/** The full (s, a) model of an exact environment, so the relabeled env's ground truth uses the new reward too. */
export function modelRows(env: Env): Transition[] {
  if (env.groundTruth !== "exact") return [];
  const out: Transition[] = [];
  const S = env.width * env.height;
  for (let s = 0; s < S; s++) {
    if (env.blocked.has(s) || env.terminals.has(s)) continue;
    for (let a = 0; a < 4; a++) {
      const { s2, r, done } = env.step(s, a);
      out.push({ s, a, r, s2, done });
    }
  }
  return out;
}

export interface RewardJobData {
  rows: Int32Array;
  original: Float64Array;
  info: RewardEnvInfo;
  /** Rows [0, nData) are the dataset's transitions; the rest are the env model (exact envs only). */
  nData: number;
}

export function buildRewardJob(base: Dataset): RewardJobData {
  const model = modelRows(base.env);
  const all = [...base.transitions, ...model];
  const original = new Float64Array(all.length);
  all.forEach((t, i) => (original[i] = t.r));
  return { rows: packRows(all), original, info: rewardEnvInfo(base.env), nData: base.transitions.length };
}

const ORIGIN = new WeakMap<Dataset, Dataset>();

/** The logged dataset a relabeled dataset was derived from (or the dataset itself). */
export function originalDataset(ds: Dataset): Dataset {
  return ORIGIN.get(ds) ?? ds;
}

/**
 * Trusted main-thread rebuild: same transitions with r replaced, statistics recomputed by
 * buildDataset, and an env whose step() returns the NEW reward (table lookup for exact envs,
 * empirical model of the relabeled data otherwise) so ground truth and V* stay consistent.
 */
export function relabelDataset(base: Dataset, rewards: Float64Array, label: string): Dataset {
  const model = modelRows(base.env);
  const nData = base.transitions.length;
  if (rewards.length !== nData + model.length) throw new Error(`Expected ${nData + model.length} rewards, got ${rewards.length}.`);
  for (let i = 0; i < rewards.length; i++) if (!Number.isFinite(rewards[i])) throw new Error(`Reward #${i} is not a finite number.`);
  const transitions: Transition[] = base.transitions.map((t, i) => ({ s: t.s, a: t.a, r: rewards[i], s2: t.s2, done: t.done }));
  const env: Env = {
    ...base.env,
    name: `${base.env.name} · ${label}`,
    description: `${base.env.description} Rewards relabeled by a custom function (${label}).`,
    step: base.env.step,
  };
  const ds = buildDataset(env, transitions, base.episodes);
  if (base.env.groundTruth === "exact") {
    const table = new Map<number, StepResult>();
    model.forEach((m, i) => table.set(m.s * 4 + m.a, { s2: m.s2, r: rewards[nData + i], done: m.done }));
    const fallback = base.env.step;
    env.step = (s, a) => table.get(s * 4 + a) ?? fallback(s, a);
  } else {
    const rMin = ds.rewardMin;
    env.step = (s, a) => {
      if (env.terminals.has(s)) return { s2: s, r: 0, done: true };
      const k = s * 4 + a;
      if (ds.counts[k] === 0) return { s2: s, r: rMin, done: false };
      const s2 = ds.modeNext[k];
      return { s2, r: ds.rewardMean[k], done: env.terminals.has(s2) };
    };
  }
  ORIGIN.set(ds, base);
  return ds;
}

export interface RewardHeatmap {
  /** Mean (relabeled or Δ) reward per state; NaN where the filter matches no transition. */
  values: Float64Array;
  counts: Float64Array;
  lo: number;
  hi: number;
}

/** Mean relabeled reward (or relabeled − original) per origin state s, optionally for one action. */
export function rewardHeatmap(base: Dataset, rewards: Float64Array, opts: { action: number | "all"; diff: boolean }): RewardHeatmap {
  const S = base.env.width * base.env.height;
  const sum = new Float64Array(S);
  const counts = new Float64Array(S);
  const T = base.transitions;
  for (let i = 0; i < T.length && i < rewards.length; i++) {
    const t = T[i];
    if (opts.action !== "all" && t.a !== opts.action) continue;
    sum[t.s] += opts.diff ? rewards[i] - t.r : rewards[i];
    counts[t.s]++;
  }
  const values = new Float64Array(S);
  let lo = Infinity;
  let hi = -Infinity;
  for (let s = 0; s < S; s++) {
    values[s] = counts[s] ? sum[s] / counts[s] : NaN;
    if (counts[s]) {
      lo = Math.min(lo, values[s]);
      hi = Math.max(hi, values[s]);
    }
  }
  if (!Number.isFinite(lo)) {
    lo = 0;
    hi = 0;
  }
  return { values, counts, lo, hi };
}

export interface RewardSummary {
  n: number;
  changed: number;
  min: number;
  max: number;
  mean: number;
  originalMean: number;
}

export function rewardSummary(base: Dataset, rewards: Float64Array): RewardSummary {
  const T = base.transitions;
  let changed = 0;
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let osum = 0;
  for (let i = 0; i < T.length; i++) {
    const r = rewards[i];
    if (Math.abs(r - T[i].r) > 1e-9) changed++;
    min = Math.min(min, r);
    max = Math.max(max, r);
    sum += r;
    osum += T[i].r;
  }
  const n = T.length;
  return { n, changed, min: n ? min : 0, max: n ? max : 0, mean: n ? sum / n : 0, originalMean: n ? osum / n : 0 };
}
