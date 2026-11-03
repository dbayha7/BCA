// Risk statistics for the lab: histograms, quantiles, lower-tail VaR / CVaR of returns,
// the logged reward distribution of a Dataset, and the episode-return distribution of a
// deterministic greedy policy on a grid environment. Pure functions — no React, no DOM.
//
// Sign convention: everything here is about RETURNS (higher is better), so "risk" lives in
// the LOWER tail. VaR_α is the return level at or below which the worst α fraction of
// episodes fall, and CVaR_α is the average return of that worst α fraction
// (expected shortfall). CVaR_α ≤ VaR_α ≤ max and CVaR_α ≤ mean always hold.

import type { Dataset, Env } from "./rl";

/* ---------------- Histograms ---------------- */

export interface Histogram {
  /** Bin edges, strictly increasing, length = bins + 1. Bin i is [edges[i], edges[i+1]); the last bin also includes its right edge. */
  edges: number[];
  /** Count of values in each bin (length = bins). */
  counts: number[];
  /** Number of finite values that landed inside [edges[0], edges[last]]. */
  total: number;
  /** Finite values below edges[0] (only possible with explicit edges or range). */
  below: number;
  /** Finite values above edges[last] (only possible with explicit edges or range). */
  above: number;
  /** Non-finite values (NaN, ±Infinity) that were ignored. */
  dropped: number;
}

/** Evenly spaced edges over [lo, hi]; a degenerate range is widened by ±0.5. */
export function uniformEdges(lo: number, hi: number, bins: number): number[] {
  if (!Number.isInteger(bins) || bins < 1) throw new Error("bins must be a positive integer");
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    lo = 0;
    hi = 1;
  }
  if (hi < lo) [lo, hi] = [hi, lo];
  if (hi - lo < 1e-12) {
    lo -= 0.5;
    hi += 0.5;
  }
  const edges = new Array<number>(bins + 1);
  for (let i = 0; i <= bins; i++) edges[i] = lo + ((hi - lo) * i) / bins;
  edges[bins] = hi;
  return edges;
}

function finiteRange(values: ArrayLike<number>): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? [lo, hi] : null;
}

/** Index of the bin containing v (edges strictly increasing, v within [edges[0], edges[last]]). */
function binIndex(edges: number[], v: number): number {
  const last = edges.length - 2;
  if (v >= edges[last]) return last;
  let lo = 0;
  let hi = last;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (edges[mid] <= v) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/**
 * Histogram of `values`. Pass a bin COUNT (edges span `range`, or the data's finite min..max)
 * or an explicit array of strictly increasing EDGES. Values outside the edges are counted in
 * `below` / `above`; non-finite values in `dropped`.
 */
export function histogram(values: ArrayLike<number>, binsOrEdges: number | number[], range?: [number, number]): Histogram {
  let edges: number[];
  if (typeof binsOrEdges === "number") {
    const r = range ?? finiteRange(values) ?? [0, 1];
    edges = uniformEdges(r[0], r[1], binsOrEdges);
  } else {
    edges = [...binsOrEdges];
    if (edges.length < 2) throw new Error("histogram needs at least two edges");
    for (let i = 0; i < edges.length; i++) {
      if (!Number.isFinite(edges[i])) throw new Error("histogram edges must be finite");
      if (i > 0 && !(edges[i] > edges[i - 1])) throw new Error("histogram edges must be strictly increasing");
    }
  }
  const counts = new Array<number>(edges.length - 1).fill(0);
  let total = 0;
  let below = 0;
  let above = 0;
  let dropped = 0;
  const lo = edges[0];
  const hi = edges[edges.length - 1];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) {
      dropped++;
      continue;
    }
    if (v < lo) {
      below++;
      continue;
    }
    if (v > hi) {
      above++;
      continue;
    }
    counts[binIndex(edges, v)]++;
    total++;
  }
  return { edges, counts, total, below, above, dropped };
}

/* ---------------- Quantiles & moments ---------------- */

function sortedFinite(values: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) out.push(values[i]);
  return out.sort((a, b) => a - b);
}

/** Linear-interpolation quantile (Hyndman–Fan type 7, numpy's default) of an ASCENDING array. NaN if empty. */
export function quantileSorted(sorted: ArrayLike<number>, q: number): number {
  const n = sorted.length;
  if (n === 0) return NaN;
  const p = Math.min(1, Math.max(0, q)) * (n - 1);
  const i = Math.floor(p);
  const f = p - i;
  return i + 1 < n ? sorted[i] + f * (sorted[i + 1] - sorted[i]) : sorted[i];
}

/** Type-7 quantiles of the finite values for each probability in `qs` (one sort). */
export function quantiles(values: ArrayLike<number>, qs: number[]): number[] {
  const s = sortedFinite(values);
  return qs.map((q) => quantileSorted(s, q));
}

export function mean(values: ArrayLike<number>): number {
  let s = 0;
  let n = 0;
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) continue;
    s += values[i];
    n++;
  }
  return n ? s / n : NaN;
}

/** Population standard deviation (divides by n, matching benchmarkThruster's stdReturn). */
export function std(values: ArrayLike<number>): number {
  const m = mean(values);
  if (!Number.isFinite(m)) return NaN;
  let s = 0;
  let n = 0;
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) continue;
    s += (values[i] - m) * (values[i] - m);
    n++;
  }
  return Math.sqrt(s / n);
}

/* ---------------- Lower-tail VaR / CVaR ---------------- */

const EPS = 1e-9;

function checkLevel(level: number) {
  if (!(level > 0 && level <= 1)) throw new Error("tail level must be in (0, 1]");
}

function varSorted(s: number[], level: number): number {
  if (!s.length) return NaN;
  const k = Math.ceil(level * s.length - EPS);
  return s[Math.min(s.length - 1, Math.max(0, k - 1))];
}

function cvarSorted(s: number[], level: number): number {
  const n = s.length;
  if (!n) return NaN;
  const k = level * n; // fractional number of samples in the tail
  let whole = Math.floor(k + EPS);
  let frac = k - whole;
  if (frac < EPS) frac = 0;
  whole = Math.min(whole, n);
  let sum = 0;
  for (let i = 0; i < whole; i++) sum += s[i];
  if (frac > 0 && whole < n) sum += frac * s[whole];
  return sum / k;
}

/**
 * Lower-tail Value-at-Risk of returns: the empirical α-quantile using the inverse-CDF
 * definition, i.e. the ⌈α·n⌉-th smallest return. "With probability ≥ α the return is ≤ VaR."
 */
export function valueAtRisk(values: ArrayLike<number>, level: number): number {
  checkLevel(level);
  return varSorted(sortedFinite(values), level);
}

/**
 * Lower-tail Conditional VaR (expected shortfall) of returns: the mean of the worst α·n
 * returns, weighting the boundary sample fractionally when α·n is not an integer
 * (Rockafellar–Uryasev on the empirical distribution). If α·n < 1 this is the minimum.
 */
export function conditionalValueAtRisk(values: ArrayLike<number>, level: number): number {
  checkLevel(level);
  return cvarSorted(sortedFinite(values), level);
}

export interface ReturnRisk {
  /** Finite samples used. */
  n: number;
  mean: number;
  /** Population standard deviation. */
  std: number;
  min: number;
  max: number;
  median: number;
  /** Tail level α in (0, 1]. */
  level: number;
  /** Lower-tail VaR_α (a return level). */
  var: number;
  /** Lower-tail CVaR_α (mean of the worst α fraction). */
  cvar: number;
  /** α·n — how many samples the tail average is taken over (fractional). */
  tailSize: number;
}

/** Summary + lower-tail risk of a sample of returns. Empty input → n = 0 and NaN statistics. */
export function returnRisk(values: ArrayLike<number>, level = 0.05): ReturnRisk {
  checkLevel(level);
  const s = sortedFinite(values);
  const n = s.length;
  return {
    n,
    mean: mean(s),
    std: std(s),
    min: n ? s[0] : NaN,
    max: n ? s[n - 1] : NaN,
    median: quantileSorted(s, 0.5),
    level,
    var: varSorted(s, level),
    cvar: cvarSorted(s, level),
    tailSize: level * n,
  };
}

/* ---------------- Dataset reward distribution ---------------- */

export type TerminalKind = "goal" | "hazard" | "other" | "none";

export interface ActionRewardStats {
  action: number;
  name: string;
  /** Transitions logged with this action. */
  n: number;
  /** n / total transitions. */
  share: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  /** Counts on the dataset-wide `edges` (so actions stack into the overall histogram). */
  counts: number[];
}

export interface TerminalShare {
  /** Transitions whose reward came from entering this kind of terminal. */
  count: number;
  /** count / total transitions. */
  share: number;
  /** Σ r over these transitions. */
  rewardSum: number;
  /** Σ|r| over these transitions / Σ|r| over all transitions (0 if the dataset has no reward mass). */
  absRewardShare: number;
}

export interface DatasetRewardDistribution {
  n: number;
  edges: number[];
  overall: Histogram;
  summary: { mean: number; std: number; min: number; max: number; q05: number; q25: number; median: number; q75: number; q95: number };
  /** Number of distinct reward values (small ⇒ rewards are effectively categorical, e.g. gridworld −0.1 / +10 / −5). */
  distinctValues: number;
  perAction: ActionRewardStats[];
  /**
   * Where rewards come from: `goal` / `hazard` = transitions with done=true that land in a
   * goal / hazard terminal of env.terminals; `other` = done=true into a state not marked as a
   * terminal; `none` = non-terminal transitions.
   */
  terminal: Record<TerminalKind, TerminalShare>;
}

/** Reward distribution of the logged transitions: overall, per action, and goal/hazard-terminal shares. */
export function datasetRewardDistribution(ds: Dataset, bins = 24): DatasetRewardDistribution {
  const T = ds.transitions;
  const n = T.length;
  const rewards = new Float64Array(n);
  for (let i = 0; i < n; i++) rewards[i] = T[i].r;
  const range = finiteRange(rewards) ?? [0, 1];
  const edges = uniformEdges(range[0], range[1], bins);
  const overall = histogram(rewards, edges);

  const byAction: number[][] = [[], [], [], []];
  const kinds: TerminalKind[] = ["goal", "hazard", "other", "none"];
  const term = Object.fromEntries(kinds.map((k) => [k, { count: 0, share: 0, rewardSum: 0, absRewardShare: 0 }])) as Record<TerminalKind, TerminalShare>;
  const absByKind: Record<TerminalKind, number> = { goal: 0, hazard: 0, other: 0, none: 0 };
  let absTotal = 0;
  const distinct = new Set<number>();
  for (const t of T) {
    if (t.a >= 0 && t.a < 4) byAction[t.a].push(t.r);
    const kind: TerminalKind = t.done ? (ds.env.terminals.get(t.s2) ?? "other") : "none";
    term[kind].count++;
    term[kind].rewardSum += t.r;
    absByKind[kind] += Math.abs(t.r);
    absTotal += Math.abs(t.r);
    if (distinct.size <= 1000) distinct.add(t.r);
  }
  for (const k of kinds) {
    term[k].share = n ? term[k].count / n : 0;
    term[k].absRewardShare = absTotal > 0 ? absByKind[k] / absTotal : 0;
  }

  const perAction: ActionRewardStats[] = byAction.map((rs, a) => {
    const s = [...rs].sort((x, y) => x - y);
    return {
      action: a,
      name: ds.env.actionNames[a] ?? `a${a}`,
      n: rs.length,
      share: n ? rs.length / n : 0,
      mean: mean(s),
      std: std(s),
      min: s.length ? s[0] : NaN,
      max: s.length ? s[s.length - 1] : NaN,
      counts: histogram(rs, edges).counts,
    };
  });

  const sorted = Array.from(rewards).sort((a, b) => a - b);
  return {
    n,
    edges,
    overall,
    summary: {
      mean: mean(sorted),
      std: std(sorted),
      min: n ? sorted[0] : NaN,
      max: n ? sorted[n - 1] : NaN,
      q05: quantileSorted(sorted, 0.05),
      q25: quantileSorted(sorted, 0.25),
      median: quantileSorted(sorted, 0.5),
      q75: quantileSorted(sorted, 0.75),
      q95: quantileSorted(sorted, 0.95),
    },
    distinctValues: distinct.size,
    perAction,
    terminal: term,
  };
}

/* ---------------- Episode returns of a greedy policy from many starts ---------------- */

export type EpisodeOutcome = "goal" | "hazard" | "terminal" | "timeout";

export interface StartStateReturns {
  /** Which start set was used: explicit list, every data-visited state, or every free state. */
  startSet: "custom" | "data" | "all";
  /** Start states (ascending for "data"/"all"). */
  starts: number[];
  /** Undiscounted Σr over at most `horizon` steps, aligned with `starts`. */
  returns: number[];
  /** γ-discounted return Σ γ^t r_t, aligned with `starts`. */
  discounted: number[];
  /** Steps taken (≤ horizon). */
  steps: number[];
  outcomes: EpisodeOutcome[];
  outcomeCounts: Record<EpisodeOutcome, number>;
  /** Timeouts whose trajectory revisited a state — a deterministic cycle the policy never leaves. */
  loops: number;
  horizon: number;
  gamma: number;
}

export interface PolicyReturnOptions {
  /** Explicit start states (overrides `ds`). */
  starts?: number[];
  /** Dataset whose visited states define the start set (see below). */
  ds?: Dataset;
  /** Step cap per episode (default 100). */
  horizon?: number;
  /** Discount for the `discounted` returns (default 0.95, the lab's default γ). */
  gamma?: number;
}

/**
 * Episode-return distribution of a deterministic policy (e.g. the greedy Int8Array from
 * `evaluate`) on a grid Env, from MANY START STATES. Because env.step and the policy are both
 * deterministic, each start yields exactly one trajectory, so the "distribution" is over
 * start states with equal weight — it shows how the return depends on where the agent begins.
 *
 * Start set, in order of precedence:
 *  1. `opts.starts` if given (blocked/terminal states are skipped);
 *  2. with `opts.ds`: every non-terminal, non-blocked state the dataset visited
 *     (ds.visits[s] > 0), plus env.start — "the states the data says the agent can be in";
 *  3. otherwise every non-terminal, non-blocked state.
 * Each rollout is capped at `horizon` steps (default 100) and ends early on done.
 */
export function policyReturnDistribution(env: Env, policy: ArrayLike<number>, opts: PolicyReturnOptions = {}): StartStateReturns {
  const S = env.width * env.height;
  const horizon = Math.max(1, Math.floor(opts.horizon ?? 100));
  const gamma = opts.gamma ?? 0.95;
  const valid = (s: number) => Number.isInteger(s) && s >= 0 && s < S && !env.blocked.has(s) && !env.terminals.has(s);

  let startSet: StartStateReturns["startSet"];
  let starts: number[];
  if (opts.starts) {
    startSet = "custom";
    starts = opts.starts.filter(valid);
  } else if (opts.ds) {
    startSet = "data";
    const visits = opts.ds.visits;
    starts = [];
    for (let s = 0; s < S; s++) if ((visits[s] > 0 || s === env.start) && valid(s)) starts.push(s);
  } else {
    startSet = "all";
    starts = [];
    for (let s = 0; s < S; s++) if (valid(s)) starts.push(s);
  }

  const returns: number[] = [];
  const discounted: number[] = [];
  const steps: number[] = [];
  const outcomes: EpisodeOutcome[] = [];
  const outcomeCounts: Record<EpisodeOutcome, number> = { goal: 0, hazard: 0, terminal: 0, timeout: 0 };
  let loops = 0;
  const seenStamp = new Int32Array(S).fill(-1);

  starts.forEach((start, run) => {
    let s = start;
    let ret = 0;
    let disc = 0;
    let g = 1;
    let t = 0;
    let outcome: EpisodeOutcome = "timeout";
    let revisited = false;
    for (; t < horizon; t++) {
      if (seenStamp[s] === run) revisited = true;
      seenStamp[s] = run;
      const { s2, r, done } = env.step(s, policy[s]);
      ret += r;
      disc += g * r;
      g *= gamma;
      if (done) {
        outcome = env.terminals.get(s2) ?? "terminal";
        t++;
        break;
      }
      s = s2;
    }
    if (outcome === "timeout" && revisited) loops++;
    returns.push(ret);
    discounted.push(disc);
    steps.push(t);
    outcomes.push(outcome);
    outcomeCounts[outcome]++;
  });

  return { startSet, starts, returns, discounted, steps, outcomes, outcomeCounts, loops, horizon, gamma };
}
