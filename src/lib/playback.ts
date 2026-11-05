// Pure helpers for frame-by-frame episode playback and live policy overlays.
// No React and no DOM, so everything here is unit-tested in tests/lab/playback.test.ts.
// Imports stay relative so `bun test` can load this module without the "@/" alias.

import { greedy } from "./rl";
import { THRUSTER_PROJECTION } from "./thrusterchain";

/* ---------------- Returns ---------------- */

/**
 * Discounted return-to-go for every step of ONE episode:
 * G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + … (rewards after the last frame count as 0).
 */
export function discountedReturnsToGo(rewards: ArrayLike<number>, gamma: number): Float64Array {
  const n = rewards.length;
  const out = new Float64Array(n);
  let g = 0;
  for (let t = n - 1; t >= 0; t--) {
    g = rewards[t] + gamma * g;
    out[t] = g;
  }
  return out;
}

/** One step of an episode as the inspector needs it: tabular state, action taken, reward. */
export interface PlaybackStep {
  s: number;
  a: number;
  r: number;
}

export interface ValueTrace {
  /** Q(s_t, a_t) as the critic currently predicts it. */
  predicted: Float64Array;
  /** Discounted return-to-go actually obtained from step t in this episode. */
  actual: Float64Array;
}

/** Critic prediction vs. realised discounted return for every step of an episode. */
export function episodeValueTrace(Q: ArrayLike<number>, steps: readonly PlaybackStep[], gamma: number): ValueTrace {
  const predicted = new Float64Array(steps.length);
  for (let t = 0; t < steps.length; t++) predicted[t] = Q[steps[t].s * 4 + steps[t].a] ?? 0;
  return { predicted, actual: discountedReturnsToGo(steps.map((st) => st.r), gamma) };
}

/* ---------------- Per-state inspection ---------------- */

export interface ActionInspection {
  a: number;
  q: number;
  /** Logged (s, a) samples in the dataset: ds.counts[s*4+a]. */
  count: number;
  /** True when the dataset never tried this action in this state. */
  ood: boolean;
  greedy: boolean;
}

export interface StateInspection {
  greedy: number;
  actions: ActionInspection[];
  /** Total logged samples in this state (all actions). */
  visits: number;
}

/** The four Q-values at a state, the greedy action, and the data support of every action. */
export function inspectState(Q: Float64Array, counts: ArrayLike<number>, s: number): StateInspection {
  const g = greedy(Q, s);
  const actions: ActionInspection[] = [];
  let visits = 0;
  for (let a = 0; a < 4; a++) {
    const count = counts[s * 4 + a] ?? 0;
    visits += count;
    actions.push({ a, q: Q[s * 4 + a] ?? 0, count, ood: count === 0, greedy: a === g });
  }
  return { greedy: g, actions, visits };
}

/** Data-support level in [0, 1] on a log scale, so ×1 vs ×10 is as visible as ×10 vs ×100. */
export function supportLevel(count: number, maxCount: number): number {
  if (count <= 0 || maxCount <= 0) return 0;
  return Math.min(1, Math.log1p(count) / Math.log1p(maxCount));
}

/* ---------------- Greedy-policy change tracking ---------------- */

export interface PolicyChangeTracker {
  /** Training version this tracker last ingested (NaN before the first ingest). */
  version: number;
  /** Greedy action per state at `version`. */
  policy: Int8Array | null;
  /** Version at which each state's greedy action last changed (−1 = never seen a change). */
  lastChanged: Int32Array;
  /** Greedy action each state had before its most recent change (−1 = none). */
  fromAction: Int8Array;
  /** States whose greedy action changed at `version`. */
  changedNow: number[];
}

export function createPolicyTracker(S: number): PolicyChangeTracker {
  return {
    version: Number.NaN,
    policy: null,
    lastChanged: new Int32Array(S).fill(-1),
    fromAction: new Int8Array(S).fill(-1),
    changedNow: [],
  };
}

/** States whose greedy action differs between two policies (optionally restricted to `eligible` states). */
export function diffPolicies(prev: ArrayLike<number>, next: ArrayLike<number>, eligible?: ArrayLike<number>): number[] {
  if (prev.length !== next.length) return [];
  const out: number[] = [];
  for (let s = 0; s < next.length; s++) {
    if (eligible && !eligible[s]) continue;
    if (prev[s] !== next[s]) out.push(s);
  }
  return out;
}

/**
 * Feed the greedy policy of a training `version` into the tracker.
 * Idempotent for a version it has already seen (safe under React StrictMode double-render).
 * A policy of a different size (new dataset) restarts tracking with no changes reported.
 */
export function ingestPolicy(
  tr: PolicyChangeTracker,
  policy: Int8Array,
  version: number,
  eligible?: ArrayLike<number>,
): PolicyChangeTracker {
  if (tr.lastChanged.length !== policy.length) tr = createPolicyTracker(policy.length);
  const prev = tr.policy;
  if (tr.version === version && prev) return tr;
  const changedNow = prev ? diffPolicies(prev, policy, eligible) : [];
  const { lastChanged, fromAction } = tr;
  for (const s of changedNow) {
    lastChanged[s] = version;
    fromAction[s] = prev ? prev[s] : -1;
  }
  return { version, policy: new Int8Array(policy), lastChanged, fromAction, changedNow };
}

/** Highlight strength in [0, 1]: 1 when the change happened at `version`, fading to 0 over `fade` versions. */
export function changeRecency(tr: PolicyChangeTracker, s: number, version: number, fade: number): number {
  const at = tr.lastChanged[s];
  if (at === undefined || at < 0) return 0;
  const age = version - at;
  if (age < 0 || age >= fade) return 0;
  return 1 - age / Math.max(1, fade);
}

/* ---------------- ThrusterChain projection ---------------- */

/** (p₁, v₁) grid cell of a full 13-dim ThrusterChain state, identical to the projection the tabular agent learns on. */
export function thrusterCell(x: ArrayLike<number>): { cell: number; xi: number; yi: number } {
  const { xAxis, yAxis } = THRUSTER_PROJECTION;
  const clampI = (v: number, n: number) => Math.max(0, Math.min(n - 1, v));
  const xi = clampI(Math.floor(((x[xAxis.stateIndex] - xAxis.lo) / (xAxis.hi - xAxis.lo)) * xAxis.bins), xAxis.bins);
  const yi = clampI(Math.floor(((yAxis.hi - x[yAxis.stateIndex]) / (yAxis.hi - yAxis.lo)) * yAxis.bins), yAxis.bins);
  return { cell: yi * xAxis.bins + xi, xi, yi };
}

export interface ThrusterSummary {
  p1: number;
  v1: number;
  /** Oscillator energy Σ x[0..9]² (the state part of the cost). */
  energy: number;
  current: number;
  waveAmp: number;
}

export function thrusterSummary(x: ArrayLike<number>): ThrusterSummary {
  let energy = 0;
  for (let j = 0; j < 10; j++) energy += x[j] * x[j];
  return { p1: x[0], v1: x[1], energy, current: x[10], waveAmp: Math.hypot(x[11], x[12]) };
}

/* ---------------- Transport controls ---------------- */

export function clampIndex(i: number, total: number): number {
  if (total <= 0) return 0;
  return Math.max(0, Math.min(total - 1, Math.round(i)));
}

export type PlaybackCommand = { type: "step"; delta: number } | { type: "toggle" } | { type: "start" } | { type: "end" };

/** Keyboard map for a focused player: ←/→ (Shift = 10 steps), ,/. frame step, Space play/pause, Home/End. */
export function playbackKey(key: string, shift = false): PlaybackCommand | null {
  switch (key) {
    case "ArrowLeft":
      return { type: "step", delta: shift ? -10 : -1 };
    case "ArrowRight":
      return { type: "step", delta: shift ? 10 : 1 };
    case ",":
      return { type: "step", delta: -1 };
    case ".":
      return { type: "step", delta: 1 };
    case " ":
    case "Spacebar":
      return { type: "toggle" };
    case "Home":
      return { type: "start" };
    case "End":
      return { type: "end" };
    default:
      return null;
  }
}
