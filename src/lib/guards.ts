// Numerical guards: detect a diverged training step or physics run before it reaches the
// charts, so the lab rolls back to the last finite state instead of plotting NaN or freezing.

import type { Metrics } from "./rl";

/** Values beyond this magnitude are treated as diverged even if still finite. */
export const DIVERGENCE_LIMIT = 1e8;

export function isSane(v: number): boolean {
  return Number.isFinite(v) && Math.abs(v) <= DIVERGENCE_LIMIT;
}

/** Index of the first non-finite or exploding entry, or -1 when the whole array is sane. */
export function firstInsane(values: ArrayLike<number>): number {
  for (let i = 0; i < values.length; i++) if (!isSane(values[i])) return i;
  return -1;
}

export function metricsSane(m: Pick<Metrics, "estQ" | "trueQ" | "loss" | "tdLoss" | "cqlGap" | "oodRate" | "divergence">): boolean {
  return [m.estQ, m.trueQ, m.loss, m.tdLoss, m.cqlGap, m.oodRate, m.divergence].every(isSane);
}

export interface QCheck {
  ok: boolean;
  /** Human-readable reason when not ok. */
  reason?: string;
}

/** Explains which (state, action) entry broke, for the error log. */
export function checkQ(q: Float64Array, width: number): QCheck {
  const i = firstInsane(q);
  if (i < 0) return { ok: true };
  const s = Math.floor(i / 4);
  const a = i % 4;
  const v = q[i];
  const what = Number.isNaN(v) ? "NaN" : !Number.isFinite(v) ? (v > 0 ? "+Infinity" : "-Infinity") : `${v.toExponential(2)} (beyond ±${DIVERGENCE_LIMIT.toExponential(0)})`;
  return { ok: false, reason: `Q(${s % width},${Math.floor(s / width)}; action ${a}) became ${what}` };
}

/** Runs a physics/benchmark computation and turns throws or non-finite numeric results into a clear error. */
export function guardPhysics<T>(label: string, fn: () => T, numbers: (r: T) => number[]): T {
  let result: T;
  try {
    result = fn();
  } catch (e) {
    throw new Error(`${label} crashed: ${e instanceof Error ? e.message : String(e)}`);
  }
  const bad = numbers(result).findIndex((v) => !isSane(v));
  if (bad >= 0) throw new Error(`${label} diverged: a result became non-finite or exceeded ±${DIVERGENCE_LIMIT.toExponential(0)}.`);
  return result;
}
