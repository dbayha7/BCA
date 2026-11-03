import { test, expect } from "bun:test";
import {
  conditionalValueAtRisk,
  datasetRewardDistribution,
  histogram,
  policyReturnDistribution,
  quantiles,
  returnRisk,
  uniformEdges,
  valueAtRisk,
} from "../../src/lib/risk";
import { evaluate, generateDataset, createQ, mulberry32, trainUpdates, policyEvaluation } from "../../src/lib/rl";

test("histogram with a bin count spans min..max and counts every finite value", () => {
  const h = histogram([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5);
  expect(h.edges).toEqual([0, 2, 4, 6, 8, 10]);
  expect(h.counts).toEqual([2, 2, 2, 2, 3]); // max lands in the last (closed) bin
  expect(h.total).toBe(11);
  expect(h.below + h.above + h.dropped).toBe(0);
});

test("histogram with explicit edges reports out-of-range and non-finite values", () => {
  const h = histogram([-1, 0, 0.5, 1, 1.5, 3, NaN, Infinity], [0, 1, 2]);
  expect(h.counts).toEqual([2, 2]);
  expect(h.below).toBe(1);
  expect(h.above).toBe(1);
  expect(h.dropped).toBe(2);
  expect(() => histogram([1], [0, 0])).toThrow();
  expect(() => histogram([1], 0)).toThrow();
});

test("histogram of a constant sample widens the range instead of dividing by zero", () => {
  const h = histogram([3, 3, 3], 4);
  expect(h.edges[0]).toBeCloseTo(2.5);
  expect(h.edges[4]).toBeCloseTo(3.5);
  expect(h.total).toBe(3);
  expect(uniformEdges(0, 1, 4)).toEqual([0, 0.25, 0.5, 0.75, 1]);
});

test("quantiles use linear interpolation (numpy default)", () => {
  expect(quantiles([4, 1, 3, 2], [0, 0.5, 1, 0.25])).toEqual([1, 2.5, 4, 1.75]);
  expect(Number.isNaN(quantiles([], [0.5])[0])).toBe(true);
});

test("lower-tail VaR / CVaR on 1..100", () => {
  const v = Array.from({ length: 100 }, (_, i) => i + 1).reverse();
  expect(valueAtRisk(v, 0.05)).toBe(5);
  expect(conditionalValueAtRisk(v, 0.05)).toBe(3); // mean of 1..5
  expect(valueAtRisk(v, 0.1)).toBe(10);
  expect(conditionalValueAtRisk(v, 0.1)).toBe(5.5);
  expect(conditionalValueAtRisk(v, 1)).toBe(50.5); // whole sample = mean
});

test("CVaR weights the boundary sample when alpha*n is fractional", () => {
  // n = 10, alpha = 0.25 -> 2.5 samples: (1 + 2 + 0.5*3) / 2.5
  const v = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
  expect(conditionalValueAtRisk(v, 0.25)).toBeCloseTo(4.5 / 2.5, 12);
  expect(valueAtRisk(v, 0.25)).toBe(3);
  // alpha*n < 1 -> CVaR is the minimum
  expect(conditionalValueAtRisk(v, 0.05)).toBe(1);
  // floating alpha*n = 3.0000000000000004 must not round up to 4 samples
  const w = Array.from({ length: 30 }, (_, i) => i);
  expect(valueAtRisk(w, 0.1)).toBe(2);
  expect(conditionalValueAtRisk(w, 0.1)).toBeCloseTo(1, 12);
});

test("returnRisk invariants: CVaR <= VaR, CVaR <= mean, std is population std", () => {
  const rng = mulberry32(5);
  const v = Array.from({ length: 257 }, () => -50 + 30 * rng() + (rng() < 0.1 ? -40 : 0));
  for (const level of [0.01, 0.05, 0.1, 0.25, 0.5]) {
    const r = returnRisk(v, level);
    expect(r.n).toBe(257);
    expect(r.cvar).toBeLessThanOrEqual(r.var + 1e-12);
    expect(r.cvar).toBeLessThanOrEqual(r.mean + 1e-12);
    expect(r.var).toBeGreaterThanOrEqual(r.min);
  }
  expect(returnRisk([1, 3]).std).toBe(1);
  const empty = returnRisk([]);
  expect(empty.n).toBe(0);
  expect(Number.isNaN(empty.cvar)).toBe(true);
  expect(() => returnRisk([1], 0)).toThrow();
});

test("dataset reward distribution: per-action histograms stack into the overall one", () => {
  const ds = generateDataset("gridworld-random", 7);
  const d = datasetRewardDistribution(ds, 20);
  expect(d.n).toBe(ds.transitions.length);
  expect(d.overall.total).toBe(d.n);
  const stacked = d.overall.counts.map((_, i) => d.perAction.reduce((s, a) => s + a.counts[i], 0));
  expect(stacked).toEqual(d.overall.counts);
  expect(d.perAction.reduce((s, a) => s + a.n, 0)).toBe(d.n);
  expect(d.perAction[0].name).toBe("Up");
  // gridworld rewards are categorical: -0.1 step, +10 goal, -5 hazard
  expect(d.distinctValues).toBe(3);
  const goals = ds.transitions.filter((t) => t.done && t.r === 10).length;
  const hazards = ds.transitions.filter((t) => t.done && t.r === -5).length;
  expect(d.terminal.goal.count).toBe(goals);
  expect(d.terminal.hazard.count).toBe(hazards);
  expect(d.terminal.goal.rewardSum).toBeCloseTo(10 * goals, 9);
  expect(d.terminal.goal.count + d.terminal.hazard.count + d.terminal.other.count + d.terminal.none.count).toBe(d.n);
  const shareSum = d.terminal.goal.absRewardShare + d.terminal.hazard.absRewardShare + d.terminal.other.absRewardShare + d.terminal.none.absRewardShare;
  expect(shareSum).toBeCloseTo(1, 12);
  expect(d.summary.min).toBe(-5);
  expect(d.summary.max).toBe(10);
});

test("greedy-policy return distribution covers every data-visited non-terminal state", () => {
  const ds = generateDataset("gridworld-random", 7);
  const Q = createQ(ds);
  trainUpdates(Q, ds, { alpha: 1, gamma: 0.95, lr: 0.3, batch: 64 }, mulberry32(1234), 400);
  const { policy } = evaluate(Q, ds, 0.95);
  const out = policyReturnDistribution(ds.env, policy, { ds, horizon: 100, gamma: 0.95 });
  const expected: number[] = [];
  for (let s = 0; s < ds.visits.length; s++) if (ds.visits[s] > 0 && !ds.env.terminals.has(s) && !ds.env.blocked.has(s)) expected.push(s);
  expect(out.startSet).toBe("data");
  expect(out.starts).toEqual(expected);
  expect(out.returns.length).toBe(expected.length);
  const oc = out.outcomeCounts;
  expect(oc.goal + oc.hazard + oc.terminal + oc.timeout).toBe(expected.length);
  expect(out.loops).toBeLessThanOrEqual(oc.timeout);
  // Discounted returns match exact policy evaluation: exactly for episodes that terminate,
  // and within the truncation bound gamma^H * max|V| for the ones cut at the horizon.
  const V = policyEvaluation(ds.env, policy, 0.95);
  const maxAbsV = Math.max(...Array.from(V, Math.abs));
  out.starts.forEach((s, i) => {
    const err = Math.abs(out.discounted[i] - V[s]);
    if (out.outcomes[i] === "timeout") expect(err).toBeLessThanOrEqual(Math.pow(0.95, 100) * maxAbsV + 1e-6);
    else expect(err).toBeLessThan(1e-6);
  });
  // Undiscounted return of a goal-reaching episode = 10 - 0.1 * (steps - 1)
  out.starts.forEach((_, i) => {
    if (out.outcomes[i] === "goal") expect(out.returns[i]).toBeCloseTo(10 - 0.1 * (out.steps[i] - 1), 9);
    expect(out.steps[i]).toBeLessThanOrEqual(100);
  });
});

test("policyReturnDistribution start-set fallbacks", () => {
  const ds = generateDataset("gridworld-random", 7);
  const env = ds.env;
  const allRight = new Int8Array(env.width * env.height).fill(1);
  const all = policyReturnDistribution(env, allRight, { horizon: 30 });
  expect(all.startSet).toBe("all");
  expect(all.starts.length).toBe(env.width * env.height - env.blocked.size - env.terminals.size);
  const custom = policyReturnDistribution(env, allRight, { starts: [env.start, [...env.terminals.keys()][0]], horizon: 30 });
  expect(custom.starts).toEqual([env.start]); // terminal start dropped
  // From the bottom-left corner, "always right" walks along the bottom row and times out against the wall.
  expect(custom.outcomes[0]).toBe("timeout");
  expect(custom.returns[0]).toBeCloseTo(-3, 9);
  expect(custom.loops).toBe(1);
});
