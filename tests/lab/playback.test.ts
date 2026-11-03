import { test, expect } from "bun:test";
import {
  changeRecency,
  clampIndex,
  createPolicyTracker,
  diffPolicies,
  discountedReturnsToGo,
  episodeValueTrace,
  ingestPolicy,
  inspectState,
  playbackKey,
  supportLevel,
  thrusterCell,
  thrusterSummary,
} from "../../src/lib/playback";
import { rolloutThruster, THRUSTER_PROJECTION } from "../../src/lib/thrusterchain";
import { generateDataset } from "../../src/lib/rl";

test("discounted return-to-go matches the forward definition", () => {
  const r = [1, -2, 0.5, 3];
  const g = 0.9;
  const G = discountedReturnsToGo(r, g);
  for (let t = 0; t < r.length; t++) {
    let fwd = 0;
    for (let k = t; k < r.length; k++) fwd += Math.pow(g, k - t) * r[k];
    expect(G[t]).toBeCloseTo(fwd, 12);
  }
  expect(G[r.length - 1]).toBe(3);
});

test("return-to-go edge cases: empty episode, gamma 0 and gamma 1", () => {
  expect(discountedReturnsToGo([], 0.95).length).toBe(0);
  expect(Array.from(discountedReturnsToGo([1, 2, 3], 0))).toEqual([1, 2, 3]);
  expect(Array.from(discountedReturnsToGo([1, 2, 3], 1))).toEqual([6, 5, 3]);
});

test("episodeValueTrace pairs Q(s_t,a_t) with the realised return", () => {
  const Q = new Float64Array(3 * 4);
  Q[0 * 4 + 1] = 5;
  Q[2 * 4 + 3] = -1;
  const steps = [
    { s: 0, a: 1, r: 1 },
    { s: 2, a: 3, r: 2 },
  ];
  const tr = episodeValueTrace(Q, steps, 0.5);
  expect(Array.from(tr.predicted)).toEqual([5, -1]);
  expect(Array.from(tr.actual)).toEqual([2, 2]);
});

test("inspectState reports Q, greedy action and data support with OOD flags", () => {
  const Q = new Float64Array(8);
  Q.set([0.1, 0.9, -0.3, 0.9], 4); // state 1: tie between actions 1 and 3 -> lowest index wins (matches rl.greedy)
  const counts = new Float64Array(8);
  counts.set([3, 0, 7, 0], 4);
  const ins = inspectState(Q, counts, 1);
  expect(ins.greedy).toBe(1);
  expect(ins.visits).toBe(10);
  expect(ins.actions.map((a) => a.q)).toEqual([0.1, 0.9, -0.3, 0.9]);
  expect(ins.actions.map((a) => a.count)).toEqual([3, 0, 7, 0]);
  expect(ins.actions.map((a) => a.ood)).toEqual([false, true, false, true]);
  expect(ins.actions.filter((a) => a.greedy).map((a) => a.a)).toEqual([1]);
});

test("supportLevel is a monotone log scale clamped to [0, 1]", () => {
  expect(supportLevel(0, 50)).toBe(0);
  expect(supportLevel(5, 0)).toBe(0);
  expect(supportLevel(50, 50)).toBeCloseTo(1, 12);
  expect(supportLevel(500, 50)).toBe(1);
  expect(supportLevel(1, 50)).toBeLessThan(supportLevel(10, 50));
  expect(supportLevel(10, 50)).toBeLessThan(supportLevel(40, 50));
});

test("diffPolicies finds changed states and respects the eligibility mask", () => {
  const a = new Int8Array([0, 1, 2, 3, 0]);
  const b = new Int8Array([0, 2, 2, 1, 3]);
  expect(diffPolicies(a, b)).toEqual([1, 3, 4]);
  expect(diffPolicies(a, b, new Uint8Array([1, 1, 1, 0, 1]))).toEqual([1, 4]);
  expect(diffPolicies(a, new Int8Array(3))).toEqual([]);
});

test("policy tracker records changes per version and is idempotent per version", () => {
  let tr = createPolicyTracker(4);
  tr = ingestPolicy(tr, new Int8Array([0, 0, 0, 0]), 1);
  expect(tr.changedNow).toEqual([]); // nothing to compare against yet

  tr = ingestPolicy(tr, new Int8Array([0, 2, 0, 1]), 2);
  expect(tr.changedNow).toEqual([1, 3]);
  expect(tr.lastChanged[1]).toBe(2);
  expect(tr.fromAction[1]).toBe(0);

  // StrictMode re-render: same version again must not wipe the diff
  const again = ingestPolicy(tr, new Int8Array([0, 2, 0, 1]), 2);
  expect(again).toBe(tr);
  expect(again.changedNow).toEqual([1, 3]);

  // the tracker keeps its own copy, so mutating the caller's array does not leak in
  const p3 = new Int8Array([0, 2, 0, 1]);
  tr = ingestPolicy(tr, p3, 3);
  p3[0] = 3;
  expect(tr.changedNow).toEqual([]);
  expect(tr.policy![0]).toBe(0);

  tr = ingestPolicy(tr, new Int8Array([1, 2, 0, 1]), 4);
  expect(tr.changedNow).toEqual([0]);
  expect(changeRecency(tr, 0, 4, 5)).toBe(1);
  expect(changeRecency(tr, 0, 6, 5)).toBeCloseTo(0.6, 12);
  expect(changeRecency(tr, 0, 9, 5)).toBe(0);
  expect(changeRecency(tr, 1, 4, 5)).toBeCloseTo(0.6, 12); // changed at version 2
  expect(changeRecency(tr, 2, 4, 5)).toBe(0); // never changed
});

test("policy tracker restarts when the state space changes size", () => {
  let tr = createPolicyTracker(3);
  tr = ingestPolicy(tr, new Int8Array([0, 1, 2]), 1);
  tr = ingestPolicy(tr, new Int8Array([3, 3, 3, 3, 3]), 2);
  expect(tr.lastChanged.length).toBe(5);
  expect(tr.changedNow).toEqual([]);
});

test("thrusterCell reproduces the projection the rollout policy is queried with", () => {
  const cells: number[] = [];
  const frames = rolloutThruster(
    {
      mode: "policy",
      policy: (c) => {
        cells.push(c);
        return c % 4;
      },
    },
    3,
  );
  expect(frames.length).toBe(cells.length);
  for (let t = 0; t < frames.length; t++) {
    const { cell, xi, yi } = thrusterCell(frames[t].x);
    expect(cell).toBe(cells[t]);
    expect(cell).toBe(yi * THRUSTER_PROJECTION.xAxis.bins + xi);
    expect(frames[t].a).toBe(cells[t] % 4);
  }
});

test("thrusterCell clamps far-out states to the grid edges", () => {
  const x = new Float64Array(13);
  x[0] = 99;
  x[1] = 99;
  expect(thrusterCell(x)).toEqual({ cell: THRUSTER_PROJECTION.xAxis.bins - 1, xi: THRUSTER_PROJECTION.xAxis.bins - 1, yi: 0 });
  x[0] = -99;
  x[1] = -99;
  const bottom = THRUSTER_PROJECTION.yAxis.bins - 1;
  expect(thrusterCell(x)).toEqual({ cell: bottom * THRUSTER_PROJECTION.xAxis.bins, xi: 0, yi: bottom });
});

test("thrusterSummary computes oscillator energy, current and wave amplitude", () => {
  const x = new Float64Array(13);
  x[0] = 1;
  x[1] = -2;
  x[9] = 2;
  x[10] = 0.4;
  x[11] = 0.3;
  x[12] = 0.4;
  const sum = thrusterSummary(x);
  expect(sum.energy).toBeCloseTo(9, 12);
  expect(sum.p1).toBe(1);
  expect(sum.v1).toBe(-2);
  expect(sum.current).toBe(0.4);
  expect(sum.waveAmp).toBeCloseTo(0.5, 12);
});

test("gridworld behavior rollout: return-to-go at t=0 equals the discounted episode return", () => {
  const ds = generateDataset("gridworld-random", 7);
  const env = ds.env;
  let s = env.start;
  const rewards: number[] = [];
  for (let t = 0; t < 30; t++) {
    const { s2, r, done } = env.step(s, t % 2 === 0 ? 1 : 0);
    rewards.push(r);
    if (done) break;
    s = s2;
  }
  const G = discountedReturnsToGo(rewards, 0.95);
  const direct = rewards.reduce((acc, r, k) => acc + Math.pow(0.95, k) * r, 0);
  expect(G[0]).toBeCloseTo(direct, 10);
});

test("transport helpers: clampIndex and keyboard map", () => {
  expect(clampIndex(-3, 10)).toBe(0);
  expect(clampIndex(4, 10)).toBe(4);
  expect(clampIndex(12, 10)).toBe(9);
  expect(clampIndex(5, 0)).toBe(0);

  expect(playbackKey("ArrowLeft")).toEqual({ type: "step", delta: -1 });
  expect(playbackKey("ArrowRight")).toEqual({ type: "step", delta: 1 });
  expect(playbackKey("ArrowRight", true)).toEqual({ type: "step", delta: 10 });
  expect(playbackKey(" ")).toEqual({ type: "toggle" });
  expect(playbackKey("Home")).toEqual({ type: "start" });
  expect(playbackKey("End")).toEqual({ type: "end" });
  expect(playbackKey("a")).toBeNull();
});
