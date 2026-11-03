import { expect, test } from "bun:test";
import {
  buildProjection,
  buildRouteGeometry,
  crossesSlab,
  drawStart,
  edgeTouches,
  generateRouteDataset,
  geometryDeviation,
  recorderAction,
  slabProgress,
  taskStep,
  trackingController,
  ROUTE_HORIZON,
} from "../../src/lib/thrusterchain-route";
import { BUILTIN_STATES } from "../../src/lib/snapshot";
import { createQ, evaluate, mulberry32 } from "../../src/lib/rl";

const geometry = buildRouteGeometry();

test("the lab rebuilds the research geometry exactly (route, danger center, radius, separation)", () => {
  expect(geometryDeviation(geometry)).toBe(0);
  expect(geometry.danger.radius).toBe(0.1);
  expect(slabProgress(geometry.B, geometry)).toBeCloseTo(0, 12);
  expect(slabProgress(geometry.C, geometry)).toBeCloseTo(1, 12);
});

test("noise-free: the nominal route reaches the goal, five opposite steps from B reach the danger", () => {
  const g = geometry;
  expect(edgeTouches(g.excursion[4], g.excursion[5], g.danger)).toBe(true);
  // The nominal route never touches the danger.
  for (let t = 0; t < 100; t++) expect(edgeTouches(g.route[t], g.route[t + 1], g.danger)).toBe(false);
});

test("projection keeps A, B, C, the goal and the danger in distinct cells; snapshot grid size matches", () => {
  const p = buildProjection(geometry);
  const cells = [geometry.A, geometry.B, geometry.C, geometry.goal.center, geometry.danger.center].map((x) => p.cellOf(x));
  expect(new Set(cells).size).toBe(5);
  expect(BUILTIN_STATES["route-full"]).toBe(p.width * p.height);
});

const sets = {
  full: generateRouteDataset("full", 2026),
  honeypot: generateRouteDataset("honeypot", 2026),
  bottleneck: generateRouteDataset("bottleneck", 2026),
};

test("same recorders in every condition; recorders only ever take Controller or Opposite", () => {
  for (const r of Object.values(sets)) {
    expect(r.counts.recorded).toBe(sets.full.counts.recorded);
    const used = new Set(r.dataset.transitions.map((t) => t.a));
    expect([...used].sort()).toEqual([0, 1]);
  }
  expect(sets.full.counts.removed).toBe(0);
});

test("Honeypot removes exactly the danger-touching edges: no −100 survives, successes unchanged", () => {
  const h = sets.honeypot;
  expect(h.counts.removed).toBe(h.counts.removedDanger);
  expect(h.counts.removed).toBeGreaterThan(0);
  expect(Math.min(...h.dataset.transitions.map((t) => t.r))).toBeGreaterThan(-2);
  expect(sets.full.dataset.transitions.some((t) => t.r < -100)).toBe(true);
  expect(h.counts.intactSuccesses).toBe(sets.full.counts.intactSuccesses);
});

test("Bottleneck cuts every complete success but keeps the fragments on both sides", () => {
  const b = sets.bottleneck;
  expect(b.counts.removed).toBe(b.counts.removedSlab);
  expect(b.counts.intactSuccesses).toBe(0);
  expect(sets.full.counts.intactSuccesses).toBeGreaterThan(1000);
  // Goal-reaching fragments after C and approach fragments before B both remain.
  expect(b.dataset.transitions.some((t) => t.r > 90)).toBe(true);
  expect(b.dataset.transitions.some((t) => t.s === b.dataset.env.start)).toBe(true);
});

test("slab rule matches connector_cut: a straight B→C edge crosses, a far-away edge does not", () => {
  expect(crossesSlab(geometry.B, geometry.C, geometry)).toBe(true);
  expect(crossesSlab(geometry.route[60], geometry.route[61], geometry)).toBe(false);
});

test("reference model: from B, Opposite heads up toward the danger and Controller stays on the route", () => {
  const ds = sets.full.dataset;
  const p = buildProjection(geometry);
  const b = p.cellOf(geometry.B);
  const row = (c: number) => Math.floor(c / p.width);
  const up = ds.env.step(b, 1);
  const on = ds.env.step(b, 0);
  expect(row(up.s2)).toBeLessThan(row(b));
  expect(row(on.s2)).toBe(row(b));
  // Every action has truth at B, even the two the data never contain.
  for (let a = 0; a < 4; a++) expect(ds.env.model!.counts[b * 4 + a]).toBeGreaterThan(0);
});

test("datasets are deterministic for a seed and train/evaluate without NaN", () => {
  const again = generateRouteDataset("honeypot", 2026);
  expect(again.dataset.transitions.length).toBe(sets.honeypot.dataset.transitions.length);
  const r = evaluate(createQ(sets.full.dataset), sets.full.dataset, 0.98);
  expect(Number.isFinite(r.trueQ)).toBe(true);
});

test("an episode under the recorders follows the task semantics (terminal flags, horizon)", () => {
  const control = trackingController();
  const rng = mulberry32(5);
  let x = drawStart(geometry, rng);
  let t = 0;
  for (; t < ROUTE_HORIZON; t++) {
    const step = taskStep(x, t, recorderAction(0, control, x, t, rng).command, geometry);
    if (step.terminal) {
      expect(step.goal || step.danger || t + 1 === ROUTE_HORIZON).toBe(true);
      break;
    }
    x = step.next;
  }
  expect(t).toBeLessThan(ROUTE_HORIZON);
});
