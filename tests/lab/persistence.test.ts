import { expect, test } from "bun:test";
import { resumableRng } from "../../src/lib/rng";
import { createQ, evaluate, generateDataset, mulberry32, trainUpdates, type Hyper, type Metrics } from "../../src/lib/rl";
import { DEFAULT_QUERY, filterExperiments, matchesText, parseTags, summarize, tailStability, type ExperimentRecord } from "../../src/lib/experiments";
import { checkQ, firstInsane, guardPhysics, metricsSane } from "../../src/lib/guards";
import { auditMarkdown, buildAuditLog, metricsCsv } from "../../src/lib/log-export";
import { buildLogEntries } from "../../src/components/lab/experiment-log";

const H: Hyper = { alpha: 1, gamma: 0.95, lr: 0.3, batch: 64 };

test("resumable stream is identical to mulberry32 and resumes exactly from saved state", () => {
  const a = mulberry32(1234);
  const b = resumableRng(1234);
  for (let i = 0; i < 1000; i++) expect(b()).toBe(a());
  const saved = b.state();
  const next = Array.from({ length: 50 }, () => b());
  const resumed = resumableRng(saved);
  expect(Array.from({ length: 50 }, () => resumed())).toEqual(next);
});

test("a run saved mid-training and resumed matches an uninterrupted run bit for bit", () => {
  const ds = generateDataset("gridworld-random", 7);
  const straightQ = createQ(ds);
  const straight = resumableRng(1234);
  for (let k = 0; k < 10; k++) trainUpdates(straightQ, ds, H, straight, 20);

  const q = createQ(ds);
  const rng = resumableRng(1234);
  for (let k = 0; k < 4; k++) trainUpdates(q, ds, H, rng, 20);
  // "Reload": copy Q and the stream state, then continue in fresh objects.
  const savedQ = new Float64Array(q);
  const resumedRng = resumableRng(rng.state());
  for (let k = 4; k < 10; k++) trainUpdates(savedQ, ds, H, resumedRng, 20);
  expect(Array.from(savedQ)).toEqual(Array.from(straightQ));
});

function metric(iter: number, trueQ: number, estQ = trueQ + 0.5): Metrics {
  return { iter, updates: iter * 20, estQ, trueQ, loss: 1, tdLoss: 0.5, cqlGap: 0.5, oodRate: 0.1, divergence: 1.2 };
}

function rec(over: Partial<ExperimentRecord>): ExperimentRecord {
  return {
    id: Math.random().toString(36).slice(2),
    name: "run",
    createdAt: 1,
    updatedAt: 1,
    datasetId: "gridworld-random",
    datasetName: "Gridworld-Random",
    seed: 1234,
    rngState: 1234,
    hyper: { ...H },
    hyperLog: [{ atUpdates: 0, hyper: { ...H } }],
    history: [metric(1, 0), metric(2, 1)],
    q: new Float64Array(4),
    optimal: 3,
    status: "paused",
    tags: [],
    notes: "",
    anomalies: 0,
    guardEvents: [],
    ...over,
  };
}

test("search matches every term across name, notes, tags and dataset", () => {
  const r = rec({ name: "Baseline", notes: "low alpha sanity check", tags: ["thruster"] });
  expect(matchesText(r, "baseline sanity")).toBe(true);
  expect(matchesText(r, "THRUSTER gridworld")).toBe(true);
  expect(matchesText(r, "baseline cheetah")).toBe(false);
});

test("filters and sorting", () => {
  const a = rec({ name: "a", hyper: { ...H, alpha: 0 }, history: [metric(1, 2)], updatedAt: 3, tags: ["x"] });
  const b = rec({ name: "b", hyper: { ...H, alpha: 2 }, history: [metric(1, 5)], updatedAt: 2, datasetId: "halfcheetah-medium" });
  const c = rec({ name: "c", hyper: { ...H, alpha: 1 }, history: [metric(1, 1), metric(2, 3)], updatedAt: 1, status: "diverged" });
  const all = [a, b, c];
  expect(filterExperiments(all, DEFAULT_QUERY).map((r) => r.name)).toEqual(["a", "b", "c"]);
  expect(filterExperiments(all, { ...DEFAULT_QUERY, sort: "trueQ" }).map((r) => r.name)).toEqual(["b", "c", "a"]);
  expect(filterExperiments(all, { ...DEFAULT_QUERY, alphaMin: 0.5 }).map((r) => r.name)).toEqual(["b", "c"]);
  expect(filterExperiments(all, { ...DEFAULT_QUERY, datasetId: "halfcheetah-medium" }).map((r) => r.name)).toEqual(["b"]);
  expect(filterExperiments(all, { ...DEFAULT_QUERY, status: "diverged" }).map((r) => r.name)).toEqual(["c"]);
  expect(filterExperiments(all, { ...DEFAULT_QUERY, tag: "x" }).map((r) => r.name)).toEqual(["a"]);
  expect(filterExperiments(all, { ...DEFAULT_QUERY, minUpdates: 30 }).map((r) => r.name)).toEqual(["c"]);
  expect(filterExperiments(all, { ...DEFAULT_QUERY, sort: "name", descending: false }).map((r) => r.name)).toEqual(["a", "b", "c"]);
});

test("summary and tail stability", () => {
  const steady = Array.from({ length: 20 }, (_, i) => metric(i + 1, i < 10 ? i : 10));
  const shaky = Array.from({ length: 20 }, (_, i) => metric(i + 1, i % 2 ? 12 : 6));
  expect(summarize({ history: steady }).bestTrueQ).toBe(10);
  expect(tailStability(steady)!.std).toBe(0);
  expect(tailStability(shaky)!.std).toBeGreaterThan(2);
  expect(tailStability(shaky)!.maxDrawdown).toBe(6);
  expect(tailStability(steady.slice(0, 3))).toBeNull();
  expect(parseTags(" A, b ,a,, c ")).toEqual(["a", "b", "c"]);
});

test("guards flag NaN, infinities and explosions and say where", () => {
  const q = new Float64Array(8);
  expect(checkQ(q, 2).ok).toBe(true);
  q[5] = NaN;
  const c = checkQ(q, 2);
  expect(c.ok).toBe(false);
  expect(c.reason).toContain("NaN");
  expect(c.reason).toContain("action 1");
  expect(firstInsane([1, 2, 1e9])).toBe(2);
  expect(metricsSane(metric(1, 1))).toBe(true);
  expect(metricsSane({ ...metric(1, 1), loss: Infinity })).toBe(false);
  expect(() => guardPhysics("bench", () => ({ v: NaN }), (r) => [r.v])).toThrow(/diverged/);
  expect(() => guardPhysics("bench", () => { throw new Error("boom"); }, () => [])).toThrow(/crashed: boom/);
  expect(guardPhysics("bench", () => ({ v: 1 }), (r) => [r.v]).v).toBe(1);
});

test("audit log: CSV has one row per step; markdown lists hyperparameter changes and guard events", () => {
  const r = rec({
    history: [metric(1, 0), metric(2, 1)],
    hyperLog: [
      { atUpdates: 0, hyper: { ...H } },
      { atUpdates: 20, hyper: { ...H, alpha: 2 } },
    ],
    guardEvents: [{ atUpdates: 40, message: "rolled back", at: 0 }],
  });
  const csv = metricsCsv(r.history).trim().split("\n");
  expect(csv[0]).toBe("iter,updates,estQ,trueQ,gap,loss,tdLoss,cqlGap,oodRate,divergence");
  expect(csv).toHaveLength(3);
  const audit = buildAuditLog({ ...r, transitions: 100 }, [], [], 20);
  expect(audit.summary.updates).toBe(40);
  const md = auditMarkdown(audit);
  expect(md).toContain("| 20 | 2 |");
  expect(md).toContain("rolled back");
});

test("overestimation anomaly blames extrapolation only when the policy takes unseen actions", () => {
  const ds = generateDataset("gridworld-random", 7);
  const base = Array.from({ length: 6 }, (_, i) => ({ ...metric(i + 1, 1, 1.2), oodRate: 0 }));
  const jumpNoOod = [...base, { ...metric(7, 1, 3), oodRate: 0 }];
  const jumpOod = [...base, { ...metric(7, 1, 3), oodRate: 0.3 }];
  const textOf = (h: Metrics[]) => buildLogEntries(h, ds, H, 3).find((e) => e.kind === "anomaly")?.text ?? "";
  expect(textOf(jumpNoOod)).toContain("OOD 0%");
  expect(textOf(jumpOod)).toContain("extrapolation error is a likely cause");
});

test("evaluate is untouched by the persistence work (smoke)", () => {
  const ds = generateDataset("gridworld-random", 7);
  const r = evaluate(createQ(ds), ds, 0.95);
  expect(Number.isFinite(r.trueQ)).toBe(true);
});

import { buildSnapshot, parseSnapshot, BUILTIN_STATES } from "../../src/lib/snapshot";

test("built-in grid sizes used to validate snapshots match the datasets", () => {
  for (const id of ["gridworld-random", "halfcheetah-medium"] as const) {
    const d = generateDataset(id, 7);
    expect(BUILTIN_STATES[id]).toBe(d.env.width * d.env.height);
  }
});

test("snapshot round-trip yields a new paused run with identical Q, history and stream state", () => {
  const r = rec({ name: "shared", tags: ["x"], rngState: 987654321, q: Float64Array.from({ length: 400 }, (_, i) => i / 7) });
  const text = JSON.stringify(buildSnapshot(r, undefined, "Priya"));
  const { run, upload } = parseSnapshot(text, 5, () => "new-id");
  expect(upload).toBeUndefined();
  expect(run.id).toBe("new-id");
  expect(run.name).toBe("shared (imported from Priya)");
  expect(Array.from(run.q)).toEqual(Array.from(r.q));
  expect(run.history).toEqual(r.history);
  expect(run.rngState).toBe(987654321);
  expect(run.tags).toEqual(["x", "imported"]);
  expect(run.status).toBe("paused");
});

test("snapshot import rejects wrong schema, wrong Q size, non-finite values and missing uploads", () => {
  const good = buildSnapshot(rec({ q: new Float64Array(400) }));
  const bad = (mut: (s: any) => void) => {
    const s = JSON.parse(JSON.stringify(good));
    mut(s);
    return () => parseSnapshot(JSON.stringify(s), 0, () => "id");
  };
  expect(() => parseSnapshot("{not json", 0)).toThrow(/valid JSON/);
  expect(bad((s) => (s.schema = "other"))).toThrow(/schema/);
  expect(bad((s) => (s.run.q = s.run.q.slice(0, 8)))).toThrow(/needs 400/);
  expect(bad((s) => (s.run.history[0].trueQ = "NaN"))).toThrow(/metrics row/);
  expect(bad((s) => (s.run.datasetId = "custom"))).toThrow(/must include that dataset/);
  expect(bad((s) => (s.run.datasetId = "evil"))).toThrow(/unknown dataset/);
});
