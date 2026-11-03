import { test, expect } from "bun:test";
import { createQ, evaluate, generateDataset, mulberry32, trainUpdates, type Dataset, type Hyper } from "../../src/lib/rl";
import {
  buildRuns,
  buildSweepDataset,
  CHUNK,
  createInlineExecutor,
  DEFAULT_SWEEP,
  driveSweep,
  rngFromState,
  SweepScheduler,
  trainSegment,
  type SegmentTask,
  type SweepRun,
  type SweepSettings,
} from "../../src/lib/sweep-core";

/**
 * The pre-pool sequential sweep loop, copied verbatim (minus the UI yields) from
 * the old src/lib/sweep.ts. Used as the ground truth that the pooled engine must
 * reproduce exactly.
 */
function legacySweep(settings: SweepSettings, dataset: Dataset, runs: SweepRun[]) {
  const checkpoints = Math.max(1, Math.ceil(settings.totalUpdates / CHUNK));
  const bestAt = new Float64Array(checkpoints + 1).fill(-Infinity);
  for (const run of runs) {
    run.status = "running";
    const Q = createQ(dataset);
    const rng = mulberry32(1337 + run.id * 7919);
    const h: Hyper = { alpha: run.alpha, gamma: settings.gamma, lr: run.lr, batch: settings.batch };
    let done = 0;
    let ck = 0;
    while (done < settings.totalUpdates) {
      const n = Math.min(CHUNK, settings.totalUpdates - done);
      const { tdLoss, cqlGap } = trainUpdates(Q, dataset, h, rng, n);
      done += n;
      ck++;
      run.updatesDone = done;
      run.progress = done / settings.totalUpdates;
      const ev = evaluate(Q, dataset, settings.gamma);
      run.trueQ = ev.trueQ;
      run.estQ = ev.estQ;
      run.gap = ev.estQ - ev.trueQ;
      run.oodRate = ev.oodRate;
      run.loss = tdLoss + h.alpha * cqlGap;
      if (
        settings.earlyStop &&
        run.progress >= settings.pruneAfterFrac &&
        isFinite(bestAt[ck]) &&
        bestAt[ck] - run.trueQ > settings.pruneGap &&
        done < settings.totalUpdates
      ) {
        run.status = "pruned";
      }
      if (run.trueQ > bestAt[ck]) bestAt[ck] = run.trueQ;
      if (run.status === "pruned") break;
    }
    if (run.status === "running") run.status = "done";
  }
  return runs;
}

const pick = (r: SweepRun) => ({
  id: r.id,
  lr: r.lr,
  alpha: r.alpha,
  status: r.status,
  updatesDone: r.updatesDone,
  trueQ: r.trueQ,
  estQ: r.estQ,
  gap: r.gap,
  oodRate: r.oodRate,
  loss: r.loss,
});

const datasetCache = new Map<string, Dataset>();
const resolveDs = (key: { id: SweepSettings["datasetId"]; seed: number }) => {
  const k = `${key.id}:${key.seed}`;
  if (!datasetCache.has(k)) datasetCache.set(k, buildSweepDataset(key));
  return datasetCache.get(k)!;
};

/** Run the pooled engine with `size` emulated workers that finish in a scrambled order. */
async function pooled(settings: SweepSettings, size: number, jitterSeed: number, maxTotalUpdates: number | null = null) {
  const sch = new SweepScheduler(settings, { maxActive: size * 2, maxTotalUpdates });
  const jitter = mulberry32(jitterSeed);
  const ex = createInlineExecutor(resolveDs, { size, delayMs: () => (size === 1 ? 0 : Math.floor(jitter() * 4)) });
  await driveSweep(sch, ex);
  return sch;
}

const SMALL: SweepSettings = {
  ...DEFAULT_SWEEP,
  lrSteps: 3,
  alphaSteps: 3,
  totalUpdates: 600,
  pruneAfterFrac: 0.3,
  pruneGap: 0.6,
};

test("stateful RNG replica matches rl.mulberry32 bit for bit", () => {
  for (const seed of [0, 1, 1337, 1337 + 7919 * 15, 0xffffffff]) {
    const a = mulberry32(seed);
    const st = { a: seed >>> 0 };
    const b = rngFromState(st);
    for (let i = 0; i < 2000; i++) expect(b()).toBe(a());
    // Resuming from a shipped state continues the same stream.
    const c = rngFromState({ a: st.a });
    for (let i = 0; i < 50; i++) expect(c()).toBe(a());
  }
});

test("splitting a run into segments does not change its result", () => {
  const ds = generateDataset("gridworld-random", 7);
  const base: SegmentTask = {
    taskId: 1,
    index: 0,
    dataset: { id: "gridworld-random", seed: 7 },
    lr: 0.3,
    alpha: 1,
    gamma: 0.95,
    batch: 64,
    totalUpdates: 610,
    fromUpdates: 0,
    toUpdates: 610,
    q: null,
    rngState: 42,
    pruneGap: 0,
    pruneRefs: null,
  };
  const whole = trainSegment(ds, base);
  const a = trainSegment(ds, { ...base, toUpdates: 180 });
  const b = trainSegment(ds, { ...base, fromUpdates: 180, toUpdates: 420, q: a.q, rngState: a.rngState });
  const c = trainSegment(ds, { ...base, fromUpdates: 420, toUpdates: 610, q: b.q, rngState: b.rngState });
  expect(Array.from(c.q)).toEqual(Array.from(whole.q));
  expect(c.rngState).toBe(whole.rngState);
  const cps = [...a.checkpoints, ...b.checkpoints, ...c.checkpoints];
  expect(cps).toEqual(whole.checkpoints);
  expect(cps.at(-1)!.updates).toBe(610);
  expect(cps.length).toBe(Math.ceil(610 / CHUNK));
});

test("pooled sweep reproduces the old sequential sweep exactly (1 worker and N workers)", async () => {
  const ds = resolveDs({ id: SMALL.datasetId, seed: 7 });
  const legacy = legacySweep(SMALL, ds, buildRuns(SMALL)).map(pick);
  // The settings must exercise both outcomes, or the test proves nothing about pruning.
  expect(legacy.some((r) => r.status === "pruned")).toBe(true);
  expect(legacy.some((r) => r.status === "done")).toBe(true);

  const seq = (await pooled(SMALL, 1, 1)).runs.map(pick);
  expect(seq).toEqual(legacy);
  for (const [size, jitterSeed] of [
    [2, 11],
    [3, 5],
    [4, 99],
    [8, 2024],
  ]) {
    const par = (await pooled(SMALL, size, jitterSeed)).runs.map(pick);
    expect(par).toEqual(seq);
  }
});

test("pruning stays order-independent on HalfCheetah and ThrusterChain too", async () => {
  for (const datasetId of ["halfcheetah-medium", "thrusterchain-stall"] as const) {
    const s: SweepSettings = { ...SMALL, datasetId, lrSteps: 3, alphaSteps: 2, totalUpdates: 420, pruneGap: 0.3 };
    const legacy = legacySweep(s, resolveDs({ id: datasetId, seed: 7 }), buildRuns(s)).map(pick);
    const seq = (await pooled(s, 1, 3)).runs.map(pick);
    const par = (await pooled(s, 4, 17)).runs.map(pick);
    expect(seq).toEqual(legacy);
    expect(par).toEqual(seq);
  }
});

test("updates budget skips the same runs for every worker count and is never exceeded", async () => {
  const cap = 3000; // 600 updates per run → at most 5 full runs, more if pruning frees budget
  const seq = await pooled(SMALL, 1, 1, cap);
  const par = await pooled(SMALL, 4, 7, cap);
  expect(par.runs.map(pick)).toEqual(seq.runs.map(pick));
  expect(seq.runs.some((r) => r.status === "skipped")).toBe(true);
  expect(seq.stopReason).toBe("updates-budget");
  for (const sch of [seq, par]) {
    const used = sch.runs.reduce((a, r) => a + r.updatesDone, 0);
    expect(used).toBeLessThanOrEqual(cap);
    expect(sch.updatesDone).toBe(used);
  }
});

test("a different sweep seed changes the runs, the same seed reproduces them", async () => {
  const a = (await pooled({ ...SMALL, earlyStop: false, seed: 1 }, 3, 4)).runs.map(pick);
  const b = (await pooled({ ...SMALL, earlyStop: false, seed: 1 }, 1, 4)).runs.map(pick);
  const c = (await pooled({ ...SMALL, earlyStop: false, seed: 2 }, 3, 4)).runs.map(pick);
  expect(a).toEqual(b);
  expect(a).not.toEqual(c);
  expect(a.every((r) => r.status === "done")).toBe(true);
});

test("cancel and manual stop leave a consistent state", async () => {
  const sch = new SweepScheduler(SMALL, { maxActive: 4 });
  const ex = createInlineExecutor(resolveDs, { size: 2, delayMs: () => 1 });
  let chunks = 0;
  await driveSweep(sch, ex, {
    onUpdate: () => {
      chunks++;
      if (chunks === 5) sch.stopRun(0);
      if (chunks === 25) sch.cancel();
    },
  });
  expect(sch.runs[0].status).toBe("stopped");
  expect(sch.runs.every((r) => r.status !== "running" && r.status !== "pending")).toBe(true);
  expect(sch.runs.some((r) => r.status === "skipped")).toBe(true);
});

test("with one worker every run is a single segment; pruning is applied inside the segment", async () => {
  const sch = new SweepScheduler(SMALL, { maxActive: 1 });
  const ex = createInlineExecutor(resolveDs, { size: 1 });
  let dispatches = 0;
  let segmentsWithRefs = 0;
  await driveSweep(sch, ex, {
    onDispatch: (_w, task) => {
      dispatches++;
      if (task.pruneRefs?.some((r) => r != null && isFinite(r))) segmentsWithRefs++;
    },
  });
  expect(dispatches).toBe(sch.runs.length);
  expect(segmentsWithRefs).toBe(sch.runs.length - 1); // run 0 has no lower runs to compare against
  const legacy = legacySweep(SMALL, resolveDs({ id: SMALL.datasetId, seed: 7 }), buildRuns(SMALL)).map(pick);
  expect(sch.runs.map(pick)).toEqual(legacy);
  expect(sch.runs.filter((r) => r.status === "pruned").every((r) => r.prunedAt === Math.ceil(r.updatesDone / CHUNK))).toBe(true);
});

test("benchmark suite gives identical results for 1 and 4 workers", async () => {
  const { buildSuite, runBenchmark, aggregateBenchmark, benchmarkCsv } = await import("../../src/lib/benchmark");
  const cases = buildSuite()
    .filter((c) => c.seed === 101)
    .map((c, index) => ({ ...c, index, updates: 300 }));
  expect(cases.length).toBe(9);
  const one = await runBenchmark(cases, createInlineExecutor(resolveDs, { size: 1 }));
  const jitter = mulberry32(8);
  const four = await runBenchmark(cases, createInlineExecutor(resolveDs, { size: 4, delayMs: () => Math.floor(jitter() * 5) }));
  const strip = (rs: typeof one) => rs.map((r) => ({ idx: r.case.index, final: r.final, curve: r.curve }));
  expect(strip(four)).toEqual(strip(one));
  const csv = benchmarkCsv(aggregateBenchmark(one, { "gridworld-random": 1, "halfcheetah-medium": 2, "thrusterchain-stall": 3 }));
  expect(csv.trim().split("\n").length).toBe(1 + 9);
  expect(csv.split("\n")[0]).toContain("true_return_mean");
});
