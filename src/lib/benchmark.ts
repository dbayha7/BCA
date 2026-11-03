// Automated benchmarking suite: a FIXED set of runs (every built-in dataset ×
// a small set of conservative weights × fixed training seeds) trained through
// the same pure trainSegment as the sweep, so the leaderboard is reproducible
// on any machine and for any worker count. Pure: no DOM, no Worker APIs.

import type { CheckpointMetrics, SegmentExecutor, SegmentResult, SegmentTask, SweepDatasetId } from "./sweep-core";

export const BENCH_VERSION = 1;

/** Dataset seeds match the main lab (gridworld 7, HalfCheetah 11, ThrusterChain 7). */
export const BENCH_DATASETS: { id: SweepDatasetId; name: string; seed: number }[] = [
  { id: "gridworld-random", name: "Gridworld-Random", seed: 7 },
  { id: "halfcheetah-medium", name: "HalfCheetah-Medium", seed: 11 },
  { id: "thrusterchain-stall", name: "ThrusterChain-Stall", seed: 7 },
];
export const BENCH_ALPHAS = [0, 1, 3];
export const BENCH_SEEDS = [101, 202, 303];
/** Defaults of the main lab: lr 0.3, γ 0.95, batch 64. */
export const BENCH_HYPER = { lr: 0.3, gamma: 0.95, batch: 64, updates: 3000 };

export interface BenchCase {
  index: number;
  datasetId: SweepDatasetId;
  datasetName: string;
  datasetSeed: number;
  alpha: number;
  lr: number;
  gamma: number;
  batch: number;
  updates: number;
  seed: number;
}

export function buildSuite(): BenchCase[] {
  const cases: BenchCase[] = [];
  for (const d of BENCH_DATASETS)
    for (const alpha of BENCH_ALPHAS)
      for (const seed of BENCH_SEEDS)
        cases.push({
          index: cases.length,
          datasetId: d.id,
          datasetName: d.name,
          datasetSeed: d.seed,
          alpha,
          lr: BENCH_HYPER.lr,
          gamma: BENCH_HYPER.gamma,
          batch: BENCH_HYPER.batch,
          updates: BENCH_HYPER.updates,
          seed,
        });
  return cases;
}

export function caseTask(c: BenchCase, taskId: number): SegmentTask {
  return {
    taskId,
    index: c.index,
    dataset: { id: c.datasetId, seed: c.datasetSeed },
    lr: c.lr,
    alpha: c.alpha,
    gamma: c.gamma,
    batch: c.batch,
    totalUpdates: c.updates,
    fromUpdates: 0,
    toUpdates: c.updates,
    q: null,
    rngState: c.seed >>> 0,
    pruneGap: 0,
    pruneRefs: null,
  };
}

export interface BenchRunResult {
  case: BenchCase;
  final: CheckpointMetrics;
  /** True return after every checkpoint (learning curve). */
  curve: number[];
  /** Worker compute time (≈ CPU time). */
  computeMs: number;
  /** Main-thread wall time from dispatch to result. */
  wallMs: number;
}

export interface BenchHooks {
  onDispatch?: (worker: number, c: BenchCase) => void;
  onChunk?: (c: BenchCase, m: CheckpointMetrics) => void;
  onResult?: (r: BenchRunResult) => void;
  isCancelled?: () => boolean;
  now?: () => number;
}

/** Train every case through the executor's workers. Resolves with results in case order. */
export function runBenchmark(cases: BenchCase[], ex: SegmentExecutor, hooks: BenchHooks = {}): Promise<BenchRunResult[]> {
  const now = hooks.now ?? (() => (typeof performance !== "undefined" ? performance.now() : Date.now()));
  return new Promise((resolve, reject) => {
    const results: (BenchRunResult | null)[] = cases.map(() => null);
    const idle = Array.from({ length: ex.size }, (_, i) => i);
    let next = 0;
    let inFlight = 0;
    let failed = false;
    const done = () => resolve(results.filter((r): r is BenchRunResult => r != null));
    const pump = () => {
      if (failed) return;
      const cancelled = hooks.isCancelled?.() ?? false;
      while (!cancelled && idle.length > 0 && next < cases.length) {
        const c = cases[next++];
        const w = idle.shift() as number;
        inFlight++;
        hooks.onDispatch?.(w, c);
        const t0 = now();
        ex.run(w, caseTask(c, c.index + 1), (m) => hooks.onChunk?.(c, m)).then(
          (res: SegmentResult) => {
            inFlight--;
            const final = res.checkpoints[res.checkpoints.length - 1];
            const r: BenchRunResult = { case: c, final, curve: res.checkpoints.map((m) => m.trueQ), computeMs: res.computeMs, wallMs: now() - t0 };
            results[c.index] = r;
            hooks.onResult?.(r);
            idle.push(w);
            idle.sort((a, b) => a - b);
            pump();
          },
          (err) => {
            inFlight--;
            failed = true;
            reject(err instanceof Error ? err : new Error(String(err)));
          },
        );
      }
      if (inFlight === 0 && (cancelled || next >= cases.length)) done();
    };
    pump();
  });
}

export interface BenchRow {
  datasetId: SweepDatasetId;
  datasetName: string;
  alpha: number;
  lr: number;
  gamma: number;
  updates: number;
  seeds: number[];
  n: number;
  /** Mean true value of the greedy policy over dataset states (the sweep's "true return"). */
  trueQMean: number;
  trueQStd: number;
  /** Mean V^π(start). */
  vStartMean: number;
  /** Optimal V*(start) of the dataset's (exact or empirical) MDP. */
  vStar: number;
  /** V*(start) − mean V^π(start). */
  gapToOptimal: number;
  /** Mean (estimated Q − true value): positive = overestimation. */
  overestimation: number;
  oodRate: number;
  cpuMs: number;
  wallMs: number;
  /** Rank within its dataset by mean true return (1 = best). */
  rank: number;
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const std = (xs: number[]) => {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1));
};

/** One leaderboard row per (dataset, α), aggregated over seeds; ranked within each dataset. */
export function aggregateBenchmark(results: BenchRunResult[], vStars: Partial<Record<SweepDatasetId, number>>): BenchRow[] {
  const groups = new Map<string, BenchRunResult[]>();
  for (const r of results) {
    const k = `${r.case.datasetId}|${r.case.alpha}`;
    const g = groups.get(k);
    if (g) g.push(r);
    else groups.set(k, [r]);
  }
  const rows: BenchRow[] = [];
  for (const g of groups.values()) {
    const c = g[0].case;
    const vStar = vStars[c.datasetId] ?? NaN;
    const vStartMean = mean(g.map((r) => r.final.vStart));
    rows.push({
      datasetId: c.datasetId,
      datasetName: c.datasetName,
      alpha: c.alpha,
      lr: c.lr,
      gamma: c.gamma,
      updates: c.updates,
      seeds: g.map((r) => r.case.seed),
      n: g.length,
      trueQMean: mean(g.map((r) => r.final.trueQ)),
      trueQStd: std(g.map((r) => r.final.trueQ)),
      vStartMean,
      vStar,
      gapToOptimal: vStar - vStartMean,
      overestimation: mean(g.map((r) => r.final.gap)),
      oodRate: mean(g.map((r) => r.final.oodRate)),
      cpuMs: g.reduce((a, r) => a + r.computeMs, 0),
      wallMs: g.reduce((a, r) => a + r.wallMs, 0),
      rank: 0,
    });
  }
  const order = BENCH_DATASETS.map((d) => d.id);
  rows.sort((a, b) => order.indexOf(a.datasetId) - order.indexOf(b.datasetId) || b.trueQMean - a.trueQMean || a.alpha - b.alpha);
  let prev: SweepDatasetId | null = null;
  let rank = 0;
  for (const r of rows) {
    rank = r.datasetId === prev ? rank + 1 : 1;
    prev = r.datasetId;
    r.rank = rank;
  }
  return rows;
}

const CSV_COLUMNS: [string, (r: BenchRow) => string | number][] = [
  ["dataset", (r) => r.datasetId],
  ["rank", (r) => r.rank],
  ["alpha", (r) => r.alpha],
  ["lr", (r) => r.lr],
  ["gamma", (r) => r.gamma],
  ["updates", (r) => r.updates],
  ["seeds", (r) => r.seeds.join(" ")],
  ["true_return_mean", (r) => r.trueQMean.toFixed(6)],
  ["true_return_std", (r) => r.trueQStd.toFixed(6)],
  ["v_pi_start_mean", (r) => r.vStartMean.toFixed(6)],
  ["v_star_start", (r) => r.vStar.toFixed(6)],
  ["gap_to_v_star", (r) => r.gapToOptimal.toFixed(6)],
  ["overestimation_mean", (r) => r.overestimation.toFixed(6)],
  ["ood_rate_mean", (r) => r.oodRate.toFixed(6)],
  ["cpu_s", (r) => (r.cpuMs / 1000).toFixed(4)],
  ["wall_s", (r) => (r.wallMs / 1000).toFixed(4)],
];

export function benchmarkCsv(rows: BenchRow[]): string {
  const esc = (v: string | number) => {
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [CSV_COLUMNS.map(([h]) => h).join(","), ...rows.map((r) => CSV_COLUMNS.map(([, f]) => esc(f(r))).join(","))].join("\n") + "\n";
}

export interface BenchMeta {
  workers: number;
  startedAt: string;
  elapsedMs: number;
  cancelled: boolean;
}

export function benchmarkJson(meta: BenchMeta, rows: BenchRow[], results: BenchRunResult[], vStars: Partial<Record<SweepDatasetId, number>>) {
  return {
    kind: "offline-learning-lab/benchmark",
    version: BENCH_VERSION,
    suite: {
      datasets: BENCH_DATASETS,
      alphas: BENCH_ALPHAS,
      seeds: BENCH_SEEDS,
      ...BENCH_HYPER,
      note: "True return = exact policy evaluation of the greedy policy, averaged over states seen in the data. V*(start) is the optimal value of the start state (empirical MDP for ThrusterChain).",
    },
    run: meta,
    optimalStartValues: vStars,
    leaderboard: rows,
    runs: results.map((r) => ({
      dataset: r.case.datasetId,
      alpha: r.case.alpha,
      seed: r.case.seed,
      final: r.final,
      curve: r.curve,
      cpuMs: r.computeMs,
      wallMs: r.wallMs,
    })),
  };
}
