// Pure sweep engine, shared by the main thread, the sweep Web Workers and the
// bun tests. No DOM or Worker APIs in here (the inline executor only uses
// setTimeout), so the exact same per-run training code runs everywhere.
//
// DETERMINISM CONTRACT — a configuration's result never depends on how many
// workers train the sweep or in what order the runs finish:
//
// 1. Seeds. Every configuration has its own RNG, seeded from the sweep seed and
//    its run index: seed_i = (sweepSeed + i · 7919) >>> 0. The old sequential
//    runSweep already gave each run a private RNG (mulberry32(1337 + id·7919));
//    it never shared one RNG across runs. 1337 is kept as the default sweep seed,
//    so per-run results are bit-identical to before. The RNG state is one 32-bit
//    integer that travels with the Q-table between segments, so a run can hop
//    between workers without changing its random stream.
// 2. Chunking. Runs train in chunks of CHUNK updates and are evaluated after
//    every chunk, exactly as before. A dispatched segment always starts and ends
//    on chunk boundaries, so splitting a run into segments changes nothing.
// 3. Pruning rule ("prune against lower-indexed runs"). At a prune-eligible
//    checkpoint ck (early stop on, ≥ pruneAfterFrac of the run trained, not the
//    final checkpoint) run i is pruned iff
//        max{ trueQ_j(ck) : j < i, run j reached ck } − trueQ_i(ck) > pruneGap.
//    That is exactly what the old sequential loop computed (the runs trained
//    before run i were the runs j < i), but stated over run indices rather than
//    wall-clock order. A segment carries the reference values for every
//    checkpoint whose inputs are already final, and the worker applies the rule
//    in place (so with 1 worker each run is a single segment). At the first
//    checkpoint whose inputs are not yet known, run i waits until every
//    lower-indexed run has reported ck or is known never to reach it (pruned
//    earlier, stopped, skipped). Run 0 never waits, so there is no deadlock.
// 4. Updates budget. Run k only starts if Σ_{j<k} updates(j) + T ≤ cap, where
//    updates(j) counts what run j actually trained (a pruned run only counts its
//    updates up to the prune point). While a lower run is still training and the
//    answer could go either way, admission waits. → The set of started and
//    skipped runs is the same for every worker count.
//    Wall-clock caps, manual stop and cancel depend on timing by nature.

import { createQ, evaluate, generateDataset, qStar, trainUpdates, type Dataset, type Hyper } from "./rl";
import { generateThrusterDataset } from "./thrusterchain";

/* ---------------- Settings & runs ---------------- */

export type SweepDatasetId = "gridworld-random" | "halfcheetah-medium" | "thrusterchain-stall";

export interface SweepSettings {
  datasetId: SweepDatasetId;
  lrMin: number;
  lrMax: number;
  lrSteps: number;
  alphaMin: number;
  alphaMax: number;
  alphaSteps: number;
  gamma: number;
  batch: number;
  totalUpdates: number;
  earlyStop: boolean;
  /** Pruning only kicks in after this fraction of training is complete. */
  pruneAfterFrac: number;
  /** Prune when a run's true return trails the best lower-indexed run at the same checkpoint by more than this. */
  pruneGap: number;
  /** Sweep seed. Run i trains with RNG seed (seed + i·7919) >>> 0. */
  seed: number;
}

export const DEFAULT_SWEEP_SEED = 1337;
/** The sweep has always trained on dataset seed 7. */
export const SWEEP_DATASET_SEED = 7;

export const DEFAULT_SWEEP: SweepSettings = {
  datasetId: "gridworld-random",
  lrMin: 0.02,
  lrMax: 0.5,
  lrSteps: 4,
  alphaMin: 0,
  alphaMax: 3,
  alphaSteps: 4,
  gamma: 0.95,
  batch: 64,
  totalUpdates: 1200,
  earlyStop: true,
  pruneAfterFrac: 0.3,
  pruneGap: 2,
  seed: DEFAULT_SWEEP_SEED,
};

export type SweepRunStatus = "pending" | "running" | "done" | "pruned" | "stopped" | "skipped";

export interface SweepRun {
  id: number;
  lr: number;
  alpha: number;
  /** Private RNG seed of this configuration. */
  seed: number;
  status: SweepRunStatus;
  progress: number; // 0..1
  updatesDone: number;
  trueQ: number;
  estQ: number;
  gap: number;
  oodRate: number;
  loss: number;
  /** True value of the learned greedy policy at the start state, V^π(start). */
  vStart: number;
  /** Last checkpoint reached (1-based). */
  ck: number;
  /** Checkpoint at which the run was pruned, if it was. */
  prunedAt: number | null;
  /** Worker compute time spent on this run, in milliseconds (≈ CPU time). */
  computeMs: number;
  error?: string;
}

/** Updates per evaluation checkpoint. Unchanged from the sequential sweep. */
export const CHUNK = 60;

export function runSeed(sweepSeed: number, index: number): number {
  return (sweepSeed + index * 7919) >>> 0;
}

/** Log-spaced learning rates × linearly spaced alphas (alpha-major order, as before). */
export function buildRuns(s: SweepSettings): SweepRun[] {
  const lrs: number[] = [];
  for (let i = 0; i < s.lrSteps; i++) {
    const t = s.lrSteps === 1 ? 0 : i / (s.lrSteps - 1);
    lrs.push(Math.exp(Math.log(s.lrMin) + t * (Math.log(s.lrMax) - Math.log(s.lrMin))));
  }
  const alphas: number[] = [];
  for (let i = 0; i < s.alphaSteps; i++) {
    const t = s.alphaSteps === 1 ? 0 : i / (s.alphaSteps - 1);
    alphas.push(s.alphaMin + t * (s.alphaMax - s.alphaMin));
  }
  const sweepSeed = s.seed ?? DEFAULT_SWEEP_SEED;
  const runs: SweepRun[] = [];
  let id = 0;
  for (const alpha of alphas)
    for (const lr of lrs) {
      runs.push({
        id,
        lr: Math.round(lr * 1000) / 1000,
        alpha: Math.round(alpha * 100) / 100,
        seed: runSeed(sweepSeed, id),
        status: "pending",
        progress: 0,
        updatesDone: 0,
        trueQ: 0,
        estQ: 0,
        gap: 0,
        oodRate: 0,
        loss: 0,
        vStart: 0,
        ck: 0,
        prunedAt: null,
        computeMs: 0,
      });
      id++;
    }
  return runs;
}

/* ---------------- Datasets ---------------- */

export interface DatasetKey {
  id: SweepDatasetId;
  seed: number;
}

export const datasetKeyString = (k: DatasetKey) => `${k.id}:${k.seed}`;

/** Build a sweep dataset from scratch (each worker calls this once per key and caches it). */
export function buildSweepDataset(key: DatasetKey, onProgress?: (pct: number, label: string) => void): Dataset {
  if (key.id === "thrusterchain-stall") return generateThrusterDataset(key.seed, onProgress);
  return generateDataset(key.id, key.seed);
}

/** Optimal value of the start state, V*(start) = max_a Q*(start, a). */
export function optimalStartValue(ds: Dataset, gamma: number): number {
  const Qs = qStar(ds.env, gamma);
  const b = ds.env.start * 4;
  return Math.max(Qs[b], Qs[b + 1], Qs[b + 2], Qs[b + 3]);
}

/* ---------------- RNG with shippable state ---------------- */

/**
 * Exact replica of rl.mulberry32 whose state lives in `st.a`, so it can be sent
 * to another worker and resumed. Initial state for a seed is `seed >>> 0`.
 */
export function rngFromState(st: { a: number }): () => number {
  return () => {
    st.a = (st.a + 0x6d2b79f5) | 0;
    let t = Math.imul(st.a ^ (st.a >>> 15), 1 | st.a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------------- Per-run training (shared by every executor) ---------------- */

export interface SegmentTask {
  taskId: number;
  /** Index of the run inside its sweep / suite. */
  index: number;
  dataset: DatasetKey;
  lr: number;
  alpha: number;
  gamma: number;
  batch: number;
  totalUpdates: number;
  /** Chunk-aligned update range [fromUpdates, toUpdates). */
  fromUpdates: number;
  toUpdates: number;
  /** Q-table to resume from (null = fresh zero table). */
  q: Float64Array | null;
  /** RNG state to resume from. */
  rngState: number;
  pruneGap: number;
  /**
   * In-segment pruning references: pruneRefs[k] applies to checkpoint
   * ceil(fromUpdates / CHUNK) + 1 + k and holds the best true return of the
   * lower-indexed runs there (−Infinity if none), or null when no decision is
   * due at that checkpoint. Only filled where every input is already final.
   */
  pruneRefs: (number | null)[] | null;
}

export interface CheckpointMetrics {
  ck: number;
  updates: number;
  trueQ: number;
  estQ: number;
  gap: number;
  oodRate: number;
  loss: number;
  vStart: number;
}

export interface SegmentResult {
  taskId: number;
  index: number;
  toUpdates: number;
  q: Float64Array;
  rngState: number;
  checkpoints: CheckpointMetrics[];
  /** Time spent training + evaluating (excludes dataset construction). */
  computeMs: number;
  /** Checkpoint at which the in-segment pruning rule stopped the run, if it did. */
  prunedAt: number | null;
}

/** The pruning rule, shared by the scheduler and in-segment decisions. */
export function shouldPrune(bestLower: number, trueQ: number, pruneGap: number): boolean {
  return isFinite(bestLower) && bestLower - trueQ > pruneGap;
}

const nowMs = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

/**
 * Train one run from `fromUpdates` to `toUpdates`, evaluating after every chunk —
 * the same trainUpdates/evaluate sequence the sequential sweep always used.
 */
export function trainSegment(ds: Dataset, task: SegmentTask, onChunk?: (m: CheckpointMetrics) => void): SegmentResult {
  const t0 = nowMs();
  const Q = task.q ? new Float64Array(task.q) : createQ(ds);
  const st = { a: task.rngState };
  const rng = rngFromState(st);
  const h: Hyper = { alpha: task.alpha, gamma: task.gamma, lr: task.lr, batch: task.batch };
  const checkpoints: CheckpointMetrics[] = [];
  let done = task.fromUpdates;
  let ck = Math.ceil(done / CHUNK);
  const firstCk = ck + 1;
  const end = Math.min(task.toUpdates, task.totalUpdates);
  let prunedAt: number | null = null;
  while (done < end) {
    const n = Math.min(CHUNK, task.totalUpdates - done);
    const { tdLoss, cqlGap } = trainUpdates(Q, ds, h, rng, n);
    done += n;
    ck++;
    const ev = evaluate(Q, ds, task.gamma);
    const m: CheckpointMetrics = {
      ck,
      updates: done,
      trueQ: ev.trueQ,
      estQ: ev.estQ,
      gap: ev.estQ - ev.trueQ,
      oodRate: ev.oodRate,
      loss: tdLoss + h.alpha * cqlGap,
      vStart: ev.vPi[ds.env.start],
    };
    checkpoints.push(m);
    onChunk?.(m);
    const ref = task.pruneRefs?.[ck - firstCk];
    if (ref != null && shouldPrune(ref, m.trueQ, task.pruneGap)) {
      prunedAt = ck;
      break;
    }
  }
  return { taskId: task.taskId, index: task.index, toUpdates: done, q: Q, rngState: st.a, checkpoints, computeMs: nowMs() - t0, prunedAt };
}

/* ---------------- Scheduler (deterministic pruning + budget) ---------------- */

export type StopReason = "updates-budget" | "time-budget" | "cancelled";

type Phase = "pending" | "ready" | "inflight" | "await" | "terminal" | "skipped";

interface Slot {
  phase: Phase;
  q: Float64Array | null;
  rngState: number;
  /** trueQ at each checkpoint, valid where reached[ck] = 1. */
  history: Float64Array;
  reached: Uint8Array;
  lastCk: number;
  taskId: number;
}

export interface SchedulerOptions {
  /** Hard cap on total gradient updates across the sweep (null = none). */
  maxTotalUpdates?: number | null;
  /** Max configurations in flight at once (started but unfinished). Default 2. */
  maxActive?: number;
  datasetSeed?: number;
}

/**
 * Orchestrates a sweep as a stream of segment tasks. It never looks at the clock
 * or at completion order, so whatever executor drives it (1 worker, N workers,
 * inline) every configuration ends with the same metrics and status.
 */
export class SweepScheduler {
  readonly settings: SweepSettings;
  readonly runs: SweepRun[];
  /** Number of evaluation checkpoints per run. */
  readonly checkpoints: number;
  readonly dataset: DatasetKey;
  readonly maxTotalUpdates: number | null;
  maxActive: number;
  cancelled = false;
  stopReason: StopReason | null = null;
  /** Updates applied so far across all runs. */
  updatesDone = 0;
  /** Worker compute time across all segments (including discarded ones). */
  computeMs = 0;
  inFlight = 0;
  errors: string[] = [];
  /** Called after external state changes (stop, cancel, budget) so a driver can re-pump. */
  onChange: (() => void) | null = null;

  private slots: Slot[];
  private nextAdmit = 0;
  private admitting = true;
  private seq = 0;

  constructor(settings: SweepSettings, opts: SchedulerOptions & { runs?: SweepRun[] } = {}) {
    this.settings = settings;
    this.runs = opts.runs ?? buildRuns(settings);
    this.checkpoints = Math.max(1, Math.ceil(settings.totalUpdates / CHUNK));
    this.dataset = { id: settings.datasetId, seed: opts.datasetSeed ?? SWEEP_DATASET_SEED };
    this.maxTotalUpdates = opts.maxTotalUpdates ?? null;
    this.maxActive = Math.max(1, opts.maxActive ?? 2);
    const sweepSeed = settings.seed ?? DEFAULT_SWEEP_SEED;
    this.slots = this.runs.map((r) => ({
      phase: "pending",
      q: null,
      rngState: (Number.isFinite(r.seed) ? r.seed : runSeed(sweepSeed, r.id)) >>> 0,
      history: new Float64Array(this.checkpoints + 1),
      reached: new Uint8Array(this.checkpoints + 1),
      lastCk: 0,
      taskId: -1,
    }));
  }

  updatesAt(ck: number) {
    return Math.min(ck * CHUNK, this.settings.totalUpdates);
  }

  /** Same condition as the sequential sweep: early stop on, past the grace period, not the final checkpoint. */
  eligible(ck: number) {
    const s = this.settings;
    const u = this.updatesAt(ck);
    return s.earlyStop && u / s.totalUpdates >= s.pruneAfterFrac && u < s.totalUpdates;
  }

  /** Highest checkpoint up to which every lower-indexed run's outcome is final. */
  private knownUntil(i: number) {
    let k = Infinity;
    for (let j = 0; j < i; j++) {
      const p = this.slots[j].phase;
      if (p === "terminal" || p === "skipped") continue;
      k = Math.min(k, this.runs[j].ck);
    }
    return k;
  }

  /** Best true return at `ck` among lower-indexed runs that reached it (−Infinity if none). */
  private referenceAt(i: number, ck: number) {
    let best = -Infinity;
    for (let j = 0; j < i; j++) {
      const sj = this.slots[j];
      if (sj.reached[ck] && sj.history[ck] > best) best = sj.history[ck];
    }
    return best;
  }

  private activeCount() {
    let n = 0;
    for (const s of this.slots) if (s.phase === "ready" || s.phase === "inflight" || s.phase === "await") n++;
    return n;
  }

  /** Runs that trained to a checkpoint and are waiting for lower-indexed runs to get there. */
  get waiting() {
    return this.slots.filter((s) => s.phase === "await").length;
  }

  get finished() {
    return this.slots.every((s) => s.phase === "terminal" || s.phase === "skipped");
  }

  /** Upper bound on the updates still to train (pruning can only make it smaller). */
  plannedRemaining() {
    const T = this.settings.totalUpdates;
    let rem = 0;
    this.slots.forEach((s, i) => {
      if (s.phase === "pending") rem += this.admitting ? T : 0;
      else if (s.phase !== "terminal" && s.phase !== "skipped") rem += T - this.runs[i].updatesDone;
    });
    return rem;
  }

  /** Next task to dispatch, or null if nothing can run right now. */
  next(): SegmentTask | null {
    if (this.cancelled) return null;
    for (let i = 0; i < this.slots.length; i++) if (this.slots[i].phase === "ready") return this.dispatch(i);
    if (this.nextAdmit < this.slots.length && this.activeCount() < this.maxActive) {
      const verdict = this.admission(this.nextAdmit);
      if (verdict === "admit") {
        const i = this.nextAdmit++;
        this.slots[i].phase = "ready";
        this.runs[i].status = "running";
        return this.dispatch(i);
      }
      if (verdict === "skip") this.skipRemaining(this.stopReason ?? "updates-budget");
    }
    return null;
  }

  private admission(k: number): "admit" | "skip" | "wait" {
    if (!this.admitting) return "skip";
    const cap = this.maxTotalUpdates;
    if (cap == null) return "admit";
    const T = this.settings.totalUpdates;
    let lo = 0;
    let hi = 0;
    for (let j = 0; j < k; j++) {
      const done = this.runs[j].updatesDone;
      lo += done;
      hi += this.slots[j].phase === "terminal" || this.slots[j].phase === "skipped" ? done : T;
    }
    if (hi + T <= cap) return "admit";
    if (lo + T > cap) return "skip";
    return "wait";
  }

  private skipRemaining(reason: StopReason) {
    this.admitting = false;
    if (!this.stopReason) this.stopReason = reason;
    for (let i = this.nextAdmit; i < this.slots.length; i++) {
      if (this.slots[i].phase !== "pending") continue;
      this.slots[i].phase = "skipped";
      this.runs[i].status = "skipped";
    }
    this.nextAdmit = this.slots.length;
  }

  private dispatch(i: number): SegmentTask {
    const slot = this.slots[i];
    const run = this.runs[i];
    const s = this.settings;
    // Train through every checkpoint whose pruning inputs are already final; stop
    // at the first one that still depends on a lower run (or at the end).
    let toCk = this.checkpoints;
    let pruneRefs: (number | null)[] | null = null;
    if (s.earlyStop) {
      const known = this.knownUntil(i);
      pruneRefs = [];
      for (let ck = slot.lastCk + 1; ck < this.checkpoints; ck++) {
        if (!this.eligible(ck)) {
          pruneRefs.push(null);
          continue;
        }
        if (ck > known) {
          toCk = ck;
          break;
        }
        pruneRefs.push(this.referenceAt(i, ck));
      }
    }
    const task: SegmentTask = {
      taskId: ++this.seq,
      index: i,
      dataset: this.dataset,
      lr: run.lr,
      alpha: run.alpha,
      gamma: s.gamma,
      batch: s.batch,
      totalUpdates: s.totalUpdates,
      fromUpdates: this.updatesAt(slot.lastCk),
      toUpdates: this.updatesAt(toCk),
      q: slot.q,
      rngState: slot.rngState,
      pruneGap: s.pruneGap,
      pruneRefs,
    };
    slot.phase = "inflight";
    slot.taskId = task.taskId;
    this.inFlight++;
    return task;
  }

  private apply(i: number, m: CheckpointMetrics) {
    const slot = this.slots[i];
    if (m.ck < 0 || m.ck > this.checkpoints || slot.reached[m.ck]) return;
    slot.reached[m.ck] = 1;
    slot.history[m.ck] = m.trueQ;
    const run = this.runs[i];
    this.updatesDone += m.updates - run.updatesDone;
    run.updatesDone = m.updates;
    run.progress = m.updates / this.settings.totalUpdates;
    run.ck = m.ck;
    run.trueQ = m.trueQ;
    run.estQ = m.estQ;
    run.gap = m.gap;
    run.oodRate = m.oodRate;
    run.loss = m.loss;
    run.vStart = m.vStart;
  }

  /** Live per-chunk progress from an executor. */
  onChunk(task: SegmentTask, m: CheckpointMetrics) {
    const slot = this.slots[task.index];
    if (!slot || slot.taskId !== task.taskId || slot.phase !== "inflight") return;
    this.apply(task.index, m);
  }

  complete(task: SegmentTask, res: SegmentResult) {
    this.inFlight = Math.max(0, this.inFlight - 1);
    const i = task.index;
    const slot = this.slots[i];
    const run = this.runs[i];
    this.computeMs += res.computeMs;
    run.computeMs += res.computeMs;
    if (slot.taskId !== task.taskId || slot.phase !== "inflight") return; // run was stopped meanwhile
    slot.taskId = -1;
    for (const m of res.checkpoints) this.apply(i, m);
    slot.q = res.q;
    slot.rngState = res.rngState;
    slot.lastCk = Math.ceil(res.toUpdates / CHUNK);
    if (res.prunedAt != null) {
      slot.phase = "terminal";
      run.status = "pruned";
      run.prunedAt = res.prunedAt;
    } else if (res.toUpdates >= this.settings.totalUpdates) {
      slot.phase = "terminal";
      run.status = "done";
      run.progress = 1;
    } else {
      slot.phase = this.eligible(slot.lastCk) ? "await" : "ready";
    }
    this.resolveDecisions();
  }

  fail(task: SegmentTask, err: unknown) {
    this.inFlight = Math.max(0, this.inFlight - 1);
    const slot = this.slots[task.index];
    if (slot.taskId !== task.taskId || slot.phase !== "inflight") return;
    const msg = err instanceof Error ? err.message : String(err);
    this.errors.push(msg);
    this.runs[task.index].error = msg;
    this.runs[task.index].status = "stopped";
    slot.phase = "terminal";
    slot.taskId = -1;
    this.resolveDecisions();
  }

  /**
   * Make every pruning decision whose inputs are now known. Ascending order, so a
   * lower run's decision (which tells whether it will reach later checkpoints)
   * is settled before the runs above it look at it.
   */
  private resolveDecisions() {
    for (let i = 0; i < this.slots.length; i++) {
      const slot = this.slots[i];
      if (slot.phase !== "await") continue;
      const ck = slot.lastCk;
      if (this.knownUntil(i) < ck) continue; // a lower run is still training below ck
      if (shouldPrune(this.referenceAt(i, ck), slot.history[ck], this.settings.pruneGap)) {
        slot.phase = "terminal";
        this.runs[i].status = "pruned";
        this.runs[i].prunedAt = ck;
      } else {
        slot.phase = "ready";
      }
    }
  }

  /** Manual stop: freeze the run at its last reported checkpoint. */
  stopRun(index: number) {
    const slot = this.slots[index];
    if (!slot) return;
    if (slot.phase === "ready" || slot.phase === "inflight" || slot.phase === "await") {
      slot.phase = "terminal";
      slot.taskId = -1;
      this.runs[index].status = "stopped";
      this.resolveDecisions();
      this.onChange?.();
    }
  }

  /** Stop admitting new configurations; the rest are marked skipped. Started runs finish. */
  stopAdmitting(reason: StopReason) {
    if (!this.admitting) return;
    this.skipRemaining(reason);
    this.onChange?.();
  }

  cancel() {
    if (this.cancelled) return;
    this.cancelled = true;
    this.stopReason = "cancelled";
    this.admitting = false;
    this.slots.forEach((s, i) => {
      if (s.phase === "pending") {
        s.phase = "skipped";
        this.runs[i].status = "skipped";
      } else if (s.phase !== "terminal" && s.phase !== "skipped") {
        s.phase = "terminal";
        s.taskId = -1;
        this.runs[i].status = "stopped";
      }
    });
    this.nextAdmit = this.slots.length;
    this.onChange?.();
  }
}

/* ---------------- Executors & driver ---------------- */

export interface SegmentExecutor {
  readonly size: number;
  run(worker: number, task: SegmentTask, onChunk: (m: CheckpointMetrics) => void): Promise<SegmentResult>;
}

/**
 * Runs segments on the calling thread. Used by tests (optionally with random
 * per-task delays to emulate N workers finishing in arbitrary order) and as the
 * fallback when Web Workers are unavailable.
 */
export function createInlineExecutor(
  resolveDataset: (key: DatasetKey) => Dataset,
  opts: { size?: number; delayMs?: (worker: number, task: SegmentTask) => number } = {},
): SegmentExecutor {
  const cache = new Map<string, Dataset>();
  return {
    size: Math.max(1, opts.size ?? 1),
    run(worker, task, onChunk) {
      const delay = opts.delayMs?.(worker, task) ?? 0;
      return new Promise<SegmentResult>((resolve, reject) => {
        setTimeout(() => {
          try {
            const k = datasetKeyString(task.dataset);
            let ds = cache.get(k);
            if (!ds) {
              ds = resolveDataset(task.dataset);
              cache.set(k, ds);
            }
            resolve(trainSegment(ds, task, onChunk));
          } catch (err) {
            reject(err instanceof Error ? err : new Error(String(err)));
          }
        }, delay);
      });
    },
  };
}

export interface DriveHooks {
  /** Any state change worth re-rendering (chunks, completions). */
  onUpdate?: () => void;
  onDispatch?: (worker: number, task: SegmentTask) => void;
  /** Called before each dispatch round — the place to enforce a wall-clock cap. */
  tick?: () => void;
}

/**
 * Feed scheduler tasks to an executor's workers until the sweep is finished or
 * cancelled. Idle workers are filled lowest index first; the scheduler decides
 * which run goes next, so the outcome is independent of executor size.
 */
export function driveSweep(sch: SweepScheduler, ex: SegmentExecutor, hooks: DriveHooks = {}): Promise<void> {
  return new Promise<void>((resolve) => {
    const idle: number[] = Array.from({ length: ex.size }, (_, i) => i);
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      sch.onChange = null;
      hooks.onUpdate?.();
      resolve();
    };
    // pump() can be re-entered (a tick or stopRun fires sch.onChange); fold nested calls into a loop.
    let pumping = false;
    let again = false;
    const pump = () => {
      if (settled) return;
      if (pumping) {
        again = true;
        return;
      }
      pumping = true;
      try {
        do {
          again = false;
          pumpOnce();
        } while (again && !settled);
      } finally {
        pumping = false;
      }
    };
    const pumpOnce = () => {
      hooks.tick?.();
      if (sch.cancelled) return finish();
      while (idle.length > 0) {
        const task = sch.next();
        if (!task) break;
        const w = idle.shift() as number;
        hooks.onDispatch?.(w, task);
        let result: SegmentResult | null = null;
        let error: unknown = null;
        ex.run(w, task, (m) => {
          sch.onChunk(task, m);
          hooks.onUpdate?.();
        })
          .then(
            (r) => {
              result = r;
            },
            (e) => {
              error = e ?? new Error("Worker task failed.");
            },
          )
          .finally(() => {
            if (result) sch.complete(task, result);
            else sch.fail(task, error);
            idle.push(w);
            idle.sort((a, b) => a - b);
            hooks.onUpdate?.();
            pump();
          });
      }
      // Nothing in flight and nothing dispatchable: either finished, or (defensively) stalled.
      if (idle.length === ex.size && (sch.finished || sch.inFlight === 0)) finish();
    };
    sch.onChange = pump;
    pump();
  });
}
