// Main-thread Web Worker pool for sweeps and the benchmark suite. Each worker
// owns a dataset cache and trains segments with the shared pure trainSegment.
// The pool tracks per-worker status and busy time for the resource monitor.
// If Workers are unavailable (or a worker script fails to load) that slot falls
// back to running the same code on the main thread.

import type { Dataset } from "./rl";
import {
  buildSweepDataset,
  datasetKeyString,
  optimalStartValue,
  trainSegment,
  type CheckpointMetrics,
  type DatasetKey,
  type SegmentExecutor,
  type SegmentResult,
  type SegmentTask,
} from "./sweep-core";
import type { DatasetInfo, SweepWorkerRequest, SweepWorkerResponse } from "./sweep-worker";

export type WorkerPhase = "idle" | "preparing" | "training" | "error";

export interface WorkerStatus {
  index: number;
  phase: WorkerPhase;
  /** Run index currently being trained, if any. */
  runIndex: number | null;
  detail: string;
  /** Segments completed. */
  tasks: number;
  /** Training + evaluation time (≈ CPU time of this worker). */
  busyMs: number;
  /** Dataset construction time. */
  setupMs: number;
  /** True when this slot runs on the main thread (no Worker support). */
  inline: boolean;
}

export function hardwareCores(): number {
  const n = typeof navigator !== "undefined" ? navigator.hardwareConcurrency : 0;
  return Number.isFinite(n) && n > 0 ? n : 4;
}

/** Default pool size: min(4, cores − 1), at least 1. */
export function defaultPoolSize(): number {
  return Math.max(1, Math.min(4, hardwareCores() - 1));
}

type RequestBody =
  | { type: "prepare"; dataset: DatasetKey }
  | { type: "segment"; task: SegmentTask }
  | { type: "optimal"; dataset: DatasetKey; gamma: number };

interface Pending {
  worker: number;
  body: RequestBody;
  resolve: (v: never) => void;
  reject: (e: Error) => void;
  onChunk?: (m: CheckpointMetrics) => void;
  onProgress?: (pct: number, label: string) => void;
}

export class SweepPool implements SegmentExecutor {
  readonly size: number;
  readonly status: WorkerStatus[];
  /** Called whenever a worker's status changes (throttle re-renders on your side). */
  onStatus: (() => void) | null = null;
  /** Optional human label for a training task, shown in per-worker status. */
  labelFor: ((task: SegmentTask) => string) | null = null;

  private workers: (Worker | null)[] = [];
  private pending = new Map<number, Pending>();
  private seq = 0;
  private inlineCache = new Map<string, Dataset>();
  private alive = true;

  constructor(size: number) {
    this.size = Math.max(1, Math.floor(size));
    this.status = Array.from({ length: this.size }, (_, index) => ({
      index,
      phase: "idle" as WorkerPhase,
      runIndex: null,
      detail: "Idle",
      tasks: 0,
      busyMs: 0,
      setupMs: 0,
      inline: false,
    }));
    for (let i = 0; i < this.size; i++) this.workers.push(this.spawn(i));
  }

  private spawn(i: number): Worker | null {
    if (typeof window === "undefined" || typeof Worker === "undefined") {
      this.status[i].inline = true;
      return null;
    }
    try {
      const w = new Worker(new URL("./sweep-worker.ts", import.meta.url), { type: "module" });
      w.onmessage = (e: MessageEvent<SweepWorkerResponse>) => this.handle(e.data);
      w.onerror = (e) => {
        e.preventDefault();
        this.fallBackToInline(i);
      };
      return w;
    } catch {
      this.status[i].inline = true;
      return null;
    }
  }

  /** Worker script failed to load or crashed: finish its queued work on the main thread. */
  private fallBackToInline(i: number) {
    const w = this.workers[i];
    if (!w) return;
    w.terminate();
    this.workers[i] = null;
    this.status[i].inline = true;
    for (const [id, p] of this.pending) if (p.worker === i) this.runInline(id, p);
  }

  private label(task: SegmentTask) {
    return this.labelFor?.(task) ?? `Run #${task.index + 1}`;
  }

  private setStatus(i: number, patch: Partial<WorkerStatus>) {
    Object.assign(this.status[i], patch);
    this.onStatus?.();
  }

  private handle(m: SweepWorkerResponse) {
    const p = this.pending.get(m.id);
    if (!p) return;
    const i = p.worker;
    const st = this.status[i];
    switch (m.type) {
      case "progress":
        p.onProgress?.(m.pct, m.label);
        this.setStatus(i, { detail: m.label });
        return;
      case "chunk": {
        p.onChunk?.(m.metrics);
        if (p.body.type === "segment") {
          const t = p.body.task;
          this.setStatus(i, { detail: `${this.label(t)} · ${m.metrics.updates.toLocaleString()}/${t.totalUpdates.toLocaleString()} updates` });
        }
        return;
      }
      case "prepared":
        this.pending.delete(m.id);
        this.setStatus(i, { phase: "idle", detail: "Dataset ready", setupMs: st.setupMs + m.setupMs });
        (p.resolve as (v: DatasetInfo) => void)(m.info);
        return;
      case "segment-done":
        this.pending.delete(m.id);
        this.setStatus(i, {
          phase: "idle",
          runIndex: null,
          detail: "Idle",
          tasks: st.tasks + 1,
          busyMs: st.busyMs + m.result.computeMs,
          setupMs: st.setupMs + m.setupMs,
        });
        (p.resolve as (v: SegmentResult) => void)(m.result);
        return;
      case "optimal-done":
        this.pending.delete(m.id);
        this.setStatus(i, { phase: "idle", detail: "Idle", setupMs: st.setupMs + m.setupMs });
        (p.resolve as (v: number) => void)(m.vStar);
        return;
      case "error":
        this.pending.delete(m.id);
        this.setStatus(i, { phase: "error", runIndex: null, detail: m.message });
        p.reject(new Error(m.message));
        return;
    }
  }

  private inlineDataset(key: DatasetKey) {
    const k = datasetKeyString(key);
    let ds = this.inlineCache.get(k);
    let setupMs = 0;
    if (!ds) {
      const t0 = performance.now();
      ds = buildSweepDataset(key);
      setupMs = performance.now() - t0;
      this.inlineCache.set(k, ds);
    }
    return { ds, setupMs };
  }

  /** Same work as the worker, on the main thread (async so callers never block synchronously). */
  private runInline(id: number, p: Pending) {
    setTimeout(() => {
      if (!this.pending.has(id)) return;
      try {
        const b = p.body;
        if (b.type === "prepare") {
          const { ds, setupMs } = this.inlineDataset(b.dataset);
          this.handle({ id, type: "prepared", setupMs, info: { transitions: ds.transitions.length, states: ds.env.width * ds.env.height, name: ds.env.name } });
        } else if (b.type === "segment") {
          const { ds, setupMs } = this.inlineDataset(b.task.dataset);
          const result = trainSegment(ds, b.task, (metrics) => this.handle({ id, type: "chunk", metrics }));
          this.handle({ id, type: "segment-done", result, setupMs });
        } else {
          const { ds, setupMs } = this.inlineDataset(b.dataset);
          this.handle({ id, type: "optimal-done", vStar: optimalStartValue(ds, b.gamma), setupMs });
        }
      } catch (err) {
        this.handle({ id, type: "error", message: err instanceof Error ? err.message : "Processing failed." });
      }
    }, 0);
  }

  private send<T>(worker: number, body: RequestBody, extra: Pick<Pending, "onChunk" | "onProgress"> = {}): Promise<T> {
    if (!this.alive) return Promise.reject(new Error("Worker pool was shut down."));
    const id = ++this.seq;
    return new Promise<T>((resolve, reject) => {
      const p: Pending = { worker, body, resolve: resolve as (v: never) => void, reject, ...extra };
      this.pending.set(id, p);
      const w = this.workers[worker];
      if (w) w.postMessage({ id, ...body } as SweepWorkerRequest);
      else this.runInline(id, p);
    });
  }

  /** Build (and cache) the dataset in every worker in parallel. */
  async prepare(dataset: DatasetKey, onProgress?: (pct: number, label: string) => void): Promise<DatasetInfo> {
    const all = this.status.map((_, i) => {
      this.setStatus(i, { phase: "preparing", detail: "Building dataset…" });
      return this.send<DatasetInfo>(i, { type: "prepare", dataset }, { onProgress: i === 0 ? onProgress : undefined });
    });
    const infos = await Promise.all(all);
    return infos[0];
  }

  run(worker: number, task: SegmentTask, onChunk: (m: CheckpointMetrics) => void): Promise<SegmentResult> {
    this.setStatus(worker, {
      phase: "training",
      runIndex: task.index,
      detail: `${this.label(task)} · ${task.fromUpdates.toLocaleString()}/${task.totalUpdates.toLocaleString()} updates`,
    });
    return this.send<SegmentResult>(worker, { type: "segment", task }, { onChunk });
  }

  /** V*(start) of a dataset, computed in a worker. */
  optimal(dataset: DatasetKey, gamma: number, worker = 0): Promise<number> {
    this.setStatus(worker, { phase: "preparing", detail: "Computing V*(start)…" });
    return this.send<number>(worker, { type: "optimal", dataset, gamma });
  }

  /** Reset per-worker counters (keeps the workers and their dataset caches). */
  resetCounters() {
    this.status.forEach((s) => Object.assign(s, { tasks: 0, busyMs: 0, setupMs: 0 }));
    this.onStatus?.();
  }

  terminate() {
    this.alive = false;
    this.workers.forEach((w) => w?.terminate());
    this.workers = [];
    for (const p of this.pending.values()) p.reject(new Error("Worker pool was shut down."));
    this.pending.clear();
  }
}
