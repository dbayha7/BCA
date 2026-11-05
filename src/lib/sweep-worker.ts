/// <reference lib="webworker" />
// Sweep pool worker. Each worker builds and caches its own copy of every dataset
// it is asked about (env.step closures are not structured-cloneable, so datasets
// never cross threads) and trains segments with the shared pure trainSegment —
// the very same code the 1-worker / inline path and the bun tests run.

import type { Dataset } from "./rl";
import {
  buildSweepDataset,
  datasetKeyString,
  optimalStartValue,
  trainSegment,
  type CheckpointMetrics,
  type DatasetKey,
  type SegmentResult,
  type SegmentTask,
} from "./sweep-core";

export type SweepWorkerRequest =
  | { id: number; type: "prepare"; dataset: DatasetKey }
  | { id: number; type: "segment"; task: SegmentTask }
  | { id: number; type: "optimal"; dataset: DatasetKey; gamma: number };

export interface DatasetInfo {
  transitions: number;
  states: number;
  name: string;
}

export type SweepWorkerResponse =
  | { id: number; type: "progress"; pct: number; label: string }
  | { id: number; type: "prepared"; setupMs: number; info: DatasetInfo }
  | { id: number; type: "chunk"; metrics: CheckpointMetrics }
  | { id: number; type: "segment-done"; result: SegmentResult; setupMs: number }
  | { id: number; type: "optimal-done"; vStar: number; setupMs: number }
  | { id: number; type: "error"; message: string };

const cache = new Map<string, Dataset>();

function getDataset(key: DatasetKey, onProgress?: (pct: number, label: string) => void): { ds: Dataset; setupMs: number } {
  const k = datasetKeyString(key);
  const hit = cache.get(k);
  if (hit) return { ds: hit, setupMs: 0 };
  const t0 = performance.now();
  const ds = buildSweepDataset(key, onProgress);
  cache.set(k, ds);
  return { ds, setupMs: performance.now() - t0 };
}

const post = (m: SweepWorkerResponse, transfer?: Transferable[]) => (self as unknown as Worker).postMessage(m, transfer ?? []);

self.onmessage = (e: MessageEvent<SweepWorkerRequest>) => {
  const msg = e.data;
  try {
    if (msg.type === "prepare") {
      const { ds, setupMs } = getDataset(msg.dataset, (pct, label) => post({ id: msg.id, type: "progress", pct, label }));
      post({ id: msg.id, type: "prepared", setupMs, info: { transitions: ds.transitions.length, states: ds.env.width * ds.env.height, name: ds.env.name } });
    } else if (msg.type === "segment") {
      const { ds, setupMs } = getDataset(msg.task.dataset);
      const result = trainSegment(ds, msg.task, (metrics) => post({ id: msg.id, type: "chunk", metrics }));
      post({ id: msg.id, type: "segment-done", result, setupMs }, [result.q.buffer]);
    } else if (msg.type === "optimal") {
      const { ds, setupMs } = getDataset(msg.dataset);
      post({ id: msg.id, type: "optimal-done", vStar: optimalStartValue(ds, msg.gamma), setupMs });
    }
  } catch (err) {
    post({ id: msg.id, type: "error", message: err instanceof Error ? err.message : "Sweep worker failed." });
  }
};
