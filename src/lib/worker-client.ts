// Main-thread client for the lab Web Worker. Spawns a module worker per task,
// forwards progress to the UI, rehydrates the serialized dataset (re-attaching
// the empirical env.step from the shipped statistics), and falls back to
// synchronous main-thread processing when Workers are unavailable (e.g. SSR).

import { parseCustomDataset, type Dataset, type Env } from "./rl";
import { generateThrusterDataset } from "./thrusterchain";
import type { LabWorkerRequest, LabWorkerResponse, SerializedDataset } from "./worker";

export type ProgressFn = (pct: number, label: string) => void;

let seq = 0;

/** Re-attach the (non-cloneable) env.step for empirical datasets from shipped stats. */
function rehydrate(s: SerializedDataset): Dataset {
  const env = s.env as Env;
  const ds = { ...s, env } as Dataset;
  const rMin = ds.rewardMin;
  const terminals = env.terminals;
  const isCustom = env.id === "custom";
  env.step = (st, a) => {
    if (isCustom && terminals.has(st)) return { s2: st, r: 0, done: true };
    const k = st * 4 + a;
    if (ds.counts[k] === 0) return { s2: st, r: rMin, done: false };
    const s2 = ds.modeNext[k];
    return { s2, r: ds.rewardMean[k], done: isCustom && terminals.has(s2) };
  };
  return ds;
}

type TaskRequest = { type: "generate-thruster"; seed: number } | { type: "parse-custom"; text: string };

function runTask(req: TaskRequest, onProgress: ProgressFn | undefined, fallback: () => Dataset): Promise<Dataset> {
  if (typeof window === "undefined" || typeof Worker === "undefined") {
    return Promise.resolve(fallback());
  }
  return new Promise<Dataset>((resolve, reject) => {
    let worker: Worker;
    try {
      worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    } catch {
      try {
        resolve(fallback());
      } catch (err) {
        reject(err instanceof Error ? err : new Error("Processing failed."));
      }
      return;
    }
    const id = ++seq;
    worker.onmessage = (e: MessageEvent<LabWorkerResponse>) => {
      const m = e.data;
      if (!m || m.id !== id) return;
      if (m.type === "progress") {
        onProgress?.(m.pct, m.label);
      } else if (m.type === "done") {
        worker.terminate();
        resolve(rehydrate(m.dataset));
      } else {
        worker.terminate();
        reject(new Error(m.message));
      }
    };
    worker.onerror = () => {
      worker.terminate();
      // Worker failed to load (e.g. unsupported environment) — do the work on the main thread.
      try {
        resolve(fallback());
      } catch (err) {
        reject(err instanceof Error ? err : new Error("Processing failed."));
      }
    };
    worker.postMessage({ ...req, id } as LabWorkerRequest);
  });
}

/** Simulate the ThrusterChain system in a background worker (float64 physics + aggregation). */
export function generateThrusterInWorker(seed: number, onProgress?: ProgressFn): Promise<Dataset> {
  return runTask({ type: "generate-thruster", seed }, onProgress, () => generateThrusterDataset(seed));
}

/** Parse + validate a large custom JSON upload in a background worker. */
export function parseCustomInWorker(text: string, onProgress?: ProgressFn): Promise<Dataset> {
  return runTask({ type: "parse-custom", text }, onProgress, () => parseCustomDataset(JSON.parse(text) as unknown));
}
