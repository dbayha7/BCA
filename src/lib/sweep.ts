// Hyperparameter sweep API: builds a grid of (learning rate × conservative
// weight) configurations and trains them. The engine (seeding, chunked training,
// the order-independent pruning rule and the updates budget) lives in the pure
// module ./sweep-core so the Web Worker pool, the main thread and the tests all
// share one implementation; see the DETERMINISM CONTRACT at the top of that file.

import { generateDataset, type Dataset } from "./rl";
import { createInlineExecutor, driveSweep, SweepScheduler, type SweepRun, type SweepSettings } from "./sweep-core";
import { generateThrusterInWorker, type ProgressFn } from "./worker-client";

export {
  buildRuns,
  CHUNK,
  DEFAULT_SWEEP,
  DEFAULT_SWEEP_SEED,
  runSeed,
  SWEEP_DATASET_SEED,
  SweepScheduler,
  driveSweep,
  type SweepDatasetId,
  type SweepRun,
  type SweepRunStatus,
  type SweepSettings,
  type StopReason,
} from "./sweep-core";

export interface SweepSignal {
  cancelled: boolean;
  stopRequested: Set<number>;
}

export function getSweepDataset(id: SweepSettings["datasetId"], seed: number, onProgress?: ProgressFn): Promise<Dataset> {
  if (id === "thrusterchain-stall") return generateThrusterInWorker(seed, onProgress);
  return Promise.resolve(generateDataset(id, seed));
}

/**
 * Sequential, main-thread sweep over an already-built dataset (the original API,
 * kept for compatibility). It is the 1-worker case of the pooled engine, so it
 * produces exactly the same per-run results as any pool size. Mutates `runs` in
 * place and calls `onUpdate` as chunks complete.
 */
export async function runSweep(
  settings: SweepSettings,
  dataset: Dataset,
  runs: SweepRun[],
  onUpdate: () => void,
  signal: SweepSignal,
): Promise<void> {
  const sch = new SweepScheduler(settings, { runs, maxActive: 1 });
  const ex = createInlineExecutor(() => dataset, { size: 1 });
  const stopped = new Set<number>();
  await driveSweep(sch, ex, {
    onUpdate,
    tick: () => {
      if (signal.cancelled) sch.cancel();
      signal.stopRequested.forEach((id) => {
        if (stopped.has(id)) return;
        stopped.add(id);
        const idx = runs.findIndex((r) => r.id === id);
        if (idx >= 0) sch.stopRun(idx);
      });
    },
  });
}
