// Hyperparameter sweep engine: builds a grid of (learning rate × conservative
// weight) configurations, trains them sequentially in small async chunks so the
// UI stays responsive, evaluates each at checkpoints, and supports early
// termination of poorly performing runs (automatic pruning + manual stop).

import {
  createQ,
  evaluate,
  generateDataset,
  mulberry32,
  trainUpdates,
  type Dataset,
  type Hyper,
} from "./rl";
import { generateThrusterInWorker, type ProgressFn } from "./worker-client";

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
  /** Prune when a run's true return trails the best seen at the same checkpoint by more than this. */
  pruneGap: number;
}

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
};

export type SweepRunStatus = "pending" | "running" | "done" | "pruned" | "stopped";

export interface SweepRun {
  id: number;
  lr: number;
  alpha: number;
  status: SweepRunStatus;
  progress: number; // 0..1
  updatesDone: number;
  trueQ: number;
  estQ: number;
  gap: number;
  oodRate: number;
  loss: number;
}

/** Log-spaced learning rates × linearly spaced alphas. */
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
  const runs: SweepRun[] = [];
  let id = 0;
  for (const alpha of alphas)
    for (const lr of lrs)
      runs.push({
        id: id++,
        lr: Math.round(lr * 1000) / 1000,
        alpha: Math.round(alpha * 100) / 100,
        status: "pending",
        progress: 0,
        updatesDone: 0,
        trueQ: 0,
        estQ: 0,
        gap: 0,
        oodRate: 0,
        loss: 0,
      });
  return runs;
}

export interface SweepSignal {
  cancelled: boolean;
  stopRequested: Set<number>;
}

export function getSweepDataset(id: SweepDatasetId, seed: number, onProgress?: ProgressFn): Promise<Dataset> {
  if (id === "thrusterchain-stall") return generateThrusterInWorker(seed, onProgress);
  return Promise.resolve(generateDataset(id, seed));
}

const CHUNK = 60; // updates per async slice — keeps the main thread free for painting

/**
 * Runs all configurations sequentially. Mutates `runs` in place and calls
 * `onUpdate` after every chunk so the UI can re-render live. Early termination:
 * automatic pruning against the best true return seen at the same checkpoint,
 * plus manual per-run stop and global cancel via `signal`.
 */
export async function runSweep(
  settings: SweepSettings,
  dataset: Dataset,
  runs: SweepRun[],
  onUpdate: () => void,
  signal: SweepSignal,
): Promise<void> {
  const checkpoints = Math.max(1, Math.ceil(settings.totalUpdates / CHUNK));
  const bestAt = new Float64Array(checkpoints + 1).fill(-Infinity);

  for (const run of runs) {
    if (signal.cancelled) {
      if (run.status === "pending") run.status = "stopped";
      continue;
    }
    run.status = "running";
    onUpdate();
    const Q = createQ(dataset);
    const rng = mulberry32(1337 + run.id * 7919);
    const h: Hyper = { alpha: run.alpha, gamma: settings.gamma, lr: run.lr, batch: settings.batch };
    let done = 0;
    let ck = 0;

    while (done < settings.totalUpdates) {
      if (signal.cancelled || signal.stopRequested.has(run.id)) {
        run.status = "stopped";
        break;
      }
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

      // Early pruning: trailing the best-at-this-checkpoint by more than pruneGap
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
      onUpdate();
      if (run.status === "pruned") break;
      // Yield to the event loop so the UI can paint
      await new Promise<void>((r) => setTimeout(r, 0));
    }

    if (run.status === "running") run.status = "done";
    onUpdate();
  }
}
