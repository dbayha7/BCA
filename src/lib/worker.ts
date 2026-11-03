/// <reference lib="webworker" />
// Lab background worker — heavy data processing off the main thread.
// Handles: (1) exact float64 ThrusterChain simulation + dataset aggregation,
// (2) large custom JSON upload parsing/validation + empirical-MDP construction.
// The main thread only receives progress messages and a serialized dataset
// (env.step functions are not structured-cloneable, so they are stripped here
// and re-attached by the client from the shipped statistics).

import { parseCustomDataset, type Dataset, type Env } from "./rl";
import { generateThrusterDataset } from "./thrusterchain";
import { generateRouteDataset, type RouteCondition } from "./thrusterchain-route";

export type LabWorkerRequest =
  | { id: number; type: "generate-thruster"; seed: number }
  | { id: number; type: "parse-custom"; text: string }
  | { id: number; type: "generate-route"; condition: RouteCondition; seed: number };

export interface SerializedDataset extends Omit<Dataset, "env"> {
  env: Omit<Env, "step">;
}

export type LabWorkerResponse =
  | { id: number; type: "progress"; pct: number; label: string }
  | { id: number; type: "done"; dataset: SerializedDataset }
  | { id: number; type: "error"; message: string };

function serializeDataset(ds: Dataset): SerializedDataset {
  const { step: _omit, ...env } = ds.env;
  void _omit;
  return { ...ds, env };
}

self.onmessage = (e: MessageEvent<LabWorkerRequest>) => {
  const msg = e.data;
  const post = (m: LabWorkerResponse) => (self as unknown as Worker).postMessage(m);
  const progress = (pct: number, label: string) => post({ id: msg.id, type: "progress", pct, label });
  try {
    let ds: Dataset;
    if (msg.type === "generate-thruster") {
      ds = generateThrusterDataset(msg.seed, progress);
    } else if (msg.type === "generate-route") {
      ds = generateRouteDataset(msg.condition, msg.seed, progress).dataset;
    } else {
      progress(5, "Parsing JSON…");
      const raw = JSON.parse(msg.text) as unknown;
      ds = parseCustomDataset(raw, progress);
    }
    post({ id: msg.id, type: "done", dataset: serializeDataset(ds) });
  } catch (err) {
    post({ id: msg.id, type: "error", message: err instanceof Error ? err.message : "Background processing failed." });
  }
};
