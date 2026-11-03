// Run snapshots: one self-contained file that lets a teammate reopen exactly your run — dataset
// (built-in id, or the raw upload itself), hyperparameter history, Q-table, metrics and the
// minibatch-stream state — in their own lab. Data only: nothing in a snapshot is executed.

import type { ExperimentRecord } from "./experiments";
import type { DatasetId } from "./rl";

export const SNAPSHOT_SCHEMA = "offline-learning-lab.snapshot.v1";
export const MAX_SNAPSHOT_BYTES = 60 * 1024 * 1024;

/** Grid sizes of the built-in datasets, so a snapshot's Q-table can be checked before it is opened. */
export const BUILTIN_STATES: Partial<Record<DatasetId, number>> = {
  "gridworld-random": 100,
  "halfcheetah-medium": 108,
  "thrusterchain-stall": 99,
  // The route task's 23 × 15 projection grid (checked against buildProjection in the tests).
  "route-full": 345,
  "route-honeypot": 345,
  "route-bottleneck": 345,
};

export interface RunSnapshot {
  schema: typeof SNAPSHOT_SCHEMA;
  exportedAt: string;
  sharedBy?: string;
  run: Omit<ExperimentRecord, "q"> & { q: number[] };
  /** Present only for runs on uploaded datasets. */
  upload?: { name: string; text: string };
}

export function buildSnapshot(run: ExperimentRecord, upload?: { name: string; text: string }, sharedBy?: string): RunSnapshot {
  return {
    schema: SNAPSHOT_SCHEMA,
    exportedAt: new Date().toISOString(),
    sharedBy: sharedBy || undefined,
    run: { ...run, q: Array.from(run.q) },
    upload: run.datasetId === "custom" ? upload : undefined,
  };
}

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isStr = (v: unknown): v is string => typeof v === "string";
const METRIC_KEYS = ["iter", "updates", "estQ", "trueQ", "loss", "tdLoss", "cqlGap", "oodRate", "divergence"] as const;
const HYPER_KEYS = ["alpha", "gamma", "lr", "batch"] as const;

function hyperOk(h: unknown): boolean {
  return !!h && typeof h === "object" && HYPER_KEYS.every((k) => isNum((h as Record<string, unknown>)[k]));
}

/**
 * Validates an untrusted snapshot file and returns a fresh run record (new id, so it never
 * overwrites a local run) plus the upload text if the run needs one. Throws a readable error.
 */
export function parseSnapshot(raw: string, now = Date.now(), newId: () => string = () => crypto.randomUUID()): { run: ExperimentRecord; upload?: { name: string; text: string } } {
  if (raw.length > MAX_SNAPSHOT_BYTES) throw new Error("Snapshot is larger than 60 MB.");
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    throw new Error("This file is not valid JSON.");
  }
  const s = data as Partial<RunSnapshot>;
  if (!s || s.schema !== SNAPSHOT_SCHEMA) throw new Error("Not an Offline Learning Lab run snapshot (schema mismatch).");
  const r = s.run as Record<string, unknown> | undefined;
  if (!r || typeof r !== "object") throw new Error("Snapshot has no run.");
  const datasetId = r.datasetId as DatasetId;
  if (!["gridworld-random", "halfcheetah-medium", "thrusterchain-stall", "route-full", "route-honeypot", "route-bottleneck", "custom"].includes(datasetId)) throw new Error("Snapshot names an unknown dataset.");
  if (!Array.isArray(r.q) || !r.q.every(isNum)) throw new Error("Snapshot Q-table is missing or has non-finite values.");
  const states = BUILTIN_STATES[datasetId];
  if (states !== undefined && r.q.length !== states * 4) throw new Error(`Snapshot Q-table has ${r.q.length} values; ${datasetId} needs ${states * 4}.`);
  if (r.q.length === 0 || r.q.length % 4 !== 0) throw new Error("Snapshot Q-table size is not a whole number of states × 4 actions.");
  if (!Array.isArray(r.history) || r.history.length === 0) throw new Error("Snapshot has no training history.");
  for (const m of r.history) {
    if (!m || typeof m !== "object" || !METRIC_KEYS.every((k) => isNum((m as Record<string, unknown>)[k]))) throw new Error("Snapshot history has a malformed or non-finite metrics row.");
  }
  if (!hyperOk(r.hyper)) throw new Error("Snapshot hyperparameters are malformed.");
  if (!Array.isArray(r.hyperLog) || !r.hyperLog.every((h) => h && isNum(h.atUpdates) && hyperOk(h.hyper))) throw new Error("Snapshot hyperparameter log is malformed.");
  if (!isNum(r.rngState) || !isNum(r.seed)) throw new Error("Snapshot is missing its minibatch-stream state.");
  let upload: { name: string; text: string } | undefined;
  if (datasetId === "custom") {
    const u = s.upload;
    if (!u || !isStr(u.name) || !isStr(u.text) || !u.text.length) throw new Error("Snapshot of an uploaded dataset must include that dataset.");
    upload = { name: u.name.slice(0, 200), text: u.text };
  }
  const tags = Array.isArray(r.tags) ? r.tags.filter(isStr).map((t) => t.slice(0, 40)).slice(0, 12) : [];
  const name = isStr(r.name) && r.name.trim() ? r.name.trim().slice(0, 120) : "Imported run";
  const run: ExperimentRecord = {
    id: newId(),
    name: `${name} (imported${s.sharedBy ? ` from ${String(s.sharedBy).slice(0, 60)}` : ""})`,
    createdAt: now,
    updatedAt: now,
    datasetId,
    datasetName: isStr(r.datasetName) ? r.datasetName.slice(0, 120) : datasetId,
    transitions: isNum(r.transitions) ? r.transitions : undefined,
    seed: r.seed,
    rngState: r.rngState >>> 0,
    hyper: r.hyper as ExperimentRecord["hyper"],
    hyperLog: r.hyperLog as ExperimentRecord["hyperLog"],
    history: r.history as ExperimentRecord["history"],
    q: Float64Array.from(r.q),
    optimal: isNum(r.optimal) ? r.optimal : 0,
    status: "paused",
    tags: [...new Set([...tags, "imported"])],
    notes: isStr(r.notes) ? r.notes.slice(0, 4000) : "",
    anomalies: isNum(r.anomalies) ? r.anomalies : 0,
    guardEvents: Array.isArray(r.guardEvents)
      ? (r.guardEvents as unknown[])
          .filter((g): g is { atUpdates: number; message: string; at: number } => !!g && typeof g === "object" && isNum((g as Record<string, unknown>).atUpdates) && isStr((g as Record<string, unknown>).message))
          .map((g) => ({ atUpdates: g.atUpdates, message: g.message.slice(0, 500), at: isNum(g.at) ? g.at : now }))
      : [],
    computeMs: isNum(r.computeMs) ? r.computeMs : undefined,
    // A custom reward travels as code; it only ever runs inside the sandbox when the run is opened.
    reward:
      r.reward && typeof r.reward === "object" && isStr((r.reward as Record<string, unknown>).code) && isStr((r.reward as Record<string, unknown>).name)
        ? { name: String((r.reward as Record<string, unknown>).name).slice(0, 120), code: String((r.reward as Record<string, unknown>).code).slice(0, 20_000) }
        : undefined,
  };
  return { run, upload };
}
