// Experiment records: one per training run, saved locally as it trains so nothing is lost
// on reload, and searchable afterwards. Pure helpers only (no storage calls) so they can be
// tested outside the browser.

import type { DatasetId, Hyper, Metrics } from "./rl";

export type ExperimentStatus = "training" | "paused" | "diverged";

export interface HyperChange {
  /** Cumulative updates at which the new hyperparameters took effect. */
  atUpdates: number;
  hyper: Hyper;
}

export interface ExperimentRecord {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  datasetId: DatasetId;
  datasetName: string;
  /** For uploaded datasets: key of the raw upload in local storage, so the run can be reopened. */
  uploadKey?: string;
  /** Number of offline transitions in the dataset. */
  transitions?: number;
  /** Seed the minibatch stream started from. */
  seed: number;
  /** Internal state of the minibatch stream after the last saved update. */
  rngState: number;
  hyper: Hyper;
  /** Every hyperparameter change made during the run, in order (the first entry is the start). */
  hyperLog: HyperChange[];
  /** Full metrics history, one row per training step (never truncated). */
  history: Metrics[];
  /** Q-table after the last saved update. */
  q: Float64Array;
  optimal: number;
  status: ExperimentStatus;
  tags: string[];
  notes: string;
  /** Count of anomaly entries the experiment log raised. */
  anomalies: number;
  /** Training guard events (non-finite values rolled back), newest last. */
  guardEvents: { atUpdates: number; message: string; at: number }[];
}

export interface ExperimentSummary {
  updates: number;
  trueQ: number | null;
  estQ: number | null;
  gap: number | null;
  oodRate: number | null;
  bestTrueQ: number | null;
}

export function summarize(rec: Pick<ExperimentRecord, "history">): ExperimentSummary {
  const last = rec.history[rec.history.length - 1];
  if (!last) return { updates: 0, trueQ: null, estQ: null, gap: null, oodRate: null, bestTrueQ: null };
  let best = -Infinity;
  for (const m of rec.history) if (m.trueQ > best) best = m.trueQ;
  return { updates: last.updates, trueQ: last.trueQ, estQ: last.estQ, gap: last.estQ - last.trueQ, oodRate: last.oodRate, bestTrueQ: best };
}

export function defaultExperimentName(datasetName: string, hyper: Hyper, at: number): string {
  const d = new Date(at);
  const stamp = `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  return `${datasetName} · α ${hyper.alpha.toFixed(1)} · ${stamp}`;
}

export function hyperEqual(a: Hyper, b: Hyper): boolean {
  return a.alpha === b.alpha && a.gamma === b.gamma && a.lr === b.lr && a.batch === b.batch;
}

export type SortKey = "updated" | "created" | "trueQ" | "gap" | "updates" | "name";

export interface ExperimentQuery {
  text: string;
  datasetId: DatasetId | "all";
  status: ExperimentStatus | "all";
  tag: string | "all";
  alphaMin: number | null;
  alphaMax: number | null;
  minUpdates: number | null;
  sort: SortKey;
  descending: boolean;
}

export const DEFAULT_QUERY: ExperimentQuery = {
  text: "",
  datasetId: "all",
  status: "all",
  tag: "all",
  alphaMin: null,
  alphaMax: null,
  minUpdates: null,
  sort: "updated",
  descending: true,
};

/** Free-text search over name, notes, tags and dataset; every whitespace-separated term must match. */
export function matchesText(rec: ExperimentRecord, text: string): boolean {
  const terms = text.toLowerCase().split(/\s+/).filter(Boolean);
  if (!terms.length) return true;
  const hay = [rec.name, rec.notes, rec.datasetName, rec.datasetId, ...rec.tags, rec.status].join(" ").toLowerCase();
  return terms.every((t) => hay.includes(t));
}

export function filterExperiments(all: ExperimentRecord[], q: ExperimentQuery): ExperimentRecord[] {
  const out = all.filter((r) => {
    if (q.datasetId !== "all" && r.datasetId !== q.datasetId) return false;
    if (q.status !== "all" && r.status !== q.status) return false;
    if (q.tag !== "all" && !r.tags.includes(q.tag)) return false;
    if (q.alphaMin !== null && r.hyper.alpha < q.alphaMin) return false;
    if (q.alphaMax !== null && r.hyper.alpha > q.alphaMax) return false;
    if (q.minUpdates !== null && summarize(r).updates < q.minUpdates) return false;
    return matchesText(r, q.text);
  });
  const val = (r: ExperimentRecord): number | string => {
    const s = summarize(r);
    switch (q.sort) {
      case "created":
        return r.createdAt;
      case "trueQ":
        return s.trueQ ?? -Infinity;
      case "gap":
        return s.gap ?? -Infinity;
      case "updates":
        return s.updates;
      case "name":
        return r.name.toLowerCase();
      default:
        return r.updatedAt;
    }
  };
  out.sort((a, b) => {
    const va = val(a);
    const vb = val(b);
    const c = va < vb ? -1 : va > vb ? 1 : 0;
    return q.descending ? -c : c;
  });
  return out;
}

export function allTags(all: ExperimentRecord[]): string[] {
  return [...new Set(all.flatMap((r) => r.tags))].sort();
}

export function parseTags(raw: string): string[] {
  return [...new Set(raw.split(",").map((t) => t.trim().toLowerCase()).filter(Boolean))].slice(0, 12);
}

export interface TailStability {
  /** Steps in the tail window (last 20% of the run, at least 5). */
  window: number;
  mean: number;
  std: number;
  /** Largest drop from a running best inside the window. */
  maxDrawdown: number;
}

/** How steady the true return is at the end of a run — a stable policy barely moves once trained. */
export function tailStability(history: Pick<ExperimentRecord["history"][number], "trueQ">[]): TailStability | null {
  if (history.length < 5) return null;
  const w = Math.max(5, Math.ceil(history.length * 0.2));
  const tail = history.slice(-w).map((m) => m.trueQ);
  const mean = tail.reduce((a, b) => a + b, 0) / tail.length;
  const std = Math.sqrt(tail.reduce((a, b) => a + (b - mean) * (b - mean), 0) / tail.length);
  let best = -Infinity;
  let dd = 0;
  for (const v of tail) {
    best = Math.max(best, v);
    dd = Math.max(dd, best - v);
  }
  return { window: w, mean, std, maxDrawdown: dd };
}
