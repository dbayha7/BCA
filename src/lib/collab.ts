// Researcher collaboration: discussion threads pinned to specific experiment-log entries
// (anomalies / milestones) and portable "share bundles" that carry a run's results plus
// its discussion, so teammates can review offline and merge each other's comments.

import type { LogEntry } from "@/components/lab/experiment-log";
import type { Hyper, Metrics } from "@/lib/rl";

export interface ThreadComment {
  id: string;
  author: string;
  text: string;
  createdAt: string;
}

export interface Thread {
  key: string;
  datasetId: string;
  datasetName: string;
  atUpdates: number;
  kind: "anomaly" | "milestone";
  severity: LogEntry["severity"];
  excerpt: string;
  status: "open" | "resolved";
  updatedAt: string;
  comments: ThreadComment[];
}

export type Threads = Record<string, Thread>;

export interface SharedRun {
  id: string;
  label: string;
  sharedBy: string;
  sharedAt: string;
  note: string;
  dataset: { id: string; name: string; transitions: number };
  hyper: Hyper;
  iterations: number;
  final: { updates: number; estQ: number; trueQ: number; loss: number; oodRate: number } | null;
  optimal: number;
  curve: { u: number; est: number; tru: number }[];
  log: { atUpdates: number; kind: LogEntry["kind"]; severity: LogEntry["severity"]; text: string }[];
  threads: Thread[];
}

export interface ShareBundle {
  format: "offline-learning-lab/experiment-share";
  version: 1;
  run: SharedRun;
}

const FORMAT = "offline-learning-lab/experiment-share";

function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

export const isDiscussable = (e: LogEntry) => e.kind === "anomaly" || e.kind === "milestone";

/** Stable key for a log entry — independent of list position, unique per run event. */
export function threadKeyOf(e: Pick<LogEntry, "atUpdates" | "text">, datasetId: string): string {
  return `${datasetId}:${e.atUpdates}:${fnv1a(e.text)}`;
}

export function threadFromEntry(e: LogEntry, datasetId: string, datasetName: string): Thread {
  return {
    key: threadKeyOf(e, datasetId),
    datasetId,
    datasetName,
    atUpdates: e.atUpdates,
    kind: e.kind === "milestone" ? "milestone" : "anomaly",
    severity: e.severity,
    excerpt: e.text,
    status: "open",
    updatedAt: new Date().toISOString(),
    comments: [],
  };
}

export function buildShareBundle(opts: {
  sharedBy: string;
  note: string;
  datasetId: string;
  datasetName: string;
  transitions: number;
  hyper: Hyper;
  history: Metrics[];
  optimal: number;
  log: LogEntry[];
  threads: Threads;
}): ShareBundle {
  const last = opts.history[opts.history.length - 1] ?? null;
  const step = Math.max(1, Math.ceil(opts.history.length / 120));
  const curve = opts.history.filter((_, i) => i % step === 0 || i === opts.history.length - 1).map((m) => ({ u: m.updates, est: round(m.estQ), tru: round(m.trueQ) }));
  return {
    format: FORMAT,
    version: 1,
    run: {
      id: crypto.randomUUID(),
      label: `${opts.datasetName} · ${(last?.updates ?? 0).toLocaleString()} updates`,
      sharedBy: opts.sharedBy.trim() || "Anonymous researcher",
      sharedAt: new Date().toISOString(),
      note: opts.note.trim(),
      dataset: { id: opts.datasetId, name: opts.datasetName, transitions: opts.transitions },
      hyper: { ...opts.hyper },
      iterations: opts.history.length,
      final: last ? { updates: last.updates, estQ: round(last.estQ), trueQ: round(last.trueQ), loss: round(last.loss), oodRate: round(last.oodRate) } : null,
      optimal: round(opts.optimal),
      curve,
      log: opts.log.map((e) => ({ atUpdates: e.atUpdates, kind: e.kind, severity: e.severity, text: e.text })),
      threads: Object.values(opts.threads).filter((t) => t.datasetId === opts.datasetId && (t.comments.length > 0 || t.status === "resolved")),
    },
  };
}

const round = (v: number) => Math.round(v * 1e4) / 1e4;
const str = (v: unknown, max: number) => (typeof v === "string" ? v.slice(0, max) : "");
const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : 0);
const SEVERITIES = ["info", "warn", "bad", "good"] as const;
const sev = (v: unknown): LogEntry["severity"] => (SEVERITIES.includes(v as LogEntry["severity"]) ? (v as LogEntry["severity"]) : "info");

function sanitizeThread(raw: unknown): Thread | null {
  const t = raw as Record<string, unknown>;
  if (!t || typeof t.key !== "string" || !Array.isArray(t.comments)) return null;
  const comments: ThreadComment[] = (t.comments as unknown[])
    .slice(0, 200)
    .map((c) => {
      const o = c as Record<string, unknown>;
      return { id: str(o?.id, 80), author: str(o?.author, 60) || "Unknown", text: str(o?.text, 2000), createdAt: str(o?.createdAt, 40) };
    })
    .filter((c) => c.id && c.text.trim());
  return {
    key: t.key.slice(0, 200),
    datasetId: str(t.datasetId, 80),
    datasetName: str(t.datasetName, 120),
    atUpdates: num(t.atUpdates),
    kind: t.kind === "milestone" ? "milestone" : "anomaly",
    severity: sev(t.severity),
    excerpt: str(t.excerpt, 600),
    status: t.status === "resolved" ? "resolved" : "open",
    updatedAt: str(t.updatedAt, 40) || new Date(0).toISOString(),
    comments,
  };
}

export function parseShareBundle(raw: string): ShareBundle {
  let doc: Record<string, unknown>;
  try {
    doc = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    throw new Error("That file is not valid JSON.");
  }
  if (doc?.format !== FORMAT) throw new Error("Not an Offline Learning Lab experiment share file.");
  const r = doc.run as Record<string, unknown> | undefined;
  if (!r || typeof r.id !== "string") throw new Error("The share file has no run data.");
  const ds = (r.dataset ?? {}) as Record<string, unknown>;
  const h = (r.hyper ?? {}) as Record<string, unknown>;
  const f = r.final as Record<string, unknown> | null;
  const run: SharedRun = {
    id: r.id.slice(0, 80),
    label: str(r.label, 160) || "Shared run",
    sharedBy: str(r.sharedBy, 60) || "Anonymous researcher",
    sharedAt: str(r.sharedAt, 40),
    note: str(r.note, 1000),
    dataset: { id: str(ds.id, 80), name: str(ds.name, 120) || "Unknown dataset", transitions: num(ds.transitions) },
    hyper: { alpha: num(h.alpha), gamma: num(h.gamma), lr: num(h.lr), batch: num(h.batch) },
    iterations: num(r.iterations),
    final: f ? { updates: num(f.updates), estQ: num(f.estQ), trueQ: num(f.trueQ), loss: num(f.loss), oodRate: num(f.oodRate) } : null,
    optimal: num(r.optimal),
    curve: Array.isArray(r.curve) ? (r.curve as Record<string, unknown>[]).slice(0, 400).map((p) => ({ u: num(p?.u), est: num(p?.est), tru: num(p?.tru) })) : [],
    log: Array.isArray(r.log)
      ? (r.log as Record<string, unknown>[]).slice(0, 500).map((e) => ({
          atUpdates: num(e?.atUpdates),
          kind: e?.kind === "anomaly" || e?.kind === "milestone" ? e.kind : "summary",
          severity: sev(e?.severity),
          text: str(e?.text, 600),
        }))
      : [],
    threads: Array.isArray(r.threads) ? (r.threads as unknown[]).slice(0, 500).map(sanitizeThread).filter((t): t is Thread => !!t) : [],
  };
  return { format: FORMAT, version: 1, run };
}

/** Merges incoming threads into local ones: comments are unioned by id, status follows the newer edit. */
export function mergeThreads(local: Threads, incoming: Thread[]): { threads: Threads; newComments: number } {
  const next: Threads = { ...local };
  let newComments = 0;
  for (const t of incoming) {
    const cur = next[t.key];
    if (!cur) {
      next[t.key] = { ...t, comments: [...t.comments] };
      newComments += t.comments.length;
      continue;
    }
    const ids = new Set(cur.comments.map((c) => c.id));
    const added = t.comments.filter((c) => !ids.has(c.id));
    newComments += added.length;
    const newer = Date.parse(t.updatedAt) > Date.parse(cur.updatedAt);
    next[t.key] = {
      ...cur,
      status: newer ? t.status : cur.status,
      updatedAt: newer ? t.updatedAt : cur.updatedAt,
      comments: [...cur.comments, ...added].sort((a, b) => a.createdAt.localeCompare(b.createdAt)),
    };
  }
  return { threads: next, newComments };
}

export function bundleToMarkdown(run: SharedRun): string {
  const f = run.final;
  const lines = [
    `# Experiment report — ${run.dataset.name}`,
    "",
    `Shared by **${run.sharedBy}** on ${new Date(run.sharedAt).toLocaleString()}`,
    run.note ? `\n> ${run.note.replace(/\n/g, "\n> ")}` : "",
    "",
    "## Setup",
    `- Dataset: ${run.dataset.name} (${run.dataset.transitions.toLocaleString()} transitions)`,
    `- Hyperparameters: α=${run.hyper.alpha}, γ=${run.hyper.gamma}, lr=${run.hyper.lr}, batch ${run.hyper.batch}`,
    `- Training: ${run.iterations} iterations${f ? `, ${f.updates.toLocaleString()} updates` : ""}`,
    "",
    "## Results",
    f
      ? [
          `- True return V^π(start): **${f.trueQ.toFixed(3)}** (optimal ${run.optimal.toFixed(3)})`,
          `- Estimated Q: ${f.estQ.toFixed(3)} (gap ${(f.estQ - f.trueQ >= 0 ? "+" : "") + (f.estQ - f.trueQ).toFixed(3)})`,
          `- Loss: ${f.loss.toFixed(3)} · OOD action rate: ${(f.oodRate * 100).toFixed(1)}%`,
        ].join("\n")
      : "- No training steps recorded yet.",
    "",
    "## Flagged events",
  ];
  const flagged = run.log.filter((e) => e.kind !== "summary");
  if (flagged.length === 0) lines.push("- None flagged.");
  for (const e of [...flagged].reverse()) {
    lines.push(`- **${e.kind === "milestone" ? "Milestone" : "Anomaly"} @ ${e.atUpdates.toLocaleString()} updates** — ${e.text}`);
    const t = run.threads.find((th) => th.key === threadKeyOf(e, run.dataset.id));
    if (t) {
      if (t.status === "resolved") lines.push("  - _Resolved_");
      for (const c of t.comments) lines.push(`  - ${c.author}: ${c.text.replace(/\n/g, " ")}`);
    }
  }
  return lines.filter((l, i, a) => !(l === "" && a[i - 1] === "")).join("\n");
}

export function initialsOf(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  return (parts[0][0] + (parts.length > 1 ? parts[parts.length - 1][0] : "")).toUpperCase();
}

const TONES = ["bg-primary/15 text-primary", "bg-success/15 text-success", "bg-warning/15 text-warning", "bg-destructive/15 text-destructive", "bg-accent text-accent-foreground"];

export function toneOf(name: string): string {
  return TONES[parseInt(fnv1a(name.trim().toLowerCase()), 16) % TONES.length];
}

export function relTime(iso: string): string {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "";
  const s = Math.round((Date.now() - t) / 1000);
  if (s < 45) return "just now";
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return new Date(t).toLocaleDateString();
}
