// Audit export of a training run: what was trained, how, every metric step, every flagged
// anomaly and every guard/crash event — as JSON (complete), CSV (metrics) or Markdown (readable).

import type { LogEntry } from "@/components/lab/experiment-log";
import type { ErrorEntry } from "./error-log";
import type { ExperimentRecord } from "./experiments";
import { summarize } from "./experiments";

export const AUDIT_SCHEMA = "offline-learning-lab.audit.v1";

export type AuditInput = Pick<
  ExperimentRecord,
  "id" | "name" | "datasetId" | "datasetName" | "seed" | "hyper" | "hyperLog" | "history" | "optimal" | "status" | "tags" | "notes" | "guardEvents" | "createdAt" | "updatedAt"
> & { transitions: number };

export interface AuditLog {
  schema: typeof AUDIT_SCHEMA;
  exportedAt: string;
  experiment: {
    id: string;
    name: string;
    status: string;
    tags: string[];
    notes: string;
    createdAt: string;
    updatedAt: string;
  };
  dataset: { id: string; name: string; transitions: number };
  training: { seed: number; updatesPerStep: number; hyper: AuditInput["hyper"]; hyperLog: AuditInput["hyperLog"] };
  summary: ReturnType<typeof summarize> & { optimal: number };
  metrics: AuditInput["history"];
  logEntries: LogEntry[];
  guardEvents: AuditInput["guardEvents"];
  errors: ErrorEntry[];
}

export function buildAuditLog(run: AuditInput, logEntries: LogEntry[], errors: ErrorEntry[], updatesPerStep: number): AuditLog {
  return {
    schema: AUDIT_SCHEMA,
    exportedAt: new Date().toISOString(),
    experiment: {
      id: run.id,
      name: run.name,
      status: run.status,
      tags: run.tags,
      notes: run.notes,
      createdAt: new Date(run.createdAt).toISOString(),
      updatedAt: new Date(run.updatedAt).toISOString(),
    },
    dataset: { id: run.datasetId, name: run.datasetName, transitions: run.transitions },
    training: { seed: run.seed, updatesPerStep, hyper: run.hyper, hyperLog: run.hyperLog },
    summary: { ...summarize(run), optimal: run.optimal },
    metrics: run.history,
    // Oldest first reads like a timeline.
    logEntries: [...logEntries].sort((a, b) => a.atUpdates - b.atUpdates),
    guardEvents: run.guardEvents,
    errors,
  };
}

const CSV_COLUMNS = ["iter", "updates", "estQ", "trueQ", "gap", "loss", "tdLoss", "cqlGap", "oodRate", "divergence"] as const;

function csvCell(v: unknown): string {
  const s = typeof v === "number" ? (Number.isFinite(v) ? String(v) : "") : String(v ?? "");
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function metricsCsv(history: AuditInput["history"]): string {
  const rows = history.map((m) => [m.iter, m.updates, m.estQ, m.trueQ, m.estQ - m.trueQ, m.loss, m.tdLoss, m.cqlGap, m.oodRate, m.divergence].map(csvCell).join(","));
  return [CSV_COLUMNS.join(","), ...rows].join("\n") + "\n";
}

const f = (v: number | null | undefined, d = 2) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(d));

export function auditMarkdown(log: AuditLog): string {
  const s = log.summary;
  const lines = [
    `# ${log.experiment.name}`,
    "",
    `Exported ${log.exportedAt} · status **${log.experiment.status}**${log.experiment.tags.length ? ` · tags: ${log.experiment.tags.join(", ")}` : ""}`,
    "",
    `- Dataset: ${log.dataset.name} (${log.dataset.transitions.toLocaleString()} offline transitions)`,
    `- Seed ${log.training.seed}, ${log.training.updatesPerStep} updates per step`,
    `- Final: ${s.updates.toLocaleString()} updates, true return ${f(s.trueQ)}, estimate ${f(s.estQ)} (gap ${f(s.gap)}), OOD rate ${s.oodRate === null ? "—" : (s.oodRate * 100).toFixed(1) + "%"}, best true return ${f(s.bestTrueQ)}, optimal V*(start) ${f(s.optimal)}`,
    "",
    "## Hyperparameters over time",
    "",
    "| From update | α | γ | lr | batch |",
    "|---:|---:|---:|---:|---:|",
    ...log.training.hyperLog.map((h) => `| ${h.atUpdates.toLocaleString()} | ${h.hyper.alpha} | ${h.hyper.gamma} | ${h.hyper.lr} | ${h.hyper.batch} |`),
    "",
    "## Log",
    "",
    ...(log.logEntries.length ? log.logEntries.map((e) => `- **${e.kind}** @ ${e.atUpdates.toLocaleString()} updates: ${e.text}`) : ["- (no entries)"]),
    "",
    "## Guard and error events",
    "",
    ...(log.guardEvents.length || log.errors.length
      ? [
          ...log.guardEvents.map((g) => `- guard @ ${g.atUpdates.toLocaleString()} updates (${new Date(g.at).toISOString()}): ${g.message}`),
          ...log.errors.map((e) => `- ${e.source} (${new Date(e.at).toISOString()}): ${e.message}${e.context ? ` — ${e.context}` : ""}`),
        ]
      : ["- none"]),
    "",
    log.experiment.notes ? `## Notes\n\n${log.experiment.notes}\n` : "",
  ];
  return lines.join("\n");
}

export function slugify(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60) || "run";
}

export function downloadText(filename: string, text: string, type: string): void {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
