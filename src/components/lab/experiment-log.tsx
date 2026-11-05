// AI-driven experiment log: generates natural-language summaries of training runs
// and flags performance anomalies (loss spikes, overestimation blow-ups, OOD surges,
// true-return regressions) directly from the metrics stream — no backend required.

import { useMemo } from "react";
import { VirtualList } from "@/components/lab/virtual-list";
import type { Dataset, Hyper, Metrics } from "@/lib/rl";

export interface LogEntry {
  id: string;
  atUpdates: number;
  kind: "summary" | "anomaly" | "milestone";
  severity: "info" | "warn" | "bad" | "good";
  text: string;
}

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

/** Rule-based narrative generator — scans the metrics history and emits dated log entries. */
export function buildLogEntries(history: Metrics[], ds: Dataset | null, hyper: Hyper, optimal: number): LogEntry[] {
  if (!ds || history.length === 0) return [];
  const out: LogEntry[] = [];
  const push = (m: Metrics, kind: LogEntry["kind"], severity: LogEntry["severity"], text: string) =>
    out.push({ id: `${kind}-${m.updates}-${out.length}`, atUpdates: m.updates, kind, severity, text });

  const first = history[0];
  push(
    first,
    "summary",
    "info",
    `Run started on ${ds.env.name} (${ds.transitions.length.toLocaleString()} offline transitions) with α=${hyper.alpha.toFixed(1)}, γ=${hyper.gamma.toFixed(2)}, lr=${hyper.lr.toFixed(2)}, batch ${hyper.batch}. Initial estimate ${first.estQ.toFixed(2)} vs ground truth ${first.trueQ.toFixed(2)}.`,
  );

  let bestTrue = first.trueQ;
  let crossedHalf = false;
  let lastAnomalyIter = -10;

  for (let i = 1; i < history.length; i++) {
    const m = history[i];
    const prevWindow = history.slice(Math.max(0, i - 8), i);
    const lossBase = mean(prevWindow.map((p) => p.loss));
    const gap = m.estQ - m.trueQ;
    const prevGap = history[i - 1].estQ - history[i - 1].trueQ;

    // Anomalies (rate-limited so the log stays readable)
    if (i - lastAnomalyIter >= 5) {
      if (lossBase > 0.05 && m.loss > lossBase * 2.5) {
        push(m, "anomaly", "warn", `Loss spiked to ${m.loss.toFixed(2)} — ${(m.loss / lossBase).toFixed(1)}× the recent average. Likely a batch heavy in rarely-visited (s,a) pairs; expect it to settle unless the learning rate is too high.`);
        lastAnomalyIter = i;
      } else if (gap > 1 && gap > prevGap + 0.4) {
        push(
          m,
          "anomaly",
          "bad",
          m.oodRate > 0
            ? `Overestimation jumped to +${gap.toFixed(2)} while ${(m.oodRate * 100).toFixed(0)}% of greedy actions are ones the dataset never tried — extrapolation error is a likely cause. Raising α would push those estimates back down.`
            : `Overestimation jumped to +${gap.toFixed(2)}, yet every greedy action is one the dataset contains (OOD 0%). The error comes from supported actions — few samples per (s,a) and bootstrapped targets — not from extrapolation, so α alone may not fix it; a lower learning rate or more data would.`,
        );
        lastAnomalyIter = i;
      } else if (m.oodRate > 0.2 && m.oodRate > history[i - 1].oodRate + 0.05) {
        push(m, "anomaly", "warn", `Out-of-distribution action rate surged to ${(m.oodRate * 100).toFixed(0)}% — the greedy policy is drifting away from the behavior data's support.`);
        lastAnomalyIter = i;
      } else if (i >= 8 && m.divergence > mean(prevWindow.map((p) => p.divergence)) + 0.35) {
        // Drift monitor: the greedy policy is moving away from what the behavior data did.
        push(m, "anomaly", "warn", `Policy drift: the policy↔data NLL rose to ${m.divergence.toFixed(2)} from a recent average of ${mean(prevWindow.map((p) => p.divergence)).toFixed(2)}. The learned policy is choosing actions the logger rarely took, so its value estimates rest on thinner data.`);
        lastAnomalyIter = i;
      } else if (m.trueQ < bestTrue - Math.max(0.5, Math.abs(bestTrue) * 0.15)) {
        push(m, "anomaly", "bad", `True return regressed to ${m.trueQ.toFixed(2)} from a best of ${bestTrue.toFixed(2)}. The policy got worse while the estimate ${m.estQ > m.trueQ ? "stayed optimistic" : "tracked it"} — a sign of unstable targets.`);
        lastAnomalyIter = i;
      }
    }

    // Milestones
    if (m.trueQ > bestTrue) bestTrue = m.trueQ;
    if (!crossedHalf && optimal > 0 && m.trueQ >= optimal * 0.5) {
      crossedHalf = true;
      push(m, "milestone", "good", `Policy reached half of optimal: true return ${m.trueQ.toFixed(2)} vs V*(start) ${optimal.toFixed(2)} after ${m.updates.toLocaleString()} updates.`);
    }
  }

  // Closing summary
  const last = history[history.length - 1];
  const gap = last.estQ - last.trueQ;
  const trend = history.length >= 6 ? last.trueQ - history[history.length - 6].trueQ : 0;
  const verdict =
    gap > 1 && last.oodRate > 0.1
      ? "the critic remains optimistic about unseen actions — consider a higher α"
      : Math.abs(gap) < 0.3
        ? "estimates track the true return closely; the conservative penalty is doing its job"
        : gap < -0.5
          ? "the critic is pessimistic (under-estimating) — α may be stronger than needed"
          : "estimation error is moderate and stable";
  push(
    last,
    "summary",
    gap > 1 ? "warn" : "info",
    `After ${last.updates.toLocaleString()} updates: true return ${last.trueQ.toFixed(2)} (${trend >= 0 ? "improving" : "declining"} ${trend >= 0 ? "+" : ""}${trend.toFixed(2)} over the last 5 iterations), estimate ${last.estQ.toFixed(2)} (gap ${gap >= 0 ? "+" : ""}${gap.toFixed(2)}), OOD rate ${(last.oodRate * 100).toFixed(1)}%. Overall, ${verdict}.`,
  );

  return out.reverse(); // newest first
}

const TONE: Record<LogEntry["severity"], string> = {
  info: "border-border bg-surface/60 text-muted-foreground",
  warn: "border-warning/40 bg-warning/10 text-foreground",
  bad: "border-destructive/40 bg-destructive/10 text-foreground",
  good: "border-success/40 bg-success/10 text-foreground",
};

const BADGE: Record<LogEntry["kind"], string> = {
  summary: "Summary",
  anomaly: "Anomaly",
  milestone: "Milestone",
};

export function ExperimentLog({ history, ds, hyper, optimal }: { history: Metrics[]; ds: Dataset | null; hyper: Hyper; optimal: number }) {
  const entries = useMemo(() => buildLogEntries(history, ds, hyper, optimal), [history, ds, hyper, optimal]);

  if (entries.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border px-4 py-6 text-center text-[11.5px] text-muted-foreground">
        The log narrates your run as it trains — start pressing <span className="font-medium text-foreground">Step Training</span> to generate summaries and anomaly reports.
      </div>
    );
  }

  const anomalies = entries.filter((e) => e.kind === "anomaly").length;
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-[10.5px] text-muted-foreground">
        <span>{entries.length} entries · newest first</span>
        <span className={anomalies > 0 ? "font-medium text-warning" : ""}>{anomalies} anomal{anomalies === 1 ? "y" : "ies"} flagged</span>
      </div>
      <VirtualList
        items={entries}
        rowHeight={86}
        height={entries.length >= 4 ? 344 : entries.length * 86}
        scrollKey={`exp-log-${ds?.env.id ?? "none"}`}
        ariaLabel="Experiment log"
        className="rounded-md"
        renderRow={(e) => (
          <div className="h-full py-1">
            <div className={`flex h-full flex-col justify-center gap-1 rounded-lg border px-3 py-2 ${TONE[e.severity]}`}>
              <div className="flex items-center gap-2 text-[10px]">
                <span className={`rounded px-1.5 py-0.5 font-semibold uppercase tracking-wider ${e.kind === "anomaly" ? "bg-warning/20 text-foreground" : e.kind === "milestone" ? "bg-success/20 text-foreground" : "bg-muted text-muted-foreground"}`}>{BADGE[e.kind]}</span>
                <span className="font-mono tabular-nums">@ {e.atUpdates.toLocaleString()} updates</span>
              </div>
              <p className="line-clamp-2 text-[11px] leading-snug">{e.text}</p>
            </div>
          </div>
        )}
      />
    </div>
  );
}
