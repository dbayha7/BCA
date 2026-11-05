// Compute resource management for sweeps and benchmarks: worker-count selector,
// budget caps, and a live monitor (updates/s, elapsed, ETA, per-worker status,
// CPU-seconds summary, cancel).

import type { WorkerStatus } from "@/lib/sweep-pool";
import type { StopReason } from "@/lib/sweep-core";

export interface BudgetCaps {
  /** Max total gradient updates across the sweep (null = no cap). */
  maxUpdates: number | null;
  /** Max wall-clock seconds, including dataset preparation (null = no cap). */
  maxSeconds: number | null;
}

export interface MonitorSnapshot {
  phase: "idle" | "preparing" | "running" | "finished";
  /** Wall-clock since the run button was pressed. */
  elapsedMs: number;
  /** Wall-clock of the training phase only. */
  trainingMs: number;
  updatesDone: number;
  updatesPerSec: number;
  /** Upper bound on updates still to train (pruning can only shrink it). */
  remainingUpdates: number;
  cpuMs: number;
  setupMs: number;
  runsFinished: number;
  runsTotal: number;
  waiting: number;
  stopReason: StopReason | null;
  /** Training wall-clock of the same sweep on 1 worker, if it was measured this session. */
  baselineMs: number | null;
}

const secs = (ms: number) => (ms >= 100_000 ? `${Math.round(ms / 1000)} s` : `${(ms / 1000).toFixed(1)} s`);
const big = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e4 ? `${(n / 1e3).toFixed(1)}k` : Math.round(n).toLocaleString());

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-[11px] font-medium text-muted-foreground">{label}</span>
      {children}
    </label>
  );
}

const inputCls =
  "h-9 w-full rounded-md border border-input bg-card px-3 font-mono text-sm tabular-nums text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50";

export function WorkerCountSelect({ value, onChange, cores, disabled }: { value: number; onChange: (n: number) => void; cores: number; disabled?: boolean }) {
  const def = Math.max(1, Math.min(4, cores - 1));
  return (
    <select value={value} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} className={inputCls} aria-label="Parallel workers">
      {Array.from({ length: Math.max(1, cores) }, (_, i) => i + 1).map((n) => (
        <option key={n} value={n}>
          {n} worker{n === 1 ? "" : "s"}
          {n === def ? " (default)" : ""}
          {n === 1 ? " · sequential" : ""}
        </option>
      ))}
    </select>
  );
}

/** Inputs: pool size and budget caps. */
export function ComputeSettings({
  poolSize,
  onPoolSize,
  cores,
  budget,
  onBudget,
  plannedUpdates,
  updatesPerRun,
  disabled,
}: {
  poolSize: number;
  onPoolSize: (n: number) => void;
  cores: number;
  budget: BudgetCaps;
  onBudget: (b: BudgetCaps) => void;
  plannedUpdates: number;
  updatesPerRun: number;
  disabled: boolean;
}) {
  const capRuns = budget.maxUpdates != null ? Math.floor(budget.maxUpdates / Math.max(1, updatesPerRun)) : null;
  return (
    <div className="space-y-4">
      <Field label={`Parallel workers (${cores} logical cores detected)`}>
        <WorkerCountSelect value={poolSize} onChange={onPoolSize} cores={cores} disabled={disabled} />
      </Field>

      <fieldset className="space-y-3 rounded-lg border border-border bg-card/50 p-3.5">
        <legend className="px-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Budget caps</legend>
        <CapRow
          label="Max total updates"
          enabled={budget.maxUpdates != null}
          value={budget.maxUpdates ?? Math.max(updatesPerRun, Math.round(plannedUpdates / 2 / 100) * 100)}
          min={100}
          step={500}
          disabled={disabled}
          onChange={(v) => onBudget({ ...budget, maxUpdates: v })}
        />
        <CapRow
          label="Max wall-clock (s)"
          enabled={budget.maxSeconds != null}
          value={budget.maxSeconds ?? 10}
          min={1}
          step={1}
          disabled={disabled}
          onChange={(v) => onBudget({ ...budget, maxSeconds: v })}
        />
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          When a cap is hit no new configurations start and the rest are marked <em>skipped</em>; runs already started finish. A configuration only starts if its full {updatesPerRun.toLocaleString()} updates still fit under the update cap
          {capRuns != null ? ` (at least ${capRuns} of them fit)` : ""}, so the same runs are skipped for every worker count. The wall-clock cap counts dataset preparation too.
        </p>
      </fieldset>
      <div className="flex items-center justify-between rounded-md border border-border bg-muted px-3 py-2 text-[12px]">
        <span className="text-muted-foreground">Planned compute (worst case)</span>
        <span className="font-mono font-semibold tabular-nums text-foreground">{big(plannedUpdates)} updates</span>
      </div>
    </div>
  );
}

function CapRow({
  label,
  enabled,
  value,
  min,
  step,
  disabled,
  onChange,
}: {
  label: string;
  enabled: boolean;
  value: number;
  min: number;
  step: number;
  disabled: boolean;
  onChange: (v: number | null) => void;
}) {
  return (
    <div className="grid grid-cols-[auto_minmax(0,1fr)] items-center gap-2.5">
      <label className="flex items-center gap-2 text-[12px] text-foreground">
        <input type="checkbox" checked={enabled} disabled={disabled} onChange={(e) => onChange(e.target.checked ? value : null)} className="h-4 w-4 accent-[var(--primary)]" />
        {label}
      </label>
      <input
        type="number"
        aria-label={label}
        value={value}
        min={min}
        step={step}
        disabled={disabled || !enabled}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v) && v >= min) onChange(Math.round(v));
        }}
        className={`${inputCls} h-8`}
      />
    </div>
  );
}

function Stat({ label, value, hint, tone }: { label: string; value: string; hint?: string; tone?: "ok" | "warn" }) {
  const c = tone === "warn" ? "text-warning-foreground dark:text-warning" : tone === "ok" ? "text-success" : "text-foreground";
  return (
    <div className="rounded-lg border border-border bg-surface/60 px-3 py-2" title={hint}>
      <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className={`mt-0.5 font-mono text-[15px] font-medium tabular-nums ${c}`}>{value}</div>
    </div>
  );
}

function Meter({ label, used, cap, fmt }: { label: string; used: number; cap: number; fmt: (v: number) => string }) {
  const frac = Math.min(1, used / Math.max(1e-9, cap));
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between text-[11px]">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono tabular-nums text-foreground">
          {fmt(used)} / {fmt(cap)}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-muted" role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={cap} aria-valuenow={used}>
        <div className={`h-full rounded-full transition-all duration-300 ${frac >= 1 ? "bg-warning" : "bg-primary"}`} style={{ width: `${frac * 100}%` }} />
      </div>
    </div>
  );
}

const PHASE_DOT: Record<WorkerStatus["phase"], string> = {
  idle: "bg-muted-foreground/40",
  preparing: "bg-info animate-pulse",
  training: "bg-success animate-pulse",
  error: "bg-destructive",
};

export function WorkerList({ workers, wallMs }: { workers: WorkerStatus[]; wallMs: number }) {
  return (
    <ul className="divide-y divide-border/60 rounded-lg border border-border">
      {workers.map((w) => {
        const util = wallMs > 0 ? Math.min(1, w.busyMs / wallMs) : 0;
        return (
          <li key={w.index} className="grid grid-cols-[70px_minmax(0,1fr)_auto] items-center gap-3 px-3 py-2 text-[12px]" data-testid="worker-row">
            <span className="flex items-center gap-2 font-medium text-foreground">
              <span className={`h-2 w-2 shrink-0 rounded-full ${PHASE_DOT[w.phase]}`} aria-hidden />
              W{w.index + 1}
              {w.inline && <span className="rounded bg-muted px-1 text-[9px] font-semibold uppercase text-muted-foreground" title="No Web Worker support — running on the main thread">main</span>}
            </span>
            <span className="truncate text-muted-foreground" title={w.detail}>
              {w.phase === "idle" && w.tasks > 0 ? `Idle · ${w.tasks} segment${w.tasks === 1 ? "" : "s"} done` : w.detail}
            </span>
            <span className="flex items-center gap-2 font-mono text-[11px] tabular-nums text-muted-foreground" title="Busy time (training + evaluation) and utilisation of wall-clock">
              {(w.busyMs / 1000).toFixed(2)} s
              <span className="h-1.5 w-12 overflow-hidden rounded-full bg-muted">
                <span className="block h-full rounded-full bg-chart-2" style={{ width: `${util * 100}%` }} />
              </span>
            </span>
          </li>
        );
      })}
    </ul>
  );
}

const STOP_TEXT: Record<StopReason, string> = {
  "updates-budget": "Update budget reached — no further configurations were started.",
  "time-budget": "Wall-clock budget reached — no further configurations were started.",
  cancelled: "Cancelled — running configurations were stopped and the rest skipped.",
};

/** Live monitor: throughput, time, budget use, per-worker status and the compute summary. */
export function ResourceMonitor({
  snap,
  workers,
  poolSize,
  budget,
  onCancel,
  skipped,
}: {
  snap: MonitorSnapshot;
  workers: WorkerStatus[];
  poolSize: number;
  budget: BudgetCaps;
  onCancel: () => void;
  skipped: number;
}) {
  const busy = snap.phase === "preparing" || snap.phase === "running";
  const rate = snap.updatesPerSec;
  const eta = busy && rate > 0 ? (snap.remainingUpdates / rate) * 1000 : null;
  const busyWorkers = snap.trainingMs > 0 ? snap.cpuMs / snap.trainingMs : 0;
  const measured = snap.phase === "finished" && snap.baselineMs != null && snap.trainingMs > 0 ? snap.baselineMs / snap.trainingMs : null;
  const perRun = snap.runsFinished > 0 ? snap.cpuMs / snap.runsFinished : 0;

  return (
    <section className="rounded-xl border border-border bg-card shadow-sm" aria-label="Resource monitor">
      <header className="flex items-center justify-between gap-2 border-b border-border px-4 py-2.5">
        <h2 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Resource monitor</h2>
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
            {poolSize} worker{poolSize === 1 ? "" : "s"}
          </span>
          {busy && (
            <button
              type="button"
              onClick={onCancel}
              className="rounded border border-destructive/40 px-2.5 py-1 text-[11px] font-semibold text-destructive transition-colors duration-200 hover:bg-destructive/10"
            >
              Cancel
            </button>
          )}
        </div>
      </header>
      <div className="space-y-4 p-4">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Updates / s" value={rate > 0 ? big(rate) : "—"} hint="Gradient updates per second across all workers (last 2 s while running, average once finished)" />
          <Stat label="Elapsed" value={secs(snap.elapsedMs)} hint="Wall-clock since the run started, including dataset preparation" />
          <Stat label="ETA" value={eta != null ? `≤ ${secs(eta)}` : snap.phase === "finished" ? "done" : "—"} hint="Upper bound: assumes no further run is pruned" />
          <Stat label="Updates" value={big(snap.updatesDone)} hint="Gradient updates trained so far" />
        </div>

        {(budget.maxUpdates != null || budget.maxSeconds != null) && (
          <div className="grid gap-3 sm:grid-cols-2">
            {budget.maxUpdates != null && <Meter label="Update budget" used={snap.updatesDone} cap={budget.maxUpdates} fmt={big} />}
            {budget.maxSeconds != null && <Meter label="Wall-clock budget" used={snap.elapsedMs / 1000} cap={budget.maxSeconds} fmt={(v) => `${v.toFixed(1)} s`} />}
          </div>
        )}

        {snap.stopReason && (
          <div className="rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-[12px] text-warning-foreground dark:text-warning" role="status">
            {STOP_TEXT[snap.stopReason]}
            {skipped > 0 ? ` ${skipped} configuration${skipped === 1 ? "" : "s"} skipped.` : ""}
          </div>
        )}

        <div>
          <div className="mb-2 flex items-baseline justify-between">
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Workers</h3>
            {snap.waiting > 0 && busy && (
              <span className="text-[11px] text-muted-foreground" title="Runs that reached a pruning checkpoint and wait until every lower-indexed run has reached it too">
                {snap.waiting} run{snap.waiting === 1 ? "" : "s"} waiting at a pruning checkpoint
              </span>
            )}
          </div>
          <WorkerList workers={workers} wallMs={snap.trainingMs} />
        </div>

        <div>
          <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Compute summary</h3>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat label="CPU-seconds" value={(snap.cpuMs / 1000).toFixed(2)} hint="Sum over workers of time spent training and evaluating" />
            <Stat label="Per finished run" value={perRun > 0 ? `${(perRun / 1000).toFixed(3)} s` : "—"} hint="CPU-seconds divided by finished (done or pruned) runs; per-run values are in the results table" />
            <Stat
              label="Avg. busy workers"
              value={busyWorkers > 0 ? `${busyWorkers.toFixed(2)} / ${poolSize}` : "—"}
              hint="CPU-seconds ÷ training wall-clock. This overstates the real speed-up when workers compete for cores, because CPU time per run then rises."
            />
            <Stat
              label="Speed-up vs 1 worker"
              value={poolSize === 1 ? (snap.phase === "finished" ? "baseline" : "—") : measured != null ? `${measured.toFixed(2)}×` : "—"}
              hint={
                poolSize === 1
                  ? "This run is the 1-worker baseline for the same sweep at other worker counts."
                  : "Measured wall-clock ratio against a 1-worker run of the identical sweep in this session (run it with 1 worker first)."
              }
              tone={measured != null && measured > 1.2 ? "ok" : undefined}
            />
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">
            Dataset preparation took {(snap.setupMs / 1000).toFixed(2)} CPU-s across workers (each worker builds and caches its own copy). CPU time is measured inside each worker; with more workers than free cores it includes time the OS spent on other threads.
          </p>
        </div>
      </div>
    </section>
  );
}
