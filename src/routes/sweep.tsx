// Hyperparameter sweep tool: define LR × alpha ranges, train configurations in a
// pool of Web Workers (or sequentially with 1 worker — same results), rank by
// final true return, prune poor runs early, stay inside compute budgets, and
// explain what drives good runs.

import { createFileRoute, Link } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { useTheme } from "@/components/theme-provider";
import { SweepConfig } from "@/components/lab/sweep-config";
import { ComputeSettings, ResourceMonitor, type BudgetCaps, type MonitorSnapshot } from "@/components/lab/sweep-resources";
import { SweepSensitivity } from "@/components/lab/sweep-sensitivity";
import { buildRuns, DEFAULT_SWEEP, driveSweep, SweepScheduler, type SweepRun, type SweepSettings } from "@/lib/sweep";
import { defaultPoolSize, hardwareCores, SweepPool } from "@/lib/sweep-pool";

export const Route = createFileRoute("/sweep")({ component: SweepPage });

function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const dark = theme === "dark" || (theme === "system" && typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  return (
    <button
      onClick={() => setTheme(dark ? "light" : "dark")}
      className="flex h-8 w-8 items-center justify-center rounded-md border border-border bg-card text-muted-foreground transition-colors duration-200 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {dark ? (
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
        </svg>
      ) : (
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
        </svg>
      )}
    </button>
  );
}

const STATUS_STYLE: Record<SweepRun["status"], { label: string; cls: string }> = {
  pending: { label: "Pending", cls: "bg-muted text-muted-foreground" },
  running: { label: "Running", cls: "bg-info/15 text-info" },
  done: { label: "Done", cls: "bg-success/15 text-success" },
  pruned: { label: "Pruned early", cls: "bg-warning/20 text-warning-foreground dark:text-warning" },
  stopped: { label: "Stopped", cls: "bg-destructive/15 text-destructive" },
  skipped: { label: "Skipped", cls: "bg-muted text-muted-foreground" },
};

const f2 = (v: number) => v.toFixed(2);

/** Re-render at most once per animation frame, however many worker messages arrive. */
function useFrameBump() {
  const [version, setVersion] = useState(0);
  const pending = useRef(false);
  const bump = useCallback(() => {
    if (pending.current) return;
    pending.current = true;
    requestAnimationFrame(() => {
      pending.current = false;
      setVersion((n) => n + 1);
    });
  }, []);
  return [version, bump] as const;
}

const EMPTY_SNAP: MonitorSnapshot = {
  phase: "idle",
  elapsedMs: 0,
  trainingMs: 0,
  updatesDone: 0,
  updatesPerSec: 0,
  remainingUpdates: 0,
  cpuMs: 0,
  setupMs: 0,
  runsFinished: 0,
  runsTotal: 0,
  waiting: 0,
  stopReason: null,
  baselineMs: null,
};

type Phase = "idle" | "preparing" | "running" | "finished";

function SweepPage() {
  const [settings, setSettings] = useState<SweepSettings>({ ...DEFAULT_SWEEP });
  const [runs, setRuns] = useState<SweepRun[]>([]);
  const [phase, setPhase] = useState<Phase>("idle");
  const [prepLabel, setPrepLabel] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [cores] = useState(hardwareCores);
  const [poolSize, setPoolSize] = useState(defaultPoolSize);
  const [budget, setBudget] = useState<BudgetCaps>({ maxUpdates: null, maxSeconds: null });
  const [snap, setSnap] = useState<MonitorSnapshot>(EMPTY_SNAP);
  const [version, bump] = useFrameBump();

  const poolRef = useRef<SweepPool | null>(null);
  const schRef = useRef<SweepScheduler | null>(null);
  const phaseRef = useRef<Phase>("idle");
  const timing = useRef({ startedAt: 0, trainStartedAt: 0, finishedAt: 0, samples: [] as { t: number; u: number }[] });
  // 1-worker training times per identical sweep (settings + update cap), to report a measured speed-up.
  const baselines = useRef(new Map<string, number>());
  const sweepKey = useRef("");

  const busy = phase === "preparing" || phase === "running";

  const setPhaseBoth = (p: Phase) => {
    phaseRef.current = p;
    setPhase(p);
  };

  // Shut the workers down when leaving the page.
  useEffect(
    () => () => {
      schRef.current?.cancel();
      poolRef.current?.terminate();
      poolRef.current = null;
    },
    [],
  );

  const computeSnap = useCallback((): MonitorSnapshot => {
    const sch = schRef.current;
    const pool = poolRef.current;
    const tm = timing.current;
    const p = phaseRef.current;
    if (!sch) return EMPTY_SNAP;
    const now = p === "finished" ? tm.finishedAt : performance.now();
    const trainingMs = tm.trainStartedAt ? now - tm.trainStartedAt : 0;
    let rate = 0;
    if (p === "running") {
      const s = tm.samples;
      if (s.length >= 2) rate = ((s[s.length - 1].u - s[0].u) / Math.max(1, s[s.length - 1].t - s[0].t)) * 1000;
    } else if (trainingMs > 0) {
      rate = (sch.updatesDone / trainingMs) * 1000;
    }
    return {
      phase: p,
      elapsedMs: tm.startedAt ? now - tm.startedAt : 0,
      trainingMs,
      updatesDone: sch.updatesDone,
      updatesPerSec: rate,
      remainingUpdates: sch.plannedRemaining(),
      cpuMs: sch.computeMs,
      setupMs: pool ? pool.status.reduce((a, w) => a + w.setupMs, 0) : 0,
      runsFinished: sch.runs.filter((r) => r.status === "done" || r.status === "pruned").length,
      runsTotal: sch.runs.length,
      waiting: sch.waiting,
      stopReason: sch.stopReason,
      baselineMs: baselines.current.get(sweepKey.current) ?? null,
    };
  }, []);

  // While busy: sample throughput, enforce the wall-clock cap, refresh the monitor.
  useEffect(() => {
    if (!busy) return;
    const id = window.setInterval(() => {
      const sch = schRef.current;
      const tm = timing.current;
      if (!sch) return;
      if (phaseRef.current === "running") {
        tm.samples.push({ t: performance.now(), u: sch.updatesDone });
        while (tm.samples.length > 2 && tm.samples[tm.samples.length - 1].t - tm.samples[0].t > 2000) tm.samples.shift();
      }
      setSnap(computeSnap());
    }, 250);
    return () => window.clearInterval(id);
  }, [busy, computeSnap]);

  const ensurePool = (size: number) => {
    if (poolRef.current && poolRef.current.size === size) return poolRef.current;
    poolRef.current?.terminate();
    const pool = new SweepPool(size);
    pool.onStatus = bump;
    poolRef.current = pool;
    return pool;
  };

  const start = async () => {
    setError(null);
    const pool = ensurePool(poolSize);
    pool.resetCounters();
    const newRuns = buildRuns(settings);
    pool.labelFor = (t) => {
      const r = newRuns[t.index];
      return r ? `Run #${r.id + 1} · lr ${r.lr}, α ${r.alpha}` : `Run #${t.index + 1}`;
    };
    const sch = new SweepScheduler(settings, { runs: newRuns, maxActive: poolSize * 2, maxTotalUpdates: budget.maxUpdates });
    schRef.current = sch;
    const caps = { ...budget };
    sweepKey.current = JSON.stringify({ settings, maxUpdates: caps.maxUpdates });
    const tm = timing.current;
    tm.startedAt = performance.now();
    tm.trainStartedAt = 0;
    tm.finishedAt = 0;
    tm.samples = [];
    const checkTime = () => {
      if (caps.maxSeconds != null && (performance.now() - tm.startedAt) / 1000 >= caps.maxSeconds) sch.stopAdmitting("time-budget");
    };
    setRuns(newRuns);
    setPhaseBoth("preparing");
    setPrepLabel(`Building the dataset in ${poolSize} worker${poolSize === 1 ? "" : "s"}…`);
    try {
      await pool.prepare(sch.dataset, (pct, label) => setPrepLabel(`${label} (${pct}%)`));
      if (!sch.cancelled) {
        checkTime();
        tm.trainStartedAt = performance.now();
        setPhaseBoth("running");
        await driveSweep(sch, pool, { onUpdate: bump, tick: checkTime });
      }
    } catch (err) {
      sch.cancel();
      setError(err instanceof Error ? err.message : "Sweep failed.");
    }
    if (schRef.current !== sch) return; // a newer sweep took over
    tm.finishedAt = performance.now();
    if (!tm.trainStartedAt) tm.trainStartedAt = tm.finishedAt;
    if (sch.errors.length > 0) setError(`${sch.errors.length} run${sch.errors.length === 1 ? "" : "s"} failed: ${sch.errors[0]}`);
    else if (pool.size === 1 && sch.stopReason == null) baselines.current.set(sweepKey.current, tm.finishedAt - tm.trainStartedAt);
    setPhaseBoth("finished");
    setSnap(computeSnap());
    bump();
  };

  const cancelAll = () => {
    schRef.current?.cancel();
    bump();
  };
  const stopRun = (index: number) => {
    schRef.current?.stopRun(index);
    bump();
  };

  // Runs are mutated in place by the scheduler; `version` re-renders this list.
  const ranked = runs.filter((r) => r.updatesDone > 0).sort((a, b) => b.trueQ - a.trueQ || a.id - b.id);
  const bestId = ranked.length > 0 ? ranked[0].id : -1;
  const doneCount = runs.filter((r) => r.status === "done").length;
  const prunedCount = runs.filter((r) => r.status === "pruned").length;
  const skippedCount = runs.filter((r) => r.status === "skipped").length;
  const planned = settings.lrSteps * settings.alphaSteps * settings.totalUpdates;
  const workers = poolRef.current?.status ?? [];
  const activePool = poolRef.current?.size ?? poolSize;

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="sticky top-0 z-40 border-b border-border bg-background/85 backdrop-blur-md">
        <div className="flex h-14 items-center justify-between gap-4 px-4 lg:px-6">
          <div className="flex min-w-0 items-center gap-4">
            <Link
              to="/"
              className="flex shrink-0 items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors duration-200 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              Back to lab
            </Link>
            <div className="min-w-0">
              <h1 className="font-display text-[15px] font-bold tracking-tight leading-none">Hyperparameter sweep</h1>
              <p className="mt-0.5 hidden truncate text-[11px] text-muted-foreground sm:block">Parallel grid over learning rate × conservative weight, ranked by true return</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Link
              to="/benchmark"
              className="flex items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0V4z" />
                <path d="M17 6h3a3 3 0 0 1-3 4M7 6H4a3 3 0 0 0 3 4" />
              </svg>
              Benchmark
            </Link>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-7xl flex-1 grid-cols-1 gap-5 p-4 lg:grid-cols-[340px_minmax(0,1fr)] lg:p-6">
        {/* Config column */}
        <aside className="space-y-4">
          <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
            <h2 className="mb-4 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Sweep configuration</h2>
            <SweepConfig settings={settings} onChange={setSettings} disabled={busy} />
          </div>
          <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
            <h2 className="mb-4 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Compute resources</h2>
            <ComputeSettings
              poolSize={poolSize}
              onPoolSize={setPoolSize}
              cores={cores}
              budget={budget}
              onBudget={setBudget}
              plannedUpdates={planned}
              updatesPerRun={settings.totalUpdates}
              disabled={busy}
            />
          </div>
          {busy ? (
            <button
              type="button"
              onClick={cancelAll}
              className="w-full rounded-md bg-destructive px-5 py-2.5 text-sm font-semibold text-white shadow-md transition-all duration-200 hover:opacity-90 active:scale-95"
            >
              Stop sweep
            </button>
          ) : (
            <button
              type="button"
              onClick={() => void start()}
              className="w-full rounded-md bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground shadow-md transition-all duration-200 hover:bg-primary/90 hover:shadow-lg active:scale-95"
            >
              {phase === "finished" ? "Run sweep again" : "Run sweep"}
            </button>
          )}
          {phase === "preparing" && (
            <div className="flex items-center gap-2.5 rounded-md border border-border bg-card px-3.5 py-2.5 text-xs text-muted-foreground" role="status">
              <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              {prepLabel}
            </div>
          )}
          {error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3.5 py-2.5 text-xs text-destructive" role="alert">
              {error}
            </div>
          )}
        </aside>

        {/* Results column */}
        <section className="min-w-0 space-y-4">
          {runs.length > 0 && <ResourceMonitor snap={snap} workers={workers} poolSize={activePool} budget={budget} onCancel={cancelAll} skipped={skippedCount} />}

          {runs.length > 0 && (
            <div className="flex flex-wrap items-center gap-3 rounded-xl border border-border bg-card px-4 py-3 text-[12px] shadow-sm">
              <span className="font-semibold text-foreground">{runs.length} configurations</span>
              <span className="text-border">|</span>
              <span className="text-success">{doneCount} completed</span>
              <span className="text-border">|</span>
              <span className="text-warning-foreground dark:text-warning">{prunedCount} pruned early</span>
              {skippedCount > 0 && (
                <>
                  <span className="text-border">|</span>
                  <span className="text-muted-foreground">{skippedCount} skipped</span>
                </>
              )}
              {phase === "running" && (
                <span className="ml-auto flex items-center gap-2 text-muted-foreground">
                  <span className="h-3 w-3 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                  {activePool === 1 ? "Training sequentially…" : `Training on ${activePool} workers…`}
                </span>
              )}
              {phase === "finished" && bestId >= 0 && (
                <span className="ml-auto font-mono tabular-nums text-foreground" data-testid="sweep-best">
                  Best: lr {f2(ranked[0].lr)} · α {f2(ranked[0].alpha)} → true return {f2(ranked[0].trueQ)}
                </span>
              )}
            </div>
          )}

          <div className="overflow-hidden rounded-xl border border-border bg-card shadow-sm">
            {ranked.length === 0 ? (
              <div className="flex flex-col items-center gap-3 px-6 py-16 text-center">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-muted-foreground">
                  <path d="M3 3v18h18" strokeLinecap="round" />
                  <path d="M7 14l4-4 3 3 5-6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <div>
                  <p className="text-sm font-medium text-foreground">No sweep results yet</p>
                  <p className="mt-1 max-w-sm text-xs leading-relaxed text-muted-foreground">
                    Set your learning-rate and conservative-weight ranges on the left, choose how many workers to use, then press Run sweep. Configurations train in parallel and appear here ranked by final true return.
                  </p>
                </div>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[12px]" data-testid="sweep-table">
                  <thead>
                    <tr className="border-b border-border bg-muted/60 text-[10px] uppercase tracking-wider text-muted-foreground">
                      <th className="px-3.5 py-2.5 font-semibold">Rank</th>
                      <th className="px-3.5 py-2.5 font-semibold">Run</th>
                      <th className="px-3.5 py-2.5 font-semibold">LR</th>
                      <th className="px-3.5 py-2.5 font-semibold normal-case">α</th>
                      <th className="px-3.5 py-2.5 font-semibold">Status</th>
                      <th className="px-3.5 py-2.5 font-semibold">Progress</th>
                      <th className="px-3.5 py-2.5 text-right font-semibold">True return ↓</th>
                      <th className="px-3.5 py-2.5 text-right font-semibold">Est. Q</th>
                      <th className="px-3.5 py-2.5 text-right font-semibold">Overest. gap</th>
                      <th className="px-3.5 py-2.5 text-right font-semibold">OOD %</th>
                      <th className="px-3.5 py-2.5 text-right font-semibold">Loss</th>
                      <th className="px-3.5 py-2.5 text-right font-semibold" title="Worker compute time spent on this run">
                        CPU s
                      </th>
                      <th className="px-3.5 py-2.5" />
                    </tr>
                  </thead>
                  <tbody className="font-mono tabular-nums">
                    {ranked.map((r, i) => {
                      const st = STATUS_STYLE[r.status];
                      const isBest = r.id === bestId && phase === "finished";
                      return (
                        <tr key={r.id} className={`border-b border-border/60 last:border-0 ${isBest ? "bg-primary/5" : ""}`} data-testid="sweep-row">
                          <td className="px-3.5 py-2.5 text-muted-foreground">
                            {i + 1}
                            {isBest && <span className="ml-1.5 rounded bg-primary/15 px-1.5 py-0.5 font-sans text-[10px] font-semibold text-primary">best</span>}
                          </td>
                          <td className="px-3.5 py-2.5 text-muted-foreground" title={`RNG seed ${r.seed}`}>
                            #{r.id + 1}
                          </td>
                          <td className="px-3.5 py-2.5">{r.lr}</td>
                          <td className="px-3.5 py-2.5">{r.alpha}</td>
                          <td className="px-3.5 py-2.5">
                            <span
                              className={`whitespace-nowrap rounded px-2 py-0.5 font-sans text-[10px] font-semibold ${st.cls}`}
                              title={r.status === "pruned" ? `Pruned at ${r.updatesDone.toLocaleString()} updates (checkpoint ${r.prunedAt})` : r.error}
                            >
                              {st.label}
                            </span>
                          </td>
                          <td className="px-3.5 py-2.5">
                            <div className="flex items-center gap-2">
                              <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
                                <div
                                  className={`h-full rounded-full transition-all duration-300 ${r.status === "pruned" || r.status === "stopped" ? "bg-warning" : "bg-primary"}`}
                                  style={{ width: `${Math.round(r.progress * 100)}%` }}
                                />
                              </div>
                              <span className="text-[10px] text-muted-foreground">{Math.round(r.progress * 100)}%</span>
                            </div>
                          </td>
                          <td className="px-3.5 py-2.5 text-right font-semibold text-foreground">{f2(r.trueQ)}</td>
                          <td className="px-3.5 py-2.5 text-right">{f2(r.estQ)}</td>
                          <td className={`px-3.5 py-2.5 text-right ${r.gap > 1 ? "text-destructive" : r.gap < 0.3 ? "text-success" : "text-foreground"}`}>
                            {r.gap >= 0 ? "+" : ""}
                            {f2(r.gap)}
                          </td>
                          <td className="px-3.5 py-2.5 text-right">{(r.oodRate * 100).toFixed(1)}</td>
                          <td className="px-3.5 py-2.5 text-right">{f2(r.loss)}</td>
                          <td className="px-3.5 py-2.5 text-right text-muted-foreground">{(r.computeMs / 1000).toFixed(3)}</td>
                          <td className="px-3.5 py-2.5 text-right">
                            {r.status === "running" && (
                              <button
                                type="button"
                                onClick={() => stopRun(r.id)}
                                className="rounded border border-destructive/40 px-2.5 py-1 font-sans text-[10px] font-semibold text-destructive transition-colors duration-200 hover:bg-destructive/10"
                              >
                                Stop
                              </button>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <p className="px-1 text-[11px] leading-relaxed text-muted-foreground">
            Runs are ranked by the <strong className="text-foreground">true return</strong> of the learned greedy policy (computed with exact policy evaluation on the environment), not the agent's own estimate — so inflated Q-values can't win the ranking. Every configuration has its own seed derived from the sweep seed and its run number, and early pruning compares each run only with lower-numbered runs at the same checkpoint, so the table is identical whether you train on 1 worker or many. A small overestimation gap with a low OOD rate marks a trustworthy configuration.
          </p>

          {phase === "finished" && <SweepSensitivity runs={runs} version={version} />}
        </section>
      </main>
    </div>
  );
}
