// Hyperparameter sweep tool: define LR × alpha ranges, run configurations
// sequentially, rank by final evaluation score, prune poor runs early.

import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useRef, useState } from "react";
import { useTheme } from "@/components/theme-provider";
import { SweepConfig } from "@/components/lab/sweep-config";
import {
  buildRuns,
  DEFAULT_SWEEP,
  getSweepDataset,
  runSweep,
  type SweepRun,
  type SweepSettings,
  type SweepSignal,
} from "@/lib/sweep";

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

function SweepPage() {
  const [settings, setSettings] = useState<SweepSettings>({ ...DEFAULT_SWEEP });
  const [runs, setRuns] = useState<SweepRun[]>([]);
  const [phase, setPhase] = useState<"idle" | "preparing" | "running" | "finished">("idle");
  const [prepLabel, setPrepLabel] = useState("");
  const signalRef = useRef<SweepSignal>({ cancelled: false, stopRequested: new Set() });
  const [, bump] = useState(0);

  const busy = phase === "preparing" || phase === "running";

  const start = async () => {
    const sig: SweepSignal = { cancelled: false, stopRequested: new Set() };
    signalRef.current = sig;
    const newRuns = buildRuns(settings);
    setRuns(newRuns);
    setPhase("preparing");
    setPrepLabel("Preparing dataset…");
    try {
      const ds = await getSweepDataset(settings.datasetId, 7, (pct, label) => setPrepLabel(`${label} (${pct}%)`));
      if (sig.cancelled) {
        setPhase("idle");
        return;
      }
      setPhase("running");
      await runSweep(settings, ds, newRuns, () => bump((n) => n + 1), sig);
      setPhase("finished");
    } catch (err) {
      setPrepLabel(err instanceof Error ? err.message : "Sweep failed.");
      setPhase("idle");
    }
  };

  const cancelAll = () => {
    signalRef.current.cancelled = true;
  };
  const stopRun = (id: number) => {
    signalRef.current.stopRequested.add(id);
  };

  const ranked = useMemo(() => {
    const started = runs.filter((r) => r.updatesDone > 0);
    return [...started].sort((a, b) => b.trueQ - a.trueQ);
  }, [runs]);

  const bestId = ranked.length > 0 ? ranked[0].id : -1;
  const doneCount = runs.filter((r) => r.status === "done").length;
  const prunedCount = runs.filter((r) => r.status === "pruned").length;

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="sticky top-0 z-40 border-b border-border bg-background/85 backdrop-blur-md">
        <div className="flex h-14 items-center justify-between gap-4 px-4 lg:px-6">
          <div className="flex items-center gap-4">
            <Link
              to="/"
              className="flex items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors duration-200 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              Back to lab
            </Link>
            <div>
              <h1 className="font-display text-[15px] font-bold tracking-tight leading-none">Hyperparameter sweep</h1>
              <p className="mt-0.5 text-[11px] text-muted-foreground">Sequential grid over learning rate × conservative weight, ranked by true return</p>
            </div>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-7xl flex-1 grid-cols-1 gap-5 p-4 lg:grid-cols-[340px_minmax(0,1fr)] lg:p-6">
        {/* Config column */}
        <aside className="space-y-4">
          <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
            <h2 className="mb-4 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Sweep configuration</h2>
            <SweepConfig settings={settings} onChange={setSettings} disabled={busy} />
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
        </aside>

        {/* Results column */}
        <section className="space-y-4">
          {runs.length > 0 && (
            <div className="flex flex-wrap items-center gap-3 rounded-xl border border-border bg-card px-4 py-3 text-[12px] shadow-sm">
              <span className="font-semibold text-foreground">{runs.length} configurations</span>
              <span className="text-border">|</span>
              <span className="text-success">{doneCount} completed</span>
              <span className="text-border">|</span>
              <span className="text-warning-foreground dark:text-warning">{prunedCount} pruned early</span>
              {phase === "running" && (
                <span className="ml-auto flex items-center gap-2 text-muted-foreground">
                  <span className="h-3 w-3 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                  Training sequentially…
                </span>
              )}
              {phase === "finished" && bestId >= 0 && (
                <span className="ml-auto font-mono tabular-nums text-foreground">
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
                    Set your learning-rate and conservative-weight ranges on the left, then press Run sweep. Configurations train one after another and appear here ranked by final true return.
                  </p>
                </div>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[12px]">
                  <thead>
                    <tr className="border-b border-border bg-muted/60 text-[10px] uppercase tracking-wider text-muted-foreground">
                      <th className="px-3.5 py-2.5 font-semibold">Rank</th>
                      <th className="px-3.5 py-2.5 font-semibold">LR</th>
                      <th className="px-3.5 py-2.5 font-semibold">α</th>
                      <th className="px-3.5 py-2.5 font-semibold">Status</th>
                      <th className="px-3.5 py-2.5 font-semibold">Progress</th>
                      <th className="px-3.5 py-2.5 text-right font-semibold">True return ↓</th>
                      <th className="px-3.5 py-2.5 text-right font-semibold">Est. Q</th>
                      <th className="px-3.5 py-2.5 text-right font-semibold">Overest. gap</th>
                      <th className="px-3.5 py-2.5 text-right font-semibold">OOD %</th>
                      <th className="px-3.5 py-2.5 text-right font-semibold">Loss</th>
                      <th className="px-3.5 py-2.5" />
                    </tr>
                  </thead>
                  <tbody className="font-mono tabular-nums">
                    {ranked.map((r, i) => {
                      const st = STATUS_STYLE[r.status];
                      return (
                        <tr
                          key={r.id}
                          className={`border-b border-border/60 last:border-0 ${r.id === bestId && phase === "finished" ? "bg-primary/5" : ""}`}
                        >
                          <td className="px-3.5 py-2.5 text-muted-foreground">
                            {i + 1}
                            {r.id === bestId && phase === "finished" && <span className="ml-1.5 rounded bg-primary/15 px-1.5 py-0.5 font-sans text-[10px] font-semibold text-primary">best</span>}
                          </td>
                          <td className="px-3.5 py-2.5">{r.lr}</td>
                          <td className="px-3.5 py-2.5">{r.alpha}</td>
                          <td className="px-3.5 py-2.5">
                            <span className={`rounded px-2 py-0.5 font-sans text-[10px] font-semibold ${st.cls}`}>{st.label}</span>
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
            Runs are ranked by the <strong className="text-foreground">true return</strong> of the learned greedy policy (computed with exact policy evaluation on the environment), not the agent's own estimate — so inflated Q-values can't win the ranking. Each run uses the same fixed seed, and a small overestimation gap with a low OOD rate marks a trustworthy configuration.
          </p>
        </section>
      </main>
    </div>
  );
}
