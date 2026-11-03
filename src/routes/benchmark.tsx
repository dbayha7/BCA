// Automated benchmarking suite: a fixed, seeded set of CQL runs on every
// built-in dataset, trained through the worker pool, summarised as a
// leaderboard and exportable as JSON / CSV.

import { createFileRoute, Link } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { useTheme } from "@/components/theme-provider";
import { WorkerCountSelect, WorkerList } from "@/components/lab/sweep-resources";
import {
  aggregateBenchmark,
  BENCH_ALPHAS,
  BENCH_DATASETS,
  BENCH_HYPER,
  BENCH_SEEDS,
  benchmarkCsv,
  benchmarkJson,
  buildSuite,
  runBenchmark,
  type BenchRow,
  type BenchRunResult,
} from "@/lib/benchmark";
import type { SweepDatasetId } from "@/lib/sweep-core";
import { defaultPoolSize, hardwareCores, SweepPool } from "@/lib/sweep-pool";

export const Route = createFileRoute("/benchmark")({ component: BenchmarkPage });

function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const dark = theme === "dark" || (theme === "system" && typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  return (
    <button
      onClick={() => setTheme(dark ? "light" : "dark")}
      className="flex h-8 w-8 items-center justify-center rounded-md border border-border bg-card text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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

function Stat({ label, value, hint, tone }: { label: string; value: string; hint?: string; tone?: "ok" }) {
  return (
    <div className="rounded-lg border border-border bg-surface/60 px-3 py-2" title={hint}>
      <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className={`mt-0.5 font-mono text-[15px] font-medium tabular-nums ${tone === "ok" ? "text-success" : "text-foreground"}`}>{value}</div>
    </div>
  );
}

const fmt = (v: number) => (!Number.isFinite(v) ? "—" : Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2));
const signed = (v: number) => (!Number.isFinite(v) ? "—" : Math.abs(v) < 0.005 ? "0.00" : `${v >= 0 ? "+" : "−"}${fmt(Math.abs(v))}`);

function download(name: string, text: string, type: string) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

type Phase = "idle" | "running" | "finished";

function BenchmarkPage() {
  const [cores] = useState(hardwareCores);
  const [poolSize, setPoolSize] = useState(defaultPoolSize);
  const [phase, setPhase] = useState<Phase>("idle");
  const [results, setResults] = useState<BenchRunResult[]>([]);
  const [vStars, setVStars] = useState<Partial<Record<SweepDatasetId, number>>>({});
  const [error, setError] = useState<string | null>(null);
  const [, setTick] = useState(0);
  const poolRef = useRef<SweepPool | null>(null);
  const cancelRef = useRef(false);
  const live = useRef(new Map<number, number>()); // case index → updates done
  const timing = useRef({ startedAt: 0, finishedAt: 0, workers: 0, startedIso: "" });
  /** Wall time of the last complete 1-worker run of the suite (for a measured speed-up). */
  const baselineMs = useRef<number | null>(null);
  const pendingFrame = useRef(false);

  const cases = useState(buildSuite)[0];
  const totalUpdates = cases.reduce((a, c) => a + c.updates, 0);

  const bump = useCallback(() => {
    if (pendingFrame.current) return;
    pendingFrame.current = true;
    requestAnimationFrame(() => {
      pendingFrame.current = false;
      setTick((n) => n + 1);
    });
  }, []);

  useEffect(
    () => () => {
      cancelRef.current = true;
      poolRef.current?.terminate();
      poolRef.current = null;
    },
    [],
  );

  // Keep the elapsed clock moving while running.
  useEffect(() => {
    if (phase !== "running") return;
    const id = window.setInterval(() => setTick((n) => n + 1), 250);
    return () => window.clearInterval(id);
  }, [phase]);

  const run = async () => {
    setError(null);
    if (!poolRef.current || poolRef.current.size !== poolSize) {
      poolRef.current?.terminate();
      poolRef.current = new SweepPool(poolSize);
      poolRef.current.onStatus = bump;
    }
    const pool = poolRef.current;
    pool.resetCounters();
    pool.labelFor = (t) => {
      const c = cases[t.index];
      return c ? `${c.datasetName} · α ${c.alpha} · seed ${c.seed}` : `Case ${t.index + 1}`;
    };
    cancelRef.current = false;
    live.current = new Map();
    const tm = timing.current;
    tm.startedAt = performance.now();
    tm.finishedAt = 0;
    tm.workers = pool.size;
    tm.startedIso = new Date().toISOString();
    setResults([]);
    setPhase("running");
    const collected: BenchRunResult[] = [];
    try {
      // V*(start) per dataset (each worker also builds and caches the dataset here).
      const vs = await Promise.all(BENCH_DATASETS.map((d, i) => pool.optimal({ id: d.id, seed: d.seed }, BENCH_HYPER.gamma, i % pool.size)));
      const vMap: Partial<Record<SweepDatasetId, number>> = {};
      BENCH_DATASETS.forEach((d, i) => (vMap[d.id] = vs[i]));
      setVStars(vMap);
      await runBenchmark(cases, pool, {
        isCancelled: () => cancelRef.current,
        onChunk: (c, m) => {
          live.current.set(c.index, m.updates);
          bump();
        },
        onResult: (r) => {
          collected.push(r);
          setResults([...collected].sort((a, b) => a.case.index - b.case.index));
        },
      });
    } catch (err) {
      if (!cancelRef.current) setError(err instanceof Error ? err.message : "Benchmark failed.");
    }
    tm.finishedAt = performance.now();
    if (pool.size === 1 && collected.length === cases.length) baselineMs.current = tm.finishedAt - tm.startedAt;
    setPhase("finished");
  };

  const running = phase === "running";
  const now = running ? performance.now() : timing.current.finishedAt;
  const elapsedMs = timing.current.startedAt ? now - timing.current.startedAt : 0;
  const rows: BenchRow[] = aggregateBenchmark(results, vStars);
  const cpuMs = results.reduce((a, r) => a + r.computeMs, 0);
  const doneUpdates = results.reduce((a, r) => a + r.case.updates, 0) + [...live.current.entries()].filter(([i]) => !results.some((r) => r.case.index === i)).reduce((a, [, u]) => a + u, 0);
  const busyWorkers = elapsedMs > 0 ? cpuMs / elapsedMs : 0;
  const complete = phase === "finished" && results.length === cases.length;
  const measured = complete && timing.current.workers > 1 && baselineMs.current != null && elapsedMs > 0 ? baselineMs.current / elapsedMs : null;
  const workers = poolRef.current?.status ?? [];

  const exportJson = () => {
    const tm = timing.current;
    const doc = benchmarkJson({ workers: tm.workers, startedAt: tm.startedIso, elapsedMs, cancelled: !complete }, rows, results, vStars);
    download(`offline-lab-benchmark-${tm.startedIso.slice(0, 19).replace(/[:T]/g, "-")}.json`, JSON.stringify(doc, null, 2), "application/json");
  };
  const exportCsv = () => {
    download(`offline-lab-benchmark-${timing.current.startedIso.slice(0, 19).replace(/[:T]/g, "-")}.csv`, benchmarkCsv(rows), "text/csv");
  };

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="sticky top-0 z-40 border-b border-border bg-background/85 backdrop-blur-md">
        <div className="flex h-14 items-center justify-between gap-4 px-4 lg:px-6">
          <div className="flex min-w-0 items-center gap-4">
            <Link
              to="/"
              className="flex shrink-0 items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              Back to lab
            </Link>
            <div className="min-w-0">
              <h1 className="font-display text-[15px] font-bold tracking-tight leading-none">Benchmark suite</h1>
              <p className="mt-0.5 hidden truncate text-[11px] text-muted-foreground sm:block">Fixed, seeded CQL runs on every built-in dataset — comparable across machines and worker counts</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Link
              to="/sweep"
              className="flex items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3" />
                <path d="M1 14h6M9 8h6M17 16h6" />
              </svg>
              Sweep
            </Link>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-7xl flex-1 grid-cols-1 gap-5 p-4 lg:grid-cols-[320px_minmax(0,1fr)] lg:p-6">
        <aside className="space-y-4">
          <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
            <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Suite (fixed)</h2>
            <dl className="space-y-2.5 text-[12px]">
              <div>
                <dt className="text-muted-foreground">Datasets · data seed</dt>
                <dd className="mt-1 space-y-0.5">
                  {BENCH_DATASETS.map((d) => (
                    <div key={d.id} className="flex justify-between gap-2">
                      <span className="text-foreground">{d.name}</span>
                      <span className="font-mono tabular-nums text-muted-foreground">{d.seed}</span>
                    </div>
                  ))}
                </dd>
              </div>
              {[
                ["Conservative weight α", BENCH_ALPHAS.join(", ")],
                ["Training seeds", BENCH_SEEDS.join(", ")],
                ["Learning rate · γ · batch", `${BENCH_HYPER.lr} · ${BENCH_HYPER.gamma} · ${BENCH_HYPER.batch}`],
                ["Updates per run", BENCH_HYPER.updates.toLocaleString()],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between gap-2">
                  <dt className="text-muted-foreground">{k}</dt>
                  <dd className="font-mono tabular-nums text-foreground">{v}</dd>
                </div>
              ))}
            </dl>
            <div className="mt-3 flex items-center justify-between rounded-md border border-border bg-muted px-3 py-2 text-[12px]">
              <span className="text-muted-foreground">Total</span>
              <span className="font-mono font-semibold tabular-nums text-foreground">
                {cases.length} runs · {(totalUpdates / 1000).toFixed(0)}k updates
              </span>
            </div>
          </div>

          <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
            <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Compute</h2>
            <label className="flex flex-col gap-1.5">
              <span className="text-[11px] font-medium text-muted-foreground">Parallel workers ({cores} logical cores detected)</span>
              <WorkerCountSelect value={poolSize} onChange={setPoolSize} cores={cores} disabled={running} />
            </label>
            <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">Results do not depend on the worker count — only the wall time does.</p>
          </div>

          {running ? (
            <button
              type="button"
              onClick={() => {
                cancelRef.current = true;
              }}
              className="w-full rounded-md bg-destructive px-5 py-2.5 text-sm font-semibold text-white shadow-md transition-all duration-200 hover:opacity-90 active:scale-95"
            >
              Cancel benchmark
            </button>
          ) : (
            <button
              type="button"
              onClick={() => void run()}
              className="w-full rounded-md bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground shadow-md transition-all duration-200 hover:bg-primary/90 hover:shadow-lg active:scale-95"
            >
              {phase === "finished" ? "Run benchmark again" : "Run benchmark"}
            </button>
          )}
          {error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3.5 py-2.5 text-xs text-destructive" role="alert">
              {error}
            </div>
          )}
        </aside>

        <section className="min-w-0 space-y-4">
          {phase !== "idle" && (
            <div className="rounded-xl border border-border bg-card shadow-sm">
              <header className="flex items-center justify-between gap-2 border-b border-border px-4 py-2.5">
                <h2 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Progress</h2>
                <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
                  {timing.current.workers} worker{timing.current.workers === 1 ? "" : "s"}
                </span>
              </header>
              <div className="space-y-4 p-4">
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
                  <Stat label="Runs" value={`${results.length} / ${cases.length}`} />
                  <Stat label="Elapsed" value={`${(elapsedMs / 1000).toFixed(1)} s`} hint="Wall-clock including dataset preparation and V* computation" />
                  <Stat label="CPU-seconds" value={(cpuMs / 1000).toFixed(2)} hint="Sum of worker training + evaluation time" />
                  <Stat label="Avg. busy workers" value={busyWorkers > 0 ? busyWorkers.toFixed(2) : "—"} hint="CPU-seconds ÷ elapsed; overstates the real speed-up when workers compete for cores" />
                  <Stat
                    label="Vs 1 worker"
                    value={timing.current.workers === 1 ? (complete ? "baseline" : "—") : measured != null ? `${measured.toFixed(2)}×` : "—"}
                    hint="Measured wall-clock speed-up against a complete 1-worker run of the suite in this session"
                    tone={measured != null && measured > 1.2 ? "ok" : undefined}
                  />
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-muted" role="progressbar" aria-valuemin={0} aria-valuemax={totalUpdates} aria-valuenow={doneUpdates} aria-label="Benchmark progress">
                  <div className="h-full rounded-full bg-primary transition-all duration-300" style={{ width: `${(doneUpdates / totalUpdates) * 100}%` }} />
                </div>
                <WorkerList workers={workers} wallMs={elapsedMs} />
              </div>
            </div>
          )}

          <div className="overflow-hidden rounded-xl border border-border bg-card shadow-sm">
            <header className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-2.5">
              <h2 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Leaderboard</h2>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  disabled={results.length === 0 || running}
                  onClick={exportJson}
                  className="rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground disabled:opacity-50"
                >
                  Export JSON
                </button>
                <button
                  type="button"
                  disabled={results.length === 0 || running}
                  onClick={exportCsv}
                  className="rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground disabled:opacity-50"
                >
                  Export CSV
                </button>
              </div>
            </header>
            {rows.length === 0 ? (
              <div className="flex flex-col items-center gap-3 px-6 py-16 text-center">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-muted-foreground" aria-hidden>
                  <path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0V4z" />
                  <path d="M17 6h3a3 3 0 0 1-3 4M7 6H4a3 3 0 0 0 3 4" />
                </svg>
                <div>
                  <p className="text-sm font-medium text-foreground">{running ? "Computing optimal values…" : "No benchmark results yet"}</p>
                  <p className="mt-1 max-w-sm text-xs leading-relaxed text-muted-foreground">
                    Press Run benchmark to train the fixed suite. Rows appear as configurations finish, ranked within each dataset by mean true return across seeds.
                  </p>
                </div>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[12px]" data-testid="bench-table">
                  <thead>
                    <tr className="border-b border-border bg-muted/60 text-[10px] uppercase tracking-wider text-muted-foreground">
                      <th className="px-3 py-2.5 font-semibold">Rank</th>
                      <th className="px-3 py-2.5 font-semibold normal-case">α</th>
                      <th className="px-3 py-2.5 text-right font-semibold">Seeds</th>
                      <th className="px-3 py-2.5 text-right font-semibold" title="Mean true value of the greedy policy over states seen in the data (± std across seeds)">
                        True return
                      </th>
                      <th className="px-3 py-2.5 text-right font-semibold" title="True value of the greedy policy at the start state">
                        <span className="normal-case">
                          V<sup>π</sup>(start)
                        </span>
                      </th>
                      <th className="px-3 py-2.5 text-right font-semibold" title="Optimal value at the start state">
                        <span className="normal-case">V*(start)</span>
                      </th>
                      <th className="px-3 py-2.5 text-right font-semibold" title="V*(start) − V^π(start)">
                        Gap to <span className="normal-case">V*</span>
                      </th>
                      <th className="px-3 py-2.5 text-right font-semibold" title="Mean estimated Q − true value; positive = overestimation">
                        Overest.
                      </th>
                      <th className="px-3 py-2.5 text-right font-semibold">OOD %</th>
                      <th className="px-3 py-2.5 text-right font-semibold" title="Worker compute time, summed over seeds">
                        CPU s
                      </th>
                      <th className="px-3 py-2.5 text-right font-semibold" title="Dispatch-to-result wall time, summed over seeds">
                        Wall s
                      </th>
                    </tr>
                  </thead>
                  {BENCH_DATASETS.map((d) => {
                    const group = rows.filter((r) => r.datasetId === d.id);
                    if (group.length === 0) return null;
                    return (
                      <tbody key={d.id} className="font-mono tabular-nums">
                        <tr className="border-b border-border bg-surface/60">
                          <td colSpan={11} className="px-3 py-2 font-sans text-[12px] font-semibold text-foreground">
                            {d.name}
                            <span className="ml-2 font-mono text-[11px] font-normal text-muted-foreground">V*(start) = {fmt(vStars[d.id] ?? NaN)}</span>
                          </td>
                        </tr>
                        {group.map((r) => {
                          const top = r.rank === 1 && r.n === BENCH_SEEDS.length;
                          return (
                            <tr key={`${r.datasetId}-${r.alpha}`} className={`border-b border-border/60 ${top ? "bg-primary/5" : ""}`} data-testid="bench-row">
                              <td className="px-3 py-2 text-muted-foreground">
                                {r.rank}
                                {top && <span className="ml-1.5 rounded bg-primary/15 px-1.5 py-0.5 font-sans text-[10px] font-semibold text-primary">best α</span>}
                              </td>
                              <td className="px-3 py-2">{r.alpha}</td>
                              <td className="px-3 py-2 text-right text-muted-foreground">
                                {r.n}/{BENCH_SEEDS.length}
                              </td>
                              <td className="px-3 py-2 text-right font-semibold text-foreground">
                                {fmt(r.trueQMean)}
                                <span className="ml-1 text-[10px] font-normal text-muted-foreground">±{fmt(r.trueQStd)}</span>
                              </td>
                              <td className="px-3 py-2 text-right">{fmt(r.vStartMean)}</td>
                              <td className="px-3 py-2 text-right text-muted-foreground">{fmt(r.vStar)}</td>
                              <td className={`px-3 py-2 text-right ${r.gapToOptimal < 0.05 * Math.max(1, Math.abs(r.vStar)) ? "text-success" : "text-foreground"}`}>{fmt(r.gapToOptimal)}</td>
                              <td className={`px-3 py-2 text-right ${r.overestimation > 1 ? "text-destructive" : r.overestimation < 0.3 ? "text-success" : "text-foreground"}`}>{signed(r.overestimation)}</td>
                              <td className="px-3 py-2 text-right">{(r.oodRate * 100).toFixed(1)}</td>
                              <td className="px-3 py-2 text-right text-muted-foreground">{(r.cpuMs / 1000).toFixed(2)}</td>
                              <td className="px-3 py-2 text-right text-muted-foreground">{(r.wallMs / 1000).toFixed(2)}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    );
                  })}
                </table>
              </div>
            )}
          </div>

          <p className="px-1 text-[11px] leading-relaxed text-muted-foreground">
            <strong className="text-foreground">True return</strong> is exact policy evaluation of the learned greedy policy averaged over states seen in the data (the same score the sweep ranks by); <strong className="text-foreground">V^π(start)</strong> and{" "}
            <strong className="text-foreground">V*(start)</strong> compare the policy with the optimum at the start state. ThrusterChain uses its empirical MDP as ground truth. Every run has a fixed seed, so the numbers are reproducible; CPU and wall times are not.
          </p>
        </section>
      </main>
    </div>
  );
}
