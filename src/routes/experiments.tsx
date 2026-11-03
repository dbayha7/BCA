// Experiments library: every saved run, searchable and filterable, with side-by-side comparison
// of any selection and per-run audit export. Everything is read from this browser's storage.

import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { ThemeToggle } from "@/components/lab/theme-toggle";
import { MultiLineChart, runStyle, type RunSeries } from "@/components/lab/multi-line-chart";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { buildLogEntries } from "@/components/lab/experiment-log";
import { experimentsStore, kv } from "@/lib/store";
import {
  allTags,
  DEFAULT_QUERY,
  filterExperiments,
  summarize,
  tailStability,
  type ExperimentQuery,
  type ExperimentRecord,
  type SortKey,
} from "@/lib/experiments";
import { auditMarkdown, buildAuditLog, downloadText, metricsCsv, slugify } from "@/lib/log-export";
import { recordError } from "@/lib/error-log";
import { UPDATES_PER_STEP, type Dataset } from "@/lib/rl";

export const Route = createFileRoute("/experiments")({ component: ExperimentsPage });

const MAX_COMPARE = 10;
const DATASETS: { id: ExperimentQuery["datasetId"]; label: string }[] = [
  { id: "all", label: "All datasets" },
  { id: "gridworld-random", label: "Gridworld-Random" },
  { id: "halfcheetah-medium", label: "HalfCheetah-Medium" },
  { id: "thrusterchain-stall", label: "ThrusterChain v4" },
  { id: "custom", label: "Custom uploads" },
];
const SORTS: { id: SortKey; label: string }[] = [
  { id: "updated", label: "Last updated" },
  { id: "created", label: "Created" },
  { id: "trueQ", label: "True return" },
  { id: "gap", label: "Overestimation" },
  { id: "updates", label: "Updates" },
  { id: "name", label: "Name" },
];

const fmt = (v: number | null | undefined, d = 2) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(d));
const control =
  "rounded-md border border-border bg-card px-2.5 py-1.5 text-[12px] text-foreground outline-none transition-colors focus:border-ring focus-visible:ring-2 focus-visible:ring-ring/40";

/** buildLogEntries only reads the dataset's name and size; a stub lets saved runs be narrated without reloading data. */
function datasetStub(rec: ExperimentRecord): Dataset {
  return { env: { name: rec.datasetName }, transitions: { length: rec.transitions ?? 0 } } as unknown as Dataset;
}

function exportRun(rec: ExperimentRecord, format: "json" | "csv" | "md") {
  const entries = buildLogEntries(rec.history, datasetStub(rec), rec.hyper, rec.optimal);
  const audit = buildAuditLog({ ...rec, transitions: rec.transitions ?? 0 }, entries, [], UPDATES_PER_STEP);
  const base = `audit-${slugify(rec.name)}-${summarize(rec).updates}u`;
  if (format === "json") downloadText(`${base}.json`, JSON.stringify(audit, null, 2), "application/json");
  else if (format === "csv") downloadText(`${base}.csv`, metricsCsv(rec.history), "text/csv");
  else downloadText(`${base}.md`, auditMarkdown(audit), "text/markdown");
}

function ExperimentsPage() {
  const [all, setAll] = useState<ExperimentRecord[] | null>(null);
  const [currentId, setCurrentId] = useState<string | undefined>();
  const [q, setQ] = useState<ExperimentQuery>(DEFAULT_QUERY);
  const [selected, setSelected] = useState<string[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    Promise.all([experimentsStore.all(), kv.get<string>("currentExperimentId")])
      .then(([recs, cur]) => {
        if (!alive) return;
        setAll(recs);
        setCurrentId(cur);
      })
      .catch((e) => {
        if (!alive) return;
        setAll([]);
        setLoadError(e instanceof Error ? e.message : "Local storage is unavailable.");
        recordError("storage", e, "loading the experiments library");
      });
    return () => {
      alive = false;
    };
  }, []);

  const rows = useMemo(() => (all ? filterExperiments(all, q) : []), [all, q]);
  const tags = useMemo(() => (all ? allTags(all) : []), [all]);
  const chosen = useMemo(() => selected.map((id) => all?.find((r) => r.id === id)).filter((r): r is ExperimentRecord => !!r), [selected, all]);
  const set = (patch: Partial<ExperimentQuery>) => setQ((cur) => ({ ...cur, ...patch }));
  const filtersActive = JSON.stringify({ ...q, sort: DEFAULT_QUERY.sort, descending: DEFAULT_QUERY.descending }) !== JSON.stringify({ ...DEFAULT_QUERY });

  const toggle = (id: string) =>
    setSelected((cur) => {
      if (cur.includes(id)) return cur.filter((x) => x !== id);
      if (cur.length >= MAX_COMPARE) {
        toast.message(`Compare up to ${MAX_COMPARE} runs at a time.`);
        return cur;
      }
      return [...cur, id];
    });

  const remove = async (ids: string[]) => {
    const names = ids.map((id) => all?.find((r) => r.id === id)?.name ?? id);
    if (!window.confirm(ids.length === 1 ? `Delete “${names[0]}”? This cannot be undone.` : `Delete ${ids.length} runs? This cannot be undone.`)) return;
    try {
      for (const id of ids) await experimentsStore.delete(id);
      if (currentId && ids.includes(currentId)) await kv.del("currentExperimentId");
      setAll((cur) => (cur ?? []).filter((r) => !ids.includes(r.id)));
      setSelected((cur) => cur.filter((id) => !ids.includes(id)));
      toast.success(ids.length === 1 ? "Run deleted" : `${ids.length} runs deleted`);
    } catch (e) {
      recordError("storage", e, "deleting runs");
      toast.error(e instanceof Error ? e.message : "Could not delete.");
    }
  };

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="sticky top-0 z-40 border-b border-border bg-background/85 backdrop-blur-md">
        <div className="flex h-14 items-center justify-between gap-4 px-4 lg:px-6">
          <div className="flex items-center gap-4">
            <Link
              to="/"
              className="flex items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              Back to lab
            </Link>
            <div>
              <h1 className="font-display text-[15px] font-bold leading-none tracking-tight">Experiments</h1>
              <p className="mt-0.5 text-[11px] text-muted-foreground">Every run you train is saved here in this browser · search, filter, compare, export</p>
            </div>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl flex-1 space-y-4 p-4 lg:p-6">
        <section className="rounded-xl border border-border bg-card p-4 shadow-sm">
          <div className="flex flex-wrap items-end gap-3">
            <label className="min-w-[220px] flex-1">
              <span className="text-[11px] text-muted-foreground">Search name, notes, tags, dataset</span>
              <input className={`${control} mt-1 w-full`} value={q.text} placeholder="e.g. baseline thruster" onChange={(e) => set({ text: e.target.value })} aria-label="Search experiments" />
            </label>
            <label>
              <span className="text-[11px] text-muted-foreground">Dataset</span>
              <select className={`${control} mt-1 block`} value={q.datasetId} onChange={(e) => set({ datasetId: e.target.value as ExperimentQuery["datasetId"] })}>
                {DATASETS.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span className="text-[11px] text-muted-foreground">Status</span>
              <select className={`${control} mt-1 block`} value={q.status} onChange={(e) => set({ status: e.target.value as ExperimentQuery["status"] })}>
                <option value="all">Any</option>
                <option value="paused">Paused</option>
                <option value="training">Training</option>
                <option value="diverged">Diverged</option>
              </select>
            </label>
            <label>
              <span className="text-[11px] text-muted-foreground">Tag</span>
              <select className={`${control} mt-1 block`} value={q.tag} onChange={(e) => set({ tag: e.target.value })}>
                <option value="all">Any</option>
                {tags.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </label>
            <div>
              <span className="text-[11px] text-muted-foreground">α range</span>
              <div className="mt-1 flex items-center gap-1">
                <input
                  className={`${control} w-16`}
                  type="number"
                  step={0.1}
                  min={0}
                  placeholder="min"
                  aria-label="Minimum alpha"
                  value={q.alphaMin ?? ""}
                  onChange={(e) => set({ alphaMin: e.target.value === "" ? null : Number(e.target.value) })}
                />
                <span className="text-muted-foreground">–</span>
                <input
                  className={`${control} w-16`}
                  type="number"
                  step={0.1}
                  min={0}
                  placeholder="max"
                  aria-label="Maximum alpha"
                  value={q.alphaMax ?? ""}
                  onChange={(e) => set({ alphaMax: e.target.value === "" ? null : Number(e.target.value) })}
                />
              </div>
            </div>
            <label>
              <span className="text-[11px] text-muted-foreground">Min updates</span>
              <input
                className={`${control} mt-1 block w-24`}
                type="number"
                min={0}
                step={100}
                value={q.minUpdates ?? ""}
                onChange={(e) => set({ minUpdates: e.target.value === "" ? null : Number(e.target.value) })}
              />
            </label>
            <div>
              <span className="text-[11px] text-muted-foreground">Sort</span>
              <div className="mt-1 flex gap-1">
                <select className={control} value={q.sort} onChange={(e) => set({ sort: e.target.value as SortKey })} aria-label="Sort by">
                  {SORTS.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.label}
                    </option>
                  ))}
                </select>
                <button type="button" className={control} onClick={() => set({ descending: !q.descending })} aria-label={q.descending ? "Sort descending" : "Sort ascending"}>
                  {q.descending ? "↓" : "↑"}
                </button>
              </div>
            </div>
            {filtersActive && (
              <button type="button" onClick={() => setQ({ ...DEFAULT_QUERY, sort: q.sort, descending: q.descending })} className="pb-1.5 text-[11px] text-muted-foreground hover:text-foreground">
                Clear filters
              </button>
            )}
          </div>
        </section>

        {chosen.length > 0 && <ComparePanel runs={chosen} onClear={() => setSelected([])} onDelete={() => void remove(chosen.map((r) => r.id))} />}

        <section className="rounded-xl border border-border bg-card shadow-sm">
          <header className="flex items-center justify-between gap-2 border-b border-border px-4 py-2.5">
            <h2 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              {all === null ? "Loading…" : `${rows.length} of ${all.length} run${all.length === 1 ? "" : "s"}`}
            </h2>
            <span className="text-[11px] text-muted-foreground">Tick runs to compare them (up to {MAX_COMPARE})</span>
          </header>
          {loadError && <p className="px-4 py-3 text-[12px] text-destructive">Could not read saved runs: {loadError}</p>}
          {all !== null && all.length === 0 && !loadError && (
            <div className="px-4 py-10 text-center text-[12.5px] text-muted-foreground">
              No saved runs yet. Train in the lab — every run saves itself here after its first step.
            </div>
          )}
          {all !== null && all.length > 0 && rows.length === 0 && <div className="px-4 py-10 text-center text-[12.5px] text-muted-foreground">No runs match these filters.</div>}
          <ul className="divide-y divide-border">
            {rows.map((r) => {
              const s = summarize(r);
              const st = tailStability(r.history);
              const isSel = selected.includes(r.id);
              return (
                <li key={r.id} className={`flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 ${isSel ? "bg-accent/40" : ""}`}>
                  <input type="checkbox" checked={isSel} onChange={() => toggle(r.id)} aria-label={`Compare ${r.name}`} className="h-4 w-4 accent-[var(--primary)]" />
                  <div className="min-w-[220px] flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[13px] font-medium">{r.name || "Untitled run"}</span>
                      {r.id === currentId && <span className="rounded bg-primary/15 px-1.5 py-0.5 text-[10px] font-medium text-primary">open in lab</span>}
                      {r.status === "diverged" && <span className="rounded bg-destructive/15 px-1.5 py-0.5 text-[10px] font-medium text-destructive">diverged</span>}
                      {r.tags.map((t) => (
                        <button key={t} type="button" onClick={() => set({ tag: t })} className="rounded border border-border bg-surface px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground hover:text-foreground">
                          {t}
                        </button>
                      ))}
                    </div>
                    <div className="mt-0.5 font-mono text-[11px] text-muted-foreground">
                      {r.datasetName} · α {r.hyper.alpha} · γ {r.hyper.gamma} · lr {r.hyper.lr} · batch {r.hyper.batch}
                      {r.hyperLog.length > 1 ? ` · ${r.hyperLog.length - 1} change${r.hyperLog.length > 2 ? "s" : ""}` : ""}
                    </div>
                    {r.notes && <p className="mt-1 line-clamp-2 text-[11.5px] text-muted-foreground">{r.notes}</p>}
                  </div>
                  <dl className="grid grid-cols-3 gap-x-4 gap-y-0.5 font-mono text-[11px] tabular-nums sm:grid-cols-6">
                    <Metric label="updates" value={s.updates.toLocaleString()} />
                    <Metric label="true return" value={fmt(s.trueQ)} />
                    <Metric label="gap" value={s.gap === null ? "—" : `${s.gap >= 0 ? "+" : ""}${s.gap.toFixed(2)}`} tone={s.gap !== null && s.gap > 1 ? "bad" : undefined} />
                    <Metric label="OOD" value={s.oodRate === null ? "—" : `${(s.oodRate * 100).toFixed(1)}%`} />
                    <Metric label="tail σ" value={fmt(st?.std)} />
                    <Metric label="anomalies" value={String(r.anomalies + r.guardEvents.length)} tone={r.guardEvents.length ? "bad" : undefined} />
                  </dl>
                  <div className="flex items-center gap-2">
                    <span className="hidden font-mono text-[10.5px] text-muted-foreground lg:inline">{new Date(r.updatedAt).toLocaleString()}</span>
                    <Link to="/" search={{ run: r.id }} className="rounded-md bg-primary px-2.5 py-1 text-[11px] font-semibold text-primary-foreground hover:bg-primary/90">
                      Open
                    </Link>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <button type="button" className="rounded-md border border-border bg-card px-2.5 py-1 text-[11px] font-medium text-muted-foreground hover:bg-surface hover:text-foreground" aria-label={`More actions for ${r.name}`}>
                          ⋯
                        </button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onSelect={() => exportRun(r, "json")}>Export audit (JSON)</DropdownMenuItem>
                        <DropdownMenuItem onSelect={() => exportRun(r, "csv")}>Export metrics (CSV)</DropdownMenuItem>
                        <DropdownMenuItem onSelect={() => exportRun(r, "md")}>Export report (Markdown)</DropdownMenuItem>
                        <DropdownMenuItem className="text-destructive" onSelect={() => void remove([r.id])}>
                          Delete…
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      </main>
    </div>
  );
}

function Metric({ label, value, tone }: { label: string; value: string; tone?: "bad" }) {
  return (
    <div>
      <dt className="text-[9.5px] uppercase tracking-wider text-muted-foreground">{label}</dt>
      <dd className={tone === "bad" ? "text-destructive" : "text-foreground"}>{value}</dd>
    </div>
  );
}

function ComparePanel({ runs, onClear, onDelete }: { runs: ExperimentRecord[]; onClear: () => void; onDelete: () => void }) {
  const series = (pick: (m: ExperimentRecord["history"][number]) => number): RunSeries[] =>
    runs.map((r, i) => ({ name: r.id, ...runStyle(i), xs: r.history.map((m) => m.updates), values: r.history.map(pick) }));
  const rows = runs.map((r, i) => ({ r, i, s: summarize(r), st: tailStability(r.history) }));
  // "Stable" = the smallest spread of true return over the tail, among runs whose tail mean is near the best.
  const withTail = rows.filter((x) => x.st);
  const bestMean = withTail.length ? Math.max(...withTail.map((x) => x.st!.mean)) : null;
  const near = bestMean === null ? [] : withTail.filter((x) => x.st!.mean >= bestMean - Math.max(0.25, Math.abs(bestMean) * 0.05));
  // Only meaningful when at least two runs are close to the best tail return.
  const stableId = near.length >= 2 ? near.reduce((a, b) => (b.st!.std < a.st!.std ? b : a)).r.id : null;

  return (
    <section className="rounded-xl border border-border bg-card shadow-sm">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-2.5">
        <h2 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Comparing {runs.length} run{runs.length === 1 ? "" : "s"}</h2>
        <div className="flex gap-2">
          <button type="button" onClick={onDelete} className="rounded-md border border-destructive/40 bg-card px-2.5 py-1 text-[11px] font-medium text-destructive hover:bg-destructive/10">
            Delete selected
          </button>
          <button type="button" onClick={onClear} className="rounded-md border border-border bg-card px-2.5 py-1 text-[11px] font-medium text-muted-foreground hover:bg-surface hover:text-foreground">
            Clear selection
          </button>
        </div>
      </header>
      <div className="grid grid-cols-1 gap-4 p-4 lg:grid-cols-2">
        {(
          [
            ["True return V^π", (m) => m.trueQ],
            ["Estimated Q", (m) => m.estQ],
            ["Overestimation (est − true)", (m) => m.estQ - m.trueQ],
            ["OOD action rate", (m) => m.oodRate],
          ] as [string, (m: ExperimentRecord["history"][number]) => number][]
        ).map(([title, pick]) => (
          <div key={title}>
            <div className="mb-1 text-[11px] font-medium text-muted-foreground">{title}</div>
            <MultiLineChart series={series(pick)} label={`${title} by update for the selected runs`} empty="Selected runs have fewer than two steps." />
          </div>
        ))}
      </div>
      <div className="overflow-x-auto border-t border-border">
        <table className="w-full min-w-[760px] text-left text-[12px]">
          <thead className="text-[10px] uppercase tracking-wider text-muted-foreground">
            <tr>
              <th className="px-4 py-2 font-medium">Run</th>
              <th className="px-2 py-2 text-right font-medium">Updates</th>
              <th className="px-2 py-2 text-right font-medium">True return</th>
              <th className="px-2 py-2 text-right font-medium">Best</th>
              <th className="px-2 py-2 text-right font-medium">Gap</th>
              <th className="px-2 py-2 text-right font-medium">OOD</th>
              <th className="px-2 py-2 text-right font-medium" title="Standard deviation of true return over the last 20% of steps">Tail σ</th>
              <th className="px-4 py-2 text-right font-medium" title="Largest drop from a running best over the last 20% of steps">Drawdown</th>
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {rows.map(({ r, i, s, st }) => {
              const style = runStyle(i);
              return (
                <tr key={r.id} className="border-t border-border">
                  <td className="px-4 py-2 font-sans">
                    <span className="mr-2 inline-block h-0.5 w-4 align-middle" style={{ background: style.color, borderTop: style.dashed ? `2px dashed ${style.color}` : undefined }} />
                    {r.name || "Untitled run"}
                    {r.id === stableId && rows.length > 1 && <span className="ml-2 rounded bg-success/15 px-1.5 py-0.5 text-[10px] font-medium text-success">most stable near the best</span>}
                  </td>
                  <td className="px-2 py-2 text-right">{s.updates.toLocaleString()}</td>
                  <td className="px-2 py-2 text-right">{fmt(s.trueQ)}</td>
                  <td className="px-2 py-2 text-right">{fmt(s.bestTrueQ)}</td>
                  <td className="px-2 py-2 text-right">{s.gap === null ? "—" : `${s.gap >= 0 ? "+" : ""}${s.gap.toFixed(2)}`}</td>
                  <td className="px-2 py-2 text-right">{s.oodRate === null ? "—" : `${(s.oodRate * 100).toFixed(1)}%`}</td>
                  <td className="px-2 py-2 text-right">{fmt(st?.std)}</td>
                  <td className="px-4 py-2 text-right">{fmt(st?.maxDrawdown)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="px-4 pb-3 pt-1 text-[11px] text-muted-foreground">
          Tail σ and drawdown use the last 20% of each run's steps. Runs on different datasets are not comparable in absolute return; compare them on the same dataset.
        </p>
      </div>
    </section>
  );
}
