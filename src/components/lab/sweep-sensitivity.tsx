// Win drivers: automated sensitivity analysis of a sweep — plain-language
// summary, lr × α heatmap of true return, main effects per parameter level
// (with overestimation gap and OOD rate), and Spearman rank correlations.

import { useMemo } from "react";
import { heatScale } from "@/components/lab/grid";
import { analyzeSweep, strength, type Correlations, type Insight, type ParamSensitivity, type SensitivityReport, type SensitivityRun } from "@/lib/sensitivity";

const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
const fmtRho = (r: number | null) => (r == null ? "n/a" : `${r >= 0 ? "+" : "−"}${Math.abs(r).toFixed(2)}`);

const TONE: Record<Insight["tone"], string> = {
  good: "bg-success",
  warn: "bg-warning",
  info: "bg-info",
};

function Summary({ items }: { items: Insight[] }) {
  return (
    <ul className="space-y-2" data-testid="win-driver-summary">
      {items.map((s, i) => (
        <li key={i} className="flex gap-2.5 text-[13px] leading-relaxed text-foreground">
          <span className={`mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full ${TONE[s.tone]}`} aria-hidden />
          <span>{s.text}</span>
        </li>
      ))}
    </ul>
  );
}

function HeatmapView({ rep }: { rep: SensitivityReport }) {
  const { lrs, alphas, cells, min, max } = rep.heatmap;
  const span = max - min || 1;
  return (
    <div>
      <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
        True return · <span className="normal-case">lr × α</span>
      </h3>
      <div className="overflow-x-auto">
        <div className="inline-grid gap-1" style={{ gridTemplateColumns: `auto repeat(${lrs.length}, minmax(52px, 1fr))` }} role="table" aria-label="Heatmap of true return by learning rate and alpha">
          <div className="pr-1.5 text-right text-[10px] font-medium text-muted-foreground" role="columnheader">
            α \ lr
          </div>
          {lrs.map((lr) => (
            <div key={lr} className="text-center font-mono text-[10px] tabular-nums text-muted-foreground" role="columnheader">
              {lr}
            </div>
          ))}
          {alphas
            .map((a, ai) => ({ a, ai }))
            .reverse()
            .map(({ a, ai }) => (
              <div key={a} className="contents" role="row">
                <div className="flex items-center justify-end pr-1.5 font-mono text-[10px] tabular-nums text-muted-foreground" role="rowheader">
                  {a}
                </div>
                {lrs.map((lr, li) => {
                  const c = cells[ai][li];
                  if (!c)
                    return (
                      <div key={lr} className="flex h-10 items-center justify-center rounded-md bg-muted text-[10px] text-muted-foreground" role="cell" title="No comparable run">
                        –
                      </div>
                    );
                  const hs = heatScale((c.trueQ - min) / span, "viridis");
                  const isBest = c.id === rep.best.id;
                  return (
                    <div
                      key={lr}
                      role="cell"
                      title={`lr ${c.lr}, α ${c.alpha}: true return ${fmt(c.trueQ)}${c.status === "pruned" ? " (pruned early)" : ""}`}
                      className={`relative flex h-10 items-center justify-center rounded-md font-mono text-[11px] font-medium tabular-nums ${isBest ? "ring-2 ring-foreground ring-offset-1 ring-offset-card" : ""}`}
                      style={{ background: hs.color, color: hs.L > 0.62 ? "oklch(0.2 0.02 275)" : "oklch(0.98 0 0)" }}
                    >
                      {fmt(c.trueQ)}
                      {c.status === "pruned" && <span className="absolute right-1 top-0.5 text-[8px] font-semibold uppercase opacity-80">p</span>}
                    </div>
                  );
                })}
              </div>
            ))}
        </div>
      </div>
      <div className="mt-2 flex items-center gap-2 text-[10px] text-muted-foreground">
        <span className="font-mono tabular-nums">{fmt(min)}</span>
        <span className="h-2 w-28 rounded-full" style={{ background: `linear-gradient(to right, ${[0, 0.25, 0.5, 0.75, 1].map((t) => heatScale(t, "viridis").color).join(", ")})` }} aria-hidden />
        <span className="font-mono tabular-nums">{fmt(max)}</span>
        <span className="ml-2">outlined = best · p = pruned early</span>
      </div>
    </div>
  );
}

function MainEffect({ p, color, lo, hi }: { p: ParamSensitivity; color: string; lo: number; hi: number }) {
  const W = 140;
  const span = hi - lo || 1;
  const x = (v: number) => 4 + ((v - lo) / span) * (W - 8);
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <h4 className="text-[12px] font-medium text-foreground">{p.label}</h4>
        <span className="font-mono text-[10px] tabular-nums text-muted-foreground">effect range {fmt(p.effectRange)}</span>
      </div>
      <table className="w-full text-[11px]">
        <thead>
          <tr className="text-[9px] uppercase tracking-wider text-muted-foreground">
            <th className="pb-1 text-left font-semibold">Level</th>
            <th className="pb-1 text-left font-semibold">Mean · min–max</th>
            <th className="pb-1 text-right font-semibold">Return</th>
            <th className="pb-1 text-right font-semibold">Gap</th>
            <th className="pb-1 text-right font-semibold">OOD</th>
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">
          {p.levels.map((l) => (
            <tr key={l.value} className={l.value === p.bestLevel ? "text-foreground" : "text-muted-foreground"}>
              <td className="py-0.5 pr-2">{l.value}</td>
              <td className="py-0.5 pr-2">
                <svg width={W} height="12" aria-hidden>
                  <line x1={4} x2={W - 4} y1="6" y2="6" className="stroke-border" strokeWidth="1" />
                  <line x1={x(l.minTrueQ)} x2={x(l.maxTrueQ)} y1="6" y2="6" stroke={color} strokeOpacity="0.45" strokeWidth="3" strokeLinecap="round" />
                  <circle cx={x(l.meanTrueQ)} cy="6" r="3.5" fill={color} />
                </svg>
              </td>
              <td className={`py-0.5 text-right ${l.value === p.bestLevel ? "font-semibold" : ""}`}>{fmt(l.meanTrueQ)}</td>
              <td className="py-0.5 text-right">
                {l.meanGap >= 0 ? "+" : ""}
                {fmt(l.meanGap)}
              </td>
              <td className="py-0.5 text-right">{(l.meanOod * 100).toFixed(0)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RhoCell({ r }: { r: number | null }) {
  const a = r == null ? 0 : Math.abs(r);
  const bg = r == null ? "transparent" : r >= 0 ? `oklch(0.72 ${0.03 + a * 0.11} 185 / ${0.15 + a * 0.45})` : `oklch(0.72 ${0.03 + a * 0.13} 50 / ${0.15 + a * 0.45})`;
  return (
    <td className="px-2 py-1.5 text-right" style={{ background: bg }} title={r == null ? "Not enough variation to compute" : `${strength(r)} ${r >= 0 ? "positive" : "negative"} rank correlation`}>
      <span className="font-mono tabular-nums text-foreground">{fmtRho(r)}</span>
      <span className="ml-1.5 font-sans text-[9px] uppercase tracking-wide text-muted-foreground">{strength(r) === "n/a" ? "" : strength(r)}</span>
    </td>
  );
}

function CorrelationTable({ rep }: { rep: SensitivityReport }) {
  const rows: { label: string; rho: Correlations }[] = rep.params.map((p) => ({ label: p.label, rho: p.rho }));
  return (
    <div>
      <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Spearman rank correlation with outcomes</h3>
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-[12px]">
          <thead>
            <tr className="border-b border-border bg-muted/60 text-[10px] uppercase tracking-wider text-muted-foreground">
              <th className="px-2 py-1.5 text-left font-semibold">Parameter</th>
              <th className="px-2 py-1.5 text-right font-semibold">True return</th>
              <th className="px-2 py-1.5 text-right font-semibold">Overest. gap</th>
              <th className="px-2 py-1.5 text-right font-semibold">OOD rate</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label} className="border-b border-border/60 last:border-0">
                <td className="px-2 py-1.5 text-foreground">{r.label}</td>
                <RhoCell r={r.rho.trueQ} />
                <RhoCell r={r.rho.gap} />
                <RhoCell r={r.rho.ood} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <dl className="mt-2.5 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] sm:grid-cols-4">
        <div>
          <dt className="text-muted-foreground">Mean overest. gap</dt>
          <dd className={`font-mono tabular-nums ${rep.overall.meanGap > 1 ? "text-destructive" : "text-foreground"}`}>
            {rep.overall.meanGap >= 0 ? "+" : ""}
            {fmt(rep.overall.meanGap)}
          </dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Mean OOD rate</dt>
          <dd className="font-mono tabular-nums text-foreground">{(rep.overall.meanOod * 100).toFixed(1)}%</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">ρ(gap, return)</dt>
          <dd className="font-mono tabular-nums text-foreground">{fmtRho(rep.overall.rhoGapReturn)}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">ρ(OOD, return)</dt>
          <dd className="font-mono tabular-nums text-foreground">{fmtRho(rep.overall.rhoOodReturn)}</dd>
        </div>
      </dl>
    </div>
  );
}

export function SweepSensitivity({ runs, version }: { runs: SensitivityRun[]; version: number }) {
  // `version` changes whenever the (mutated-in-place) runs change.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const rep = useMemo(() => analyzeSweep(runs), [runs, version]);
  if (!rep) return null;
  const lo = Math.min(...rep.params.flatMap((p) => p.levels.map((l) => l.minTrueQ)));
  const hi = Math.max(...rep.params.flatMap((p) => p.levels.map((l) => l.maxTrueQ)));
  return (
    <section className="lab-rise bca-panel" aria-label="Win drivers">
      <header className="flex items-center justify-between gap-2 border-b border-border px-4 py-2.5">
        <h2 className="bca-panel-title flex items-center gap-2"><span className="bca-tick" aria-hidden />Win drivers · sensitivity analysis</h2>
        <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
          {rep.n} runs analysed{rep.excluded > 0 ? ` · ${rep.excluded} excluded` : ""}
        </span>
      </header>
      <div className="space-y-5 p-4">
        <Summary items={rep.summary} />
        <div className="grid gap-5 xl:grid-cols-2">
          <HeatmapView rep={rep} />
          <div className="space-y-4">
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Main effects · mean true return per level</h3>
            <MainEffect p={rep.params[0]} color="var(--chart-1)" lo={lo} hi={hi} />
            <MainEffect p={rep.params[1]} color="var(--chart-2)" lo={lo} hi={hi} />
          </div>
        </div>
        <CorrelationTable rep={rep} />
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          Rule-based analysis computed in your browser. Main effects average each level over the other parameter; Spearman ρ measures monotone trends only (a sweet spot can show ρ ≈ 0). Finished and pruned runs are included — pruned runs with their true return at the prune checkpoint; stopped and skipped runs are excluded.
        </p>
      </div>
    </section>
  );
}
