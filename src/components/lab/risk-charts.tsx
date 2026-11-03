// Risk charts: overlaid return histograms + strip plot with lower-tail VaR / CVaR markers and a
// risk table (RiskCharts); the logged reward distribution of a dataset (DatasetRewardChart); and
// a ready-made return-risk view for grid environments built from start-state rollouts
// (PolicyReturnRisk). Pure presentation over src/lib/risk.ts.

import { useEffect, useMemo, useState } from "react";
import { qStar, type Dataset } from "@/lib/rl";
import {
  datasetRewardDistribution,
  histogram,
  policyReturnDistribution,
  returnRisk,
  uniformEdges,
  type ReturnRisk,
  type StartStateReturns,
} from "@/lib/risk";

export interface RiskSeries {
  label: string;
  values: number[];
  /** CSS color; defaults to var(--chart-N) by position. */
  color?: string;
  /** Optional one-line description shown under the label in the table. */
  hint?: string;
}

const PALETTE = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"];
const LEVELS = [0.05, 0.1, 0.25];

function fmt(v: number) {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

const pct = (v: number) => `${(v * 100).toFixed(v > 0 && v < 0.1 ? 1 : 0)}%`;

function EmptyState({ children }: { children: React.ReactNode }) {
  return <div className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-[11.5px] text-muted-foreground">{children}</div>;
}

/**
 * Content width (px) of the element the returned callback ref is attached to. Charts use it as
 * their viewBox width so they draw at 1:1 scale: text stays legible in a narrow sidebar and does
 * not balloon in a wide panel.
 */
export function useContainerWidth(fallback = 560): [(el: HTMLDivElement | null) => void, number] {
  const [el, setEl] = useState<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    if (!el) return;
    const update = () => {
      const w = Math.round(el.clientWidth);
      if (w > 0) setWidth(w);
    };
    update();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return [setEl, width];
}

function Segmented<T extends string | number>({ value, options, onChange, label }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div className="flex rounded-md border border-border bg-surface p-0.5 text-[11px]" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          onClick={() => onChange(o.value)}
          className={`rounded px-2 py-0.5 font-medium transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
            o.value === value ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/* ---------------- RiskCharts ---------------- */

/**
 * Return distributions of several controllers on one shared axis: overlaid histograms (share of
 * episodes per bin) with dashed VaR lines and ▼ CVaR markers, a strip of every episode with the
 * lower tail highlighted, and a table of mean / std / VaR / CVaR. The tail level starts at
 * `level` and can be switched between 5%, 10% and 25% in the header.
 */
export function RiskCharts({
  series,
  level = 0.1,
  bins = 24,
  valueLabel = "Return",
  empty = "No returns to chart yet.",
}: {
  series: RiskSeries[];
  /** Initial lower-tail level α (e.g. 0.05 or 0.1). */
  level?: number;
  bins?: number;
  /** Name of the measured quantity (axis + table header). */
  valueLabel?: string;
  empty?: string;
}) {
  const [lvl, setLvl] = useState(level);
  const [boxRef, boxW] = useContainerWidth();
  const levels = useMemo(() => [...new Set([...LEVELS, level])].sort((a, b) => a - b), [level]);

  const model = useMemo(() => {
    const items = series.map((s, i) => ({ ...s, color: s.color ?? PALETTE[i % PALETTE.length], risk: returnRisk(s.values, lvl) }));
    const all = items.flatMap((s) => s.values.filter(Number.isFinite));
    if (!all.length) return null;
    let lo = Math.min(...all);
    let hi = Math.max(...all);
    const pad = (hi - lo || Math.abs(hi) || 1) * 0.02;
    lo -= pad;
    hi += pad;
    const edges = uniformEdges(lo, hi, bins);
    const hists = items.map((s) => {
      const h = histogram(s.values, edges);
      return h.counts.map((c) => (s.risk.n ? c / s.risk.n : 0));
    });
    const yMax = Math.max(1e-9, ...hists.flat());
    return { items, edges, hists, yMax, lo: edges[0], hi: edges[edges.length - 1] };
  }, [series, lvl, bins]);

  if (!model) return <EmptyState>{empty}</EmptyState>;

  const { items, edges, hists, yMax, lo, hi } = model;
  const W = Math.max(280, boxW);
  const padL = W < 440 ? 76 : 100;
  const maxChars = Math.floor((padL - 8) / 5.8);
  const padR = 12;
  const histTop = 10;
  const histH = 120;
  const rowH = 18;
  const stripTop = histTop + histH + 12;
  const axisY = stripTop + items.length * rowH + 4;
  const H = axisY + 18;
  const sx = (v: number) => padL + ((v - lo) / (hi - lo || 1)) * (W - padL - padR);
  const sy = (d: number) => histTop + (1 - d / (yMax * 1.08)) * histH;
  const base = sy(0);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => lo + f * (hi - lo));
  const alphaTxt = `${Math.round(lvl * 1000) / 10}%`;
  const valid = items.filter((s) => s.risk.n > 0);
  const bestCvar = valid.length > 1 ? Math.max(...valid.map((s) => s.risk.cvar)) : NaN;
  const bestMean = valid.length > 1 ? Math.max(...valid.map((s) => s.risk.mean)) : NaN;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-[11px] text-muted-foreground">
          {valueLabel} distribution · {items.map((s) => s.risk.n).join(" / ")} samples
        </p>
        <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
          Tail level α
          <Segmented<number> label="Tail level" value={lvl} onChange={setLvl} options={levels.map((l) => ({ value: l, label: `${Math.round(l * 1000) / 10}%` }))} />
        </div>
      </div>

      <div className="rounded-lg border border-border bg-background p-2">
        <div ref={boxRef}>
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label={`${valueLabel} histograms with VaR and CVaR at ${alphaTxt}`}>
          {/* histogram grid */}
          {[0, 0.5, 1].map((f) => {
            const d = f * yMax;
            return (
              <g key={f}>
                <line x1={padL} x2={W - padR} y1={sy(d)} y2={sy(d)} className="stroke-border" strokeWidth="1" />
                <text x={padL - 6} y={sy(d) + 3} textAnchor="end" className="fill-muted-foreground font-mono" fontSize="10">
                  {pct(d)}
                </text>
              </g>
            );
          })}
          <text x={4} y={histTop + histH * 0.25} className="fill-muted-foreground" fontSize="10">
            share of
          </text>
          <text x={4} y={histTop + histH * 0.25 + 12} className="fill-muted-foreground" fontSize="10">
            episodes
          </text>
          {/* overlaid step histograms */}
          {items.map((s, i) => {
            if (!s.risk.n) return null;
            const d = hists[i];
            let path = `M${sx(edges[0]).toFixed(1)},${base.toFixed(1)}`;
            d.forEach((v, b) => {
              path += ` L${sx(edges[b]).toFixed(1)},${sy(v).toFixed(1)} L${sx(edges[b + 1]).toFixed(1)},${sy(v).toFixed(1)}`;
            });
            path += ` L${sx(edges[edges.length - 1]).toFixed(1)},${base.toFixed(1)} Z`;
            return <path key={s.label} d={path} fill={s.color} fillOpacity={0.1} stroke={s.color} strokeWidth="1.5" strokeLinejoin="round" />;
          })}
          {/* VaR (dashed) and CVaR (▼) markers */}
          {items.map((s) =>
            s.risk.n ? (
              <g key={`m-${s.label}`}>
                <line x1={sx(s.risk.var)} x2={sx(s.risk.var)} y1={histTop} y2={base} stroke={s.color} strokeWidth="1.25" strokeDasharray="4 3">
                  <title>{`${s.label}: VaR ${alphaTxt} = ${fmt(s.risk.var)}`}</title>
                </line>
                <polygon points={`${sx(s.risk.cvar) - 4},${histTop - 2} ${sx(s.risk.cvar) + 4},${histTop - 2} ${sx(s.risk.cvar)},${histTop + 5}`} fill={s.color}>
                  <title>{`${s.label}: CVaR ${alphaTxt} = ${fmt(s.risk.cvar)}`}</title>
                </polygon>
              </g>
            ) : null,
          )}
          <line x1={padL} x2={W - padR} y1={base} y2={base} className="stroke-muted-foreground" strokeWidth="1" opacity="0.5" />

          {/* strip: one row per series, every episode a dot, lower tail in red */}
          {items.map((s, i) => {
            const cy = stripTop + i * rowH + rowH / 2;
            return (
              <g key={`s-${s.label}`}>
                <text x={4} y={cy + 3.5} className="fill-muted-foreground" fontSize="10.5">
                  <title>{s.label}</title>
                  {s.label.length > maxChars ? `${s.label.slice(0, maxChars - 1)}…` : s.label}
                </text>
                <line x1={padL} x2={W - padR} y1={cy} y2={cy} className="stroke-border" strokeWidth="1" />
                {s.values.map((v, k) => {
                  if (!Number.isFinite(v)) return null;
                  const tail = v <= s.risk.var;
                  const jitter = (((k * 0.618034) % 1) - 0.5) * 8;
                  return <circle key={k} cx={sx(v)} cy={cy + jitter} r="2.1" fill={tail ? "var(--destructive)" : s.color} opacity={tail ? 0.85 : 0.4} />;
                })}
                {s.risk.n > 0 && (
                  <>
                    <line x1={sx(s.risk.mean)} x2={sx(s.risk.mean)} y1={cy - 7} y2={cy + 7} className="stroke-foreground" strokeWidth="1.75">
                      <title>{`${s.label}: mean ${fmt(s.risk.mean)}`}</title>
                    </line>
                    <line x1={sx(s.risk.cvar)} x2={sx(s.risk.cvar)} y1={cy - 7} y2={cy + 7} stroke="var(--destructive)" strokeWidth="1.75">
                      <title>{`${s.label}: CVaR ${alphaTxt} ${fmt(s.risk.cvar)}`}</title>
                    </line>
                  </>
                )}
              </g>
            );
          })}

          {/* x axis */}
          {ticks.map((t, i) => (
            <text key={i} x={sx(t)} y={axisY + 10} textAnchor={i === 0 ? "start" : i === ticks.length - 1 ? "end" : "middle"} className="fill-muted-foreground font-mono" fontSize="10">
              {fmt(t)}
            </text>
          ))}
        </svg>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[10.5px] text-muted-foreground">
        {items.map((s) => (
          <span key={s.label} className="flex items-center gap-1.5">
            <span className="h-2 w-3 rounded-sm" style={{ background: s.color, opacity: 0.8 }} />
            {s.label}
          </span>
        ))}
        <span className="flex items-center gap-1.5">
          <svg width="12" height="10" aria-hidden>
            <line x1="6" x2="6" y1="0" y2="10" stroke="currentColor" strokeDasharray="3 2" />
          </svg>
          VaR {alphaTxt}
        </span>
        <span className="flex items-center gap-1.5">
          <svg width="10" height="8" aria-hidden>
            <polygon points="1,0 9,0 5,7" fill="currentColor" />
          </svg>
          CVaR {alphaTxt}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-destructive" />
          episodes in the worst {alphaTxt}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-3 w-0 border-l-2 border-foreground" />
          mean
        </span>
      </div>

      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-left text-[11.5px]">
          <thead>
            <tr className="whitespace-nowrap border-b border-border bg-muted/60 text-[10px] uppercase tracking-wider text-muted-foreground">
              <th className="min-w-[170px] px-3 py-2 font-semibold">Controller</th>
              <th className="px-3 py-2 text-right font-semibold">n</th>
              <th className="px-3 py-2 text-right font-semibold">Mean</th>
              <th className="px-3 py-2 text-right font-semibold">± std</th>
              <th className="px-3 py-2 text-right font-semibold">Median</th>
              <th className="px-3 py-2 text-right font-semibold normal-case">VaR {alphaTxt}</th>
              <th className="px-3 py-2 text-right font-semibold normal-case">CVaR {alphaTxt}</th>
              <th className="px-3 py-2 text-right font-semibold" title="Mean − CVaR: how much worse the bad episodes are than the average one">
                Tail gap
              </th>
              <th className="px-3 py-2 text-right font-semibold">Worst</th>
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {items.map((s) => (
              <RiskRow key={s.label} label={s.label} hint={s.hint} color={s.color} r={s.risk} bestCvar={s.risk.cvar === bestCvar} bestMean={s.risk.mean === bestMean} />
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[10.5px] leading-relaxed text-muted-foreground">
        Higher {valueLabel.toLowerCase()} is better, so risk is the lower tail. VaR {alphaTxt} is the {valueLabel.toLowerCase()} at or below which the worst {alphaTxt} of episodes fall; CVaR {alphaTxt} is the average of those episodes (expected shortfall). Ranking by CVaR instead of the mean favors controllers whose bad episodes are less bad. With few episodes the tail holds only α·n of them, so read CVaR at small α with care.
      </p>
    </div>
  );
}

function RiskRow({ label, hint, color, r, bestCvar, bestMean }: { label: string; hint?: string; color: string; r: ReturnRisk; bestCvar: boolean; bestMean: boolean }) {
  return (
    <tr className={`border-b border-border/60 last:border-0 ${bestCvar ? "bg-success/5" : ""}`}>
      <td className="px-3 py-2 font-sans">
        <div className="flex items-center gap-1.5 font-medium text-foreground">
          <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />
          {label}
          {bestCvar && <span className="rounded bg-success/15 px-1.5 py-0.5 text-[10px] font-semibold text-success">safest tail</span>}
          {bestMean && <span className="rounded bg-primary/15 px-1.5 py-0.5 text-[10px] font-semibold text-primary">best mean</span>}
        </div>
        {hint && <div className="text-[10px] text-muted-foreground">{hint}</div>}
      </td>
      <td className="px-3 py-2 text-right text-muted-foreground">{r.n}</td>
      <td className="px-3 py-2 text-right text-foreground">{fmt(r.mean)}</td>
      <td className="px-3 py-2 text-right text-muted-foreground">{fmt(r.std)}</td>
      <td className="px-3 py-2 text-right text-muted-foreground">{fmt(r.median)}</td>
      <td className="px-3 py-2 text-right text-foreground">{fmt(r.var)}</td>
      <td className="px-3 py-2 text-right font-medium text-destructive">{fmt(r.cvar)}</td>
      <td className="px-3 py-2 text-right text-muted-foreground">{fmt(r.mean - r.cvar)}</td>
      <td className="px-3 py-2 text-right text-muted-foreground">{fmt(r.min)}</td>
    </tr>
  );
}

/* ---------------- DatasetRewardChart ---------------- */

const ACTION_COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)"];

/**
 * Distribution of the logged rewards r in a Dataset: a histogram stacked by action, summary
 * quantiles, where the reward comes from (goal / hazard terminals vs per-step), and a per-action
 * table. Log scale is the default because terminal rewards are rare next to per-step costs.
 */
export function DatasetRewardChart({ ds, bins = 24 }: { ds: Dataset; bins?: number }) {
  const d = useMemo(() => datasetRewardDistribution(ds, bins), [ds, bins]);
  const [scale, setScale] = useState<"log" | "linear">("log");
  const [boxRef, boxW] = useContainerWidth();

  if (d.n === 0) return <EmptyState>This dataset has no transitions.</EmptyState>;

  const W = Math.max(280, boxW);
  const H = 170;
  const padL = 40;
  const padR = 10;
  const padT = 10;
  const padB = 22;
  const lo = d.edges[0];
  const hi = d.edges[d.edges.length - 1];
  const sx = (v: number) => padL + ((v - lo) / (hi - lo || 1)) * (W - padL - padR);
  const maxCount = Math.max(1, ...d.overall.counts);
  const hFrac = (c: number) => (scale === "log" ? Math.log1p(c) / Math.log1p(maxCount) : c / maxCount);
  const plotH = H - padT - padB;
  const yTicks = scale === "log" ? [1, 10, 100, 1000, 10000, 100000].filter((t) => t <= maxCount) : [...new Set([0, Math.round(maxCount / 2), maxCount])];
  const term = d.terminal;
  const anyTerminal = term.goal.count + term.hazard.count + term.other.count > 0;
  const terminalParts = [
    { key: "goal", label: "goal terminal", color: "var(--success)", t: term.goal },
    { key: "hazard", label: "hazard / failure terminal", color: "var(--destructive)", t: term.hazard },
    { key: "other", label: "other terminal", color: "var(--warning)", t: term.other },
    { key: "none", label: "per-step (non-terminal)", color: "var(--muted-foreground)", t: term.none },
  ].filter((p) => p.t.count > 0);
  const zeroX = lo < 0 && hi > 0 ? sx(0) : null;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-[11px] text-muted-foreground">
          {d.n.toLocaleString()} logged rewards · {d.distinctValues > 1000 ? ">1000" : d.distinctValues} distinct value{d.distinctValues === 1 ? "" : "s"}
          {d.distinctValues <= 12 ? " (categorical)" : ""}
        </p>
        <Segmented<"log" | "linear">
          label="Count scale"
          value={scale}
          onChange={setScale}
          options={[
            { value: "log", label: "log count" },
            { value: "linear", label: "linear" },
          ]}
        />
      </div>

      <div className="rounded-lg border border-border bg-background p-2">
        <div ref={boxRef}>
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label={`Histogram of ${d.n} logged rewards from ${fmt(lo)} to ${fmt(hi)}, stacked by action`}>
          {yTicks.map((t) => {
            const y = padT + (1 - hFrac(t)) * plotH;
            return (
              <g key={t}>
                <line x1={padL} x2={W - padR} y1={y} y2={y} className="stroke-border" strokeWidth="1" />
                <text x={padL - 5} y={y + 3} textAnchor="end" className="fill-muted-foreground font-mono" fontSize="10">
                  {t >= 1000 ? `${t / 1000}k` : t}
                </text>
              </g>
            );
          })}
          {zeroX !== null && <line x1={zeroX} x2={zeroX} y1={padT} y2={padT + plotH} className="stroke-muted-foreground" strokeDasharray="2 3" strokeWidth="1" />}
          {d.overall.counts.map((total, b) => {
            if (!total) return null;
            const x0 = sx(d.edges[b]) + 0.75;
            const w = Math.max(1, sx(d.edges[b + 1]) - sx(d.edges[b]) - 1.5);
            const barH = hFrac(total) * plotH;
            let y = padT + plotH;
            return (
              <g key={b}>
                <title>{`r ∈ [${fmt(d.edges[b])}, ${fmt(d.edges[b + 1])}]: ${total.toLocaleString()} transitions — ${d.perAction
                  .filter((a) => a.counts[b] > 0)
                  .map((a) => `${a.name} ${a.counts[b]}`)
                  .join(", ")}`}</title>
                {d.perAction.map((a) => {
                  const c = a.counts[b];
                  if (!c) return null;
                  const h = (c / total) * barH;
                  y -= h;
                  return <rect key={a.action} x={x0} y={y} width={w} height={h} fill={ACTION_COLORS[a.action]} opacity={0.85} />;
                })}
              </g>
            );
          })}
          <line x1={padL} x2={W - padR} y1={padT + plotH} y2={padT + plotH} className="stroke-muted-foreground" strokeWidth="1" opacity="0.5" />
          {[lo, (lo + hi) / 2, hi].map((t, i) => (
            <text key={i} x={sx(t)} y={H - 6} textAnchor={i === 0 ? "start" : i === 2 ? "end" : "middle"} className="fill-muted-foreground font-mono" fontSize="10">
              {fmt(t)}
            </text>
          ))}
          {zeroX !== null && (
            <text x={zeroX} y={H - 6} textAnchor="middle" className="fill-muted-foreground font-mono" fontSize="10">
              0
            </text>
          )}
        </svg>
        </div>
      </div>

      <div className="grid grid-cols-[repeat(auto-fill,minmax(68px,1fr))] gap-1.5">
        {(
          [
            ["mean", d.summary.mean],
            ["std", d.summary.std],
            ["min", d.summary.min],
            ["5%", d.summary.q05],
            ["median", d.summary.median],
            ["95%", d.summary.q95],
            ["max", d.summary.max],
          ] as [string, number][]
        ).map(([k, v]) => (
          <div key={k} className="rounded-md border border-border bg-surface/60 px-2 py-1.5">
            <div className="text-[9.5px] font-medium uppercase tracking-wider text-muted-foreground">{k}</div>
            <div className="font-mono text-[12px] tabular-nums text-foreground">{fmt(v)}</div>
          </div>
        ))}
      </div>

      <div>
        <div className="mb-1.5 flex items-center justify-between text-[11px] text-muted-foreground">
          <span>Where the reward comes from</span>
          <span className="font-mono tabular-nums">share of transitions</span>
        </div>
        <div className="flex h-2.5 overflow-hidden rounded-full bg-surface" role="img" aria-label="Share of transitions by terminal type">
          {terminalParts.map((p) => (
            <div key={p.key} style={{ width: `${p.t.share * 100}%`, background: p.color, opacity: p.key === "none" ? 0.35 : 0.9 }} title={`${p.label}: ${pct(p.t.share)} of transitions`} />
          ))}
        </div>
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[10.5px] text-muted-foreground">
          {anyTerminal ? (
            terminalParts.map((p) => (
              <span key={p.key} className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ background: p.color, opacity: p.key === "none" ? 0.5 : 1 }} />
                {p.label}
                <span className="font-mono tabular-nums text-foreground">{pct(p.t.share)}</span>
                <span className="font-mono tabular-nums">· {pct(p.t.absRewardShare)} of |r| mass</span>
              </span>
            ))
          ) : (
            <span>No terminal transitions — every reward is a per-step cost or bonus.</span>
          )}
        </div>
      </div>

      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-left text-[11.5px]">
          <thead>
            <tr className="whitespace-nowrap border-b border-border bg-muted/60 text-[10px] uppercase tracking-wider text-muted-foreground">
              <th className="px-2 py-2 font-semibold">Action</th>
              <th className="px-2 py-2 text-right font-semibold">n</th>
              <th className="px-2 py-2 text-right font-semibold">Share</th>
              <th className="px-2 py-2 text-right font-semibold">Mean r</th>
              <th className="px-2 py-2 text-right font-semibold">± std</th>
              <th className="px-2 py-2 text-right font-semibold">Min</th>
              <th className="px-2 py-2 text-right font-semibold">Max</th>
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {d.perAction.map((a) => (
              <tr key={a.action} className="border-b border-border/60 last:border-0">
                <td className="px-2 py-1.5 font-sans">
                  <span className="flex items-center gap-1.5 font-medium text-foreground">
                    <span className="h-2 w-2 rounded-sm" style={{ background: ACTION_COLORS[a.action] }} />
                    {a.name}
                  </span>
                </td>
                <td className="px-2 py-1.5 text-right text-muted-foreground">{a.n.toLocaleString()}</td>
                <td className="px-2 py-1.5 text-right text-muted-foreground">{pct(a.share)}</td>
                <td className={`px-3 py-1.5 text-right ${a.mean < 0 ? "text-destructive" : "text-foreground"}`}>{fmt(a.mean)}</td>
                <td className="px-2 py-1.5 text-right text-muted-foreground">{fmt(a.std)}</td>
                <td className="px-2 py-1.5 text-right text-muted-foreground">{fmt(a.min)}</td>
                <td className="px-2 py-1.5 text-right text-muted-foreground">{fmt(a.max)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ---------------- PolicyReturnRisk (grid environments) ---------------- */

function majorityPolicy(ds: Dataset): Int8Array {
  const S = ds.env.width * ds.env.height;
  const p = new Int8Array(S);
  for (let s = 0; s < S; s++) {
    let best = 0;
    for (let a = 1; a < 4; a++) if (ds.counts[s * 4 + a] > ds.counts[s * 4 + best]) best = a;
    p[s] = best;
  }
  return p;
}

function greedyFromQ(Q: Float64Array, S: number): Int8Array {
  const p = new Int8Array(S);
  for (let s = 0; s < S; s++) {
    let best = 0;
    for (let a = 1; a < 4; a++) if (Q[s * 4 + a] > Q[s * 4 + best]) best = a;
    p[s] = best;
  }
  return p;
}

/**
 * Return risk of the learned greedy policy on a grid environment, next to the dataset's
 * majority-action (behavior-cloning) policy and the optimal policy π*. Every non-terminal state
 * the dataset visits is used as a start (plus env.start); since env.step and the policies are
 * deterministic, each start is exactly one episode (capped at `horizon` steps). For empirical
 * environments (custom uploads, ThrusterChain's grid) the rollouts run in the empirical MDP — use
 * SafetyPanel for ThrusterChain's real-physics returns.
 */
export function PolicyReturnRisk({ ds, policy, gamma = 0.95, horizon = 100, level = 0.1 }: { ds: Dataset; policy: Int8Array; gamma?: number; horizon?: number; level?: number }) {
  const [kind, setKind] = useState<"undiscounted" | "discounted">("undiscounted");
  const runs = useMemo(() => {
    const env = ds.env;
    const S = env.width * env.height;
    const empirical = env.groundTruth !== "exact";
    const defs: { label: string; hint: string; color: string; p: Int8Array }[] = [
      { label: "Learned policy", hint: "greedy action of the current Q-table", color: "var(--chart-1)", p: policy },
      { label: "Behavior (majority)", hint: "most-logged action per state", color: "var(--chart-3)", p: majorityPolicy(ds) },
      { label: "Optimal π*", hint: empirical ? "value iteration on the empirical MDP" : "value iteration on the true MDP", color: "var(--chart-2)", p: greedyFromQ(qStar(env, gamma), S) },
    ];
    return defs.map((d) => ({ ...d, out: policyReturnDistribution(env, d.p, { ds, horizon, gamma }) }));
  }, [ds, policy, gamma, horizon]);

  const first: StartStateReturns | undefined = runs[0]?.out;
  if (!first || first.starts.length === 0) return <EmptyState>No non-terminal start states in this dataset.</EmptyState>;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="max-w-xl text-[11px] leading-relaxed text-muted-foreground">
          One deterministic episode from each of the {first.starts.length} non-terminal states the dataset visits (≤ {horizon} steps{ds.env.groundTruth !== "exact" ? ", empirical MDP" : ""}). The spread shows how much the outcome depends on where the agent starts.
        </p>
        <Segmented<"undiscounted" | "discounted">
          label="Return type"
          value={kind}
          onChange={setKind}
          options={[
            { value: "undiscounted", label: "Σ r" },
            { value: "discounted", label: `Σ γᵗ r (γ=${gamma})` },
          ]}
        />
      </div>
      <div className="flex flex-wrap gap-x-5 gap-y-1 text-[10.5px] text-muted-foreground">
        {runs.map((r) => {
          const oc = r.out.outcomeCounts;
          const n = r.out.starts.length;
          return (
            <span key={r.label} className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full" style={{ background: r.color }} />
              <span className="text-foreground">{r.label}</span>
              {oc.goal > 0 && <span className="text-success">goal {pct(oc.goal / n)}</span>}
              {oc.hazard > 0 && <span className="text-destructive">hazard {pct(oc.hazard / n)}</span>}
              {oc.terminal > 0 && <span>terminal {pct(oc.terminal / n)}</span>}
              {oc.timeout > 0 && (
                <span>
                  timeout {pct(oc.timeout / n)}
                  {r.out.loops > 0 ? ` (${r.out.loops} loop${r.out.loops === 1 ? "" : "s"})` : ""}
                </span>
              )}
            </span>
          );
        })}
      </div>
      <RiskCharts
        level={level}
        valueLabel={kind === "undiscounted" ? "Return" : "Discounted return"}
        series={runs.map((r) => ({ label: r.label, hint: r.hint, color: r.color, values: kind === "undiscounted" ? r.out.returns : r.out.discounted }))}
      />
    </div>
  );
}
