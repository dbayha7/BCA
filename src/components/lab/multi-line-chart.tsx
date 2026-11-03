// Line chart for several runs of different lengths: each series carries its own x values.

import { useMemo } from "react";

export interface RunSeries {
  name: string;
  color: string;
  xs: number[];
  values: number[];
  dashed?: boolean;
}

/** Distinct colors for up to 10 runs: the five chart tokens, then the same five dashed. */
export function runStyle(i: number): { color: string; dashed: boolean } {
  return { color: `var(--chart-${(i % 5) + 1})`, dashed: i >= 5 };
}

function fmt(v: number) {
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

export function MultiLineChart({ series, height = 170, empty, label }: { series: RunSeries[]; height?: number; empty: string; label: string }) {
  const W = 420;
  const H = height;
  const pad = { l: 36, r: 10, t: 8, b: 18 };
  const usable = series.filter((s) => s.xs.length >= 2);

  const geo = useMemo(() => {
    const ys = usable.flatMap((s) => s.values).filter((v) => Number.isFinite(v));
    const xsAll = usable.flatMap((s) => s.xs);
    let lo = ys.length ? Math.min(...ys) : 0;
    let hi = ys.length ? Math.max(...ys) : 1;
    if (hi - lo < 1e-6) {
      hi += 0.5;
      lo -= 0.5;
    }
    const span = hi - lo;
    lo -= span * 0.08;
    hi += span * 0.08;
    const x0 = xsAll.length ? Math.min(...xsAll) : 0;
    const x1 = xsAll.length ? Math.max(...xsAll) : 1;
    const xr = x1 - x0 || 1;
    return {
      px: (x: number) => pad.l + ((x - x0) / xr) * (W - pad.l - pad.r),
      py: (y: number) => pad.t + (1 - (y - lo) / (hi - lo)) * (H - pad.t - pad.b),
      ticks: [lo + (hi - lo) * 0.1, (lo + hi) / 2, hi - (hi - lo) * 0.1],
      x0,
      x1,
    };
  }, [usable, H, pad.l, pad.r, pad.t, pad.b]);

  if (!usable.length) {
    return (
      <div className="flex items-center justify-center rounded-md border border-dashed border-border bg-surface/50 px-6 text-center text-xs text-muted-foreground" style={{ height }}>
        {empty}
      </div>
    );
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={label}>
      {geo.ticks.map((t, i) => (
        <g key={i}>
          <line x1={pad.l} x2={W - pad.r} y1={geo.py(t)} y2={geo.py(t)} stroke="currentColor" className="text-border" strokeWidth="1" />
          <text x={pad.l - 5} y={geo.py(t) + 3} textAnchor="end" className="fill-muted-foreground font-mono" fontSize="8.5">
            {fmt(t)}
          </text>
        </g>
      ))}
      <text x={pad.l} y={H - 4} className="fill-muted-foreground font-mono" fontSize="8.5">
        {geo.x0}
      </text>
      <text x={W - pad.r} y={H - 4} textAnchor="end" className="fill-muted-foreground font-mono" fontSize="8.5">
        {geo.x1} upd
      </text>
      {usable.map((s) => {
        const pts = s.values.map((v, i) => [s.xs[i], v] as const).filter(([, v]) => Number.isFinite(v));
        const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${geo.px(x).toFixed(1)},${geo.py(y).toFixed(1)}`).join(" ");
        const end = pts[pts.length - 1];
        return (
          <g key={s.name}>
            <path d={d} fill="none" stroke={s.color} strokeWidth="1.6" strokeDasharray={s.dashed ? "4 3" : undefined} strokeLinejoin="round" strokeLinecap="round" />
            {end && <circle cx={geo.px(end[0])} cy={geo.py(end[1])} r="2.4" fill={s.color} />}
          </g>
        );
      })}
    </svg>
  );
}
