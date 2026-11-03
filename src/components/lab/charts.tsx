import { useMemo } from "react";

export interface Series {
  name: string;
  color: string;
  values: number[];
  dashed?: boolean;
}

function fmt(v: number) {
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

export function LineChart({ series, xs, height = 150, empty }: { series: Series[]; xs: number[]; height?: number; empty: string }) {
  const W = 320;
  const H = height;
  const pad = { l: 34, r: 8, t: 8, b: 18 };
  const n = xs.length;

  const geo = useMemo(() => {
    const all = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v));
    let lo = Math.min(...all);
    let hi = Math.max(...all);
    if (!all.length) {
      lo = 0;
      hi = 1;
    }
    if (hi - lo < 1e-6) {
      hi += 0.5;
      lo -= 0.5;
    }
    const span = hi - lo;
    lo -= span * 0.08;
    hi += span * 0.08;
    const x0 = xs[0] ?? 0;
    const x1 = xs[n - 1] ?? 1;
    const xr = x1 - x0 || 1;
    const px = (x: number) => pad.l + ((x - x0) / xr) * (W - pad.l - pad.r);
    const py = (y: number) => pad.t + (1 - (y - lo) / (hi - lo)) * (H - pad.t - pad.b);
    const ticks = [lo + (hi - lo) * 0.1, (lo + hi) / 2, hi - (hi - lo) * 0.1];
    return { px, py, ticks, x0, x1 };
  }, [series, xs, n, H, pad.l, pad.r, pad.t, pad.b]);

  if (n < 2) {
    return (
      <div className="flex items-center justify-center rounded-md border border-dashed border-border bg-surface/50 text-center text-xs text-muted-foreground" style={{ height }}>
        <div className="px-6">
          <svg width="28" height="20" viewBox="0 0 28 20" fill="none" className="mx-auto mb-2 text-muted-foreground/70">
            <path d="M1 18 L8 11 L13 14 L20 5 L27 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" strokeDasharray="3 3" />
          </svg>
          {empty}
        </div>
      </div>
    );
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height }} role="img">
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
      {series.map((s) => {
        const d = s.values.map((v, i) => `${i ? "L" : "M"}${geo.px(xs[i]).toFixed(1)},${geo.py(v).toFixed(1)}`).join(" ");
        const last = s.values[s.values.length - 1];
        return (
          <g key={s.name}>
            <path d={d} fill="none" stroke={s.color} strokeWidth="1.75" strokeDasharray={s.dashed ? "4 3" : undefined} strokeLinejoin="round" strokeLinecap="round" />
            <circle cx={geo.px(xs[n - 1])} cy={geo.py(last)} r="2.5" fill={s.color} />
          </g>
        );
      })}
    </svg>
  );
}

export function Legend({ items }: { items: { name: string; color: string; value?: string; dashed?: boolean }[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1">
      {items.map((it) => (
        <div key={it.name} className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <svg width="14" height="6">
            <line x1="0" x2="14" y1="3" y2="3" stroke={it.color} strokeWidth="2" strokeDasharray={it.dashed ? "3 2" : undefined} />
          </svg>
          {it.name}
          {it.value && <span className="font-mono tabular-nums text-foreground">{it.value}</span>}
        </div>
      ))}
    </div>
  );
}
