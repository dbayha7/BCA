import { useMemo, useRef, useState } from "react";
import type { Dataset, EvalResult } from "@/lib/rl";
import { changeRecency, createPolicyTracker, ingestPolicy, supportLevel, type PolicyChangeTracker } from "@/lib/playback";

export type HeatMode = "estimate" | "truth" | "error";

/** Diverging reward color: orange (negative) → neutral → teal (positive). Colorblind-safe pair. */
export function rewardColor(r: number, lo: number, hi: number) {
  if (r < 0) {
    const t = Math.min(1, Math.abs(r) / Math.max(1e-6, Math.abs(lo)));
    return `oklch(${0.78 - t * 0.12} ${0.04 + t * 0.15} 50)`;
  }
  const t = Math.min(1, r / Math.max(1e-6, hi));
  return `oklch(${0.78 - t * 0.08} ${0.04 + t * 0.12} 185)`;
}

export type Palette = "viridis" | "diverging" | "mono";

const PALETTE_LABELS: Record<Palette, string> = {
  viridis: "Violet → Yellow",
  diverging: "Orange ↔ Teal",
  mono: "Light → Dark",
};

/** Push mid values toward the extremes so high vs. low Q-values separate sharply. */
function sharpen(t: number, contrast: number) {
  const c = Math.max(0, Math.min(1, t));
  const d = 2 * c - 1;
  return 0.5 + (Math.sign(d) * Math.pow(Math.abs(d), 1 / contrast)) / 2;
}

export function heatScale(t: number, palette: Palette = "viridis", contrast = 1) {
  const c = sharpen(t, contrast);
  let L: number;
  let C: number;
  let h: number;
  if (palette === "diverging") {
    const u = Math.abs(2 * c - 1);
    L = 0.9 - u * (c < 0.5 ? 0.38 : 0.4);
    C = 0.03 + u * (c < 0.5 ? 0.17 : 0.13);
    h = c < 0.5 ? 50 : 185;
  } else if (palette === "mono") {
    L = 0.95 - c * 0.7;
    C = 0.02 + c * 0.2;
    h = 285;
  } else {
    L = 0.28 + c * 0.62;
    C = 0.12 + Math.sin(c * Math.PI) * 0.05;
    h = 290 - c * 190;
  }
  return { color: `oklch(${L} ${C} ${h})`, L };
}

/** Sequential heat color: deep violet (low) → teal → pale yellow (high). */
export function heatColor(t: number, palette: Palette = "viridis", contrast = 1) {
  return heatScale(t, palette, contrast).color;
}

function Arrow({ x1, y1, x2, y2, color, width, opacity, dash }: { x1: number; y1: number; x2: number; y2: number; color: string; width: number; opacity: number; dash?: string }) {
  const ang = Math.atan2(y2 - y1, x2 - x1);
  const hl = 4 + width;
  const p1 = `${x2 - hl * Math.cos(ang - 0.45)},${y2 - hl * Math.sin(ang - 0.45)}`;
  const p2 = `${x2 - hl * Math.cos(ang + 0.45)},${y2 - hl * Math.sin(ang + 0.45)}`;
  return (
    <g opacity={opacity}>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={width} strokeLinecap="round" strokeDasharray={dash} />
      <polygon points={`${x2},${y2} ${p1} ${p2}`} fill={dash ? "none" : color} stroke={dash ? color : "none"} strokeWidth={dash ? 1 : 0} />
    </g>
  );
}

/** Policy-mode overlays. Any combination can be on; all of them follow the live Q-table. */
export interface PolicyOverlays {
  /** Greedy action arrows. */
  arrows: boolean;
  /** Ring cells whose greedy action changed in the last few training steps (ghost arrow = previous action). */
  changed: boolean;
  /** Fog cells by how often the dataset logged the greedy action there (log scale), with the count in the corner. */
  support: boolean;
  /** Dashed outline where the greedy action was never logged (out-of-distribution). */
  ood: boolean;
}

export const DEFAULT_OVERLAYS: PolicyOverlays = { arrows: true, changed: false, support: false, ood: true };

// Remember the overlay choice for the session so it survives switching to Watch Episodes and back.
let sessionOverlays: PolicyOverlays = DEFAULT_OVERLAYS;

const OVERLAY_LABELS: { key: keyof PolicyOverlays; label: string; title: string }[] = [
  { key: "arrows", label: "Greedy arrows", title: "Arrow per state for argmax_a Q(s, a)" },
  { key: "changed", label: "Changed", title: "Ring states whose greedy action changed since the previous training step (fades over a few steps)" },
  { key: "support", label: "Data support", title: "Fog states by how many logged samples back the greedy action (log scale)" },
  { key: "ood", label: "OOD outline", title: "Dashed outline where the greedy action never appears in the data" },
];

/** Number of training steps over which a "changed" highlight fades out. */
const CHANGE_FADE = 6;

const DIR_X = [0, 1, 0, -1];
const DIR_Y = [-1, 0, 1, 0];

interface GridProps {
  ds: Dataset;
  mode: "dataset" | "policy";
  Q: Float64Array;
  evalRes: EvalResult;
  heat: HeatMode;
  version: number;
}

export function StateGrid({ ds, mode, Q, evalRes, heat, version }: GridProps) {
  const { env } = ds;
  const W = env.width;
  const H = env.height;
  const cell = 46;
  const [hover, setHover] = useState<number | null>(null);
  const [palette, setPalette] = useState<Palette>("viridis");
  const [contrast, setContrast] = useState(1.6);
  const [overlays, setOverlays] = useState<PolicyOverlays>(() => sessionOverlays);
  const [tipPos, setTipPos] = useState<{ x: number; y: number; w: number } | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const S = W * H;

  const maxVisit = useMemo(() => Math.max(1, ...Array.from(ds.visits)), [ds]);

  // States that get a policy arrow (logged, not a wall or terminal) — the only ones overlays annotate.
  const eligible = useMemo(() => {
    const m = new Uint8Array(S);
    for (let s = 0; s < S; s++) m[s] = env.blocked.has(s) || env.terminals.has(s) || ds.visits[s] === 0 ? 0 : 1;
    return m;
  }, [ds, env, S]);

  const maxCount = useMemo(() => {
    let m = 0;
    for (let k = 0; k < ds.counts.length; k++) if (ds.counts[k] > m) m = ds.counts[k];
    return m;
  }, [ds]);

  // Previous greedy policy lives in a ref; ingesting is O(S) and idempotent per version, so it is
  // cheap enough to run on every Auto-train tick and safe under StrictMode double renders.
  const trackerRef = useRef<PolicyChangeTracker | null>(null);
  const tracker = useMemo(() => {
    const prev = trackerRef.current ?? createPolicyTracker(S);
    const next = ingestPolicy(prev, evalRes.policy, version, eligible);
    trackerRef.current = next;
    return next;
  }, [evalRes.policy, version, eligible, S]);

  const recentChanges = useMemo(() => {
    let n = 0;
    for (let s = 0; s < S; s++) if (eligible[s] && changeRecency(tracker, s, version, CHANGE_FADE) > 0) n++;
    return n;
  }, [tracker, version, eligible, S]);

  const toggleOverlay = (key: keyof PolicyOverlays) => {
    const next = { ...overlays, [key]: !overlays[key] };
    sessionOverlays = next;
    setOverlays(next);
  };

  const heatVals = useMemo(() => {
    const vals = new Float64Array(S);
    for (let s = 0; s < S; s++) {
      const b = s * 4;
      const qm = Math.max(Q[b], Q[b + 1], Q[b + 2], Q[b + 3]);
      vals[s] = heat === "estimate" ? qm : heat === "truth" ? evalRes.vPi[s] : qm - evalRes.vPi[s];
    }
    let lo = Infinity;
    let hi = -Infinity;
    for (let s = 0; s < S; s++) {
      if (env.blocked.has(s) || env.terminals.has(s) || ds.visits[s] === 0) continue;
      lo = Math.min(lo, vals[s]);
      hi = Math.max(hi, vals[s]);
    }
    if (!isFinite(lo)) {
      lo = 0;
      hi = 1;
    }
    if (hi - lo < 1e-6) hi = lo + 1;
    return { vals, lo, hi };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [Q, evalRes, heat, version, S, ds, env]);

  const cx = (s: number) => (s % W) * cell + cell / 2;
  const cy = (s: number) => Math.floor(s / W) * cell + cell / 2;

  const hoverInfo = hover !== null ? hover : null;

  return (
    <div className="flex flex-col gap-3">
      {mode === "policy" && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-border bg-card px-3 py-2 text-xs">
          <label className="flex items-center gap-2 text-muted-foreground">
            Color scale
            <select
              value={palette}
              onChange={(e) => setPalette(e.target.value as Palette)}
              className="h-8 rounded-md border border-input bg-background px-2.5 text-foreground"
            >
              {(Object.keys(PALETTE_LABELS) as Palette[]).map((p) => (
                <option key={p} value={p}>
                  {PALETTE_LABELS[p]}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-muted-foreground">
            Contrast
            <input type="range" min={1} max={4} step={0.1} value={contrast} onChange={(e) => setContrast(Number(e.target.value))} className="w-24 accent-[var(--primary)]" />
            <span className="w-8 font-mono tabular-nums text-foreground">{contrast.toFixed(1)}×</span>
          </label>
          <div className="flex flex-wrap items-center gap-2 text-muted-foreground" role="group" aria-label="Policy overlays">
            Overlays
            <div className="flex flex-wrap rounded-md border border-border bg-surface p-0.5 text-[11px]">
              {OVERLAY_LABELS.map(({ key, label, title }) => (
                <button
                  key={key}
                  type="button"
                  title={title}
                  aria-pressed={overlays[key]}
                  onClick={() => toggleOverlay(key)}
                  className={`flex items-center gap-1 rounded px-2 py-1 font-medium transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                    overlays[key] ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <span className={`h-1.5 w-1.5 rounded-full ${overlays[key] ? "bg-primary" : "bg-border"}`} aria-hidden />
                  {label}
                </button>
              ))}
            </div>
          </div>
          <div className="ml-auto flex min-w-[160px] flex-1 items-center gap-2 sm:max-w-[260px]">
            <span className="font-mono tabular-nums text-muted-foreground">{heatVals.lo.toFixed(2)}</span>
            <div
              className="h-3 flex-1 rounded-sm border border-border"
              style={{
                background: `linear-gradient(to right, ${Array.from({ length: 13 }, (_, i) => heatColor(i / 12, palette, contrast)).join(",")})`,
              }}
            />
            <span className="font-mono tabular-nums text-muted-foreground">{heatVals.hi.toFixed(2)}</span>
          </div>
        </div>
      )}
      {mode === "policy" && (overlays.changed || overlays.support) && (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 px-1 text-[11px] text-muted-foreground" data-testid="overlay-legend">
          {overlays.changed && (
            <span className="flex items-center gap-1.5">
              <span className="h-3 w-3 rounded-sm border-2" style={{ borderColor: "var(--chart-5)" }} />
              greedy action changed (fades over {CHANGE_FADE} steps; dashed ghost = previous action) ·{" "}
              <span className="font-mono tabular-nums text-foreground" data-testid="changed-count">
                {tracker.changedNow.length}
              </span>{" "}
              changed at the last step, <span className="font-mono tabular-nums text-foreground">{recentChanges}</span> in the last {CHANGE_FADE}
            </span>
          )}
          {overlays.support && (
            <span className="flex items-center gap-1.5">
              <span className="h-3 w-3 rounded-sm border border-border" style={{ background: "repeating-linear-gradient(135deg, var(--card) 0 2px, transparent 2px 4px)" }} />
              fog = few logged samples of the greedy action (corner = count, log scale up to ×{maxCount.toLocaleString()})
            </span>
          )}
        </div>
      )}
      <div ref={boxRef} className="relative overflow-hidden rounded-lg border border-border bg-surface/60 p-3 lab-grid-bg">
        <svg
          viewBox={`-2 -2 ${W * cell + 4} ${H * cell + 4}`}
          className="mx-auto block max-h-[440px] w-full"
          onMouseMove={(e) => {
            const r = boxRef.current?.getBoundingClientRect();
            if (r) setTipPos({ x: e.clientX - r.left, y: e.clientY - r.top, w: r.width });
          }}
          onMouseLeave={() => {
            setHover(null);
            setTipPos(null);
          }}
        >
          {Array.from({ length: S }, (_, s) => {
            const x = (s % W) * cell;
            const y = Math.floor(s / W) * cell;
            const blocked = env.blocked.has(s);
            const term = env.terminals.get(s);
            const unseen = ds.visits[s] === 0 && !term;
            let fill = "var(--card)";
            if (blocked) fill = "color-mix(in oklch, var(--foreground) 55%, var(--background))";
            else if (term === "goal") fill = "color-mix(in oklch, var(--success) 28%, var(--card))";
            else if (term === "hazard") fill = "color-mix(in oklch, var(--destructive) 22%, var(--card))";
            else if (mode === "dataset") {
              const t = Math.sqrt(ds.visits[s] / maxVisit);
              fill = `color-mix(in oklch, var(--primary) ${Math.round(t * 34)}%, var(--card))`;
            } else if (!unseen) {
              fill = heatColor((heatVals.vals[s] - heatVals.lo) / (heatVals.hi - heatVals.lo), palette, contrast);
            }
            const supportCount = mode === "policy" && overlays.support && eligible[s] ? ds.counts[s * 4 + evalRes.policy[s]] : -1;
            return (
              <g key={s} onMouseEnter={() => setHover(s)}>
                <rect
                  x={x + 1.5}
                  y={y + 1.5}
                  width={cell - 3}
                  height={cell - 3}
                  rx={5}
                  fill={fill}
                  stroke={hover === s ? "var(--primary)" : "var(--border)"}
                  strokeWidth={hover === s ? 2 : 1}
                  className="transition-[fill,stroke] duration-500 ease-out"
                />
                {supportCount >= 0 && (
                  <>
                    <rect
                      x={x + 2}
                      y={y + 2}
                      width={cell - 4}
                      height={cell - 4}
                      rx={4.5}
                      fill="var(--card)"
                      opacity={0.8 * (1 - supportLevel(supportCount, maxCount))}
                      className="pointer-events-none transition-opacity duration-300"
                      data-support={supportCount}
                    />
                    <text
                      x={x + cell - 5}
                      y={y + cell - 5}
                      textAnchor="end"
                      fontSize="8.5"
                      fontWeight={600}
                      stroke="var(--card)"
                      strokeWidth={2.5}
                      paintOrder="stroke"
                      className={`pointer-events-none font-mono ${supportCount === 0 ? "fill-[var(--warning)]" : "fill-foreground"}`}
                    >
                      {supportCount === 0 ? "0" : `×${supportCount}`}
                    </text>
                  </>
                )}
                {unseen && !blocked && (
                  <path d={`M${x + 8} ${y + cell - 8} L${x + cell - 8} ${y + 8}`} stroke="var(--border)" strokeWidth="1" />
                )}
                {term && (
                  <text x={x + cell / 2} y={y + cell / 2 + 4} textAnchor="middle" fontSize="12" className="fill-foreground font-mono" fontWeight={600}>
                    {term === "goal" ? "G" : "×"}
                  </text>
                )}
                {s === env.start && !term && (
                  <circle cx={x + cell / 2} cy={y + cell / 2} r={cell / 2 - 6} fill="none" stroke="var(--foreground)" strokeWidth="1.25" strokeDasharray="3 3" />
                )}
                {/* Named cells (the route task's A, B, C): a small label in the cell's top-left corner. */}
                {env.landmarks?.filter((l) => l.cell === s).map((l, i) => (
                  <text
                    key={l.label}
                    x={x + 3 + i * 9}
                    y={y + 10}
                    fontSize="9"
                    fontWeight={700}
                    stroke="var(--card)"
                    strokeWidth={2.5}
                    paintOrder="stroke"
                    className="pointer-events-none fill-foreground font-mono"
                  >
                    {l.label}
                  </text>
                ))}
              </g>
            );
          })}

          {mode === "dataset" &&
            Array.from({ length: S * 4 }, (_, k) => {
              const c = ds.counts[k];
              if (!c) return null;
              const s = Math.floor(k / 4);
              const s2 = ds.modeNext[k];
              const r = ds.rewardMean[k];
              const col = rewardColor(r, ds.rewardMin, ds.rewardMax);
              const op = 0.35 + 0.65 * Math.min(1, Math.sqrt(c / 25));
              const w = 1 + Math.min(2.2, Math.sqrt(c) / 4);
              if (s2 === s || s2 < 0) {
                const a = k % 4;
                const ox = [0, 1, 0, -1][a] * 12;
                const oy = [-1, 0, 1, 0][a] * 12;
                return <circle key={k} cx={cx(s) + ox} cy={cy(s) + oy} r={2.2} fill={col} opacity={op} />;
              }
              const dx = cx(s2) - cx(s);
              const dy = cy(s2) - cy(s);
              const len = Math.hypot(dx, dy);
              const ux = dx / len;
              const uy = dy / len;
              // offset parallel arrows so opposing transitions don't overlap
              const nx = -uy * 3;
              const ny = ux * 3;
              return (
                <Arrow
                  key={k}
                  x1={cx(s) + ux * 6 + nx}
                  y1={cy(s) + uy * 6 + ny}
                  x2={cx(s) + dx * 0.72 + nx}
                  y2={cy(s) + dy * 0.72 + ny}
                  color={col}
                  width={w}
                  opacity={op}
                />
              );
            })}

          {mode === "policy" &&
            Array.from({ length: S }, (_, s) => {
              if (!eligible[s]) return null;
              const a = evalRes.policy[s];
              const ood = evalRes.oodStates.has(s);
              const recency = overlays.changed ? changeRecency(tracker, s, version, CHANGE_FADE) : 0;
              if (!overlays.arrows && !(overlays.ood && ood) && recency === 0) return null;
              const dx = DIR_X[a];
              const dy = DIR_Y[a];
              const L = cell * 0.3;
              const x = (s % W) * cell;
              const y = Math.floor(s / W) * cell;
              const t = (heatVals.vals[s] - heatVals.lo) / (heatVals.hi - heatVals.lo);
              const ink = heatScale(t, palette, contrast).L > 0.6 ? "oklch(0.2 0.03 280)" : "oklch(0.97 0.01 280)";
              const from = recency > 0 ? tracker.fromAction[s] : -1;
              const showOod = overlays.ood && ood;
              return (
                <g key={`p${s}`} className="pointer-events-none">
                  {showOod && (
                    <rect x={x + 4} y={y + 4} width={cell - 8} height={cell - 8} rx={4} fill="none" stroke="var(--warning)" strokeWidth="1.75" strokeDasharray="3 2" />
                  )}
                  {recency > 0 && (
                    <g opacity={0.25 + 0.75 * recency} data-changed={tracker.changedNow.includes(s) ? "now" : "recent"}>
                      <rect x={x + 2} y={y + 2} width={cell - 4} height={cell - 4} rx={4.5} fill="none" stroke="var(--background)" strokeWidth={4.5} />
                      <rect x={x + 2} y={y + 2} width={cell - 4} height={cell - 4} rx={4.5} fill="none" stroke="var(--chart-5)" strokeWidth={2.25} />
                      {from >= 0 && from !== a && (
                        <Arrow
                          x1={cx(s) - DIR_X[from] * L * 0.8}
                          y1={cy(s) - DIR_Y[from] * L * 0.8}
                          x2={cx(s) + DIR_X[from] * L * 0.8}
                          y2={cy(s) + DIR_Y[from] * L * 0.8}
                          color="var(--chart-5)"
                          width={1.2}
                          opacity={0.85}
                          dash="2.5 2"
                        />
                      )}
                    </g>
                  )}
                  {overlays.arrows && (
                    <Arrow x1={cx(s) - dx * L} y1={cy(s) - dy * L} x2={cx(s) + dx * L} y2={cy(s) + dy * L} color={showOod ? "var(--warning)" : ink} width={1.6} opacity={0.95} />
                  )}
                </g>
              );
            })}
        </svg>
        {hover !== null && tipPos && (
          <HoverTip s={hover} pos={tipPos} ds={ds} Q={Q} evalRes={evalRes} mode={mode} />
        )}
        <div className="pointer-events-none absolute bottom-2 left-3 font-mono text-[10px] text-muted-foreground">{env.axisX}</div>
        <div className="pointer-events-none absolute right-3 top-2 font-mono text-[10px] text-muted-foreground">{env.axisY}</div>
      </div>

      {/* Inspector strip */}
      <div className="min-h-[64px] rounded-lg border border-border bg-card px-4 py-3 text-xs">
        {hoverInfo === null ? (
          <p className="text-muted-foreground">
            Hover a state to inspect its logged transitions{mode === "policy" ? ", Q-values and greedy action" : " and rewards"}.
          </p>
        ) : (
          <CellInspector s={hoverInfo} ds={ds} Q={Q} evalRes={evalRes} mode={mode} changeNote={mode === "policy" ? describeChange(tracker, hoverInfo, version, env.actionNames) : null} />
        )}
      </div>
    </div>
  );
}

function HoverTip({ s, pos, ds, Q, evalRes, mode }: { s: number; pos: { x: number; y: number; w: number }; ds: Dataset; Q: Float64Array; evalRes: EvalResult; mode: "dataset" | "policy" }) {
  const { env } = ds;
  const x = s % env.width;
  const y = Math.floor(s / env.width);
  const term = env.terminals.get(s);
  const blocked = env.blocked.has(s);
  const b = s * 4;
  const qm = Math.max(Q[b], Q[b + 1], Q[b + 2], Q[b + 3]);
  const truth = evalRes.vPi[s];
  const flipLeft = pos.x > pos.w * 0.6;
  return (
    <div
      className="pointer-events-none absolute z-50 w-48 rounded-md border border-border bg-card px-3 py-2 text-[11px] text-card-foreground shadow-lg"
      style={{ left: flipLeft ? pos.x - 200 : pos.x + 14, top: Math.max(4, pos.y - 12) }}
    >
      <div className="mb-1 font-mono font-semibold tabular-nums">
        s=({x}, {y}) · n={ds.visits[s]}
      </div>
      {blocked ? (
        <div className="text-muted-foreground">Wall</div>
      ) : term ? (
        <div className="text-muted-foreground">{term === "goal" ? "Goal (terminal)" : "Hazard (terminal)"}</div>
      ) : mode === "policy" ? (
        <div className="space-y-0.5 font-mono tabular-nums">
          <div className="flex justify-between"><span className="text-muted-foreground">max Q</span>{qm.toFixed(4)}</div>
          <div className="flex justify-between"><span className="text-muted-foreground">True V^π</span>{truth.toFixed(4)}</div>
          <div className="flex justify-between"><span className="text-muted-foreground">Error</span>{(qm - truth).toFixed(4)}</div>
          <div className="flex justify-between"><span className="text-muted-foreground">Greedy</span>{env.actionNames[evalRes.policy[s]]}</div>
          <div className="mt-1 border-t border-border pt-1">
            {env.actionNames.map((n, a) => (
              <div key={a} className="flex justify-between">
                <span className="text-muted-foreground">{n}</span>
                <span>{Q[b + a].toFixed(3)}{ds.counts[b + a] ? <span className="ml-1 text-muted-foreground">×{ds.counts[b + a]}</span> : <span className="ml-1 text-warning">OOD</span>}</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="space-y-0.5 font-mono tabular-nums">
          {env.actionNames.map((n, a) => (
            <div key={a} className="flex justify-between">
              <span className="text-muted-foreground">{n}</span>
              <span>{ds.counts[b + a] ? `×${ds.counts[b + a]} · r̄ ${ds.rewardMean[b + a].toFixed(2)}` : "unseen"}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** "Greedy changed 2 steps ago (was Up)" for the hovered state, or null when it never changed while tracked. */
function describeChange(tr: PolicyChangeTracker, s: number, version: number, names: readonly string[]): string | null {
  const at = tr.lastChanged[s];
  if (at === undefined || at < 0) return null;
  const ago = version - at;
  const was = tr.fromAction[s] >= 0 ? names[tr.fromAction[s]] : "?";
  return `Greedy action changed ${ago === 0 ? "at the last training step" : `${ago} training step${ago === 1 ? "" : "s"} ago`} (was ${was})`;
}

function CellInspector({ s, ds, Q, evalRes, mode, changeNote }: { s: number; ds: Dataset; Q: Float64Array; evalRes: EvalResult; mode: "dataset" | "policy"; changeNote?: string | null }) {
  const { env } = ds;
  const x = s % env.width;
  const y = Math.floor(s / env.width);
  const term = env.terminals.get(s);
  const blocked = env.blocked.has(s);
  return (
    <div className="flex flex-wrap items-start gap-x-6 gap-y-2">
      <div>
        <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">State</div>
        <div className="font-mono tabular-nums text-foreground">
          s=({x}, {y}) · n={ds.visits[s]}
        </div>
        <div className="mt-0.5 text-muted-foreground">{blocked ? "Wall" : term === "goal" ? "Goal (terminal)" : term === "hazard" ? "Hazard (terminal)" : ds.visits[s] === 0 ? "Never visited in data" : "Logged state"}</div>
        {env.landmarks?.filter((l) => l.cell === s).map((l) => (
          <div key={l.label} className="mt-0.5 text-[11px] text-foreground">
            {l.hint ?? l.label}
          </div>
        ))}
        {changeNote && <div className="mt-0.5 text-[11px]" style={{ color: "var(--chart-5)" }}>{changeNote}</div>}
      </div>
      {!blocked && !term && (
        <div className="grid flex-1 grid-cols-4 gap-2">
          {env.actionNames.map((name, a) => {
            const k = s * 4 + a;
            const c = ds.counts[k];
            const greedyA = mode === "policy" && evalRes.policy[s] === a;
            return (
              <div key={a} className={`rounded-md border px-2 py-1.5 ${greedyA ? "border-primary bg-accent" : "border-border bg-surface/60"}`}>
                <div className="truncate text-[10px] text-muted-foreground">{name}</div>
                {mode === "dataset" ? (
                  <div className="font-mono tabular-nums">
                    {c ? (
                      <>
                        ×{c} <span className="text-muted-foreground">r̄</span> {ds.rewardMean[k].toFixed(2)}
                      </>
                    ) : (
                      <span className="text-warning">unseen</span>
                    )}
                  </div>
                ) : (
                  <div className="font-mono tabular-nums">
                    Q {Q[k].toFixed(2)} {c ? <span className="text-muted-foreground">×{c}</span> : <span className="text-warning">OOD</span>}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
