// Hardware safety for ThrusterChain: a limits editor, a chunked constraint audit of the trained
// policy (with and without a command-clipping shield) against the behavior recorder and coast
// baseline on the Physics test bench's episodes, and StressOverlay — per-channel |value|/limit
// bars for one playback frame.

import { useEffect, useMemo, useRef, useState } from "react";
import { KNEE, SPEC_PHYSICS, THRUSTER_PROJECTION, type BenchMode, type PhysicsOverrides, type ThrusterFrame } from "@/lib/thrusterchain";
import {
  auditEpisode,
  cloneLimits,
  DEFAULT_SAFETY_LIMITS,
  measureStress,
  STRESS_CHANNELS,
  summarizeAudit,
  VIOLATION_BIT,
  VIOLATION_LABELS,
  VIOLATION_TYPES,
  type AuditResult,
  type EpisodeAudit,
  type Five,
  type Quad,
  type SafetyLimits,
  type ViolationType,
} from "@/lib/safety";
import { RiskCharts, useContainerWidth } from "./risk-charts";

/* ---------------- Stress bars (shared by StressOverlay and the audit table) ---------------- */

const STRESS_CAP = 1.5;

function stressColor(v: number, counted: boolean) {
  if (!counted) return "var(--muted-foreground)";
  if (v > 1) return "var(--destructive)";
  if (v > 0.8) return "var(--warning)";
  return "var(--success)";
}

const GROUPS: { group: ViolationType; caption: string }[] = [
  { group: "command", caption: "|a| / rating" },
  { group: "stall", caption: "/ knee" },
  { group: "position", caption: "|p|" },
  { group: "velocity", caption: "|v|" },
  { group: "energy", caption: "Σx²" },
];

function StressBars({ values, stallCounted, titles, height = 44 }: { values: ArrayLike<number>; stallCounted: boolean; titles?: string[]; height?: number }) {
  const limitPct = (1 / STRESS_CAP) * 100;
  return (
    <div className="flex flex-wrap items-end gap-x-3 gap-y-2">
      {GROUPS.map(({ group, caption }) => {
        const idx = STRESS_CHANNELS.map((c, i) => (c.group === group ? i : -1)).filter((i) => i >= 0);
        const counted = group !== "stall" || stallCounted;
        return (
          <div key={group} className="flex flex-col items-center gap-1">
            <div className="flex items-end gap-[3px]">
              {idx.map((i) => {
                const v = values[i];
                const shown = Number.isFinite(v) ? Math.min(v, STRESS_CAP) : STRESS_CAP;
                const over = !(v <= STRESS_CAP);
                const ch = STRESS_CHANNELS[i];
                return (
                  <div key={ch.key} className="flex flex-col items-center gap-0.5">
                    <div className="relative w-2.5 overflow-hidden rounded-[3px] border border-border bg-surface" style={{ height }} title={titles?.[i] ?? `${ch.label}: ${Number.isFinite(v) ? v.toFixed(2) : "∞"} × limit`}>
                      <div className="absolute inset-x-0 bottom-0 transition-[height] duration-100" style={{ height: `${(shown / STRESS_CAP) * 100}%`, background: stressColor(v, counted), opacity: counted ? 0.9 : 0.45 }} />
                      <div className="absolute inset-x-0 border-t border-dashed border-foreground/50" style={{ bottom: `${limitPct}%` }} />
                      {over && counted && <div className="absolute inset-x-0 top-0 h-[3px] bg-destructive" />}
                    </div>
                    <span className="font-mono text-[8.5px] leading-none text-muted-foreground">{ch.label}</span>
                  </div>
                );
              })}
            </div>
            <span className="text-[9px] leading-none text-muted-foreground">
              {caption}
              {!counted ? " (info)" : ""}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/**
 * Compact per-channel stress bars for one playback frame: |a_k| / rating, max|a| / knee,
 * |p_i| / limit, |v_i| / limit and Σx² / limit. The dashed line is the limit (1.0); bars turn
 * amber above 0.8 and red past 1.0. Uses the frame's PRE-transition state and executed command,
 * the same pair the audit checks.
 */
export function StressOverlay({ frame, limits = DEFAULT_SAFETY_LIMITS, knee = KNEE }: { frame: ThrusterFrame; limits?: SafetyLimits; knee?: number }) {
  const { stress, mask } = measureStress(frame.x, frame.cmd, limits, knee);
  const violated = VIOLATION_TYPES.filter((t) => mask & VIOLATION_BIT[t]);
  const titles = STRESS_CHANNELS.map((ch, i) => {
    const s = stress[i];
    const val =
      ch.group === "command"
        ? `|a${i}| = ${Math.abs(frame.cmd[i]).toFixed(3)} / rating ${limits.maxCommand[i].toFixed(2)}`
        : ch.group === "stall"
          ? `max|a| = ${Math.max(...frame.cmd.map(Math.abs)).toFixed(3)} / knee ${knee.toFixed(2)}`
          : ch.group === "position"
            ? `|p${i - 4}| = ${Math.abs(frame.x[(i - 5) * 2]).toFixed(3)} / ${limits.maxPosition[i - 5].toFixed(2)}`
            : ch.group === "velocity"
              ? `|v${i - 9}| = ${Math.abs(frame.x[(i - 10) * 2 + 1]).toFixed(3)} / ${limits.maxVelocity[i - 10].toFixed(2)}`
              : `Σx² = ${(s * limits.maxEnergy).toFixed(3)} / ${limits.maxEnergy.toFixed(1)}`;
    return `${val} → ${Number.isFinite(s) ? (s * 100).toFixed(0) : "∞"}%`;
  });
  return (
    <div className="rounded-lg border border-border bg-surface/60 px-3 py-2" role="img" aria-label={violated.length ? `Safety limits exceeded: ${violated.map((t) => VIOLATION_LABELS[t]).join(", ")}` : "All channels within safety limits"}>
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Stress vs limits</span>
        {violated.length ? (
          <span className="rounded bg-destructive/15 px-1.5 py-0.5 text-[10px] font-semibold text-destructive">{violated.map((t) => VIOLATION_LABELS[t].toLowerCase()).join(" · ")}</span>
        ) : (
          <span className="rounded bg-success/15 px-1.5 py-0.5 text-[10px] font-semibold text-success">within limits</span>
        )}
      </div>
      <StressBars values={stress} stallCounted={limits.stallIsViolation} titles={titles} height={40} />
    </div>
  );
}

/* ---------------- Limits editor ---------------- */

function LimitSlider({ label, value, min, max, step, onChange, disabled, fmt }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; disabled: boolean; fmt: (v: number) => string }) {
  return (
    <label className="block">
      <div className="mb-1 flex items-center justify-between text-[11px]">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono tabular-nums text-foreground">{fmt(value)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-primary disabled:opacity-50" aria-label={label} />
    </label>
  );
}

function sameLimits(a: SafetyLimits, b: SafetyLimits) {
  return (
    a.stallIsViolation === b.stallIsViolation &&
    a.maxEnergy === b.maxEnergy &&
    a.maxCommand.every((v, i) => v === b.maxCommand[i]) &&
    a.maxPosition.every((v, i) => v === b.maxPosition[i]) &&
    a.maxVelocity.every((v, i) => v === b.maxVelocity[i])
  );
}

/* ---------------- Violation rate over time ---------------- */

type RunKey = "policy" | "shielded" | "behavior" | "coast";

interface Run {
  key: RunKey;
  label: string;
  hint: string;
  color: string;
  dashed?: boolean;
  result: AuditResult;
}

const CONTROLLERS: { key: RunKey; label: string; hint: string; color: string; dashed?: boolean; shield: boolean }[] = [
  { key: "policy", label: "Trained policy", hint: "greedy tabular action per (p₁,v₁) cell", color: "var(--chart-1)", shield: false },
  { key: "shielded", label: "Shielded policy", hint: "same policy, commands clipped to the limits", color: "var(--chart-2)", dashed: true, shield: true },
  { key: "behavior", label: "Behavior recorder", hint: "ε=0.25 random commands that made the data", color: "var(--chart-3)", shield: false },
  { key: "coast", label: "Coast baseline", hint: "all thrusters off — passive physics", color: "var(--chart-4)", shield: false },
];

type ChartType = "any" | ViolationType;

function RateChart({ runs, type, width }: { runs: Run[]; type: ChartType; width: number }) {
  const W = Math.max(280, width);
  const H = 180;
  const padL = 40;
  const padR = 12;
  const padT = 10;
  const padB = 24;
  const series = runs.map((r) => ({ ...r, values: type === "any" ? r.result.rateByT : r.result.rateByTypeT[type] }));
  const T = series[0]?.values.length ?? 0;
  const peak = Math.max(0, ...series.flatMap((s) => s.values));
  const yMax = [0.1, 0.25, 0.5, 1].find((v) => peak <= v) ?? 1;
  const sx = (t: number) => padL + (t / Math.max(1, T - 1)) * (W - padL - padR);
  const sy = (v: number) => padT + (1 - v / yMax) * (H - padT - padB);
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label={`Fraction of episodes violating ${type === "any" ? "any limit" : VIOLATION_LABELS[type]} at each timestep`}>
      {[0, 0.5, 1].map((f) => (
        <g key={f}>
          <line x1={padL} x2={W - padR} y1={sy(f * yMax)} y2={sy(f * yMax)} className="stroke-border" strokeWidth="1" />
          <text x={padL - 6} y={sy(f * yMax) + 3} textAnchor="end" className="fill-muted-foreground font-mono" fontSize="10">
            {Math.round(f * yMax * 100)}%
          </text>
        </g>
      ))}
      {[0, 25, 50, 75, T - 1].filter((t, i, a) => t >= 0 && a.indexOf(t) === i).map((t) => (
        <text key={t} x={sx(t)} y={H - 8} textAnchor={t === 0 ? "start" : t === T - 1 ? "end" : "middle"} className="fill-muted-foreground font-mono" fontSize="10">
          {t === T - 1 ? `t=${t}` : t}
        </text>
      ))}
      {series.map((s) => (
        <polyline
          key={s.key}
          points={s.values.map((v, t) => `${sx(t).toFixed(1)},${sy(v).toFixed(1)}`).join(" ")}
          fill="none"
          stroke={s.color}
          strokeWidth={s.key === "policy" ? 2 : 1.5}
          strokeDasharray={s.dashed ? "4 3" : undefined}
          strokeLinejoin="round"
          opacity={0.9}
        />
      ))}
    </svg>
  );
}

/* ---------------- SafetyPanel ---------------- */

const CHUNK = 10;
const POLICY_CELLS = THRUSTER_PROJECTION.xAxis.bins * THRUSTER_PROJECTION.yAxis.bins;
const pct = (v: number) => `${(v * 100).toFixed(v > 0 && v < 0.1 ? 1 : 0)}%`;

/**
 * Hardware safety constraints for ThrusterChain. Edit per-thruster command ratings, the stall
 * rule, link position/velocity limits and the energy cap, then audit the trained policy, the
 * same policy behind a clipping shield, the behavior recorder and the coast baseline over the
 * Physics test bench's paired episodes (same seeds and starts; returns match the bench when the
 * physics match). Work is chunked with setTimeout so the page stays responsive.
 */
export function SafetyPanel({ policy, trained, physics = SPEC_PHYSICS }: { policy: (cell: number) => number; trained: boolean; physics?: PhysicsOverrides }) {
  const [limits, setLimits] = useState<SafetyLimits>(() => cloneLimits(DEFAULT_SAFETY_LIMITS));
  const [episodes, setEpisodes] = useState(40);
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [ranWith, setRanWith] = useState<{ limits: SafetyLimits; physics: PhysicsOverrides; episodes: number } | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [chartType, setChartType] = useState<ChartType>("any");
  const [chartRef, chartW] = useContainerWidth();
  const cancelled = useRef(false);
  const timer = useRef<number | null>(null);

  useEffect(
    () => () => {
      cancelled.current = true;
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );

  const busy = progress !== null;
  const isDefault = sameLimits(limits, DEFAULT_SAFETY_LIMITS);
  const stale = ranWith !== null && (!sameLimits(limits, ranWith.limits) || ranWith.episodes !== episodes);

  const setCmd = (k: number, v: number) => setLimits((l) => ({ ...l, maxCommand: l.maxCommand.map((x, i) => (i === k ? v : x)) as Quad }));
  const setAll = (field: "maxPosition" | "maxVelocity", v: number) => setLimits((l) => ({ ...l, [field]: [v, v, v, v, v] as Five }));

  const run = () => {
    // Freeze the policy and settings now: training may keep mutating the Q-table while we audit.
    const table = new Int8Array(POLICY_CELLS);
    for (let c = 0; c < POLICY_CELLS; c++) table[c] = policy(c);
    const frozen = (c: number) => table[c];
    const lim = cloneLimits(limits);
    const ph = { ...physics };
    const n = episodes;
    const jobs = CONTROLLERS.map((c) => ({ ...c, mode: (c.key === "behavior" ? { mode: "behavior" } : c.key === "coast" ? { mode: "coast" } : { mode: "policy", policy: frozen }) as BenchMode, eps: [] as EpisodeAudit[] }));
    const total = jobs.length * n;
    let j = 0;
    cancelled.current = false;
    setProgress({ done: 0, total });

    const tick = () => {
      if (cancelled.current) return;
      const job = jobs[j];
      const end = Math.min(n, job.eps.length + CHUNK);
      for (let e = job.eps.length; e < end; e++) job.eps.push(auditEpisode(job.mode, e, ph, lim, { shield: job.shield }));
      const done = jobs.reduce((s, jb) => s + jb.eps.length, 0);
      if (job.eps.length >= n) j++;
      if (j >= jobs.length) {
        setRuns(jobs.map((jb) => ({ key: jb.key, label: jb.label, hint: jb.hint, color: jb.color, dashed: jb.dashed, result: summarizeAudit(jb.eps, { shielded: jb.shield }) })));
        setRanWith({ limits: lim, physics: ph, episodes: n });
        setProgress(null);
        timer.current = null;
        return;
      }
      setProgress({ done, total });
      timer.current = window.setTimeout(tick, 0);
    };
    timer.current = window.setTimeout(tick, 16);
  };

  const stop = () => {
    cancelled.current = true;
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = null;
    setProgress(null);
  };

  const policyRun = runs?.find((r) => r.key === "policy");
  const shieldRun = runs?.find((r) => r.key === "shielded");
  const riskSeries = useMemo(() => (runs ? runs.map((r) => ({ label: r.label, hint: r.hint, color: r.color, values: r.result.returns })) : []), [runs]);
  const shownLimits = ranWith?.limits ?? limits;

  return (
    <section className="rounded-xl border border-border bg-card p-4 shadow-sm">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-[13px] font-semibold text-foreground">Hardware safety limits</h3>
          <p className="mt-0.5 max-w-xl text-[11px] leading-relaxed text-muted-foreground">
            Set the limits the real thruster chain must respect — gear ratings, stall, link travel and speed, total oscillation energy — then audit every controller on the test bench's paired episodes. The shield clips commands to the limits; it cannot stop state limits from being breached.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {!trained && <span className="rounded border border-warning/40 bg-warning/10 px-2 py-0.5 text-[10.5px] text-foreground">Untrained Q — policy is arbitrary</span>}
          <button
            type="button"
            disabled={busy || isDefault}
            onClick={() => setLimits(cloneLimits(DEFAULT_SAFETY_LIMITS))}
            className="rounded-md border border-border bg-card px-2.5 py-1 text-[11px] font-medium text-muted-foreground transition-colors duration-200 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-45"
          >
            Reset to defaults
          </button>
        </div>
      </div>

      <fieldset className="rounded-lg border border-border bg-card/50 p-3.5">
        <legend className="px-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Thruster gear ratings — max |command|</legend>
        <div className="grid grid-cols-2 gap-x-5 gap-y-3 sm:grid-cols-4">
          {limits.maxCommand.map((v, k) => (
            <LimitSlider key={k} label={`Thruster a${k}`} value={v} min={0.1} max={1} step={0.05} disabled={busy} fmt={(x) => x.toFixed(2)} onChange={(x) => setCmd(k, x)} />
          ))}
        </div>
        <label className="mt-3 flex items-start gap-2 text-[11px] text-muted-foreground">
          <input type="checkbox" checked={limits.stallIsViolation} disabled={busy} onChange={(e) => setLimits((l) => ({ ...l, stallIsViolation: e.target.checked }))} className="mt-0.5 accent-primary" />
          <span>
            Count stall-knee exceedance (|a| &gt; {physics.knee.toFixed(2)}) as a violation — thrust collapses past the knee and loads the gears. The policy's Burst action (0.95) always stalls.
          </span>
        </label>
      </fieldset>

      <fieldset className="mt-3 rounded-lg border border-border bg-card/50 p-3.5">
        <legend className="px-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Structure limits — all 5 links</legend>
        <div className="grid grid-cols-1 gap-x-5 gap-y-3 sm:grid-cols-3">
          <LimitSlider label="Max |link position|" value={limits.maxPosition[0]} min={0.5} max={3} step={0.1} disabled={busy} fmt={(x) => x.toFixed(1)} onChange={(x) => setAll("maxPosition", x)} />
          <LimitSlider label="Max |link velocity|" value={limits.maxVelocity[0]} min={0.3} max={3} step={0.1} disabled={busy} fmt={(x) => x.toFixed(1)} onChange={(x) => setAll("maxVelocity", x)} />
          <LimitSlider label="Max energy Σx[0..9]²" value={limits.maxEnergy} min={1} max={20} step={0.5} disabled={busy} fmt={(x) => x.toFixed(1)} onChange={(x) => setLimits((l) => ({ ...l, maxEnergy: x }))} />
        </div>
      </fieldset>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-[11px] text-muted-foreground">
          Episodes
          <select
            value={episodes}
            disabled={busy}
            onChange={(e) => setEpisodes(Number(e.target.value))}
            className="h-8 rounded-md border border-input bg-background px-2.5 text-[11px] text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            {[20, 40, 80, 160].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        {busy ? (
          <button type="button" onClick={stop} className="flex items-center gap-2 rounded-md border border-border bg-card px-4 py-2 text-[12px] font-medium text-foreground transition-colors duration-200 hover:bg-muted">
            <span className="h-3 w-3 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            Stop ({progress.done}/{progress.total} episodes)
          </button>
        ) : (
          <button type="button" onClick={run} className="rounded-md bg-primary px-5 py-2 text-[12px] font-semibold text-primary-foreground shadow-md transition-all duration-200 hover:bg-primary/90 active:scale-95">
            Run safety audit
          </button>
        )}
        {stale && !busy && <span className="text-[10.5px] text-warning-foreground dark:text-warning">Limits or episodes changed — re-run to update.</span>}
      </div>

      <div className="mt-4">
        {!runs ? (
          <div className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-[11.5px] text-muted-foreground">
            No audit yet. Press <span className="font-medium text-foreground">Run safety audit</span> to count limit violations for each controller in real physics.
          </div>
        ) : (
          <div className="space-y-4">
            {ranWith && (
              <p className="text-[11px] text-muted-foreground">
                {ranWith.episodes} episodes × {runs[0].result.horizon} steps · bench seeds {runs[0].result.baseSeed}–{runs[0].result.baseSeed + ranWith.episodes - 1} · knee {ranWith.physics.knee.toFixed(2)} · current {ranWith.physics.currentScale.toFixed(1)}× · waves {ranWith.physics.waveScale.toFixed(1)}×
                {policyRun && shieldRun && (
                  <>
                    {" · "}
                    <span className={shieldRun.result.stepViolationRate <= policyRun.result.stepViolationRate ? "text-success" : "text-destructive"}>
                      shield: {pct(policyRun.result.stepViolationRate)} → {pct(shieldRun.result.stepViolationRate)} violating steps, return {policyRun.result.meanReturn.toFixed(1)} → {shieldRun.result.meanReturn.toFixed(1)}
                    </span>
                  </>
                )}
              </p>
            )}

            <div>
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <h4 className="text-[12px] font-semibold text-foreground">Violation rate over time</h4>
                <div className="flex flex-wrap rounded-md border border-border bg-surface p-0.5 text-[11px]" role="radiogroup" aria-label="Violation type">
                  {(["any", ...VIOLATION_TYPES] as ChartType[]).map((t) => (
                    <button
                      key={t}
                      type="button"
                      role="radio"
                      aria-checked={chartType === t}
                      onClick={() => setChartType(t)}
                      className={`rounded px-2 py-0.5 font-medium transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${chartType === t ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
                    >
                      {t === "any" ? "Any" : t === "command" ? "Command" : t === "stall" ? "Stall" : t === "position" ? "|p|" : t === "velocity" ? "|v|" : "Energy"}
                    </button>
                  ))}
                </div>
              </div>
              <div className="rounded-lg border border-border bg-background p-2">
                <div ref={chartRef}>
                  <RateChart runs={runs} type={chartType} width={chartW} />
                </div>
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[10.5px] text-muted-foreground">
                {runs.map((r) => (
                  <span key={r.key} className="flex items-center gap-1.5">
                    <svg width="16" height="6" aria-hidden>
                      <line x1="0" x2="16" y1="3" y2="3" stroke={r.color} strokeWidth="2" strokeDasharray={r.dashed ? "4 3" : undefined} />
                    </svg>
                    {r.label}
                  </span>
                ))}
                <span>· share of episodes violating at step t; t = 0 is the reset state, which no controller can change.</span>
              </div>
            </div>

            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-left text-[11.5px]">
                <thead>
                  <tr className="whitespace-nowrap border-b border-border bg-muted/60 text-[10px] uppercase tracking-wider text-muted-foreground">
                    <th className="min-w-[160px] px-3 py-2 font-semibold">Controller</th>
                    <th className="px-2 py-2 text-right font-semibold" title="Episodes with at least one violation">Episodes hit</th>
                    <th className="px-2 py-2 text-right font-semibold" title="Share of all steps with any violation">Steps hit</th>
                    <th className="px-2 py-2 text-right font-semibold" title="Mean first-violation step over episodes that violated">First t</th>
                    {VIOLATION_TYPES.map((t) => (
                      <th key={t} className={`px-2 py-2 text-right font-semibold ${t === "position" || t === "velocity" ? "normal-case" : ""}`} title={`${VIOLATION_LABELS[t]}: share of steps`}>
                        {t === "command" ? "Cmd" : t === "stall" ? "Stall" : t === "position" ? "|p|" : t === "velocity" ? "|v|" : "Energy"}
                      </th>
                    ))}
                    <th className="px-2 py-2 text-right font-semibold" title="Worst |value|/limit over all channels and episodes">Peak stress</th>
                    <th className="px-2 py-2 text-right font-semibold">Mean return</th>
                    <th className="px-2 py-2 text-right font-semibold" title="Share of steps where the shield clipped a command">Clipped</th>
                  </tr>
                </thead>
                <tbody className="font-mono tabular-nums">
                  {runs.map((r) => {
                    const a = r.result;
                    let peakIdx = 0;
                    a.peakStress.forEach((v, i) => {
                      if ((STRESS_CHANNELS[i].group !== "stall" || shownLimits.stallIsViolation) && v > a.peakStress[peakIdx]) peakIdx = i;
                    });
                    const peak = a.peakStress[peakIdx];
                    return (
                      <tr key={r.key} className="border-b border-border/60 last:border-0">
                        <td className="px-3 py-2 font-sans">
                          <div className="flex items-center gap-1.5 font-medium text-foreground">
                            <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: r.color }} />
                            {r.label}
                          </div>
                          <div className="text-[10px] text-muted-foreground">{r.hint}</div>
                        </td>
                        <td className={`px-2 py-2 text-right ${a.episodeViolationRate > 0 ? "text-foreground" : "text-success"}`}>{pct(a.episodeViolationRate)}</td>
                        <td className={`px-2 py-2 text-right ${a.stepViolationRate > 0.1 ? "text-destructive" : a.stepViolationRate > 0 ? "text-foreground" : "text-success"}`}>{pct(a.stepViolationRate)}</td>
                        <td className="px-2 py-2 text-right text-muted-foreground">{a.meanFirstViolation === null ? "—" : a.meanFirstViolation.toFixed(1)}</td>
                        {VIOLATION_TYPES.map((t) => (
                          <td key={t} className={`px-2 py-2 text-right ${a.stepViolationRateByType[t] > 0 ? "text-foreground" : "text-muted-foreground/60"}`}>
                            {a.stepViolationRateByType[t] > 0 ? pct(a.stepViolationRateByType[t]) : "0"}
                          </td>
                        ))}
                        <td className={`px-2 py-2 text-right ${peak > 1 ? "text-destructive" : "text-muted-foreground"}`}>
                          {peak.toFixed(2)}× <span className="text-[10px] text-muted-foreground">{STRESS_CHANNELS[peakIdx].label}</span>
                        </td>
                        <td className="px-2 py-2 text-right text-foreground">{a.meanReturn.toFixed(1)}</td>
                        <td className="px-2 py-2 text-right text-muted-foreground">{a.shielded ? pct(a.shieldInterventionRate) : "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div>
              <h4 className="mb-2 text-[12px] font-semibold text-foreground">Worst stress per channel</h4>
              <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
                {runs.map((r) => (
                  <div key={r.key} className="rounded-lg border border-border bg-surface/60 px-3 py-2">
                    <div className="mb-1.5 flex items-center gap-1.5 text-[11px] font-medium text-foreground">
                      <span className="h-2 w-2 rounded-full" style={{ background: r.color }} />
                      {r.label}
                    </div>
                    <StressBars
                      values={r.result.peakStress}
                      stallCounted={shownLimits.stallIsViolation}
                      height={36}
                      titles={STRESS_CHANNELS.map((ch, i) => `${ch.label}: worst ${r.result.peakStress[i].toFixed(2)}× limit, mean episode peak ${r.result.meanPeakStress[i].toFixed(2)}×`)}
                    />
                  </div>
                ))}
              </div>
              <p className="mt-2 text-[10.5px] leading-relaxed text-muted-foreground">
                Bar height = max |value| / limit over all audited episodes (dashed line = limit, capped at {STRESS_CAP}×). Thrusters a1–a3 stay at zero under the policy and coast baselines — the tabular policy only drives thruster a0.
              </p>
            </div>

            <div>
              <h4 className="mb-2 text-[12px] font-semibold text-foreground">Return risk on the same episodes</h4>
              <RiskCharts series={riskSeries} level={0.1} />
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
