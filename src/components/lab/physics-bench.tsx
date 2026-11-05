// Physics test bench: evaluates the trained tabular policy against the REAL 13-dim
// ThrusterChain dynamics over many paired episodes, with optional physics perturbations
// (stall knee, slope, current, waves) to stress-test transfer beyond the frozen spec.

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { benchmarkThruster, SPEC_PHYSICS, type BenchMode, type BenchStats, type PhysicsOverrides } from "@/lib/thrusterchain";
import { guardPhysics } from "@/lib/guards";
import { recordError } from "@/lib/error-log";

type Row = { key: string; label: string; hint: string; stats: BenchStats };

const PRESETS: { id: string; label: string; ph: PhysicsOverrides }[] = [
  { id: "spec", label: "Spec (frozen v4)", ph: SPEC_PHYSICS },
  { id: "harsh-stall", label: "Harsher stall", ph: { ...SPEC_PHYSICS, knee: 0.6, slope: 2.5 } },
  { id: "rough-sea", label: "Rough sea", ph: { ...SPEC_PHYSICS, currentScale: 1.8, waveScale: 2 } },
  { id: "calm", label: "Calm water", ph: { ...SPEC_PHYSICS, currentScale: 0.3, waveScale: 0.3 } },
];

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  disabled,
  fmt,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  disabled: boolean;
  fmt: (v: number) => string;
}) {
  return (
    <label className="block">
      <div className="mb-1 flex items-center justify-between text-[11px]">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono tabular-nums text-foreground">{fmt(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-primary disabled:opacity-50"
      />
    </label>
  );
}

export function PhysicsBench({ policy, trained }: { policy: (cell: number) => number; trained: boolean }) {
  const [ph, setPh] = useState<PhysicsOverrides>({ ...SPEC_PHYSICS });
  const [episodes, setEpisodes] = useState(40);
  const [rows, setRows] = useState<Row[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [ranWith, setRanWith] = useState<PhysicsOverrides | null>(null);

  const isSpec =
    ph.knee === SPEC_PHYSICS.knee &&
    ph.slope === SPEC_PHYSICS.slope &&
    ph.currentScale === 1 &&
    ph.waveScale === 1 &&
    ph.energyWeight === 1 &&
    ph.cmdWeight === SPEC_PHYSICS.cmdWeight &&
    ph.stallPenalty === 0;

  const run = () => {
    setBusy(true);
    const snapshot = { ...ph };
    window.setTimeout(() => {
      // Physics guard: a throw or a non-finite result reports a clear error instead of a broken table.
      const bench = (mode: BenchMode, label: string) =>
        guardPhysics(`Physics bench (${label})`, () => benchmarkThruster(mode, episodes, snapshot), (st) => [st.meanReturn, st.stdReturn, st.finalEnergy, ...st.returns]);
      try {
        const out: Row[] = [
          { key: "policy", label: "Trained policy", hint: "greedy tabular action per (p₁,v₁) cell", stats: bench({ mode: "policy", policy }, "trained policy") },
          { key: "behavior", label: "Behavior recorder", hint: "ε=0.25 random commands that made the data", stats: bench({ mode: "behavior" }, "behavior recorder") },
          { key: "coast", label: "Coast baseline", hint: "all thrusters off — passive physics", stats: bench({ mode: "coast" }, "coast") },
        ];
        setRows(out);
        setRanWith(snapshot);
      } catch (e) {
        recordError("physics", e, `knee ${snapshot.knee} · slope ${snapshot.slope} · current ×${snapshot.currentScale} · waves ×${snapshot.waveScale}`);
        toast.error(e instanceof Error ? e.message : "The physics bench failed.");
      } finally {
        setBusy(false);
      }
    }, 20);
  };

  const best = rows ? Math.max(...rows.map((r) => r.stats.meanReturn)) : 0;
  const lo = rows ? Math.min(...rows.flatMap((r) => r.stats.returns)) : 0;
  const hi = rows ? Math.max(...rows.flatMap((r) => r.stats.returns)) : 1;
  const span = hi - lo || 1;
  const policyRow = rows?.find((r) => r.key === "policy");
  const behaviorRow = rows?.find((r) => r.key === "behavior");
  const lift = policyRow && behaviorRow ? policyRow.stats.meanReturn - behaviorRow.stats.meanReturn : 0;

  return (
    <section className="rounded-xl border border-border bg-card p-4 shadow-sm">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-[13px] font-semibold text-foreground">Physics test bench</h3>
          <p className="mt-0.5 max-w-xl text-[11px] leading-relaxed text-muted-foreground">
            Runs the trained policy in the exact 13-dim float64 ThrusterChain system across many paired episodes (same starting states for every controller). Perturb the physics to check whether the policy holds up outside the frozen spec.
          </p>
        </div>
        {!trained && (
          <span className="rounded border border-warning/40 bg-warning/10 px-2 py-0.5 text-[10.5px] text-foreground">Untrained Q — policy is arbitrary</span>
        )}
      </div>

      <div className="mb-3 flex flex-wrap gap-1.5">
        {PRESETS.map((p) => {
          const active = p.ph.knee === ph.knee && p.ph.slope === ph.slope && p.ph.currentScale === ph.currentScale && p.ph.waveScale === ph.waveScale;
          return (
            <button
              key={p.id}
              type="button"
              disabled={busy}
              onClick={() => setPh({ ...p.ph })}
              className={`rounded-md border px-3 py-1.5 text-[11px] font-medium transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50 ${
                active ? "border-primary bg-primary/10 text-primary" : "border-border bg-card text-muted-foreground hover:text-foreground"
              }`}
            >
              {p.label}
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 gap-x-5 gap-y-3 sm:grid-cols-2">
        <Slider label="Stall knee |a|" value={ph.knee} min={0.4} max={1} step={0.05} disabled={busy} fmt={(v) => v.toFixed(2)} onChange={(v) => setPh((p) => ({ ...p, knee: v }))} />
        <Slider label="Stall slope" value={ph.slope} min={0} max={3} step={0.1} disabled={busy} fmt={(v) => v.toFixed(1)} onChange={(v) => setPh((p) => ({ ...p, slope: v }))} />
        <Slider label="Ocean current ×" value={ph.currentScale} min={0} max={2.5} step={0.1} disabled={busy} fmt={(v) => `${v.toFixed(1)}×`} onChange={(v) => setPh((p) => ({ ...p, currentScale: v }))} />
        <Slider label="Wave amplitude ×" value={ph.waveScale} min={0} max={2.5} step={0.1} disabled={busy} fmt={(v) => `${v.toFixed(1)}×`} onChange={(v) => setPh((p) => ({ ...p, waveScale: v }))} />
      </div>

      <fieldset className="mt-4 rounded-lg border border-border bg-card/50 p-3.5">
        <legend className="px-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Reward logic &amp; success criterion</legend>
        <p className="mb-3 text-[10.5px] leading-relaxed text-muted-foreground">
          Redefine what the evaluation reward punishes — state energy, command effort, and an optional extra penalty per stalled step — and set the final-energy bar an episode must beat to count as a success. Reward edits only affect evaluation here; the policy was trained on the spec reward.
        </p>
        <div className="grid grid-cols-1 gap-x-5 gap-y-3 sm:grid-cols-2">
          <Slider label="State-energy weight" value={ph.energyWeight} min={0} max={3} step={0.1} disabled={busy} fmt={(v) => v.toFixed(1)} onChange={(v) => setPh((p) => ({ ...p, energyWeight: v }))} />
          <Slider label="Command-effort weight" value={ph.cmdWeight} min={0} max={1.5} step={0.05} disabled={busy} fmt={(v) => v.toFixed(2)} onChange={(v) => setPh((p) => ({ ...p, cmdWeight: v }))} />
          <Slider label="Stall penalty / step" value={ph.stallPenalty} min={0} max={5} step={0.25} disabled={busy} fmt={(v) => v.toFixed(2)} onChange={(v) => setPh((p) => ({ ...p, stallPenalty: v }))} />
          <Slider label="Success: final energy <" value={ph.successEnergy} min={0.1} max={5} step={0.1} disabled={busy} fmt={(v) => v.toFixed(1)} onChange={(v) => setPh((p) => ({ ...p, successEnergy: v }))} />
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
        <button
          type="button"
          onClick={run}
          disabled={busy}
          className="flex items-center gap-2 rounded-md bg-primary px-5 py-2 text-[12px] font-semibold text-primary-foreground shadow-md transition-all duration-200 hover:bg-primary/90 active:scale-95 disabled:opacity-60"
        >
          {busy && <span className="h-3 w-3 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />}
          {busy ? "Simulating…" : "Run test bench"}
        </button>
        {!isSpec && <span className="text-[10.5px] text-warning-foreground dark:text-warning">Perturbed physics or reward — the policy was trained on spec data.</span>}
      </div>

      <div className="mt-4">
        {!rows ? (
          <div className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-[11.5px] text-muted-foreground">
            No results yet. Press <span className="font-medium text-foreground">Run test bench</span> to evaluate the policy against the baselines in real physics.
          </div>
        ) : (
          <div className="space-y-3">
            {ranWith && (
              <p className="text-[11px] text-muted-foreground">
                {rows[0].stats.episodes} episodes × 100 steps · knee {ranWith.knee.toFixed(2)} · slope {ranWith.slope.toFixed(1)} · current {ranWith.currentScale.toFixed(1)}× · waves {ranWith.waveScale.toFixed(1)}× · reward {ranWith.energyWeight.toFixed(1)}·energy + {ranWith.cmdWeight.toFixed(2)}·effort{ranWith.stallPenalty > 0 ? ` + ${ranWith.stallPenalty.toFixed(2)}/stall` : ""} · success: energy &lt; {ranWith.successEnergy.toFixed(1)} ·{" "}
                <span className={lift >= 0 ? "text-success" : "text-destructive"}>
                  policy vs recorder: {lift >= 0 ? "+" : ""}
                  {lift.toFixed(1)} return
                </span>
              </p>
            )}
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-left text-[11.5px]">
                <thead>
                  <tr className="border-b border-border bg-muted/60 text-[10px] uppercase tracking-wider text-muted-foreground">
                    <th className="px-3 py-2 font-semibold">Controller</th>
                    <th className="px-3 py-2 text-right font-semibold">Mean return</th>
                    <th className="px-3 py-2 text-right font-semibold">± std</th>
                    <th className="px-3 py-2 text-right font-semibold">Success</th>
                    <th className="px-3 py-2 text-right font-semibold">Stall rate</th>
                    <th className="px-3 py-2 text-right font-semibold">Mean |p₁|</th>
                    <th className="px-3 py-2 text-right font-semibold">Final energy</th>
                    <th className="px-3 py-2 font-semibold">Return distribution</th>
                  </tr>
                </thead>
                <tbody className="font-mono tabular-nums">
                  {rows.map((r) => (
                    <tr key={r.key} className={`border-b border-border/60 last:border-0 ${r.stats.meanReturn === best ? "bg-primary/5" : ""}`}>
                      <td className="px-3 py-2 font-sans">
                        <div className="font-medium text-foreground">
                          {r.label}
                          {r.stats.meanReturn === best && <span className="ml-1.5 rounded bg-primary/15 px-1.5 py-0.5 text-[10px] font-semibold text-primary">best</span>}
                        </div>
                        <div className="text-[10px] text-muted-foreground">{r.hint}</div>
                      </td>
                      <td className="px-3 py-2 text-right text-foreground">{r.stats.meanReturn.toFixed(1)}</td>
                      <td className="px-3 py-2 text-right text-muted-foreground">{r.stats.stdReturn.toFixed(1)}</td>
                      <td className={`px-3 py-2 text-right ${r.stats.successRate >= 0.5 ? "text-success" : "text-muted-foreground"}`}>{(r.stats.successRate * 100).toFixed(0)}%</td>
                      <td className={`px-3 py-2 text-right ${r.stats.stallRate > 0.1 ? "text-warning-foreground dark:text-warning" : "text-muted-foreground"}`}>{(r.stats.stallRate * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2 text-right text-muted-foreground">{r.stats.meanAbsP1.toFixed(3)}</td>
                      <td className="px-3 py-2 text-right text-muted-foreground">{r.stats.finalEnergy.toFixed(2)}</td>
                      <td className="px-3 py-2">
                        <svg width="140" height="18" viewBox="0 0 140 18" aria-label={`Returns from ${r.stats.minReturn.toFixed(1)} to ${r.stats.maxReturn.toFixed(1)}`}>
                          <line x1="0" y1="9" x2="140" y2="9" className="stroke-border" strokeWidth="1" />
                          {r.stats.returns.map((v, i) => (
                            <circle key={i} cx={2 + ((v - lo) / span) * 136} cy="9" r="2" className="fill-primary" opacity="0.35" />
                          ))}
                          <line x1={2 + ((r.stats.meanReturn - lo) / span) * 136} x2={2 + ((r.stats.meanReturn - lo) / span) * 136} y1="2" y2="16" className="stroke-foreground" strokeWidth="1.5" />
                        </svg>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-[10.5px] leading-relaxed text-muted-foreground">
              Higher (less negative) return is better. Stall rate counts steps where thruster 1 exceeded the knee; the tabular policy only commands thruster 1, so thrusters 2–4 stay at zero under the policy and coast baselines.
            </p>
          </div>
        )}
      </div>

      <RobustnessSweep policy={policy} trained={trained} />
    </section>
  );
}

/* ---------------- Robustness sweep ---------------- */

type ParamKey = keyof PhysicsOverrides;

const SWEEP_PARAMS: { key: ParamKey; label: string; min: number; max: number; fmt: (v: number) => string }[] = [
  { key: "knee", label: "Stall knee |a|", min: 0.4, max: 1, fmt: (v) => v.toFixed(2) },
  { key: "slope", label: "Stall slope", min: 0, max: 3, fmt: (v) => v.toFixed(1) },
  { key: "currentScale", label: "Ocean current ×", min: 0, max: 2.5, fmt: (v) => `${v.toFixed(1)}×` },
  { key: "waveScale", label: "Wave amplitude ×", min: 0, max: 2.5, fmt: (v) => `${v.toFixed(1)}×` },
];

type SweepPoint = { value: number; policy: number; behavior: number; coast: number };

const SWEEP_STEPS = 7;
const SWEEP_EPISODES = 24;

function RobustnessSweep({ policy, trained }: { policy: (cell: number) => number; trained: boolean }) {
  const [paramKey, setParamKey] = useState<ParamKey>("knee");
  const [points, setPoints] = useState<SweepPoint[]>([]);
  const [sweptKey, setSweptKey] = useState<ParamKey | null>(null);
  const [busy, setBusy] = useState(false);
  const cancelled = useRef(false);
  const timer = useRef<number | null>(null);

  useEffect(
    () => () => {
      cancelled.current = true;
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );

  const param = SWEEP_PARAMS.find((p) => p.key === paramKey) ?? SWEEP_PARAMS[0];
  const shownParam = SWEEP_PARAMS.find((p) => p.key === sweptKey) ?? param;

  const run = () => {
    const p = param;
    const values = Array.from({ length: SWEEP_STEPS }, (_, i) => p.min + ((p.max - p.min) * i) / (SWEEP_STEPS - 1));
    cancelled.current = false;
    setBusy(true);
    setPoints([]);
    setSweptKey(p.key);
    const acc: SweepPoint[] = [];
    const tick = (i: number) => {
      if (cancelled.current || i >= values.length) {
        setBusy(false);
        timer.current = null;
        return;
      }
      const ph: PhysicsOverrides = { ...SPEC_PHYSICS, [p.key]: values[i] };
      acc.push({
        value: values[i],
        policy: benchmarkThruster({ mode: "policy", policy }, SWEEP_EPISODES, ph).meanReturn,
        behavior: benchmarkThruster({ mode: "behavior" }, SWEEP_EPISODES, ph).meanReturn,
        coast: benchmarkThruster({ mode: "coast" }, SWEEP_EPISODES, ph).meanReturn,
      });
      setPoints([...acc]);
      timer.current = window.setTimeout(() => tick(i + 1), 16);
    };
    timer.current = window.setTimeout(() => tick(0), 16);
  };

  const stop = () => {
    cancelled.current = true;
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = null;
    setBusy(false);
  };

  // Chart geometry
  const Wc = 560;
  const Hc = 190;
  const padL = 44;
  const padR = 12;
  const padT = 12;
  const padB = 26;
  const all = points.flatMap((pt) => [pt.policy, pt.behavior, pt.coast]);
  const yMin = all.length ? Math.min(...all) : -1;
  const yMax = all.length ? Math.max(...all) : 0;
  const ySpan = yMax - yMin || 1;
  const sx = (v: number) => padL + ((v - shownParam.min) / (shownParam.max - shownParam.min || 1)) * (Wc - padL - padR);
  const sy = (v: number) => padT + (1 - (v - yMin) / ySpan) * (Hc - padT - padB);
  const line = (k: "policy" | "behavior" | "coast") => points.map((pt) => `${sx(pt.value).toFixed(1)},${sy(pt[k]).toFixed(1)}`).join(" ");
  const specX = sx(SPEC_PHYSICS[shownParam.key]);

  const wins = points.filter((pt) => pt.policy >= pt.behavior).length;
  const worst = points.length ? points.reduce((a, b) => (b.policy < a.policy ? b : a)) : null;
  const bestPt = points.length ? points.reduce((a, b) => (b.policy > a.policy ? b : a)) : null;
  const done = !busy && points.length === SWEEP_STEPS;

  return (
    <div className="mt-6 border-t border-border pt-4">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h4 className="text-[12.5px] font-semibold text-foreground">Robustness sweep</h4>
          <p className="mt-0.5 max-w-xl text-[11px] leading-relaxed text-muted-foreground">
            Sweeps one physics parameter across {SWEEP_STEPS} settings ({SWEEP_EPISODES} paired episodes each, other parameters held at spec) and plots mean return for every controller — showing exactly where the learned policy stops transferring.
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Parameter to sweep">
          {SWEEP_PARAMS.map((p) => (
            <button
              key={p.key}
              type="button"
              role="radio"
              aria-checked={p.key === paramKey}
              disabled={busy}
              onClick={() => setParamKey(p.key)}
              className={`rounded-md border px-3 py-1.5 text-[11px] font-medium transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50 ${
                p.key === paramKey ? "border-primary bg-primary/10 text-primary" : "border-border bg-card text-muted-foreground hover:text-foreground"
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
        {busy ? (
          <button
            type="button"
            onClick={stop}
            className="flex items-center gap-2 rounded-md border border-border bg-card px-4 py-1.5 text-[11.5px] font-medium text-foreground transition-colors duration-200 hover:bg-muted"
          >
            <span className="h-3 w-3 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            Stop ({points.length}/{SWEEP_STEPS})
          </button>
        ) : (
          <button
            type="button"
            onClick={run}
            className="rounded-md bg-primary px-4 py-1.5 text-[11.5px] font-semibold text-primary-foreground shadow-sm transition-all duration-200 hover:bg-primary/90 active:scale-95"
          >
            Sweep {param.label.toLowerCase()}
          </button>
        )}
        {!trained && <span className="text-[10.5px] text-muted-foreground">Tip: train first — an untrained policy is arbitrary.</span>}
      </div>

      <div className="mt-3">
        {points.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-[11.5px] text-muted-foreground">
            Pick a parameter and run a sweep to chart policy return against {param.label.toLowerCase()} from {param.fmt(param.min)} to {param.fmt(param.max)}.
          </div>
        ) : (
          <div className="space-y-2">
            <div className="rounded-lg border border-border bg-background p-2">
              <svg viewBox={`0 0 ${Wc} ${Hc}`} className="h-auto w-full" role="img" aria-label={`Mean return vs ${shownParam.label}`}>
                {[0, 0.5, 1].map((f) => {
                  const v = yMin + f * ySpan;
                  return (
                    <g key={f}>
                      <line x1={padL} x2={Wc - padR} y1={sy(v)} y2={sy(v)} className="stroke-border" strokeWidth="1" />
                      <text x={padL - 6} y={sy(v) + 3} textAnchor="end" className="fill-muted-foreground font-mono" fontSize="9">
                        {v.toFixed(0)}
                      </text>
                    </g>
                  );
                })}
                {Array.from({ length: SWEEP_STEPS }, (_, i) => shownParam.min + ((shownParam.max - shownParam.min) * i) / (SWEEP_STEPS - 1)).map((v) => (
                  <text key={v} x={sx(v)} y={Hc - 8} textAnchor="middle" className="fill-muted-foreground font-mono" fontSize="9">
                    {shownParam.fmt(v)}
                  </text>
                ))}
                <line x1={specX} x2={specX} y1={padT} y2={Hc - padB} className="stroke-warning" strokeWidth="1" strokeDasharray="3 3" />
                <text x={specX + 4} y={padT + 8} className="fill-muted-foreground" fontSize="9">
                  spec
                </text>
                <polyline points={line("coast")} fill="none" className="stroke-info" strokeWidth="1.5" />
                <polyline points={line("behavior")} fill="none" className="stroke-muted-foreground" strokeWidth="1.5" strokeDasharray="4 3" />
                <polyline points={line("policy")} fill="none" className="stroke-primary" strokeWidth="2.25" />
                {points.map((pt) => (
                  <circle key={pt.value} cx={sx(pt.value)} cy={sy(pt.policy)} r="3" className="fill-primary">
                    <title>{`${shownParam.label} ${shownParam.fmt(pt.value)} — policy ${pt.policy.toFixed(1)}, recorder ${pt.behavior.toFixed(1)}, coast ${pt.coast.toFixed(1)}`}</title>
                  </circle>
                ))}
              </svg>
            </div>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[10.5px] text-muted-foreground">
              <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 bg-primary" />Trained policy</span>
              <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 border-t border-dashed border-muted-foreground" />Behavior recorder</span>
              <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 bg-info" />Coast</span>
              <span className="flex items-center gap-1.5"><span className="h-3 w-0 border-l border-dashed border-warning" />Frozen spec value</span>
            </div>
            {done && worst && bestPt && (
              <p className="text-[11px] leading-relaxed text-muted-foreground">
                Policy beats the recorder at <span className="font-medium text-foreground">{wins} of {SWEEP_STEPS}</span> settings. Best return{" "}
                <span className="font-mono text-foreground">{bestPt.policy.toFixed(1)}</span> at {shownParam.fmt(bestPt.value)}; worst{" "}
                <span className="font-mono text-foreground">{worst.policy.toFixed(1)}</span> at {shownParam.fmt(worst.value)}.{" "}
                <span className={wins === SWEEP_STEPS ? "text-success" : wins >= SWEEP_STEPS / 2 ? "text-foreground" : "text-destructive"}>
                  {wins === SWEEP_STEPS ? "Robust across the whole range." : wins >= SWEEP_STEPS / 2 ? "Transfers partially — check the settings where the lines cross." : "Brittle — the policy mostly fails away from the training physics."}
                </span>
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
