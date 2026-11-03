import { useMemo, useState } from "react";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";
import { LineChart, Legend } from "@/components/lab/charts";
import { VirtualList } from "@/components/lab/virtual-list";
import { heatColor } from "@/components/lab/grid";
import {
  createQ,
  evaluate,
  generateDataset,
  greedy,
  mulberry32,
  trainUpdates,
  UPDATES_PER_STEP,
  type Dataset,
  type EvalResult,
  type Hyper,
  type Metrics,
} from "@/lib/rl";
import { generateThrusterDataset } from "@/lib/thrusterchain";

type PresetId = "gridworld-random" | "halfcheetah-medium" | "thrusterchain-stall";

const DATASET_LABELS: Record<PresetId, string> = {
  "gridworld-random": "Gridworld-Random",
  "halfcheetah-medium": "D4RL HalfCheetah-Medium",
  "thrusterchain-stall": "ThrusterChain v4 (stall)",
};

export interface RunConfig {
  dataset: PresetId;
  hyper: Hyper;
}

interface RunResult {
  config: RunConfig;
  ds: Dataset;
  Q: Float64Array;
  evalRes: EvalResult;
  history: Metrics[];
}

const STEPS = 60;

function buildDs(id: PresetId): Dataset {
  if (id === "thrusterchain-stall") return generateThrusterDataset(7);
  return generateDataset(id, id === "gridworld-random" ? 7 : 11);
}

function trainRun(config: RunConfig): RunResult {
  const ds = buildDs(config.dataset);
  const Q = createQ(ds);
  const rng = mulberry32(1234);
  const history: Metrics[] = [];
  for (let i = 1; i <= STEPS; i++) {
    const { tdLoss, cqlGap } = trainUpdates(Q, ds, config.hyper, rng, UPDATES_PER_STEP);
    const r = evaluate(Q, ds, config.hyper.gamma);
    history.push({
      iter: i,
      updates: i * UPDATES_PER_STEP,
      estQ: r.estQ,
      trueQ: r.trueQ,
      loss: tdLoss + config.hyper.alpha * cqlGap,
      tdLoss,
      cqlGap,
      oodRate: r.oodRate,
      divergence: r.divergence,
    });
  }
  return { config, ds, Q, evalRes: evaluate(Q, ds, config.hyper.gamma), history };
}

/* ---------- small building blocks ---------- */

function MiniSlider({ label, value, min, max, step, onChange, format }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; format: (v: number) => string }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[11.5px] font-medium">{label}</span>
        <span className="rounded border border-border bg-surface px-1.5 py-0.5 font-mono text-[10.5px] tabular-nums">{format(value)}</span>
      </div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={(v) => onChange(v[0])} aria-label={label} />
    </div>
  );
}

function ConfigEditor({ label, accent, config, onChange }: { label: string; accent: string; config: RunConfig; onChange: (c: RunConfig) => void }) {
  const h = config.hyper;
  const setH = (patch: Partial<Hyper>) => onChange({ ...config, hyper: { ...h, ...patch } });
  return (
    <div className="space-y-3 rounded-lg border border-border bg-surface/50 p-3">
      <div className="flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" style={{ background: accent }} />
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{label}</span>
      </div>
      <Select value={config.dataset} onValueChange={(v) => onChange({ ...config, dataset: v as PresetId })}>
        <SelectTrigger className="h-9 w-full px-3.5 text-xs" aria-label={`${label} dataset`}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent className="z-50">
          {(Object.keys(DATASET_LABELS) as PresetId[]).map((id) => (
            <SelectItem key={id} value={id}>{DATASET_LABELS[id]}</SelectItem>
          ))}
        </SelectContent>
      </Select>
      <MiniSlider label="Alpha (conservative)" value={h.alpha} min={0} max={5} step={0.1} onChange={(v) => setH({ alpha: v })} format={(v) => v.toFixed(1)} />
      <MiniSlider label="Discount γ" value={h.gamma} min={0.5} max={0.99} step={0.01} onChange={(v) => setH({ gamma: v })} format={(v) => v.toFixed(2)} />
      <MiniSlider label="Learning rate" value={h.lr} min={0.01} max={1} step={0.01} onChange={(v) => setH({ lr: v })} format={(v) => v.toFixed(2)} />
      <MiniSlider label="Batch size" value={h.batch} min={8} max={256} step={8} onChange={(v) => setH({ batch: v })} format={(v) => String(v)} />
    </div>
  );
}

function MiniHeatmap({ run, title }: { run: RunResult; title: string }) {
  const { ds, Q } = run;
  const { env } = ds;
  const W = env.width;
  const H = env.height;
  const { vals, lo, hi } = useMemo(() => {
    const S = W * H;
    const v = new Float64Array(S);
    let lo = Infinity;
    let hi = -Infinity;
    for (let s = 0; s < S; s++) {
      const b = s * 4;
      v[s] = Math.max(Q[b], Q[b + 1], Q[b + 2], Q[b + 3]);
      if (env.blocked.has(s) || env.terminals.has(s) || ds.visits[s] === 0) continue;
      lo = Math.min(lo, v[s]);
      hi = Math.max(hi, v[s]);
    }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 1e-6) hi = lo + 1;
    return { vals: v, lo, hi };
  }, [Q, ds, env, W, H]);

  const cell = 100 / Math.max(W, H);
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-[10.5px] text-muted-foreground">
        <span className="font-medium uppercase tracking-wider">{title}</span>
        <span className="font-mono tabular-nums">V ∈ [{lo.toFixed(1)}, {hi.toFixed(1)}]</span>
      </div>
      <svg viewBox={`0 0 ${W * cell} ${H * cell}`} className="w-full rounded-md border border-border bg-surface" role="img" aria-label={`${title} value heatmap`}>
        {Array.from({ length: W * H }, (_, s) => {
          const x = (s % W) * cell;
          const y = Math.floor(s / W) * cell;
          if (env.blocked.has(s)) return <rect key={s} x={x} y={y} width={cell} height={cell} className="fill-muted" />;
          const term = env.terminals.get(s);
          if (term) return <rect key={s} x={x} y={y} width={cell} height={cell} fill={term === "goal" ? "oklch(0.72 0.14 170)" : "oklch(0.6 0.17 25)"} opacity={0.7} />;
          if (ds.visits[s] === 0) return <rect key={s} x={x} y={y} width={cell} height={cell} className="fill-surface" />;
          const t = (vals[s] - lo) / (hi - lo);
          return <rect key={s} x={x} y={y} width={cell} height={cell} fill={heatColor(t, "viridis", 1.6)} />;
        })}
      </svg>
    </div>
  );
}

function TransitionLog({ run, label }: { run: RunResult; label: string }) {
  const { ds } = run;
  const names = ds.env.actionNames;
  const W = ds.env.width;
  const fmt = (s: number) => `(${s % W},${Math.floor(s / W)})`;
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-[10.5px] text-muted-foreground">
        <span className="font-medium uppercase tracking-wider">Transition log</span>
        <span className="font-mono tabular-nums">{ds.transitions.length.toLocaleString()} rows · virtualized</span>
      </div>
      <div className="grid grid-cols-[2.8rem_1fr_1fr_4rem_3.4rem] gap-1 border-b border-border px-2 pb-1 text-[9.5px] font-medium uppercase tracking-wider text-muted-foreground">
        <span>#</span><span>s</span><span>s′</span><span>action</span><span className="text-right">r</span>
      </div>
      <VirtualList
        items={ds.transitions}
        rowHeight={24}
        height={192}
        scrollKey={`cmp-log-${label}-${run.config.dataset}`}
        ariaLabel={`${label} transition log`}
        className="rounded-b-md border-x border-b border-border bg-surface/40"
        renderRow={(t, i) => (
          <div className={`grid h-full grid-cols-[2.8rem_1fr_1fr_4rem_3.4rem] items-center gap-1 px-2 font-mono text-[10.5px] tabular-nums ${i % 2 ? "bg-surface/60" : ""}`}>
            <span className="text-muted-foreground">{i}</span>
            <span>{fmt(t.s)}</span>
            <span>{fmt(t.s2)}{t.done ? " ⏹" : ""}</span>
            <span className="truncate text-muted-foreground">{names[t.a]}</span>
            <span className={`text-right ${t.r > 0 ? "text-success" : t.r < -0.5 ? "text-destructive" : "text-muted-foreground"}`}>{t.r.toFixed(2)}</span>
          </div>
        )}
      />
    </div>
  );
}

function MetricsLog({ run, label, accent }: { run: RunResult; label: string; accent: string }) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-[10.5px] text-muted-foreground">
        <span className="font-medium uppercase tracking-wider">Training metrics log</span>
        <span className="font-mono tabular-nums">{run.history.length} iters</span>
      </div>
      <div className="grid grid-cols-[3.4rem_1fr_1fr_1fr_3.2rem] gap-1 border-b border-border px-2 pb-1 text-[9.5px] font-medium uppercase tracking-wider text-muted-foreground">
        <span>upd</span><span className="text-right">est Q</span><span className="text-right">true</span><span className="text-right">loss</span><span className="text-right">OOD</span>
      </div>
      <VirtualList
        items={run.history}
        rowHeight={24}
        height={144}
        scrollKey={`cmp-metrics-${label}-${run.config.dataset}`}
        ariaLabel={`${label} metrics log`}
        className="rounded-b-md border-x border-b border-border bg-surface/40"
        renderRow={(m, i) => (
          <div className={`grid h-full grid-cols-[3.4rem_1fr_1fr_1fr_3.2rem] items-center gap-1 px-2 font-mono text-[10.5px] tabular-nums ${i % 2 ? "bg-surface/60" : ""}`}>
            <span style={{ color: accent }}>{m.updates}</span>
            <span className="text-right">{m.estQ.toFixed(2)}</span>
            <span className="text-right text-muted-foreground">{m.trueQ.toFixed(2)}</span>
            <span className="text-right">{m.loss.toFixed(2)}</span>
            <span className="text-right text-muted-foreground">{(m.oodRate * 100).toFixed(0)}%</span>
          </div>
        )}
      />
    </div>
  );
}

function RunPanel({ run, label, accent }: { run: RunResult; label: string; accent: string }) {
  const xs = run.history.map((m) => m.updates);
  const last = run.history[run.history.length - 1];
  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-4">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: accent }} />
          <span className="text-sm font-semibold">{label}</span>
        </div>
        <span className="rounded border border-border bg-surface px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">{DATASET_LABELS[run.config.dataset]}</span>
      </div>
      <div className="grid grid-cols-2 gap-2 font-mono text-[10.5px] tabular-nums text-muted-foreground">
        <div>α {run.config.hyper.alpha.toFixed(1)} · γ {run.config.hyper.gamma.toFixed(2)}</div>
        <div>lr {run.config.hyper.lr.toFixed(2)} · batch {run.config.hyper.batch}</div>
      </div>
      <div>
        <div className="mb-1.5 text-[10.5px] font-medium uppercase tracking-wider text-muted-foreground">Policy loss</div>
        <LineChart xs={xs} height={120} empty="No data" series={[{ name: "Loss", color: accent, values: run.history.map((m) => m.loss) }]} />
      </div>
      <div>
        <div className="mb-1.5 text-[10.5px] font-medium uppercase tracking-wider text-muted-foreground">Est. Q vs ground truth</div>
        <LineChart
          xs={xs}
          height={120}
          empty="No data"
          series={[
            { name: "Estimated Q", color: accent, values: run.history.map((m) => m.estQ) },
            { name: "Ground truth", color: "oklch(0.6 0.02 275)", values: run.history.map((m) => m.trueQ), dashed: true },
          ]}
        />
        <div className="mt-1.5">
          <Legend
            items={[
              { name: "Est. Q", color: accent, value: last.estQ.toFixed(2) },
              { name: "Truth", color: "oklch(0.6 0.02 275)", dashed: true, value: last.trueQ.toFixed(2) },
            ]}
          />
        </div>
      </div>
      <MiniHeatmap run={run} title="Final value heatmap" />
      <TransitionLog run={run} label={label} />
      <MetricsLog run={run} label={label} accent={accent} />
    </div>
  );
}

function DeltaStat({ label, a, b, fmt, lowerBetter }: { label: string; a: number; b: number; fmt: (v: number) => string; lowerBetter?: boolean }) {
  const d = b - a;
  const better = Math.abs(d) < 1e-9 ? null : lowerBetter ? d < 0 : d > 0;
  const tone = better === null ? "text-muted-foreground" : better ? "text-success" : "text-destructive";
  return (
    <div className="rounded-lg border border-border bg-surface/60 px-3 py-2">
      <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="mt-1 flex items-baseline justify-between gap-2 font-mono text-[12.5px] tabular-nums">
        <span>{fmt(a)}</span>
        <span className="text-muted-foreground">→</span>
        <span>{fmt(b)}</span>
      </div>
      <div className={`mt-0.5 font-mono text-[11px] tabular-nums ${tone}`}>Δ {d >= 0 ? "+" : ""}{fmt(d)}</div>
    </div>
  );
}

function DeltaHeatmap({ a, b }: { a: RunResult; b: RunResult }) {
  const sameGrid = a.config.dataset === b.config.dataset;
  const { env } = a.ds;
  const W = env.width;
  const H = env.height;
  const geo = useMemo(() => {
    if (!sameGrid) return null;
    const S = W * H;
    const v = new Float64Array(S);
    let m = 0;
    for (let s = 0; s < S; s++) {
      const ba = s * 4;
      const qa = Math.max(a.Q[ba], a.Q[ba + 1], a.Q[ba + 2], a.Q[ba + 3]);
      const qb = Math.max(b.Q[ba], b.Q[ba + 1], b.Q[ba + 2], b.Q[ba + 3]);
      v[s] = qb - qa;
      if (!env.blocked.has(s) && !env.terminals.has(s) && a.ds.visits[s] > 0) m = Math.max(m, Math.abs(v[s]));
    }
    return { v, m: m || 1 };
  }, [sameGrid, a, b, W, H, env]);

  if (!sameGrid || !geo) {
    return (
      <div className="flex h-full min-h-[140px] items-center justify-center rounded-md border border-dashed border-border bg-surface/50 px-6 text-center text-xs text-muted-foreground">
        Select the same dataset for both runs to see the state-by-state value delta.
      </div>
    );
  }
  const cell = 100 / Math.max(W, H);
  return (
    <div>
      <svg viewBox={`0 0 ${W * cell} ${H * cell}`} className="w-full rounded-md border border-border bg-surface" role="img" aria-label="Value delta heatmap (Run B − Run A)">
        {Array.from({ length: W * H }, (_, s) => {
          const x = (s % W) * cell;
          const y = Math.floor(s / W) * cell;
          if (env.blocked.has(s)) return <rect key={s} x={x} y={y} width={cell} height={cell} className="fill-muted" />;
          if (env.terminals.has(s) || a.ds.visits[s] === 0) return <rect key={s} x={x} y={y} width={cell} height={cell} className="fill-surface" />;
          const t = geo.v[s] / geo.m; // −1..1
          const u = Math.min(1, Math.abs(t));
          const color = t >= 0 ? `oklch(${0.95 - u * 0.35} ${0.02 + u * 0.14} 185)` : `oklch(${0.95 - u * 0.33} ${0.02 + u * 0.16} 50)`;
          const disagree = greedy(a.Q, s) !== greedy(b.Q, s);
          return (
            <g key={s}>
              <rect x={x} y={y} width={cell} height={cell} fill={color} />
              {disagree && <circle cx={x + cell / 2} cy={y + cell / 2} r={cell * 0.12} className="fill-foreground" opacity={0.6} />}
            </g>
          );
        })}
      </svg>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[10.5px] text-muted-foreground">
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: "oklch(0.63 0.16 50)" }} /> Run A higher</span>
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: "oklch(0.63 0.14 185)" }} /> Run B higher</span>
        <span className="flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-foreground/60" /> policies disagree</span>
        <span className="font-mono tabular-nums">max |Δ| {geo.m.toFixed(2)}</span>
      </div>
    </div>
  );
}

/* ---------- main view ---------- */

const ACCENT_A = "oklch(0.55 0.23 285)";
const ACCENT_B = "oklch(0.68 0.17 55)";

export function ComparisonView() {
  const [configA, setConfigA] = useState<RunConfig>({ dataset: "gridworld-random", hyper: { alpha: 0, gamma: 0.95, lr: 0.3, batch: 64 } });
  const [configB, setConfigB] = useState<RunConfig>({ dataset: "gridworld-random", hyper: { alpha: 2, gamma: 0.95, lr: 0.3, batch: 64 } });

  const runA = useMemo(() => trainRun(configA), [configA]);
  const runB = useMemo(() => trainRun(configB), [configB]);

  const lastA = runA.history[runA.history.length - 1];
  const lastB = runB.history[runB.history.length - 1];

  const disagreement = useMemo(() => {
    if (configA.dataset !== configB.dataset) return null;
    const { env } = runA.ds;
    let n = 0;
    let diff = 0;
    for (let s = 0; s < env.width * env.height; s++) {
      if (env.blocked.has(s) || env.terminals.has(s) || runA.ds.visits[s] === 0) continue;
      n++;
      if (greedy(runA.Q, s) !== greedy(runB.Q, s)) diff++;
    }
    return n ? diff / n : 0;
  }, [runA, runB, configA.dataset, configB.dataset]);

  return (
    <div className="space-y-4">
      {/* configs */}
      <section className="rounded-xl border border-border bg-card p-4 shadow-sm">
        <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Run configurations · both runs train {STEPS * UPDATES_PER_STEP} updates with a fixed seed</h2>
        <div className="grid gap-3 md:grid-cols-2">
          <ConfigEditor label="Run A" accent={ACCENT_A} config={configA} onChange={setConfigA} />
          <ConfigEditor label="Run B" accent={ACCENT_B} config={configB} onChange={setConfigB} />
        </div>
      </section>

      {/* split screen */}
      <section className="rounded-xl border border-border bg-card shadow-sm">
        <header className="border-b border-border px-4 py-2.5">
          <h2 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Side-by-side · drag the divider to resize</h2>
        </header>
        <div className="hidden md:block">
          <ResizablePanelGroup direction="horizontal" className="min-h-[520px]">
            <ResizablePanel defaultSize={50} minSize={28}>
              <RunPanel run={runA} label="Run A" accent={ACCENT_A} />
            </ResizablePanel>
            <ResizableHandle withHandle />
            <ResizablePanel defaultSize={50} minSize={28}>
              <RunPanel run={runB} label="Run B" accent={ACCENT_B} />
            </ResizablePanel>
          </ResizablePanelGroup>
        </div>
        <div className="grid gap-2 md:hidden">
          <RunPanel run={runA} label="Run A" accent={ACCENT_A} />
          <div className="mx-4 border-t border-border" />
          <RunPanel run={runB} label="Run B" accent={ACCENT_B} />
        </div>
      </section>

      {/* delta view */}
      <section className="rounded-xl border border-border bg-card shadow-sm">
        <header className="border-b border-border px-4 py-2.5">
          <h2 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Delta view · Run A → Run B</h2>
        </header>
        <div className="grid gap-4 p-4 lg:grid-cols-[1fr_320px]">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            <DeltaStat label="Final est. Q" a={lastA.estQ} b={lastB.estQ} fmt={(v) => v.toFixed(2)} />
            <DeltaStat label="Final true return" a={lastA.trueQ} b={lastB.trueQ} fmt={(v) => v.toFixed(2)} />
            <DeltaStat label="Overestimation gap" a={lastA.estQ - lastA.trueQ} b={lastB.estQ - lastB.trueQ} fmt={(v) => v.toFixed(2)} lowerBetter />
            <DeltaStat label="Policy divergence" a={lastA.divergence} b={lastB.divergence} fmt={(v) => v.toFixed(2)} lowerBetter />
            <DeltaStat label="OOD action rate" a={lastA.oodRate} b={lastB.oodRate} fmt={(v) => `${(v * 100).toFixed(0)}%`} lowerBetter />
            <DeltaStat label="Final loss" a={lastA.loss} b={lastB.loss} fmt={(v) => v.toFixed(2)} lowerBetter />
            <div className="col-span-2 rounded-lg border border-border bg-surface/60 px-3 py-2 sm:col-span-3">
              <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Policy disagreement between runs</div>
              <div className="mt-0.5 font-mono text-[15px] font-medium tabular-nums">
                {disagreement === null ? "— (different datasets)" : `${(disagreement * 100).toFixed(1)}% of visited states pick different actions`}
              </div>
            </div>
            <p className="col-span-2 text-[11.5px] leading-relaxed text-muted-foreground sm:col-span-3">
              Green deltas move in the helpful direction (higher true return, lower overestimation, divergence, OOD rate, and loss). "True return" is computed by exact policy evaluation of each run's greedy policy, so Δ true return is the real performance difference — not an estimate.
            </p>
          </div>
          <div>
            <div className="mb-1.5 text-[10.5px] font-medium uppercase tracking-wider text-muted-foreground">Value delta map (B − A)</div>
            <DeltaHeatmap a={runA} b={runB} />
          </div>
        </div>
      </section>
    </div>
  );
}
