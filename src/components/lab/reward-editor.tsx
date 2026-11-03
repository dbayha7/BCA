// Custom reward editor + dynamic reward visualizer.
// The user writes `function reward(s, a, s2, ctx)`; every logged transition is relabeled inside
// the sandbox worker (debounced while typing for the live heatmap), and "Apply" hands a dataset
// rebuilt with buildDataset (transitions unchanged except r) to the parent via onApply.

import { useEffect, useMemo, useState } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { rewardColor } from "@/components/lab/grid";
import { CodeEditor, CodeError, SandboxNote } from "@/components/lab/code-editor";
import type { Dataset } from "@/lib/rl";
import { checkUserCode } from "@/lib/sandbox/guard";
import type { HardeningReport } from "@/lib/sandbox/harden";
import { REWARD_TIMEOUT_MS, SandboxError, runRewardSandbox } from "@/lib/sandbox/client";
import {
  REWARD_PRESETS,
  buildRewardJob,
  originalDataset,
  relabelDataset,
  rewardHeatmap,
  rewardSummary,
} from "@/lib/sandbox/reward";

const ERROR_TITLES: Record<string, string> = {
  guard: "Blocked by the static guard",
  syntax: "Syntax error",
  runtime: "Runtime error",
  output: "Invalid output",
  timeout: "Timed out",
  worker: "Sandbox unavailable",
};

export function errorInfo(e: unknown): { kind: string; message: string; line?: number } {
  if (e instanceof SandboxError) return { kind: e.kind, message: e.message, line: e.line };
  return { kind: "runtime", message: e instanceof Error ? e.message : "Something went wrong." };
}

const API_REFERENCE = `// reward(s, a, s2, ctx) → finite number, called once per logged transition
// s, s2          grid cells, row-major: s = y * ctx.width + x
// a              action 0–3 (name: ctx.actionNames[a])
// ctx.original   the logged reward of this transition
// ctx.done       true if the transition ended the episode
// ctx.x(s), ctx.y(s), ctx.cell(x, y)   (cell → −1 outside the grid)
// ctx.isGoal(s), ctx.isHazard(s), ctx.isBlocked(s)
// ctx.goal       {x, y} of the goal, or null
// ctx.target     goal, else HalfCheetah: fastest level cell,
//                ThrusterChain: rest cell (p₁ = v₁ = 0)
// ctx.width, ctx.height, ctx.actionNames
// Top-level variables persist for one relabeling run.`;

type Status = "pending" | "running" | "ok" | "error";

interface Preview {
  code: string;
  base: Dataset;
  rewards: Float64Array;
  ms: number;
}

function Seg<T extends string | number>({ value, options, onChange, label }: { value: T; options: { v: T; label: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="flex flex-wrap rounded-md border border-border bg-surface p-0.5 text-[11px]">
      {options.map((o) => (
        <button
          key={String(o.v)}
          role="radio"
          aria-checked={value === o.v}
          onClick={() => onChange(o.v)}
          className={`rounded px-2 py-1 font-medium transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
            value === o.v ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function RewardHeatmapGrid({ base, rewards, action, diff, stale }: { base: Dataset; rewards: Float64Array; action: number | "all"; diff: boolean; stale: boolean }) {
  const env = base.env;
  const W = env.width;
  const H = env.height;
  const cell = 34;
  const [hover, setHover] = useState<number | null>(null);
  const shown = useMemo(() => rewardHeatmap(base, rewards, { action, diff }), [base, rewards, action, diff]);
  const rel = useMemo(() => (diff ? rewardHeatmap(base, rewards, { action, diff: false }) : shown), [base, rewards, action, diff, shown]);
  const dlt = useMemo(() => (diff ? shown : rewardHeatmap(base, rewards, { action, diff: true })), [base, rewards, action, diff, shown]);
  const M = Math.max(Math.abs(shown.lo), Math.abs(shown.hi), 1e-9);
  const color = (v: number) => (diff ? rewardColor(v, -M, M) : rewardColor(v, Math.min(shown.lo, 0), Math.max(shown.hi, 0)));
  const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
  const lo = diff ? -M : Math.min(shown.lo, 0);
  const hi = diff ? M : Math.max(shown.hi, 0);

  return (
    <div className="space-y-2">
      <svg
        viewBox={`-1 -1 ${W * cell + 2} ${H * cell + 2}`}
        className={`mx-auto block max-h-[300px] w-full transition-opacity duration-200 ${stale ? "opacity-55" : ""}`}
        role="img"
        aria-label={`Mean ${diff ? "reward change" : "relabeled reward"} per state, ${action === "all" ? "all actions" : env.actionNames[action]}`}
        onMouseLeave={() => setHover(null)}
      >
        <defs>
          <pattern id="rw-nodata" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--border)" strokeWidth="2" />
          </pattern>
        </defs>
        {Array.from({ length: W * H }, (_, s) => {
          const x = (s % W) * cell;
          const y = Math.floor(s / W) * cell;
          const v = shown.values[s];
          const blocked = env.blocked.has(s);
          const term = env.terminals.get(s);
          const fill = blocked ? "color-mix(in oklch, var(--foreground) 55%, var(--background))" : Number.isNaN(v) ? "url(#rw-nodata)" : color(v);
          return (
            <g key={s} onMouseEnter={() => setHover(s)}>
              <rect x={x + 1} y={y + 1} width={cell - 2} height={cell - 2} rx={4} fill={fill} stroke={hover === s ? "var(--foreground)" : "var(--border)"} strokeWidth={hover === s ? 1.5 : 1} />
              {term && (
                <text x={x + cell / 2} y={y + cell / 2 + 4} textAnchor="middle" fontSize="11" fontWeight={700} className="pointer-events-none fill-foreground font-mono">
                  {term === "goal" ? "G" : "×"}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="flex flex-wrap items-center justify-between gap-2 text-[10.5px] text-muted-foreground">
        <div className="flex items-center gap-1.5">
          <span className="font-mono tabular-nums">{fmt(lo)}</span>
          <div className="h-2 w-24 rounded-full" style={{ background: `linear-gradient(90deg, ${color(lo)}, ${color(0)}, ${color(hi)})` }} />
          <span className="font-mono tabular-nums">{fmt(hi)}</span>
          <span className="ml-1">{diff ? "Δ vs logged" : "mean r"}</span>
        </div>
        <span className="flex items-center gap-1">
          <svg width="10" height="10" aria-hidden>
            <rect width="10" height="10" rx="2" fill="url(#rw-nodata)" stroke="var(--border)" />
          </svg>
          no data
        </span>
      </div>
      <div className="min-h-[18px] font-mono text-[10.5px] tabular-nums text-muted-foreground" aria-live="polite">
        {hover === null ? (
          "Hover a cell for its numbers."
        ) : (
          <>
            ({hover % W}, {Math.floor(hover / W)}) · n={rel.counts[hover]} ·{" "}
            {rel.counts[hover] ? (
              <>
                r̄ <span className="text-foreground">{rel.values[hover].toFixed(3)}</span> · Δ{" "}
                <span className={dlt.values[hover] > 1e-9 ? "text-success" : dlt.values[hover] < -1e-9 ? "text-destructive" : "text-foreground"}>
                  {dlt.values[hover] >= 0 ? "+" : ""}
                  {dlt.values[hover].toFixed(3)}
                </span>
              </>
            ) : (
              "no logged transitions"
            )}
          </>
        )}
      </div>
    </div>
  );
}

export function RewardEditor({ ds, onApply }: { ds: Dataset; onApply: (relabeled: Dataset, info: { name: string; code: string }) => void }) {
  // Always relabel the LOGGED rewards, even if the parent passes an already-relabeled dataset.
  const base = originalDataset(ds);
  const job = useMemo(() => buildRewardJob(base), [base]);
  const [code, setCode] = useState(REWARD_PRESETS[0].code);
  const [stash, setStash] = useState<string | null>(null);
  const [action, setAction] = useState<number | "all">("all");
  const [diff, setDiff] = useState(false);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [status, setStatus] = useState<Status>("pending");
  const [error, setError] = useState<{ kind: string; message: string; line?: number } | null>(null);
  const [hardening, setHardening] = useState<HardeningReport | null>(null);
  const [applying, setApplying] = useState(false);
  const [showRef, setShowRef] = useState(false);

  const preset = REWARD_PRESETS.find((p) => p.code === code);
  const guard = useMemo(() => checkUserCode(code, { requireFunction: "reward" }), [code]);

  // Live preview: static guard on every keystroke, sandbox relabel debounced (400 ms).
  useEffect(() => {
    if (!guard.ok) {
      setStatus("error");
      setError({ kind: "guard", message: guard.message, line: guard.line });
      return;
    }
    setStatus("pending");
    const ctrl = new AbortController();
    const t = window.setTimeout(() => {
      setStatus("running");
      runRewardSandbox(code, job, { signal: ctrl.signal })
        .then((run) => {
          setPreview({ code, base, rewards: run.rewards, ms: run.wallMs });
          setHardening(run.hardening);
          setError(null);
          setStatus("ok");
        })
        .catch((e) => {
          if (e instanceof SandboxError && e.kind === "aborted") return;
          setError(errorInfo(e));
          setStatus("error");
        });
    }, 400);
    return () => {
      window.clearTimeout(t);
      ctrl.abort();
    };
  }, [code, job, base, guard]);

  const current = preview && preview.base === base ? preview : null;
  const fresh = current !== null && current.code === code && status === "ok";
  const summary = useMemo(() => (current ? rewardSummary(base, current.rewards) : null), [current, base]);
  const isRelabeled = ds !== base;

  const choosePreset = (id: string) => {
    if (id === "custom") {
      if (stash !== null) setCode(stash);
      return;
    }
    const p = REWARD_PRESETS.find((x) => x.id === id);
    if (!p) return;
    if (!preset) setStash(code);
    setCode(p.code);
  };

  const apply = async () => {
    if (!guard.ok) return;
    setApplying(true);
    try {
      const rewards = fresh && current ? current.rewards : (await runRewardSandbox(code, job)).rewards;
      const name = preset ? preset.name : "Custom reward";
      onApply(relabelDataset(base, rewards, name), { name, code });
    } catch (e) {
      setError(errorInfo(e));
      setStatus("error");
    } finally {
      setApplying(false);
    }
  };

  const statusText =
    status === "error"
      ? "error"
      : status === "ok" && fresh
        ? `preview · ${base.transitions.length.toLocaleString()} transitions · ${Math.round(current!.ms)} ms`
        : status === "running"
          ? "running in sandbox…"
          : "waiting for typing to pause…";

  return (
    <div className="@container space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Select value={preset?.id ?? "custom"} onValueChange={choosePreset}>
          <SelectTrigger className="h-8 w-auto min-w-[170px] px-2.5 text-xs" aria-label="Reward preset">
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="z-50">
            {REWARD_PRESETS.map((p) => (
              <SelectItem key={p.id} value={p.id} className="text-xs">
                {p.name}
              </SelectItem>
            ))}
            {(stash !== null || !preset) && (
              <SelectItem value="custom" className="text-xs">
                {preset ? "Your edits" : "Custom (edited)"}
              </SelectItem>
            )}
          </SelectContent>
        </Select>
        <span
          className={`ml-auto inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5 font-mono text-[10px] ${
            status === "error" ? "border-destructive/40 text-destructive" : "border-border text-muted-foreground"
          }`}
          aria-live="polite"
        >
          <span className={`h-1.5 w-1.5 rounded-full ${status === "error" ? "bg-destructive" : fresh ? "bg-success" : "animate-pulse bg-warning"}`} />
          {statusText}
        </span>
      </div>
      {preset && <p className="-mt-1 text-[11px] text-muted-foreground">{preset.description}</p>}

      <div className="grid gap-4 @3xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-2">
          <CodeEditor value={code} onChange={setCode} errorLine={error?.line ?? null} height={260} ariaLabel="Reward function code" />
          {error && <CodeError title={ERROR_TITLES[error.kind] ?? "Error"} message={error.message} />}
          <button
            onClick={() => setShowRef((v) => !v)}
            aria-expanded={showRef}
            className="text-[11px] font-medium text-muted-foreground underline-offset-2 hover:text-foreground hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            {showRef ? "Hide" : "Show"} the reward(s, a, s2, ctx) reference
          </button>
          {showRef && <CodeEditor value={API_REFERENCE} readOnly height={252} ariaLabel="Reward function API reference" />}
          <SandboxNote limitMs={REWARD_TIMEOUT_MS} removed={hardening?.removed.length} failed={hardening?.failed} />
        </div>

        <div className="min-w-0 space-y-2.5">
          <div className="flex flex-wrap items-center gap-2">
            <Seg
              label="Heatmap value"
              value={diff ? "diff" : "reward"}
              onChange={(v) => setDiff(v === "diff")}
              options={[
                { v: "reward", label: "Relabeled r" },
                { v: "diff", label: "Δ vs original" },
              ]}
            />
            <Seg<number | "all">
              label="Action filter"
              value={action}
              onChange={setAction}
              options={[{ v: "all", label: "All" }, ...base.env.actionNames.map((n, a) => ({ v: a, label: n }))]}
            />
          </div>
          {current ? (
            <RewardHeatmapGrid base={base} rewards={current.rewards} action={action} diff={diff} stale={!fresh} />
          ) : (
            <div className="flex h-[220px] items-center justify-center rounded-lg border border-dashed border-border text-[11.5px] text-muted-foreground">
              {status === "error" ? "Fix the error to see the reward map." : "Computing the reward map…"}
            </div>
          )}
          {summary && (
            <div className="grid grid-cols-3 gap-2 font-mono text-[11px] tabular-nums">
              <div className="rounded-md border border-border bg-surface/60 px-2 py-1.5">
                <div className="text-[9.5px] uppercase tracking-wider text-muted-foreground">changed</div>
                {((summary.changed / Math.max(1, summary.n)) * 100).toFixed(1)}%
              </div>
              <div className="rounded-md border border-border bg-surface/60 px-2 py-1.5">
                <div className="text-[9.5px] uppercase tracking-wider text-muted-foreground">r range</div>
                {summary.min.toFixed(2)} … {summary.max.toFixed(2)}
              </div>
              <div className="rounded-md border border-border bg-surface/60 px-2 py-1.5">
                <div className="text-[9.5px] uppercase tracking-wider text-muted-foreground">mean r</div>
                {summary.mean.toFixed(3)} <span className="text-muted-foreground">({summary.originalMean.toFixed(3)})</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-border pt-3">
        <button
          onClick={apply}
          disabled={!guard.ok || applying || status === "error"}
          className="rounded-md bg-primary px-3.5 py-1.5 text-xs font-semibold text-primary-foreground shadow-sm transition-colors hover:bg-primary/90 disabled:opacity-45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-card"
        >
          {applying ? "Relabeling…" : "Apply to dataset"}
        </button>
        {isRelabeled && (
          <button
            onClick={() => onApply(base, { name: REWARD_PRESETS[0].name, code: REWARD_PRESETS[0].code })}
            className="rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            Restore logged rewards
          </button>
        )}
        <span className="text-[11px] text-muted-foreground">
          {isRelabeled ? (
            <>
              Training on <span className="font-medium text-foreground">relabeled</span> rewards. Retrain to see the effect.
            </>
          ) : (
            "Applying rebuilds the dataset with only r changed; the learner then trains on it."
          )}
        </span>
      </div>
    </div>
  );
}
