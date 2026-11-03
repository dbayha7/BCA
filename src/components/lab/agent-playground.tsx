// Interactive agent playground: write `act(obs, ctx)`, run N episodes of a real environment in
// the sandbox worker, compare against built-in baselines and step through one episode.
// Every number shown is recomputed on the main thread by replaying the agent's validated
// actions/commands on the trusted simulators.

import { useEffect, useMemo, useRef, useState } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CodeEditor, CodeError, SandboxNote } from "@/components/lab/code-editor";
import { errorInfo } from "@/components/lab/reward-editor";
import { checkUserCode } from "@/lib/sandbox/guard";
import type { HardeningReport } from "@/lib/sandbox/harden";
import type { PlaygroundEnvId } from "@/lib/sandbox/protocol";
import { AGENT_TIMEOUT_MS, SandboxError, runGridAgentSandbox, runThrusterAgentSandbox } from "@/lib/sandbox/client";
import {
  AGENT_TEMPLATES,
  GRID_HORIZON,
  MAX_EPISODES,
  PLAYGROUND_ENVS,
  gridOptimalEpisodes,
  gridRandomEpisodes,
  playgroundGridEnv,
  summarizeGrid,
  summarizeThruster,
  thrusterBehaviorEpisodes,
  thrusterCoastEpisodes,
  type GridEpisode,
  type PolicySummary,
  type ThrusterEpisode,
} from "@/lib/sandbox/agent";
import { HORIZON, KNEE } from "@/lib/thrusterchain";

const ERROR_TITLES: Record<string, string> = {
  guard: "Blocked by the static guard",
  syntax: "Syntax error",
  runtime: "Runtime error",
  output: "Invalid output",
  timeout: "Timed out",
  worker: "Sandbox unavailable",
};

const GRID_REFERENCE = `// act(obs, ctx) → integer action 0–3, called every step
// obs.s         current cell, row-major: s = y * ctx.width + x
// obs.x, obs.y  cell coordinates;  obs.t  step index from 0
// ctx.width, ctx.height, ctx.actionNames, ctx.start {x, y}
// ctx.goal      {x, y} of the goal, or null (HalfCheetah has none)
// ctx.x(s), ctx.y(s), ctx.cell(x, y)   (cell → −1 outside the grid)
// ctx.isGoal(s), ctx.isHazard(s), ctx.isBlocked(s)
// ctx.random()  seeded uniform [0, 1), reset every episode
// Episodes start at the env start and end at a goal/hazard
// or after ${GRID_HORIZON} steps. Top-level variables persist for one run.`;

const THRUSTER_REFERENCE = `// act(obs, ctx) → [a0, a1, a2, a3]: four finite numbers per step
// obs.state  13 numbers: [p1, v1, p2, v2, p3, v3, p4, v4, p5, v5,
//                         current, wave cos, wave sin]
// obs.cell   the (p1, v1) grid cell the tabular lab sees; obs.t step
// ctx.knee   ${KNEE.toFixed(2)}: |a| above it stalls (thrust falls off with
//            slope ctx.stallSlope)
// ctx.horizon (${HORIZON}), ctx.stateDims (13), ctx.commandDims (4)
// ctx.random()  seeded uniform [0, 1), reset every episode
// Commands are clipped to [−1, 1] and cast to float32 (executedCommand).
// Reward per step = −(Σ x[0..9]² + 0.3·‖a‖²); same reset seeds as the
// physics bench, so every policy faces identical episodes.`;

type Results =
  | { kind: "grid"; envId: PlaygroundEnvId; rows: PolicySummary[]; episodes: Record<string, GridEpisode[]>; wallMs: number; hardening: HardeningReport }
  | { kind: "thruster"; envId: PlaygroundEnvId; rows: PolicySummary[]; episodes: Record<string, ThrusterEpisode[]>; wallMs: number; hardening: HardeningReport };

function Panel({ title, right, children }: { title: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="min-w-0 bca-panel">
      <header className="flex items-center justify-between gap-2 border-b border-border px-4 py-2.5">
        <h2 className="bca-panel-title flex items-center gap-2"><span className="bca-tick" aria-hidden />{title}</h2>
        {right}
      </header>
      <div className="p-4">{children}</div>
    </section>
  );
}

function Seg<T extends string>({ value, options, onChange, label, disabled }: { value: T; options: { v: T; label: string }[]; onChange: (v: T) => void; label: string; disabled?: boolean }) {
  return (
    <div role="radiogroup" aria-label={label} className="flex flex-wrap rounded-lg border border-border bg-surface p-0.5">
      {options.map((o) => (
        <button
          key={o.v}
          role="radio"
          aria-checked={value === o.v}
          disabled={disabled}
          onClick={() => onChange(o.v)}
          className={`rounded-md px-3 py-1.5 text-[12.5px] font-medium transition-all duration-200 disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
            value === o.v ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/* ---------------- Results ---------------- */

function ReturnStrip({ returns, lo, hi, color }: { returns: number[]; lo: number; hi: number; color: string }) {
  const W = 120;
  const span = hi - lo || 1;
  return (
    <svg viewBox={`0 0 ${W} 14`} className="h-3.5 w-[120px]" aria-hidden>
      <line x1={0} y1={7} x2={W} y2={7} stroke="var(--border)" />
      {returns.map((r, i) => (
        <circle key={i} cx={2 + ((r - lo) / span) * (W - 4)} cy={7} r={2.4} fill={color} opacity={0.55} />
      ))}
    </svg>
  );
}

const ROW_COLORS: Record<string, string> = { you: "var(--chart-1)", random: "var(--chart-3)", optimal: "var(--chart-2)", coast: "var(--chart-4)", behavior: "var(--chart-3)" };

function ResultsTable({ res }: { res: Results }) {
  const all = res.rows.flatMap((r) => r.returns);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const best = Math.max(...res.rows.map((r) => r.meanReturn));
  const grid = res.kind === "grid";
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] border-collapse text-left text-[12px]">
        <thead>
          <tr className="border-b border-border text-[10px] uppercase tracking-wider text-muted-foreground">
            <th className="py-1.5 pr-2 font-medium">Policy</th>
            <th className="py-1.5 pr-2 text-right font-medium">Mean return</th>
            <th className="py-1.5 pr-2 text-right font-medium">± sd</th>
            <th className="py-1.5 pr-2 text-right font-medium">Steps</th>
            {grid ? (
              <>
                <th className="py-1.5 pr-2 text-right font-medium">Goal</th>
                <th className="py-1.5 pr-2 text-right font-medium">Hazards hit</th>
              </>
            ) : (
              <>
                <th className="py-1.5 pr-2 text-right font-medium" title="Steps where any thruster command exceeded the stall knee">
                  Stall steps
                </th>
                <th className="py-1.5 pr-2 text-right font-medium" title="Mean oscillator energy Σx[0..9]² after the last step">
                  Final energy
                </th>
              </>
            )}
            <th className="py-1.5 font-medium">Returns per episode</th>
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">
          {res.rows.map((r) => (
            <tr key={r.id} className={`border-b border-border/60 ${r.id === "you" ? "bg-accent/40" : ""}`}>
              <td className="py-2 pr-2 font-sans font-medium">
                <span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: ROW_COLORS[r.id] }} />
                {r.label}
              </td>
              <td className={`py-2 pr-2 text-right ${r.meanReturn === best ? "font-semibold text-success" : ""}`}>{r.meanReturn.toFixed(2)}</td>
              <td className="py-2 pr-2 text-right text-muted-foreground">{r.stdReturn.toFixed(2)}</td>
              <td className="py-2 pr-2 text-right">{r.meanSteps.toFixed(1)}</td>
              {grid ? (
                <>
                  <td className="py-2 pr-2 text-right">
                    {r.goalEpisodes}/{r.episodes}
                  </td>
                  <td className={`py-2 pr-2 text-right ${r.hazardEpisodes ? "text-destructive" : ""}`}>
                    {r.hazardEpisodes}/{r.episodes}
                  </td>
                </>
              ) : (
                <>
                  <td className={`py-2 pr-2 text-right ${r.stallRate > 0 ? "text-warning" : ""}`}>{(r.stallRate * 100).toFixed(1)}%</td>
                  <td className="py-2 pr-2 text-right">{r.finalEnergy.toFixed(2)}</td>
                </>
              )}
              <td className="py-2">
                <ReturnStrip returns={r.returns} lo={lo} hi={hi} color={ROW_COLORS[r.id] ?? "var(--chart-5)"} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------------- Playback ---------------- */

function GridPlayback({ envId, ep, idx }: { envId: PlaygroundEnvId; ep: GridEpisode; idx: number }) {
  const env = playgroundGridEnv(envId as "gridworld-random" | "halfcheetah-medium");
  const cell = 36;
  const W = env.width;
  const H = env.height;
  const cx = (s: number) => (s % W) * cell + cell / 2;
  const cy = (s: number) => Math.floor(s / W) * cell + cell / 2;
  const pos = (i: number) => (i < ep.steps.length ? ep.steps[i].s : ep.steps[ep.steps.length - 1].s2);
  const trail = Array.from({ length: idx + 1 }, (_, i) => pos(i));
  const here = pos(idx);
  return (
    <svg viewBox={`-2 -2 ${W * cell + 4} ${H * cell + 4}`} className="mx-auto block max-h-[320px] w-full" role="img" aria-label={`Agent at cell (${here % W}, ${Math.floor(here / W)}) at step ${idx}`}>
      {Array.from({ length: W * H }, (_, s) => {
        const term = env.terminals.get(s);
        const fill = env.blocked.has(s)
          ? "color-mix(in oklch, var(--foreground) 55%, var(--background))"
          : term === "goal"
            ? "color-mix(in oklch, var(--success) 28%, var(--card))"
            : term === "hazard"
              ? "color-mix(in oklch, var(--destructive) 22%, var(--card))"
              : "var(--card)";
        return (
          <g key={s}>
            <rect x={(s % W) * cell + 1.5} y={Math.floor(s / W) * cell + 1.5} width={cell - 3} height={cell - 3} rx={5} fill={fill} stroke="var(--border)" />
            {term && (
              <text x={cx(s)} y={cy(s) + 4} textAnchor="middle" fontSize="11" fontWeight={600} className="fill-foreground font-mono">
                {term === "goal" ? "G" : "×"}
              </text>
            )}
          </g>
        );
      })}
      <polyline points={trail.map((s) => `${cx(s)},${cy(s)}`).join(" ")} fill="none" stroke="var(--primary)" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" opacity={0.45} />
      <circle cx={cx(env.start)} cy={cy(env.start)} r={4} fill="none" stroke="var(--primary)" strokeWidth={1.5} />
      <g style={{ transition: "transform 120ms linear" }} transform={`translate(${cx(here)}, ${cy(here)})`}>
        <circle r={cell / 2 - 9} fill="var(--primary)" stroke="var(--primary-foreground)" strokeWidth={1.5} />
      </g>
    </svg>
  );
}

function ThrusterPlayback({ ep, idx }: { ep: ThrusterEpisode; idx: number }) {
  const S = 170;
  const mx = (p: number) => ((Math.max(-2.4, Math.min(2.4, p)) + 2.4) / 4.8) * (S - 16) + 8;
  const my = (v: number) => ((1.6 - Math.max(-1.6, Math.min(1.6, v))) / 3.2) * (S - 16) + 8;
  const st = ep.steps[idx];
  const BW = 380;
  const bar = (v: number, range: number) => (Math.max(-range, Math.min(range, v)) / range) * (BW / 2 - 50);
  return (
    <div className="flex flex-col gap-3 sm:flex-row">
      <svg viewBox={`0 0 ${S} ${S}`} className="mx-auto h-[170px] w-[170px] shrink-0 rounded-lg border border-border bg-surface/60" role="img" aria-label="Link-1 phase plane (p1 vs v1)">
        <line x1={mx(0)} y1={8} x2={mx(0)} y2={S - 8} stroke="var(--border)" />
        <line x1={8} y1={my(0)} x2={S - 8} y2={my(0)} stroke="var(--border)" />
        <polyline points={ep.steps.slice(0, idx + 1).map((f) => `${mx(f.x[0])},${my(f.x[1])}`).join(" ")} fill="none" stroke="var(--primary)" strokeWidth={1.5} opacity={0.5} />
        <circle cx={mx(st.x[0])} cy={my(st.x[1])} r={4.5} fill="var(--primary)" stroke="var(--primary-foreground)" strokeWidth={1} />
        <text x={S - 10} y={my(0) - 4} textAnchor="end" fontSize="9" className="fill-muted-foreground font-mono">p₁</text>
        <text x={mx(0) + 4} y={17} fontSize="9" className="fill-muted-foreground font-mono">v₁</text>
      </svg>
      <svg viewBox={`0 0 ${BW} 172`} className="h-auto w-full min-w-0 max-w-[400px] flex-1 self-center" role="img" aria-label="Link positions and thruster commands at this step">
        <text x={0} y={10} fontSize="10.5" className="fill-muted-foreground font-mono">positions p1…p5</text>
        {[0, 1, 2, 3, 4].map((i) => {
          const v = bar(st.x[i * 2], 2);
          const y = 18 + i * 13;
          return (
            <g key={`p${i}`}>
              <text x={0} y={y + 8} fontSize="10.5" className="fill-muted-foreground font-mono">p{i + 1}</text>
              <line x1={BW / 2} y1={y} x2={BW / 2} y2={y + 10} stroke="var(--border)" />
              <rect x={BW / 2 + Math.min(0, v)} y={y + 1} width={Math.abs(v)} height={8} rx={2} fill="var(--chart-4)" opacity={0.8} />
              <text x={BW} y={y + 8} textAnchor="end" fontSize="10.5" className="fill-foreground font-mono">{st.x[i * 2].toFixed(2)}</text>
            </g>
          );
        })}
        <text x={0} y={95} fontSize="10.5" className="fill-muted-foreground font-mono">executed commands (dashed = ±knee)</text>
        {st.cmd.map((a, k) => {
          const v = bar(a, 1);
          const y = 103 + k * 16;
          const stalled = Math.abs(a) > KNEE;
          return (
            <g key={`a${k}`}>
              <text x={0} y={y + 9} fontSize="10.5" className="fill-muted-foreground font-mono">a{k}</text>
              <line x1={BW / 2} y1={y} x2={BW / 2} y2={y + 12} stroke="var(--border)" />
              {[-1, 1].map((sgn) => (
                <line key={sgn} x1={BW / 2 + bar(sgn * KNEE, 1)} y1={y - 1} x2={BW / 2 + bar(sgn * KNEE, 1)} y2={y + 13} stroke="var(--warning)" strokeDasharray="2 2" />
              ))}
              <rect x={BW / 2 + Math.min(0, v)} y={y + 2} width={Math.abs(v)} height={8} rx={2} fill={stalled ? "var(--warning)" : "var(--chart-1)"} />
              <text x={BW} y={y + 9} textAnchor="end" fontSize="10.5" className={stalled ? "fill-[var(--warning)] font-mono" : "fill-foreground font-mono"}>
                {a.toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function Playback({ res }: { res: Results }) {
  const ids = res.rows.map((r) => r.id);
  const [policy, setPolicy] = useState("you");
  const [episode, setEpisode] = useState(0);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const pid = ids.includes(policy) ? policy : "you";
  const nEps = res.rows[0].episodes;
  const ep = Math.min(episode, nEps - 1);
  const grid = res.kind === "grid";
  const gridEp = grid ? res.episodes[pid][ep] : null;
  const thrEp = !grid ? res.episodes[pid][ep] : null;
  const last = gridEp ? gridEp.steps.length : (thrEp?.steps.length ?? 1) - 1;
  const i = Math.min(idx, last);

  useEffect(() => {
    setIdx(0);
    setPlaying(false);
  }, [res, pid, ep]);

  useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => {
      setIdx((k) => {
        if (k >= last) {
          setPlaying(false);
          return k;
        }
        return k + 1;
      });
    }, 140);
    return () => window.clearInterval(t);
  }, [playing, last]);

  const gridEnv = grid ? playgroundGridEnv(res.envId as "gridworld-random" | "halfcheetah-medium") : null;
  const actionNames = gridEnv?.actionNames ?? null;
  const gw = gridEnv?.width ?? 1;
  const gStep = gridEp && i < gridEp.steps.length ? gridEp.steps[i] : null;
  const tStep = thrEp ? thrEp.steps[i] : null;
  const btn =
    "flex h-8 items-center justify-center rounded-md border border-border bg-card px-2.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-surface hover:text-foreground disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Seg label="Policy to play back" value={pid} onChange={setPolicy} options={res.rows.map((r) => ({ v: r.id, label: r.label }))} />
        <div className="ml-auto flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <button className={btn} onClick={() => setEpisode(Math.max(0, ep - 1))} disabled={ep === 0} aria-label="Previous episode">
            ‹
          </button>
          <span className="font-mono tabular-nums">
            episode {ep + 1}/{nEps}
          </span>
          <button className={btn} onClick={() => setEpisode(Math.min(nEps - 1, ep + 1))} disabled={ep >= nEps - 1} aria-label="Next episode">
            ›
          </button>
        </div>
      </div>

      {gridEp && <GridPlayback envId={res.envId} ep={gridEp} idx={i} />}
      {thrEp && <ThrusterPlayback ep={thrEp} idx={i} />}

      <div className="flex items-center gap-2">
        <button className={btn} onClick={() => setIdx(Math.max(0, i - 1))} disabled={i === 0} aria-label="Step back">
          ◀
        </button>
        <button
          onClick={() => {
            if (i >= last) setIdx(0);
            setPlaying((p) => !p);
          }}
          className="flex h-8 min-w-[64px] items-center justify-center rounded-md bg-primary px-3 text-xs font-semibold text-primary-foreground hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {playing ? "Pause" : "Play"}
        </button>
        <button className={btn} onClick={() => setIdx(Math.min(last, i + 1))} disabled={i >= last} aria-label="Step forward">
          ▶
        </button>
        <input
          type="range"
          min={0}
          max={last}
          value={i}
          onChange={(e) => {
            setPlaying(false);
            setIdx(Number(e.target.value));
          }}
          className="w-full accent-[var(--primary)]"
          aria-label="Scrub episode"
        />
        <span className="w-16 shrink-0 text-right font-mono text-[11px] tabular-nums text-muted-foreground">
          t={i}/{last}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {gridEp && (
          <>
            <Cell label="State (x, y)" value={gStep ? `(${gStep.s % gw}, ${Math.floor(gStep.s / gw)})` : "end"} />
            <Cell label="Action" value={gStep && actionNames ? `${gStep.a} · ${actionNames[gStep.a]}` : gridEp.goal ? "reached goal" : gridEp.hazards ? "hit hazard" : "horizon"} />
            <Cell label="Reward r(t)" value={gStep ? gStep.r.toFixed(3) : "—"} tone={gStep ? (gStep.r < 0 ? "bad" : "ok") : undefined} />
            <Cell label="Return Σr" value={(gStep ? gStep.cumR : gridEp.ret).toFixed(2)} />
          </>
        )}
        {tStep && (
          <>
            <Cell label="(p₁, v₁)" value={`${tStep.x[0].toFixed(2)}, ${tStep.x[1].toFixed(2)}`} />
            <Cell label="Command" value={tStep.cmd.map((c) => c.toFixed(2)).join(" ")} />
            <Cell label="Reward r(t)" value={tStep.r.toFixed(3)} tone="bad" />
            <Cell label="Return Σr" value={tStep.cumR.toFixed(2)} />
          </>
        )}
      </div>
    </div>
  );
}

function Cell({ label, value, tone }: { label: string; value: string; tone?: "ok" | "bad" }) {
  return (
    <div className="min-w-0 rounded-lg border border-border bg-surface/60 px-3 py-2">
      <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className={`mt-0.5 truncate font-mono text-[12.5px] tabular-nums ${tone === "bad" ? "text-destructive" : tone === "ok" ? "text-success" : ""}`}>{value}</div>
    </div>
  );
}

/* ---------------- Page body ---------------- */

const initialCodes = () => Object.fromEntries(PLAYGROUND_ENVS.map((e) => [e.id, AGENT_TEMPLATES[e.id][0].code])) as Record<PlaygroundEnvId, string>;

export function AgentPlayground() {
  const [envId, setEnvId] = useState<PlaygroundEnvId>("gridworld-random");
  const [codes, setCodes] = useState<Record<PlaygroundEnvId, string>>(initialCodes);
  const [episodes, setEpisodes] = useState(20);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<{ kind: string; message: string; line?: number } | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [showRef, setShowRef] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const code = codes[envId];
  const envMeta = PLAYGROUND_ENVS.find((e) => e.id === envId)!;
  const templates = AGENT_TEMPLATES[envId];
  const template = templates.find((t) => t.code === code);
  const guard = useMemo(() => checkUserCode(code, { requireFunction: "act" }), [code]);
  const shownError = !guard.ok ? { kind: "guard", message: guard.message, line: guard.line } : error;

  useEffect(() => () => abortRef.current?.abort(), []);

  const setCode = (v: string) => {
    setCodes((c) => ({ ...c, [envId]: v }));
    setError(null);
  };

  const switchEnv = (id: PlaygroundEnvId) => {
    abortRef.current?.abort();
    setEnvId(id);
    setError(null);
  };

  const run = async () => {
    if (!guard.ok || running) return;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setRunning(true);
    setError(null);
    const n = Math.max(1, Math.min(MAX_EPISODES, Math.floor(episodes) || 1));
    try {
      if (envMeta.kind === "grid") {
        const id = envId as "gridworld-random" | "halfcheetah-medium";
        const r = await runGridAgentSandbox(code, id, n, { signal: ctrl.signal });
        const env = playgroundGridEnv(id);
        const random = gridRandomEpisodes(env, n);
        const optimal = gridOptimalEpisodes(env, n);
        setResults({
          kind: "grid",
          envId,
          rows: [summarizeGrid("you", "Your agent", r.episodes), summarizeGrid("random", "Random", random), summarizeGrid("optimal", "Optimal (Q*)", optimal)],
          episodes: { you: r.episodes, random, optimal },
          wallMs: r.wallMs,
          hardening: r.hardening,
        });
      } else {
        const r = await runThrusterAgentSandbox(code, n, { signal: ctrl.signal });
        const coast = thrusterCoastEpisodes(n);
        const behavior = thrusterBehaviorEpisodes(n);
        setResults({
          kind: "thruster",
          envId,
          rows: [summarizeThruster("you", "Your agent", r.episodes), summarizeThruster("coast", "Coast", coast), summarizeThruster("behavior", "Behavior (ε=0.25)", behavior)],
          episodes: { you: r.episodes, coast, behavior },
          wallMs: r.wallMs,
          hardening: r.hardening,
        });
      }
    } catch (e) {
      if (!(e instanceof SandboxError && e.kind === "aborted")) setError(errorInfo(e));
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null;
      setRunning(false);
    }
  };

  const shown = results && results.envId === envId ? results : null;

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
      <Panel title="Agent logic" right={<span className="font-mono text-[10px] text-muted-foreground">act(obs, ctx)</span>}>
        <div className="space-y-3">
          <Seg label="Environment" value={envId} onChange={switchEnv} options={PLAYGROUND_ENVS.map((e) => ({ v: e.id, label: e.label }))} />
          <div className="flex flex-wrap items-center gap-2">
            <Select
              value={template?.id ?? "custom"}
              onValueChange={(id) => {
                const t = templates.find((x) => x.id === id);
                if (t) setCode(t.code);
              }}
            >
              <SelectTrigger className="h-8 w-auto min-w-[200px] px-2.5 text-xs" aria-label="Template">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="z-50">
                {templates.map((t) => (
                  <SelectItem key={t.id} value={t.id} className="text-xs">
                    Template: {t.name}
                  </SelectItem>
                ))}
                {!template && (
                  <SelectItem value="custom" className="text-xs">
                    Your code (edited)
                  </SelectItem>
                )}
              </SelectContent>
            </Select>
            {template && <span className="text-[11px] text-muted-foreground">{template.description}</span>}
          </div>

          <CodeEditor value={code} onChange={setCode} readOnly={running} errorLine={shownError?.line ?? null} height={360} ariaLabel={`Agent code for ${envMeta.label}`} />
          {shownError && <CodeError title={ERROR_TITLES[shownError.kind] ?? "Error"} message={shownError.message} />}

          <div className="flex flex-wrap items-center gap-2">
            <label className="flex items-center gap-2 text-xs text-muted-foreground">
              Episodes
              <input
                type="number"
                min={1}
                max={MAX_EPISODES}
                value={episodes}
                onChange={(e) => setEpisodes(Number(e.target.value))}
                className="h-8 w-20 rounded-md border border-input bg-transparent px-2 font-mono text-xs tabular-nums text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
            </label>
            <span className="text-[11px] text-muted-foreground">{envMeta.kind === "grid" ? `from the env start, ≤ ${GRID_HORIZON} steps each` : `${HORIZON} steps each, paired reset seeds`}</span>
            <div className="ml-auto flex items-center gap-2">
              {running && (
                <button
                  onClick={() => abortRef.current?.abort()}
                  className="rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  Stop
                </button>
              )}
              <button
                onClick={run}
                disabled={!guard.ok || running}
                className="flex items-center gap-2 rounded-md bg-primary px-4 py-1.5 text-[13px] font-semibold text-primary-foreground shadow-sm transition-colors hover:bg-primary/90 disabled:opacity-45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-card"
              >
                {running ? (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" className="animate-spin" aria-hidden>
                    <path d="M21 12a9 9 0 1 1-6.2-8.56" />
                  </svg>
                ) : (
                  <svg width="11" height="11" viewBox="0 0 12 12" fill="currentColor" aria-hidden>
                    <path d="M2.5 1.5v9l7.5-4.5z" />
                  </svg>
                )}
                {running ? "Running in sandbox…" : "Run episodes"}
              </button>
            </div>
          </div>

          <button
            onClick={() => setShowRef((v) => !v)}
            aria-expanded={showRef}
            className="text-[11px] font-medium text-muted-foreground underline-offset-2 hover:text-foreground hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            {showRef ? "Hide" : "Show"} the act(obs, ctx) reference
          </button>
          {showRef && <CodeEditor value={envMeta.kind === "grid" ? GRID_REFERENCE : THRUSTER_REFERENCE} readOnly height={232} ariaLabel="Agent API reference" />}
          <SandboxNote limitMs={AGENT_TIMEOUT_MS} removed={results?.hardening.removed.length} failed={results?.hardening.failed} />
        </div>
      </Panel>

      <div className="min-w-0 space-y-4">
        <Panel
          title="Comparison"
          right={shown ? <span className="font-mono text-[10px] text-muted-foreground">{shown.rows[0].episodes} episodes · sandbox {Math.round(shown.wallMs)} ms</span> : undefined}
        >
          {shown ? (
            <div className="space-y-2">
              <ResultsTable res={shown} />
              <p className="text-[11px] leading-relaxed text-muted-foreground">
                {shown.kind === "grid"
                  ? "Optimal = greedy policy of Q* from value iteration (γ = 0.95). Random and ctx.random() share seeds, so the Uniform-random template reproduces the Random row."
                  : "Coast never fires; Behavior is the spec's ε = 0.25 recorder that logged the offline dataset. All rows use the same reset seeds as the physics bench."}{" "}
                Your agent's actions were validated and replayed on the trusted simulator.
              </p>
            </div>
          ) : (
            <div className="flex min-h-[160px] items-center justify-center rounded-lg border border-dashed border-border text-center text-xs text-muted-foreground">
              {running ? "Running your agent in the sandbox…" : "Run episodes to compare your agent with the built-in baselines."}
            </div>
          )}
        </Panel>
        <Panel title="Episode playback">
          {shown ? (
            <Playback res={shown} />
          ) : (
            <div className="flex min-h-[160px] items-center justify-center rounded-lg border border-dashed border-border text-center text-xs text-muted-foreground">Step through an episode after a run.</div>
          )}
        </Panel>
      </div>
    </div>
  );
}
