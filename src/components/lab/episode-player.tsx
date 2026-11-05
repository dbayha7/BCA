import { useCallback, useEffect, useMemo, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";
import { ChevronLeft, ChevronRight, Keyboard, Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { greedy, mulberry32, type Dataset } from "@/lib/rl";
import { KNEE, THRUSTER_PROJECTION, rolloutThruster, thrustStall, type ThrusterFrame } from "@/lib/thrusterchain";
import {
  clampIndex,
  episodeValueTrace,
  inspectState,
  playbackKey,
  thrusterCell,
  thrusterSummary,
  type PlaybackStep,
  type ValueTrace,
} from "@/lib/playback";
import { PhysicsBench } from "./physics-bench";
import { SafetyPanel } from "./safety-panel";

type Mode = "policy" | "behavior";

interface GridFrame {
  s: number;
  a: number;
  r: number;
  cumR: number;
  done: boolean;
}

function rolloutGridLocal(ds: Dataset, mode: Mode, policy: (s: number) => number, seed: number): GridFrame[] {
  const rng = mulberry32(seed * 7919 + 13);
  const env = ds.env;
  let s = env.start;
  const frames: GridFrame[] = [];
  let cum = 0;
  for (let t = 0; t < 80; t++) {
    let a: number;
    if (mode === "policy") {
      a = policy(s);
    } else {
      const b = s * 4;
      const tot = ds.counts[b] + ds.counts[b + 1] + ds.counts[b + 2] + ds.counts[b + 3];
      if (tot === 0) a = Math.floor(rng() * 4);
      else {
        let u = rng() * tot;
        a = 3;
        for (let k = 0; k < 4; k++) {
          u -= ds.counts[b + k];
          if (u <= 0) {
            a = k;
            break;
          }
        }
      }
    }
    const { s2, r, done } = env.step(s, a);
    cum += r;
    frames.push({ s, a, r, cumR: cum, done });
    if (done) break;
    s = s2;
  }
  return frames;
}

const SPEEDS = [0.5, 1, 2, 4];
const DIR_X = [0, 1, 0, -1];
const DIR_Y = [-1, 0, 1, 0];

function IconButton({ label, onClick, disabled, children }: { label: string; onClick: () => void; disabled?: boolean; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border bg-card text-muted-foreground transition-colors hover:bg-surface hover:text-foreground disabled:pointer-events-none disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      {children}
    </button>
  );
}

function ControlBar({
  mode,
  setMode,
  trained,
  playing,
  onTogglePlay,
  speed,
  setSpeed,
  idx,
  total,
  onSeek,
  onStep,
  onNewEpisode,
  extra,
}: {
  mode: Mode;
  setMode: (m: Mode) => void;
  trained: boolean;
  playing: boolean;
  onTogglePlay: () => void;
  speed: number;
  setSpeed: (s: number) => void;
  idx: number;
  total: number;
  onSeek: (i: number) => void;
  onStep: (delta: number) => void;
  onNewEpisode: () => void;
  extra?: ReactNode;
}) {
  const atStart = idx <= 0;
  const atEnd = idx >= total - 1;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex rounded-md border border-border bg-surface p-0.5 text-[11px]">
          {(["policy", "behavior"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              aria-pressed={mode === m}
              className={`rounded px-2.5 py-1 font-medium transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                mode === m ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {m === "policy" ? "Trained policy" : "Behavior (recorder)"}
            </button>
          ))}
        </div>
        {extra}
        {mode === "policy" && !trained && (
          <span className="rounded border border-warning/40 bg-warning/10 px-2 py-0.5 text-[10.5px] text-foreground">Q is untrained — the greedy policy is arbitrary until you Step Training.</span>
        )}
        <div className="ml-auto flex items-center gap-2">
          <div className="flex rounded-md border border-border bg-surface p-0.5 text-[11px]" aria-label="Playback speed">
            {SPEEDS.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSpeed(s)}
                aria-pressed={speed === s}
                className={`rounded px-2 py-1 font-mono transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${speed === s ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
              >
                {s}×
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={onNewEpisode}
            className="flex h-8 items-center gap-1.5 rounded-md border border-border bg-card px-3 text-xs font-medium text-muted-foreground transition-colors hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"><path d="M21 12a9 9 0 1 1-2.6-6.4M21 3v5h-5" /></svg>
            New episode
          </button>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 sm:flex-nowrap">
        <div className="flex items-center gap-1">
          <IconButton label="Jump to start (Home)" onClick={() => onSeek(0)} disabled={atStart}>
            <SkipBack size={13} />
          </IconButton>
          <IconButton label="Previous step (←)" onClick={() => onStep(-1)} disabled={atStart}>
            <ChevronLeft size={15} />
          </IconButton>
          <button
            type="button"
            onClick={onTogglePlay}
            disabled={total === 0}
            aria-label={playing ? "Pause (Space)" : "Play (Space)"}
            title={playing ? "Pause (Space)" : "Play (Space)"}
            className="flex h-8 w-[74px] items-center justify-center gap-1.5 rounded-md bg-primary px-3 text-xs font-semibold text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-card"
          >
            {playing ? <Pause size={12} fill="currentColor" /> : <Play size={12} fill="currentColor" />}
            {playing ? "Pause" : "Play"}
          </button>
          <IconButton label="Next step (→)" onClick={() => onStep(1)} disabled={atEnd}>
            <ChevronRight size={15} />
          </IconButton>
          <IconButton label="Jump to end (End)" onClick={() => onSeek(total - 1)} disabled={atEnd}>
            <SkipForward size={13} />
          </IconButton>
        </div>
        <span className="shrink-0 rounded-md border border-border bg-surface px-2 py-1 font-mono text-[11px] tabular-nums" aria-live="off" data-testid="step-counter">
          step {total === 0 ? 0 : idx + 1} / {total}
        </span>
        <input
          type="range"
          min={0}
          max={Math.max(0, total - 1)}
          value={idx}
          onChange={(e) => onSeek(Number(e.target.value))}
          className="min-w-[120px] flex-1 accent-[var(--primary)]"
          aria-label="Scrub episode"
          aria-valuetext={`step ${idx + 1} of ${total}`}
        />
      </div>
      <p className="flex items-center gap-1.5 text-[10.5px] text-muted-foreground transition-colors group-focus-within:text-foreground/80">
        <Keyboard size={12} className="shrink-0" />
        <span>
          Click the player, then <Kbd>←</Kbd>/<Kbd>→</Kbd> step (<Kbd>Shift</Kbd> ×10) · <Kbd>Space</Kbd> play/pause · <Kbd>Home</Kbd>/<Kbd>End</Kbd> jump
        </span>
      </p>
    </div>
  );
}

function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded border border-border bg-surface px-1 font-mono text-[9.5px] text-foreground">{children}</kbd>;
}

/** Vertical force arrow: positive length points up (thrust pushes the links up the screen). */
function VArrow({ x, y, len, color, width, dashed, opacity = 1 }: { x: number; y: number; len: number; color: string; width: number; dashed?: boolean; opacity?: number }) {
  if (Math.abs(len) < 1.5) return null;
  const dir = len > 0 ? -1 : 1;
  const tip = y + dir * Math.abs(len);
  const hl = Math.min(5.5, Math.abs(len) * 0.6);
  const base = tip - dir * hl;
  return (
    <g opacity={opacity}>
      <line x1={x} y1={y} x2={x} y2={base} stroke={color} strokeWidth={width} strokeLinecap="round" strokeDasharray={dashed ? "2.5 2" : undefined} />
      <polygon
        points={`${x},${tip} ${x - hl * 0.8},${base} ${x + hl * 0.8},${base}`}
        fill={dashed ? "none" : color}
        stroke={dashed ? color : "none"}
        strokeWidth={dashed ? 1 : 0}
        strokeLinejoin="round"
      />
    </g>
  );
}

const POD_TRAIL = 8;
const TRAIL_DX = 5; // px each older trail point drifts left, like a strip chart

/** Animated side view of the real 13-dim ThrusterChain: 5 pods, 4 thrusters, current + wave forcing. */
function ThrusterScene({ frames, idx }: { frames: ThrusterFrame[]; idx: number }) {
  const f = frames[idx];
  const W = 540;
  const H = 240;
  const baseY = 114;
  const px = (i: number) => 70 + i * 100;
  const podY = (fr: ThrusterFrame, i: number) => baseY - Math.max(-1.8, Math.min(1.8, fr.x[i * 2])) * 42;
  const py = (i: number) => podY(f, i);
  const th = thrustStall(f.cmd);
  const waveAmp = Math.hypot(f.x[11], f.x[12]);
  const wavePhase = Math.atan2(f.x[12], f.x[11]);
  const current = f.x[10];
  const energy = thrusterSummary(f.x).energy;
  const wavePath = Array.from({ length: 55 }, (_, k) => {
    const x = (k / 54) * W;
    const y = 26 + Math.sin((x / W) * Math.PI * 4 + wavePhase) * waveAmp * 22;
    return `${k === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(" ");
  const trail = frames.slice(Math.max(0, idx - POD_TRAIL), idx + 1);
  const ARROW = 26; // px per unit of command / thrust
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg border border-border" role="img" aria-label={`ThrusterChain step ${idx + 1}: oscillator energy ${energy.toFixed(2)}`}>
      <defs>
        <linearGradient id="ocean" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="color-mix(in oklch, var(--info) 14%, var(--card))" />
          <stop offset="1" stopColor="color-mix(in oklch, var(--info) 4%, var(--card))" />
        </linearGradient>
      </defs>
      <rect width={W} height={H} fill="url(#ocean)" />
      {/* wave surface */}
      <path d={wavePath} fill="none" stroke="var(--info)" strokeWidth="1.5" opacity="0.6" />
      {/* current arrow */}
      <g opacity="0.8">
        <line x1={16} y1={52} x2={16 + current * 56} y2={52} stroke="var(--info)" strokeWidth="2.5" strokeLinecap="round" />
        <polygon points={`${16 + current * 56 + 6},52 ${16 + current * 56 - 1},48.5 ${16 + current * 56 - 1},55.5`} fill="var(--info)" />
        <text x={16} y={42} fontSize="9" className="fill-muted-foreground font-mono">current {current.toFixed(2)}</text>
      </g>
      <text x={W - 10} y={11} textAnchor="end" fontSize="9" className="fill-muted-foreground font-mono">
        t {idx + 1}/{frames.length} · Σx² {energy.toFixed(2)}
      </text>
      {/* rest line */}
      <line x1={30} y1={baseY} x2={W - 30} y2={baseY} stroke="var(--border)" strokeDasharray="4 4" />
      {/* per-pod motion trails: recent positions drift left like a strip chart */}
      {Array.from({ length: 5 }, (_, i) =>
        trail.map((fr, k) => {
          if (k === 0) return null;
          const prev = trail[k - 1];
          const n = trail.length - 1;
          const age = (n - k) / Math.max(1, n);
          return (
            <line
              key={`tr${i}-${k}`}
              x1={px(i) - (n - k + 1) * TRAIL_DX}
              y1={podY(prev, i)}
              x2={px(i) - (n - k) * TRAIL_DX}
              y2={podY(fr, i)}
              stroke="var(--primary)"
              strokeWidth={0.8 + (1 - age) * 1.4}
              strokeLinecap="round"
              opacity={0.04 + (1 - age) * 0.42}
            />
          );
        }),
      )}
      {/* chain */}
      <polyline points={Array.from({ length: 5 }, (_, i) => `${px(i)},${py(i)}`).join(" ")} fill="none" stroke="var(--foreground)" strokeWidth="2" opacity="0.5" />
      {/* thrusters: cmd k drives links k & k+1. Dashed = commanded, solid = thrust after the stall knee. */}
      {f.cmd.map((a, k) => {
        const mx = (px(k) + px(k + 1)) / 2;
        const my = (py(k) + py(k + 1)) / 2 + 24;
        const stalled = Math.abs(a) > KNEE;
        const col = stalled ? "var(--warning)" : "var(--chart-1)";
        const y0 = (len: number) => my + (len > 0 ? -5 : 5);
        return (
          <g key={k}>
            <line x1={mx} y1={my - 5} x2={mx} y2={(py(k) + py(k + 1)) / 2} stroke="var(--border)" strokeWidth="1" strokeDasharray="2 3" />
            <VArrow x={mx} y={y0(a)} len={a * ARROW} color={col} width={1.4} dashed opacity={0.9} />
            <VArrow x={mx} y={y0(th[k])} len={th[k] * ARROW} color={col} width={3.5} opacity={0.85} />
            <rect x={mx - 7} y={my - 5} width={14} height={10} rx={2.5} fill="var(--card)" stroke={stalled ? "var(--warning)" : "var(--border)"} strokeWidth={stalled ? 1.8 : 1} />
            <text x={mx + 10} y={my + 3.5} fontSize="8.5" className={stalled ? "fill-[var(--warning)] font-mono" : "fill-muted-foreground font-mono"}>
              a{k} {a.toFixed(2)}
              {stalled ? " ⚠" : ""}
            </text>
          </g>
        );
      })}
      {Array.from({ length: 5 }, (_, i) => {
        const v = f.x[i * 2 + 1];
        const vMag = Math.min(1, Math.abs(v) / 1.6);
        return (
          <g key={i}>
            <circle cx={px(i)} cy={py(i)} r={11} fill={`color-mix(in oklch, var(--primary) ${Math.round(30 + vMag * 65)}%, var(--card))`} stroke="var(--foreground)" strokeWidth="1.2" style={{ transition: "fill 130ms linear" }} />
            <text x={px(i)} y={py(i) + 3.5} textAnchor="middle" fontSize="9" fontWeight={700} className="fill-foreground font-mono">{i + 1}</text>
            <text x={px(i)} y={py(i) - 16} textAnchor="middle" fontSize="8.5" className="fill-muted-foreground font-mono">
              p {f.x[i * 2].toFixed(2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** (p1, v1) phase-plane trail on the tabular grid — the projection the agent actually learns on. */
function PhaseTrail({ frames, idx }: { frames: ThrusterFrame[]; idx: number }) {
  const S = 150;
  const { xAxis, yAxis } = THRUSTER_PROJECTION;
  const mx = (p: number) => ((Math.max(xAxis.lo, Math.min(xAxis.hi, p)) - xAxis.lo) / (xAxis.hi - xAxis.lo)) * (S - 16) + 8;
  const my = (v: number) => ((yAxis.hi - Math.max(yAxis.lo, Math.min(yAxis.hi, v))) / (yAxis.hi - yAxis.lo)) * (S - 16) + 8;
  const dp = (xAxis.hi - xAxis.lo) / xAxis.bins;
  const dv = (yAxis.hi - yAxis.lo) / yAxis.bins;
  const cur = frames[idx] ? thrusterCell(frames[idx].x) : null;
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="h-[150px] w-[150px] shrink-0 rounded-lg border border-border bg-surface/60" role="img" aria-label={cur ? `Phase plane, current cell (${cur.xi}, ${cur.yi})` : "Phase plane"}>
      {Array.from({ length: xAxis.bins + 1 }, (_, k) => (
        <line key={`gx${k}`} x1={mx(xAxis.lo + k * dp)} y1={8} x2={mx(xAxis.lo + k * dp)} y2={S - 8} stroke="var(--border)" strokeWidth="0.5" opacity="0.6" />
      ))}
      {Array.from({ length: yAxis.bins + 1 }, (_, k) => (
        <line key={`gy${k}`} x1={8} y1={my(yAxis.hi - k * dv)} x2={S - 8} y2={my(yAxis.hi - k * dv)} stroke="var(--border)" strokeWidth="0.5" opacity="0.6" />
      ))}
      {cur && (
        <rect
          x={mx(xAxis.lo + cur.xi * dp)}
          y={my(yAxis.hi - cur.yi * dv)}
          width={mx(xAxis.lo + (cur.xi + 1) * dp) - mx(xAxis.lo + cur.xi * dp)}
          height={my(yAxis.hi - (cur.yi + 1) * dv) - my(yAxis.hi - cur.yi * dv)}
          fill="color-mix(in oklch, var(--primary) 22%, transparent)"
          stroke="var(--primary)"
          strokeWidth="0.8"
        />
      )}
      <line x1={mx(0)} y1={8} x2={mx(0)} y2={S - 8} stroke="var(--border)" />
      <line x1={8} y1={my(0)} x2={S - 8} y2={my(0)} stroke="var(--border)" />
      {frames.slice(Math.max(0, idx - 28), idx + 1).map((f, k, arr) => {
        if (k === 0) return null;
        const prev = arr[k - 1];
        const age = (arr.length - 1 - k) / Math.max(1, arr.length - 1);
        return (
          <line
            key={`${idx}-${k}`}
            x1={mx(prev.x[0])}
            y1={my(prev.x[1])}
            x2={mx(f.x[0])}
            y2={my(f.x[1])}
            stroke="var(--primary)"
            strokeWidth={1 + (1 - age) * 1.6}
            strokeLinecap="round"
            opacity={0.08 + (1 - age) * 0.75}
          />
        );
      })}
      {frames[idx] && (
        <g style={{ transition: "transform 130ms linear" }} transform={`translate(${mx(frames[idx].x[0])}, ${my(frames[idx].x[1])})`}>
          <circle r={7} fill="var(--primary)" opacity="0.22" className="lab-pulse" />
          <circle r={4} fill="var(--primary)" stroke="var(--primary-foreground)" strokeWidth="1" />
        </g>
      )}
      <text x={S - 10} y={my(0) - 4} textAnchor="end" fontSize="8" className="fill-muted-foreground font-mono">p₁</text>
      <text x={mx(0) + 4} y={16} fontSize="8" className="fill-muted-foreground font-mono">v₁</text>
    </svg>
  );
}

function GridScene({ ds, frames, idx, Q, version, showPolicy }: { ds: Dataset; frames: GridFrame[]; idx: number; Q: Float64Array; version: number; showPolicy: boolean }) {
  const env = ds.env;
  const cell = 40;
  const W = env.width;
  const H = env.height;
  const cx = (s: number) => (s % W) * cell + cell / 2;
  const cy = (s: number) => Math.floor(s / W) * cell + cell / 2;
  const f = frames[idx];

  // Greedy action of every logged state, recomputed whenever training advances (live while Auto-train runs).
  const greedyArrows = useMemo(() => {
    void version;
    if (!showPolicy) return [];
    const out: { s: number; a: number; ood: boolean }[] = [];
    for (let s = 0; s < W * H; s++) {
      if (env.blocked.has(s) || env.terminals.has(s) || ds.visits[s] === 0) continue;
      const a = greedy(Q, s);
      out.push({ s, a, ood: ds.counts[s * 4 + a] === 0 });
    }
    return out;
  }, [ds, env, Q, version, showPolicy, W, H]);

  return (
    <svg viewBox={`-2 -2 ${W * cell + 4} ${H * cell + 4}`} className="mx-auto block max-h-[340px] w-full" role="img" aria-label={f ? `Agent at cell (${f.s % W}, ${Math.floor(f.s / W)}) taking ${env.actionNames[f.a]}` : "Episode grid"}>
      {Array.from({ length: W * H }, (_, s) => {
        const x = (s % W) * cell;
        const y = Math.floor(s / W) * cell;
        const blocked = env.blocked.has(s);
        const term = env.terminals.get(s);
        let fill = "var(--card)";
        if (blocked) fill = "color-mix(in oklch, var(--foreground) 55%, var(--background))";
        else if (term === "goal") fill = "color-mix(in oklch, var(--success) 28%, var(--card))";
        else if (term === "hazard") fill = "color-mix(in oklch, var(--destructive) 22%, var(--card))";
        return (
          <g key={s}>
            <rect x={x + 1.5} y={y + 1.5} width={cell - 3} height={cell - 3} rx={5} fill={fill} stroke="var(--border)" />
            {term && (
              <text x={x + cell / 2} y={y + cell / 2 + 4} textAnchor="middle" fontSize="11" fontWeight={600} className="fill-foreground font-mono">
                {term === "goal" ? "G" : "×"}
              </text>
            )}
          </g>
        );
      })}
      {greedyArrows.map(({ s, a, ood }) => {
        const L = cell * 0.2;
        const x0 = cx(s) - DIR_X[a] * L;
        const y0 = cy(s) - DIR_Y[a] * L;
        const x1 = cx(s) + DIR_X[a] * L;
        const y1 = cy(s) + DIR_Y[a] * L;
        const hx = DIR_X[a] * 3.5;
        const hy = DIR_Y[a] * 3.5;
        return (
          <g key={`g${s}`} opacity={ood ? 0.75 : 0.38}>
            <line x1={x0} y1={y0} x2={x1 - hx * 0.6} y2={y1 - hy * 0.6} stroke={ood ? "var(--warning)" : "var(--muted-foreground)"} strokeWidth={1.3} strokeLinecap="round" />
            <polygon points={`${x1},${y1} ${x1 - hx - hy * 0.8},${y1 - hy - hx * 0.8} ${x1 - hx + hy * 0.8},${y1 - hy + hx * 0.8}`} fill={ood ? "var(--warning)" : "var(--muted-foreground)"} />
          </g>
        );
      })}
      {frames.slice(Math.max(0, idx - 18), idx + 1).map((fr, k, arr) => {
        if (k === 0) return null;
        const prev = arr[k - 1];
        if (prev.s === fr.s) return null;
        const age = (arr.length - 1 - k) / Math.max(1, arr.length - 1);
        return (
          <line
            key={`t${idx}-${k}`}
            x1={cx(prev.s)}
            y1={cy(prev.s)}
            x2={cx(fr.s)}
            y2={cy(fr.s)}
            stroke="var(--primary)"
            strokeWidth={1.5 + (1 - age) * 2.2}
            strokeLinecap="round"
            opacity={0.06 + (1 - age) * 0.6}
          />
        );
      })}
      {frames.slice(Math.max(0, idx - 18), idx + 1).map((fr, k, arr) => {
        const age = (arr.length - 1 - k) / Math.max(1, arr.length - 1);
        if (k === arr.length - 1) return null;
        return <circle key={`d${idx}-${k}`} cx={cx(fr.s)} cy={cy(fr.s)} r={2.5 + (1 - age) * 1.5} fill="var(--primary)" opacity={0.05 + (1 - age) * 0.4} />;
      })}
      {f && (
        <g style={{ transition: "transform 130ms linear" }} transform={`translate(${cx(f.s)}, ${cy(f.s)})`}>
          <circle r={cell / 2 - 7} fill="var(--primary)" opacity="0.2" className="lab-pulse" />
          <circle r={cell / 2 - 10} fill="var(--primary)" stroke="var(--primary-foreground)" strokeWidth="1.5" />
          {/* the move taken at this step */}
          <g transform={`rotate(${f.a * 90})`}>
            <line x1={0} y1={-cell / 2 + 9} x2={0} y2={-cell / 2 + 2} stroke="var(--foreground)" strokeWidth="2.2" strokeLinecap="round" />
            <polygon points={`0,${-cell / 2 - 2} -4,${-cell / 2 + 3.5} 4,${-cell / 2 + 3.5}`} fill="var(--foreground)" />
          </g>
        </g>
      )}
    </svg>
  );
}

/** Critic prediction Q(s_t,a_t) vs. the discounted return actually obtained, across the whole episode. Click/drag to seek. */
function ValueTraceChart({ trace, idx, onSeek }: { trace: ValueTrace; idx: number; onSeek: (i: number) => void }) {
  const n = trace.predicted.length;
  const W = 600;
  const H = 70;
  const pad = 5;
  if (n < 2) return null;
  let lo = Infinity;
  let hi = -Infinity;
  for (let t = 0; t < n; t++) {
    lo = Math.min(lo, trace.predicted[t], trace.actual[t]);
    hi = Math.max(hi, trace.predicted[t], trace.actual[t]);
  }
  if (!isFinite(lo) || !isFinite(hi)) return null;
  if (hi - lo < 1e-9) {
    lo -= 0.5;
    hi += 0.5;
  }
  const xAt = (t: number) => pad + (t / (n - 1)) * (W - 2 * pad);
  const yAt = (v: number) => pad + (1 - (v - lo) / (hi - lo)) * (H - 2 * pad);
  const pts = (arr: Float64Array) => Array.from(arr, (v, t) => `${xAt(t).toFixed(1)},${yAt(v).toFixed(1)}`).join(" ");
  const seekFrom = (e: PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    if (r.width <= 0) return;
    onSeek(Math.round(((e.clientX - r.left) / r.width) * (n - 1)));
  };
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className="h-[70px] w-full cursor-crosshair touch-pan-y rounded-md border border-border bg-surface/60"
      role="img"
      aria-label="Predicted Q versus realised discounted return over the episode"
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        seekFrom(e);
      }}
      onPointerMove={(e) => {
        if (e.buttons & 1) seekFrom(e);
      }}
    >
      {lo < 0 && hi > 0 && <line x1={0} y1={yAt(0)} x2={W} y2={yAt(0)} stroke="var(--border)" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />}
      <polyline points={pts(trace.actual)} fill="none" stroke="var(--chart-2)" strokeWidth="1.6" strokeDasharray="4 3" vectorEffect="non-scaling-stroke" />
      <polyline points={pts(trace.predicted)} fill="none" stroke="var(--chart-1)" strokeWidth="1.8" vectorEffect="non-scaling-stroke" />
      <line x1={xAt(idx)} y1={0} x2={xAt(idx)} y2={H} stroke="var(--foreground)" strokeWidth="1" opacity="0.55" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function Field({ label, children, sub, tone }: { label: string; children: ReactNode; sub?: ReactNode; tone?: string }) {
  return (
    <div className="min-w-0 rounded-lg border border-border bg-surface/60 px-3 py-2">
      <div className="text-[10px] font-medium uppercase leading-tight tracking-wider text-muted-foreground">{label}</div>
      <div className={`mt-0.5 truncate font-mono text-[13px] tabular-nums ${tone ?? ""}`}>{children}</div>
      {sub && <div className="mt-0.5 text-[10.5px] leading-snug text-muted-foreground">{sub}</div>}
    </div>
  );
}

function SectionLabel({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-1.5 flex items-baseline justify-between gap-2">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{children}</div>
      {right && <div className="text-[10.5px] text-muted-foreground">{right}</div>}
    </div>
  );
}

const fmtSigned = (v: number, d = 3) => `${v >= 0 ? "+" : ""}${v.toFixed(d)}`;

function StepInspector({
  ds,
  Q,
  steps,
  idx,
  cumR,
  trace,
  gamma,
  mode,
  thruster,
  ending,
  onSeek,
}: {
  ds: Dataset;
  Q: Float64Array;
  steps: PlaybackStep[];
  idx: number;
  cumR: number;
  trace: ValueTrace;
  gamma: number;
  mode: Mode;
  thruster?: ThrusterFrame;
  ending: "terminated" | "truncated";
  onSeek: (i: number) => void;
}) {
  const { env } = ds;
  const step = steps[idx];
  const ins = inspectState(Q, ds.counts, step.s);
  const taken = ins.actions[step.a];
  const qs = ins.actions.map((x) => x.q);
  const qLo = Math.min(...qs);
  const qHi = Math.max(...qs);
  const predicted = trace.predicted[idx];
  const actual = trace.actual[idx];
  const gap = predicted - actual;
  const gx = step.s % env.width;
  const gy = Math.floor(step.s / env.width);
  const sum = thruster ? thrusterSummary(thruster.x) : null;
  const thrust = thruster ? thrustStall(thruster.cmd) : null;
  const last = idx === steps.length - 1;
  const gapScale = Math.max(1e-6, Math.abs(actual), Math.abs(predicted));
  const gapTone = Math.abs(gap) / gapScale < 0.1 ? "text-success" : gap > 0 ? "text-warning" : "text-info";

  return (
    <div className="space-y-3 rounded-lg border border-border bg-card p-3" data-testid="step-inspector">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Step inspector</h3>
        <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
          step {idx + 1} / {steps.length} · {mode === "policy" ? "trained policy" : "behavior replay"}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Field
          label={thruster ? "Grid cell (p₁, v₁)" : "State (x, y)"}
          sub={ds.visits[step.s] > 0 ? `${ds.visits[step.s].toLocaleString()} logged visits` : <span className="text-warning">never logged</span>}
        >
          ({gx}, {gy})
        </Field>
        <Field
          label="Action taken"
          sub={
            taken.ood ? (
              <span className="font-medium text-warning">OOD — never logged here</span>
            ) : step.a === ins.greedy ? (
              "= greedy action"
            ) : (
              `greedy would be ${env.actionNames[ins.greedy]}`
            )
          }
        >
          <span className="font-sans font-medium">{env.actionNames[step.a]}</span>
        </Field>
        <Field label="Reward r(t)" tone={step.r < 0 ? "text-destructive" : "text-success"}>
          {step.r.toFixed(3)}
        </Field>
        <Field label="Return Σr so far">{cumR.toFixed(2)}</Field>
      </div>

      {sum && thruster && thrust && (
        <div className="grid gap-3 lg:grid-cols-[minmax(0,5fr)_minmax(0,4fr)]">
          <div>
            <SectionLabel right="13-dim, before the step">Continuous state</SectionLabel>
            <div className="grid grid-cols-3 gap-1.5 font-mono text-[11px] tabular-nums sm:grid-cols-5">
              {[
                ["p₁", sum.p1.toFixed(3)],
                ["v₁", sum.v1.toFixed(3)],
                ["Σx²", sum.energy.toFixed(3)],
                ["current", sum.current.toFixed(3)],
                ["wave |A|", sum.waveAmp.toFixed(3)],
              ].map(([k, v]) => (
                <div key={k} className="rounded-md border border-border bg-surface/60 px-2 py-1">
                  <div className="text-[9.5px] text-muted-foreground">{k}</div>
                  <div>{v}</div>
                </div>
              ))}
            </div>
          </div>
          <div>
            <SectionLabel right="cmd → thrust">Command (4-dim)</SectionLabel>
            <div className="grid grid-cols-4 gap-1.5 font-mono text-[11px] tabular-nums">
              {thruster.cmd.map((c, k) => {
                const stalled = Math.abs(c) > KNEE;
                return (
                  <div key={k} className={`rounded-md border px-2 py-1 ${stalled ? "border-warning/60 bg-warning/10" : "border-border bg-surface/60"}`} title={stalled ? `|a${k}| > knee ${KNEE.toFixed(2)}: thrust stalls` : undefined}>
                    <div className="text-[9.5px] text-muted-foreground">
                      a{k}
                      {stalled && <span className="ml-1 font-semibold text-warning">stall</span>}
                    </div>
                    <div>{c.toFixed(2)}</div>
                    <div className="text-[9.5px] text-muted-foreground">→ {thrust[k].toFixed(2)}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      <div>
        <SectionLabel right={`${ins.visits.toLocaleString()} logged samples in this state`}>Q(s, ·) and data support</SectionLabel>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {ins.actions.map((ai) => {
            const pct = qHi - qLo < 1e-12 ? 100 : 8 + (92 * (ai.q - qLo)) / (qHi - qLo);
            return (
              <div
                key={ai.a}
                className={`min-w-0 rounded-md border px-2.5 py-2 ${ai.greedy ? "border-primary bg-accent" : "border-border bg-surface/60"}`}
                data-greedy={ai.greedy ? "true" : undefined}
                data-ood={ai.ood ? "true" : undefined}
              >
                <div className="flex flex-wrap items-center justify-between gap-1">
                  <span className="min-w-0 truncate text-[10.5px] text-muted-foreground">{env.actionNames[ai.a]}</span>
                  <span className="flex shrink-0 gap-1">
                    {ai.greedy && <span className="rounded bg-primary px-1 text-[9px] font-semibold uppercase tracking-wide text-primary-foreground">greedy</span>}
                    {ai.a === step.a && <span className="rounded border border-foreground/40 px-1 text-[9px] font-semibold uppercase tracking-wide text-foreground">taken</span>}
                  </span>
                </div>
                <div className="mt-0.5 font-mono text-[14px] font-medium tabular-nums">{ai.q.toFixed(3)}</div>
                <div className="mt-1 h-1 overflow-hidden rounded-full bg-border/60">
                  <div className={`h-full rounded-full ${ai.greedy ? "bg-primary" : "bg-muted-foreground/50"}`} style={{ width: `${pct}%` }} />
                </div>
                <div className="mt-1 font-mono text-[10.5px] tabular-nums">
                  {ai.ood ? <span className="font-semibold text-warning">OOD · 0 logged</span> : <span className="text-muted-foreground">×{ai.count.toLocaleString()} logged</span>}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div>
        <SectionLabel right={<span className="font-mono">γ = {gamma.toFixed(2)}</span>}>Predicted vs. obtained from this step</SectionLabel>
        <div className="grid grid-cols-3 gap-2">
          <Field label="Q(s, a) predicted">{predicted.toFixed(3)}</Field>
          <Field label="Return-to-go G_t">{actual.toFixed(3)}</Field>
          <Field label="Q − G" tone={gapTone} sub={Math.abs(gap) / gapScale < 0.1 ? "close" : gap > 0 ? "critic optimistic" : "critic pessimistic"}>
            {fmtSigned(gap)}
          </Field>
        </div>
        <div className="mt-2">
          <ValueTraceChart trace={trace} idx={idx} onSeek={onSeek} />
          <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[10.5px] text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span className="h-0.5 w-4 rounded-full" style={{ background: "var(--chart-1)" }} /> Q(s_t, a_t)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-0 w-4 border-t-2 border-dashed" style={{ borderColor: "var(--chart-2)" }} /> G_t = Σ γᵏ r_(t+k) in this episode
            </span>
            <span>
              {ending === "terminated"
                ? `Episode terminated after ${steps.length} steps — G_t is exact.`
                : `Episode cut at ${steps.length} steps — G_t omits rewards after the cut${last ? " (this is the last step)" : ""}.`}
            </span>
            {mode === "behavior" && <span>Q assumes greedy actions afterwards; the recorder does not follow them.</span>}
          </div>
        </div>
      </div>
    </div>
  );
}

export interface EpisodePlayerProps {
  ds: Dataset;
  Q: Float64Array;
  version: number;
  trained: boolean;
  /** Discount used for the return-to-go comparison in the inspector. Pass the training γ (hyper.gamma); defaults to 0.95. */
  gamma?: number;
  /** Rendered directly under the ThrusterChain animation for the frame on screen (e.g. a stress overlay). */
  renderFrameExtras?: (frame: ThrusterFrame, idx: number) => ReactNode;
}

export function EpisodePlayer({ ds, Q, version, trained, gamma = 0.95, renderFrameExtras }: EpisodePlayerProps) {
  const isThruster = ds.env.id === "thrusterchain-stall";
  const [mode, setMode] = useState<Mode>(trained ? "policy" : "behavior");
  const [seed, setSeed] = useState(1);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [showPolicy, setShowPolicy] = useState(true);

  const policy = useMemo(() => {
    void version;
    return (s: number) => {
      const b = s * 4;
      let best = 0;
      for (let a = 1; a < 4; a++) if (Q[b + a] > Q[b + best]) best = a;
      return best;
    };
  }, [Q, version]);

  const thrusterFrames = useMemo(
    () => (isThruster ? rolloutThruster(mode === "policy" ? { mode: "policy", policy } : { mode: "behavior" }, seed) : null),
    [isThruster, mode, policy, seed],
  );
  const gridFrames = useMemo(() => (!isThruster ? rolloutGridLocal(ds, mode, policy, seed) : null), [isThruster, ds, mode, policy, seed]);

  const total = thrusterFrames?.length ?? gridFrames?.length ?? 0;

  // Uniform (tabular state, action, reward) view of either episode type for the inspector.
  const steps = useMemo<PlaybackStep[]>(() => {
    if (thrusterFrames) return thrusterFrames.map((f) => ({ s: thrusterCell(f.x).cell, a: f.a, r: f.r }));
    if (gridFrames) return gridFrames.map((f) => ({ s: f.s, a: f.a, r: f.r }));
    return [];
  }, [thrusterFrames, gridFrames]);

  const trace = useMemo(() => {
    void version;
    return episodeValueTrace(Q, steps, gamma);
  }, [Q, steps, gamma, version]);

  useEffect(() => {
    setIdx(0);
    setPlaying(true);
  }, [mode, seed, ds]);

  useEffect(() => {
    if (!playing || total === 0) return;
    const id = window.setInterval(() => setIdx((i) => Math.min(i + 1, total - 1)), 130 / speed);
    return () => window.clearInterval(id);
  }, [playing, speed, total]);

  const safeIdx = clampIndex(idx, total);

  useEffect(() => {
    if (playing && total > 0 && safeIdx >= total - 1) setPlaying(false);
  }, [playing, safeIdx, total]);

  const seek = useCallback(
    (i: number) => {
      setPlaying(false);
      setIdx(clampIndex(i, total));
    },
    [total],
  );
  const stepBy = useCallback(
    (delta: number) => {
      setPlaying(false);
      setIdx((i) => clampIndex(clampIndex(i, total) + delta, total));
    },
    [total],
  );
  const togglePlay = useCallback(() => {
    if (playing) {
      setPlaying(false);
      return;
    }
    if (safeIdx >= total - 1) setIdx(0);
    setPlaying(true);
  }, [playing, safeIdx, total]);

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
    const target = e.target as HTMLElement;
    const tag = target.tagName;
    const isRange = tag === "INPUT" && (target as HTMLInputElement).type === "range";
    // Native controls keep their own keys: the scrubber already steps on ←/→/Home/End, buttons click on Space.
    if (tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable) return;
    if (tag === "INPUT" && !(isRange && e.key === " ")) return;
    if ((tag === "BUTTON" || target.getAttribute("role") === "button") && (e.key === " " || e.key === "Enter")) return;
    const cmd = playbackKey(e.key, e.shiftKey);
    if (!cmd) return;
    e.preventDefault();
    if (cmd.type === "step") stepBy(cmd.delta);
    else if (cmd.type === "start") seek(0);
    else if (cmd.type === "end") seek(total - 1);
    else togglePlay();
  };

  const tf = thrusterFrames?.[safeIdx];
  const gf = gridFrames?.[safeIdx];
  const cumR = tf?.cumR ?? gf?.cumR ?? 0;
  const ending: "terminated" | "truncated" = gridFrames && gridFrames.length > 0 && gridFrames[gridFrames.length - 1].done ? "terminated" : "truncated";
  const extras = tf && renderFrameExtras ? renderFrameExtras(tf, safeIdx) : null;

  return (
    <div className="space-y-4">
      <div
        role="group"
        aria-label="Episode player. Arrow keys step, Space plays or pauses, Home and End jump."
        tabIndex={0}
        onKeyDown={onKeyDown}
        data-testid="episode-player"
        className="group space-y-4 rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-4 focus-visible:ring-offset-card"
      >
        <ControlBar
          mode={mode}
          setMode={setMode}
          trained={trained}
          playing={playing}
          onTogglePlay={togglePlay}
          speed={speed}
          setSpeed={setSpeed}
          idx={safeIdx}
          total={total}
          onSeek={seek}
          onStep={stepBy}
          onNewEpisode={() => setSeed((s) => s + 1)}
          extra={
            !isThruster ? (
              <label className="flex cursor-pointer items-center gap-1.5 text-[11px] text-muted-foreground">
                <input type="checkbox" checked={showPolicy} onChange={(e) => setShowPolicy(e.target.checked)} className="accent-[var(--primary)]" />
                Greedy-policy overlay
              </label>
            ) : null
          }
        />

        {isThruster && thrusterFrames && tf ? (
          <div className="flex flex-col gap-3 sm:flex-row">
            <div className="min-w-0 flex-1 space-y-2">
              <ThrusterScene frames={thrusterFrames} idx={safeIdx} />
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[10.5px] text-muted-foreground">
                <span className="flex items-center gap-1.5">
                  <span className="h-0 w-4 border-t-2 border-dashed" style={{ borderColor: "var(--chart-1)" }} /> commanded
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-1 w-4 rounded-full" style={{ background: "var(--chart-1)" }} /> thrust after stall
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-1 w-4 rounded-full bg-warning" /> |a| &gt; knee {KNEE.toFixed(2)} (stalled)
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 rounded-full bg-primary/50" /> pod trail (last {POD_TRAIL} steps)
                </span>
              </div>
              {extras != null && extras !== false && <div data-slot="frame-extras">{extras}</div>}
            </div>
            <div className="flex flex-row items-start gap-3 sm:flex-col">
              <PhaseTrail frames={thrusterFrames} idx={safeIdx} />
              <p className="text-[10.5px] leading-relaxed text-muted-foreground sm:w-[150px]">
                Left: the real 13-dim float64 system. Right: the (p₁, v₁) projection on the agent's 11×9 grid — the only thing the tabular agent sees.
              </p>
            </div>
          </div>
        ) : gridFrames && gf ? (
          <GridScene ds={ds} frames={gridFrames} idx={safeIdx} Q={Q} version={version} showPolicy={showPolicy} />
        ) : (
          <div className="rounded-lg border border-dashed border-border p-6 text-center text-xs text-muted-foreground">No episode to play.</div>
        )}

        {steps.length > 0 && (
          <StepInspector
            ds={ds}
            Q={Q}
            steps={steps}
            idx={safeIdx}
            cumR={cumR}
            trace={trace}
            gamma={gamma}
            mode={mode}
            thruster={tf}
            ending={ending}
            onSeek={seek}
          />
        )}
      </div>
      {isThruster && <PhysicsBench policy={policy} trained={trained} />}
      {isThruster && <SafetyPanel policy={policy} trained={trained} />}
    </div>
  );
}
