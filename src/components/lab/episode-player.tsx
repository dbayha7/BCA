import { useEffect, useMemo, useState } from "react";
import { mulberry32, type Dataset } from "@/lib/rl";
import { KNEE, rolloutThruster, thrustStall, type ThrusterFrame } from "@/lib/thrusterchain";
import { PhysicsBench } from "./physics-bench";

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

function ControlBar({
  mode,
  setMode,
  trained,
  playing,
  setPlaying,
  speed,
  setSpeed,
  idx,
  total,
  setIdx,
  onNewEpisode,
}: {
  mode: Mode;
  setMode: (m: Mode) => void;
  trained: boolean;
  playing: boolean;
  setPlaying: (p: boolean) => void;
  speed: number;
  setSpeed: (s: number) => void;
  idx: number;
  total: number;
  setIdx: (i: number) => void;
  onNewEpisode: () => void;
}) {
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex rounded-md border border-border bg-surface p-0.5 text-[11px]">
          {(["policy", "behavior"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`rounded px-2.5 py-1 font-medium transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                mode === m ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {m === "policy" ? "Trained policy" : "Behavior (recorder)"}
            </button>
          ))}
        </div>
        {mode === "policy" && !trained && (
          <span className="rounded border border-warning/40 bg-warning/10 px-2 py-0.5 text-[10.5px] text-foreground">Q is untrained — the greedy policy is arbitrary until you Step Training.</span>
        )}
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setPlaying(!playing)}
            className="flex h-8 items-center gap-1.5 rounded-md bg-primary px-3.5 text-xs font-semibold text-primary-foreground transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            {playing ? (
              <svg width="11" height="11" viewBox="0 0 12 12" fill="currentColor"><path d="M2 1.5h3v9H2zM7 1.5h3v9H7z" /></svg>
            ) : (
              <svg width="11" height="11" viewBox="0 0 12 12" fill="currentColor"><path d="M2.5 1.5v9l7.5-4.5z" /></svg>
            )}
            {playing ? "Pause" : "Play"}
          </button>
          <div className="flex rounded-md border border-border bg-surface p-0.5 text-[11px]">
            {SPEEDS.map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`rounded px-2 py-1 font-mono transition-colors duration-200 ${speed === s ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
              >
                {s}×
              </button>
            ))}
          </div>
          <button
            onClick={onNewEpisode}
            className="flex h-8 items-center gap-1.5 rounded-md border border-border bg-card px-3 text-xs font-medium text-muted-foreground transition-colors hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"><path d="M21 12a9 9 0 1 1-2.6-6.4M21 3v5h-5" /></svg>
            New episode
          </button>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <span className="w-14 shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">
          t = {idx}/{Math.max(0, total - 1)}
        </span>
        <input
          type="range"
          min={0}
          max={Math.max(0, total - 1)}
          value={idx}
          onChange={(e) => {
            setPlaying(false);
            setIdx(Number(e.target.value));
          }}
          className="w-full accent-[var(--primary)]"
          aria-label="Scrub episode"
        />
      </div>
    </div>
  );
}

/** Animated side view of the real 13-dim ThrusterChain: 5 pods, 4 thrusters, current + wave forcing. */
function ThrusterScene({ f }: { f: ThrusterFrame }) {
  const W = 540;
  const H = 230;
  const baseY = 112;
  const px = (i: number) => 70 + i * 100;
  const py = (i: number) => baseY - Math.max(-1.8, Math.min(1.8, f.x[i * 2])) * 42;
  const th = thrustStall(f.cmd);
  const waveAmp = Math.hypot(f.x[11], f.x[12]);
  const wavePhase = Math.atan2(f.x[12], f.x[11]);
  const current = f.x[10];
  const wavePath = Array.from({ length: 55 }, (_, k) => {
    const x = (k / 54) * W;
    const y = 26 + Math.sin((x / W) * Math.PI * 4 + wavePhase) * waveAmp * 22;
    return `${k === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg border border-border" role="img" aria-label="ThrusterChain episode animation">
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
      {/* rest line */}
      <line x1={30} y1={baseY} x2={W - 30} y2={baseY} stroke="var(--border)" strokeDasharray="4 4" />
      {/* chain */}
      <polyline points={Array.from({ length: 5 }, (_, i) => `${px(i)},${py(i)}`).join(" ")} fill="none" stroke="var(--foreground)" strokeWidth="2" opacity="0.5" style={{ transition: "points 130ms linear" }} />
      {Array.from({ length: 5 }, (_, i) => {
        const v = f.x[i * 2 + 1];
        const vMag = Math.min(1, Math.abs(v) / 1.6);
        return (
          <g key={i}>
            <circle cx={px(i)} cy={py(i)} r={11} fill={`color-mix(in oklch, var(--primary) ${Math.round(30 + vMag * 65)}%, var(--card))`} stroke="var(--foreground)" strokeWidth="1.2" style={{ transition: "cx 130ms linear, cy 130ms linear, fill 130ms linear" }} />
            <text x={px(i)} y={py(i) + 3.5} textAnchor="middle" fontSize="9" fontWeight={700} className="fill-foreground font-mono" style={{ transition: "x 130ms linear, y 130ms linear" }}>{i + 1}</text>
            <text x={px(i)} y={py(i) - 16} textAnchor="middle" fontSize="8.5" className="fill-muted-foreground font-mono">
              p {f.x[i * 2].toFixed(2)}
            </text>
          </g>
        );
      })}
      {/* thrusters: cmd k drives links k & k+1 */}
      {f.cmd.map((a, k) => {
        const mx = (px(k) + px(k + 1)) / 2;
        const my = (py(k) + py(k + 1)) / 2 + 24;
        const stalled = Math.abs(a) > KNEE;
        const len = th[k] * 34;
        return (
          <g key={k}>
            <rect x={mx - 7} y={my - 5} width={14} height={10} rx={2.5} fill="var(--card)" stroke={stalled ? "var(--warning)" : "var(--border)"} strokeWidth={stalled ? 1.8 : 1} />
            {Math.abs(len) > 1 && (
              <line x1={mx} y1={my + (len > 0 ? 5 : -5)} x2={mx} y2={my + (len > 0 ? 5 : -5) + len * 0.6} stroke={stalled ? "var(--warning)" : "var(--chart-1)"} strokeWidth="3.5" strokeLinecap="round" opacity="0.85" />
            )}
            <text x={mx} y={my + 26} textAnchor="middle" fontSize="8.5" className={stalled ? "fill-[var(--warning)] font-mono" : "fill-muted-foreground font-mono"}>
              a{k} {a.toFixed(2)}{stalled ? " ⚠stall" : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** (p1, v1) phase-plane trail — the projection the tabular agent actually learns on. */
function PhaseTrail({ frames, idx }: { frames: ThrusterFrame[]; idx: number }) {
  const S = 150;
  const mx = (p: number) => ((Math.max(-2.4, Math.min(2.4, p)) + 2.4) / 4.8) * (S - 16) + 8;
  const my = (v: number) => ((1.6 - Math.max(-1.6, Math.min(1.6, v))) / 3.2) * (S - 16) + 8;
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="h-[150px] w-[150px] shrink-0 rounded-lg border border-border bg-surface/60">
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

function GridScene({ ds, frames, idx }: { ds: Dataset; frames: GridFrame[]; idx: number }) {
  const env = ds.env;
  const cell = 40;
  const W = env.width;
  const H = env.height;
  const cx = (s: number) => (s % W) * cell + cell / 2;
  const cy = (s: number) => Math.floor(s / W) * cell + cell / 2;
  const f = frames[idx];
  return (
    <svg viewBox={`-2 -2 ${W * cell + 4} ${H * cell + 4}`} className="mx-auto block max-h-[340px] w-full">
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
        </g>
      )}
    </svg>
  );
}

export function EpisodePlayer({ ds, Q, version, trained }: { ds: Dataset; Q: Float64Array; version: number; trained: boolean }) {
  const isThruster = ds.env.id === "thrusterchain-stall";
  const [mode, setMode] = useState<Mode>(trained ? "policy" : "behavior");
  const [seed, setSeed] = useState(1);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);

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

  useEffect(() => {
    setIdx(0);
    setPlaying(true);
  }, [mode, seed, ds]);

  useEffect(() => {
    if (!playing || total === 0) return;
    const id = window.setInterval(
      () =>
        setIdx((i) => {
          if (i + 1 >= total) {
            setPlaying(false);
            return i;
          }
          return i + 1;
        }),
      130 / speed,
    );
    return () => window.clearInterval(id);
  }, [playing, speed, total]);

  const safeIdx = Math.min(idx, Math.max(0, total - 1));
  const tf = thrusterFrames?.[safeIdx];
  const gf = gridFrames?.[safeIdx];
  const actionName = tf ? ds.env.actionNames[tf.a] : gf ? ds.env.actionNames[gf.a] : "—";
  const r = tf?.r ?? gf?.r ?? 0;
  const cumR = tf?.cumR ?? gf?.cumR ?? 0;

  return (
    <div className="space-y-4">
      <ControlBar
        mode={mode}
        setMode={setMode}
        trained={trained}
        playing={playing}
        setPlaying={setPlaying}
        speed={speed}
        setSpeed={setSpeed}
        idx={safeIdx}
        total={total}
        setIdx={setIdx}
        onNewEpisode={() => setSeed((s) => s + 1)}
      />

      {isThruster && thrusterFrames && tf ? (
        <div className="flex flex-col gap-3 sm:flex-row">
          <div className="min-w-0 flex-1">
            <ThrusterScene f={tf} />
          </div>
          <div className="flex flex-row items-start gap-3 sm:flex-col">
            <PhaseTrail frames={thrusterFrames} idx={safeIdx} />
            <p className="text-[10.5px] leading-relaxed text-muted-foreground sm:w-[150px]">
              Left: the real 13-dim float64 system. Right: the (p₁, v₁) projection — the only thing the tabular agent sees.
            </p>
          </div>
        </div>
      ) : gridFrames && gf ? (
        <GridScene ds={ds} frames={gridFrames} idx={safeIdx} />
      ) : (
        <div className="rounded-lg border border-dashed border-border p-6 text-center text-xs text-muted-foreground">No episode to play.</div>
      )}

      <div className="grid grid-cols-3 gap-2">
        <div className="rounded-lg border border-border bg-surface/60 px-3 py-2">
          <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Action</div>
          <div className="mt-0.5 truncate text-[13px] font-medium">{actionName}</div>
        </div>
        <div className="rounded-lg border border-border bg-surface/60 px-3 py-2">
          <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Reward r(t)</div>
          <div className={`mt-0.5 font-mono text-[13px] tabular-nums ${r < 0 ? "text-destructive" : "text-success"}`}>{r.toFixed(3)}</div>
        </div>
        <div className="rounded-lg border border-border bg-surface/60 px-3 py-2">
          <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Return Σr</div>
          <div className="mt-0.5 font-mono text-[13px] tabular-nums">{cumR.toFixed(2)}</div>
        </div>
      </div>
      {gf?.done && safeIdx === total - 1 && (
        <p className="text-[11.5px] text-muted-foreground">Episode terminated after {total} steps.</p>
      )}
      {isThruster && <PhysicsBench policy={policy} trained={trained} />}
    </div>
  );
}
