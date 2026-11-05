// Hardware safety limits for ThrusterChain and a constraint audit over the REAL 13-dim physics.
// Pure functions — no React, no DOM.
//
// The audit replays exactly the episodes of `benchmarkThruster` (same seeds, same reset law,
// same physics overrides, same command pipeline), so with the shield off its returns are
// bit-identical to the Physics test bench and the two views pair episode-by-episode.
//
// Timing convention (matches the spec reward and ThrusterFrame): at step t the constraint
// check uses the PRE-transition state x_t and the command applied at t. t = 0 therefore checks
// the reset state, which no controller can influence; the state after the last step is not checked.

import { mulberry32 } from "./rl";
import {
  actionToCommand,
  executedCommand,
  HORIZON,
  KNEE,
  recordCommand,
  resetRecorded,
  SPEC_PHYSICS,
  stepWith,
  toCell,
  type BenchMode,
  type PhysicsOverrides,
} from "./thrusterchain";

export type Quad = [number, number, number, number];
export type Five = [number, number, number, number, number];

export interface SafetyLimits {
  /** Max |executed command| per thruster (gear rating). Commands are already clipped to ±1 by the spec. */
  maxCommand: Quad;
  /** Count any thruster commanded past the physics stall knee (|a| > knee) as a violation. */
  stallIsViolation: boolean;
  /** Max |position| of links 1–5 (state x[0], x[2], x[4], x[6], x[8]). */
  maxPosition: Five;
  /** Max |velocity| of links 1–5 (state x[1], x[3], x[5], x[7], x[9]). */
  maxVelocity: Five;
  /** Max oscillator energy Σ x[0..9]² (the state term of the spec reward). */
  maxEnergy: number;
}

/**
 * Defaults, calibrated on the bench episodes (seeds 1000+): gear rating 0.9 (so the 0.95 "Burst"
 * action is out of spec), stall counted, |p| ≤ 2.0 and |v| ≤ 1.5 (inside the (p₁,v₁) grid's
 * ±2.4 / ±1.6 range), energy ≤ 8. Under spec physics the coast baseline breaches |p₁| or the
 * energy cap in roughly 10% of steps and the random recorder far more often.
 */
export const DEFAULT_SAFETY_LIMITS: SafetyLimits = {
  maxCommand: [0.9, 0.9, 0.9, 0.9],
  stallIsViolation: true,
  maxPosition: [2, 2, 2, 2, 2],
  maxVelocity: [1.5, 1.5, 1.5, 1.5, 1.5],
  maxEnergy: 8,
};

export function cloneLimits(l: SafetyLimits): SafetyLimits {
  return {
    maxCommand: [...l.maxCommand] as Quad,
    stallIsViolation: l.stallIsViolation,
    maxPosition: [...l.maxPosition] as Five,
    maxVelocity: [...l.maxVelocity] as Five,
    maxEnergy: l.maxEnergy,
  };
}

/* ---------------- Violation types & stress channels ---------------- */

export const VIOLATION_TYPES = ["command", "stall", "position", "velocity", "energy"] as const;
export type ViolationType = (typeof VIOLATION_TYPES)[number];

export const VIOLATION_LABELS: Record<ViolationType, string> = {
  command: "Command > rating",
  stall: "Stall-knee exceedance",
  position: "Link |position|",
  velocity: "Link |velocity|",
  energy: "Oscillator energy",
};

/** Bit for each violation type in a per-step mask. */
export const VIOLATION_BIT: Record<ViolationType, number> = { command: 1, stall: 2, position: 4, velocity: 8, energy: 16 };

export interface StressChannel {
  key: string;
  /** Short label as drawn in the episode scene (a0..a3, p1..p5, v1..v5). */
  label: string;
  group: ViolationType;
}

/**
 * Stress channels, in a fixed order shared by `measureStress`, audit `peakStress` arrays and the
 * StressOverlay. Stress = |value| / limit (1.0 = at the limit). The stall channel is
 * max_k |a_k| / knee.
 */
export const STRESS_CHANNELS: StressChannel[] = [
  ...[0, 1, 2, 3].map((k) => ({ key: `a${k}`, label: `a${k}`, group: "command" as const })),
  { key: "stall", label: "knee", group: "stall" },
  ...[1, 2, 3, 4, 5].map((i) => ({ key: `p${i}`, label: `p${i}`, group: "position" as const })),
  ...[1, 2, 3, 4, 5].map((i) => ({ key: `v${i}`, label: `v${i}`, group: "velocity" as const })),
  { key: "energy", label: "E", group: "energy" },
];
export const N_CHANNELS = STRESS_CHANNELS.length;
const CH_STALL = 4;
const CH_P = 5;
const CH_V = 10;
const CH_E = 15;

function ratio(v: number, lim: number): number {
  const a = Math.abs(v);
  if (lim > 0) return Number.isFinite(lim) ? a / lim : 0;
  return a > 0 ? Infinity : 0;
}

/**
 * Stress of every channel at one (state, command) pair, written into `out` (length N_CHANNELS),
 * and the bitmask of violated constraints. A constraint is violated when |value| > limit
 * strictly (sitting exactly on the limit is allowed). Stall is only counted in the mask when
 * `limits.stallIsViolation`, but its stress is always reported.
 */
export function measureStress(x: ArrayLike<number>, cmd: ArrayLike<number>, limits: SafetyLimits, knee: number = KNEE, out: Float64Array = new Float64Array(N_CHANNELS)): { stress: Float64Array; mask: number } {
  let mask = 0;
  let maxA = 0;
  for (let k = 0; k < 4; k++) {
    const a = Math.abs(cmd[k] ?? 0);
    out[k] = ratio(a, limits.maxCommand[k]);
    if (a > limits.maxCommand[k]) mask |= VIOLATION_BIT.command;
    if (a > maxA) maxA = a;
  }
  out[CH_STALL] = ratio(maxA, knee);
  if (limits.stallIsViolation && maxA > knee) mask |= VIOLATION_BIT.stall;
  let energy = 0;
  for (let i = 0; i < 5; i++) {
    const p = x[2 * i];
    const v = x[2 * i + 1];
    energy += p * p + v * v;
    out[CH_P + i] = ratio(p, limits.maxPosition[i]);
    out[CH_V + i] = ratio(v, limits.maxVelocity[i]);
    if (Math.abs(p) > limits.maxPosition[i]) mask |= VIOLATION_BIT.position;
    if (Math.abs(v) > limits.maxVelocity[i]) mask |= VIOLATION_BIT.velocity;
  }
  out[CH_E] = ratio(energy, limits.maxEnergy);
  if (energy > limits.maxEnergy) mask |= VIOLATION_BIT.energy;
  return { stress: out, mask };
}

/** Per-thruster |a| bound the shield enforces: the gear rating, and the stall knee when stall counts as a violation. */
export function shieldBounds(limits: SafetyLimits, knee: number = KNEE): Quad {
  return limits.maxCommand.map((m) => Math.max(0, Math.min(m, limits.stallIsViolation ? knee : Infinity))) as Quad;
}

/**
 * Safety shield: clip each executed command into [−bound_k, bound_k] (see `shieldBounds`).
 * It acts after the spec's float32 cast, like a hardware limiter on the command bus, so a
 * clipped value is the float64 bound itself. Only commands are touched — state limits cannot be
 * enforced by clipping, which is exactly what the shielded-vs-unshielded comparison exposes.
 */
export function shieldCommand(cmd: number[], limits: SafetyLimits, knee: number = KNEE): { cmd: number[]; changed: boolean } {
  const b = shieldBounds(limits, knee);
  let changed = false;
  const out = cmd.map((a, k) => {
    const c = Math.max(-b[k], Math.min(b[k], a));
    if (c !== a) changed = true;
    return c;
  });
  return { cmd: changed ? out : cmd, changed };
}

/* ---------------- Audit ---------------- */

export interface AuditOptions {
  /** Same meaning and default as benchmarkThruster's baseSeed (episode e uses seed baseSeed + e). */
  baseSeed?: number;
  /** Steps per episode (default HORIZON = 100, as in the bench). */
  horizon?: number;
  /** Clip commands to the limits before they reach the physics (see shieldCommand). */
  shield?: boolean;
}

export interface EpisodeAudit {
  /** Episode index e (seed baseSeed + e). */
  episode: number;
  /** Undiscounted return Σ r_t under the (possibly perturbed) bench reward. */
  ret: number;
  /** Timesteps with at least one violation of each type. */
  violations: Record<ViolationType, number>;
  /** Timesteps with any violation. */
  anyViolations: number;
  /** First timestep with any violation, or −1. */
  firstViolation: number;
  /** First timestep with each violation type, or −1. */
  firstByType: Record<ViolationType, number>;
  /** Max stress over the episode, per STRESS_CHANNELS entry. */
  peakStress: number[];
  /** Steps where the shield changed the command (0 when unshielded). */
  shieldInterventions: number;
  /** Per-timestep violation bitmask (see VIOLATION_BIT), length = horizon. */
  mask: Uint8Array;
}

const zeroTypes = (v: number) => Object.fromEntries(VIOLATION_TYPES.map((t) => [t, v])) as Record<ViolationType, number>;

/** Audit ONE bench episode. Identical seeding and command pipeline to benchmarkThruster. */
export function auditEpisode(mode: BenchMode, episode: number, physics: PhysicsOverrides = SPEC_PHYSICS, limits: SafetyLimits = DEFAULT_SAFETY_LIMITS, opts: AuditOptions = {}): EpisodeAudit {
  const baseSeed = opts.baseSeed ?? 1000;
  const horizon = opts.horizon ?? HORIZON;
  const rng = mulberry32((baseSeed + episode) * 104729 + 1);
  let x = resetRecorded(rng);
  x[10] *= physics.currentScale;
  x[11] *= physics.waveScale;
  x[12] *= physics.waveScale;

  const violations = zeroTypes(0);
  const firstByType = zeroTypes(-1);
  const peak = new Array<number>(N_CHANNELS).fill(0);
  const mask = new Uint8Array(horizon);
  const buf = new Float64Array(N_CHANNELS);
  let any = 0;
  let first = -1;
  let clips = 0;
  let cum = 0;

  for (let t = 0; t < horizon; t++) {
    let cmd: number[];
    if (mode.mode === "behavior") cmd = executedCommand(recordCommand(rng, 0.25));
    else if (mode.mode === "coast") cmd = [0, 0, 0, 0];
    else cmd = executedCommand(actionToCommand(mode.policy(toCell(x))));
    if (opts.shield) {
      const sh = shieldCommand(cmd, limits, physics.knee);
      cmd = sh.cmd;
      if (sh.changed) clips++;
    }

    const { mask: m } = measureStress(x, cmd, limits, physics.knee, buf);
    for (let c = 0; c < N_CHANNELS; c++) if (buf[c] > peak[c]) peak[c] = buf[c];
    mask[t] = m;
    if (m) {
      any++;
      if (first < 0) first = t;
      for (const ty of VIOLATION_TYPES) {
        if (m & VIOLATION_BIT[ty]) {
          violations[ty]++;
          if (firstByType[ty] < 0) firstByType[ty] = t;
        }
      }
    }

    const { nx, r } = stepWith(x, cmd, physics);
    cum += r;
    x = nx;
  }
  return { episode, ret: cum, violations, anyViolations: any, firstViolation: first, firstByType, peakStress: peak, shieldInterventions: clips, mask };
}

export interface AuditResult {
  episodes: number;
  horizon: number;
  baseSeed: number;
  shielded: boolean;
  /** Per-episode returns, in episode order — equal to benchmarkThruster(...).returns when unshielded. */
  returns: number[];
  meanReturn: number;
  /** Population std (as in BenchStats). */
  stdReturn: number;
  perEpisode: EpisodeAudit[];
  /** rateByT[t] = fraction of episodes with ANY violation at timestep t. */
  rateByT: number[];
  /** Same, per violation type. */
  rateByTypeT: Record<ViolationType, number[]>;
  /** Fraction of episodes with at least one violation (any type / per type). */
  episodeViolationRate: number;
  episodeViolationRateByType: Record<ViolationType, number>;
  /** Fraction of all episode-steps with a violation (any type / per type). */
  stepViolationRate: number;
  stepViolationRateByType: Record<ViolationType, number>;
  /** Mean number of violating steps per episode (any type / per type). */
  meanViolatingSteps: number;
  meanViolatingStepsByType: Record<ViolationType, number>;
  /** Mean / median first-violation timestep over episodes that violated (null if none did). */
  meanFirstViolation: number | null;
  medianFirstViolation: number | null;
  /** Worst stress per channel over all episodes (aligned with STRESS_CHANNELS). */
  peakStress: number[];
  /** Mean over episodes of each episode's peak stress per channel. */
  meanPeakStress: number[];
  /** Fraction of steps where the shield changed the command. */
  shieldInterventionRate: number;
}

/** Aggregate per-episode audits (e.g. collected in chunks by the UI) into an AuditResult. */
export function summarizeAudit(perEpisode: EpisodeAudit[], meta: { horizon?: number; baseSeed?: number; shielded?: boolean } = {}): AuditResult {
  const n = perEpisode.length;
  const horizon = meta.horizon ?? perEpisode[0]?.mask.length ?? HORIZON;
  const div = Math.max(1, n);
  const rateByT = new Array<number>(horizon).fill(0);
  const rateByTypeT = Object.fromEntries(VIOLATION_TYPES.map((t) => [t, new Array<number>(horizon).fill(0)])) as Record<ViolationType, number[]>;
  const epByType = zeroTypes(0);
  const stepsByType = zeroTypes(0);
  const peak = new Array<number>(N_CHANNELS).fill(0);
  const meanPeak = new Array<number>(N_CHANNELS).fill(0);
  let epAny = 0;
  let stepsAny = 0;
  let clips = 0;
  const firsts: number[] = [];

  for (const ep of perEpisode) {
    for (let t = 0; t < horizon; t++) {
      const m = ep.mask[t] ?? 0;
      if (!m) continue;
      rateByT[t]++;
      for (const ty of VIOLATION_TYPES) if (m & VIOLATION_BIT[ty]) rateByTypeT[ty][t]++;
    }
    if (ep.anyViolations > 0) {
      epAny++;
      firsts.push(ep.firstViolation);
    }
    stepsAny += ep.anyViolations;
    for (const ty of VIOLATION_TYPES) {
      if (ep.violations[ty] > 0) epByType[ty]++;
      stepsByType[ty] += ep.violations[ty];
    }
    for (let c = 0; c < N_CHANNELS; c++) {
      if (ep.peakStress[c] > peak[c]) peak[c] = ep.peakStress[c];
      meanPeak[c] += ep.peakStress[c] / div;
    }
    clips += ep.shieldInterventions;
  }
  for (let t = 0; t < horizon; t++) {
    rateByT[t] /= div;
    for (const ty of VIOLATION_TYPES) rateByTypeT[ty][t] /= div;
  }
  const totalSteps = Math.max(1, n * horizon);
  const returns = perEpisode.map((e) => e.ret);
  const meanR = returns.reduce((a, b) => a + b, 0) / div;
  const varR = returns.reduce((a, b) => a + (b - meanR) * (b - meanR), 0) / div;
  firsts.sort((a, b) => a - b);
  const mapTypes = (f: (ty: ViolationType) => number) => Object.fromEntries(VIOLATION_TYPES.map((ty) => [ty, f(ty)])) as Record<ViolationType, number>;

  return {
    episodes: n,
    horizon,
    baseSeed: meta.baseSeed ?? 1000,
    shielded: meta.shielded ?? false,
    returns,
    meanReturn: meanR,
    stdReturn: Math.sqrt(varR),
    perEpisode,
    rateByT,
    rateByTypeT,
    episodeViolationRate: epAny / div,
    episodeViolationRateByType: mapTypes((ty) => epByType[ty] / div),
    stepViolationRate: stepsAny / totalSteps,
    stepViolationRateByType: mapTypes((ty) => stepsByType[ty] / totalSteps),
    meanViolatingSteps: stepsAny / div,
    meanViolatingStepsByType: mapTypes((ty) => stepsByType[ty] / div),
    meanFirstViolation: firsts.length ? firsts.reduce((a, b) => a + b, 0) / firsts.length : null,
    medianFirstViolation: firsts.length ? (firsts.length % 2 ? firsts[(firsts.length - 1) / 2] : (firsts[firsts.length / 2 - 1] + firsts[firsts.length / 2]) / 2) : null,
    peakStress: peak,
    meanPeakStress: meanPeak,
    shieldInterventionRate: clips / totalSteps,
  };
}

/**
 * Constraint audit of a controller over `episodes` real 13-dim episodes — the same episodes,
 * seeds, reset states and physics as benchmarkThruster(mode, episodes, physics, baseSeed).
 * Counts per-episode violations by type, the per-timestep violation rate across episodes,
 * first-violation times, worst stress per channel, and the returns. With `opts.shield` the
 * commands are clipped to the limits first (behavior mode still consumes the same random
 * numbers, so shielded and unshielded runs stay paired).
 */
export function auditThruster(mode: BenchMode, episodes: number, physics: PhysicsOverrides = SPEC_PHYSICS, limits: SafetyLimits = DEFAULT_SAFETY_LIMITS, opts: AuditOptions = {}): AuditResult {
  const per: EpisodeAudit[] = [];
  for (let e = 0; e < episodes; e++) per.push(auditEpisode(mode, e, physics, limits, opts));
  return summarizeAudit(per, { horizon: opts.horizon ?? HORIZON, baseSeed: opts.baseSeed ?? 1000, shielded: !!opts.shield });
}
