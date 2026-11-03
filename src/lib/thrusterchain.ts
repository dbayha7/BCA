// ThrusterChain core v4 — frozen export from the user's spec (thrusterchain_spec.json).
// 13-dim linear ocean-thruster chain: 5 coupled oscillators + autonomous current & wave forcing.
// Dynamics, stall nonlinearity, reward, reset law and behavior recorder are implemented EXACTLY
// per the spec (float64 matrices, float32-cast executed commands). The continuous system is
// simulated faithfully, then the (p1, v1) plane is discretized so the tabular lab can learn on it.

import { buildDataset, mulberry32, type Dataset, type Env, type Transition } from "./rl";

export const KNEE = 0.800000011920929;
export const STALL_SLOPE = 1.5;
export const HORIZON = 100;
export const SPEC_GAMMA = 0.98;

// A (13×13) — exact frozen export values.
const A: number[][] = [
  [0.9777125812299784, -0.09672215307307072, -0.0032177474714957027, -0.0036854581926546795, 0.0022276001886515034, 5.0097360180674706e-5, 0.003607227365457634, -0.0025977450297937827, -0.0010500340801089907, 0.0009652492372484052, 0.0, 0.0, 0.0],
  [0.09940391721844581, 0.9759393868366641, 0.0001307439011167588, -0.0014584202432089749, 0.004287444772601739, 0.006358675619747226, -0.003249106414894763, -0.0022826844542754962, -0.001195739582567211, 0.002141233362877725, 0.3, 0.3, 0.0],
  [0.004727777116290652, 0.001676003589695229, 0.9551871912419472, -0.19400801385647914, 0.0066585559862546925, 0.004321981810486594, -0.002351504691480412, -0.0028796358576129743, -0.00017534637907377042, -0.0014908328910182977, 0.0, 0.0, 0.0],
  [-0.0010694645348326218, 0.003950533344859924, 0.2002191805870674, 0.9498613915785109, -0.005257280339745227, 0.003893854725563954, -0.0016027860425690422, 0.002178774672635904, 0.0032342039892158244, 0.004745834820556732, 0.3, 0.3, 0.0],
  [0.0005227925125596042, 0.0034722484793222414, 0.003327179871759413, -0.0011316590449246787, 0.9106172106519078, -0.33126273154497077, 0.0029036055232298766, -0.003299177904469415, -0.0002792636049624802, 0.0020210035590503706, 0.0, 0.0, 0.0],
  [-0.004426956214867775, 0.0026020949999520306, -0.001087741967027858, 1.0090042320887824e-5, 0.33940832566241086, 0.9089986694657229, 0.002562491688699326, 0.00026117818132252545, -0.001954887543497725, -0.003806489583919625, 0.3, 0.3, 0.0],
  [-0.002864014579333474, 0.005511555544683352, -0.0007795943457998527, 0.0010125392394960543, 0.00302278339678814, -0.0015069787253937821, 0.8217534382428903, -0.5054721930688784, 0.004390424840539214, -0.0027294911446361066, 0.0, 0.0, 0.0],
  [0.0032829775189667934, 0.0017827723597421883, 0.0008687221328898804, -0.0002566114977840717, 0.005419950373129756, -0.0005569388845530562, 0.5049667331935627, 0.8227127339382514, 0.0033464391706002545, -0.00015333122670848354, 0.3, 0.3, 0.0],
  [0.001872218383923559, 0.0106033926617112, 0.005927786293139656, -0.0012436923251405061, 0.0033156634642937137, -0.0013459488981572368, -0.0031035397212931673, 0.0018979088335239842, 0.6637979117655004, -0.6819570736534781, 0.0, 0.0, 0.0],
  [-0.0019264007165151803, 0.004132407256634719, 0.0003150500063740507, -0.0038427711853132804, 0.008823080271840861, 0.0028660639483504276, 0.004346767798996067, -0.005234195917955739, 0.6890547118528255, 0.6655616509674585, 0.3, 0.3, 0.0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9637708963658905, -0.26673143668883115],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.26673143668883115, 0.9637708963658905],
];

// B (13×4) — exact frozen export values.
const B: number[][] = [
  [0, 0, 0, 0],
  [0.3, 0, 0, 0],
  [0, 0, 0, 0],
  [0.3, 0.3, 0, 0],
  [0, 0, 0, 0],
  [0, 0.3, 0.3, 0],
  [0, 0, 0, 0],
  [0, 0, 0.3, 0.3],
  [0, 0, 0, 0],
  [0, 0, 0, 0.3],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
];

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** Requested float64 command → executed float32 command (clip then cast), per spec. */
export function executedCommand(requested: number[]): number[] {
  return requested.map((a) => Math.fround(clamp(a, -1, 1)));
}

/** Stall-world thrust: clip(a,-k,k) − slope·sign(a)·max(|a|−k, 0). */
export function thrustStall(cmd: number[]): number[] {
  return cmd.map((a) => clamp(a, -KNEE, KNEE) - STALL_SLOPE * Math.sign(a) * Math.max(Math.abs(a) - KNEE, 0));
}

/** One exact float64 step: x' = A·x + B·thrust(cmd); reward uses PRE-transition state + executed command. */
export function stepContinuous(x: Float64Array, cmd: number[]): { nx: Float64Array; r: number } {
  const th = thrustStall(cmd);
  const nx = new Float64Array(13);
  for (let i = 0; i < 13; i++) {
    let v = 0;
    for (let j = 0; j < 13; j++) v += A[i][j] * x[j];
    v += B[i][0] * th[0] + B[i][1] * th[1] + B[i][2] * th[2] + B[i][3] * th[3];
    nx[i] = v;
  }
  let cost = 0;
  for (let j = 0; j < 10; j++) cost += x[j] * x[j];
  for (let k = 0; k < 4; k++) cost += 0.3 * cmd[k] * cmd[k];
  return { nx, r: -cost };
}

/** Verify the implementation against fixture case stall_1 from the spec (tolerance 1e-6). */
export function verifyFixture(): boolean {
  const state = new Float64Array([
    -0.133562093764342, 0.3430875101501804, -0.0371746651786682, 0.6028769782910949, 0.018847601383307544,
    0.33361108636683373, 0.46965150499964325, -0.07443521906542608, 0.1017376066196697, 0.19868765487679282,
    0.5308110241558367, 0.073473966673648, 0.19349568741751344,
  ]);
  const cmd = executedCommand([-0.8, 0.8, -0.9, 0.9]);
  const { nx, r } = stepContinuous(state, cmd);
  const expected = [
    -0.16384060474721907, 0.26310728638100644, -0.15216489703460648, 0.7495471968678542, -0.09105205312396046,
    0.5376939360981143, 0.4259341423681299, 0.35741759261729333, -0.06753094816171892, 0.581527338175516,
    0.5308110241558367, 0.019200688022652956, 0.2060833287954005,
  ];
  let err = Math.abs(r - -1.7579825167028473);
  for (let i = 0; i < 13; i++) err = Math.max(err, Math.abs(nx[i] - expected[i]));
  return err < 1e-6;
}

/* ---------------- Episode rollout for interactive playback ---------------- */

export interface ThrusterFrame {
  /** Full 13-dim float64 state BEFORE the transition. */
  x: Float64Array;
  /** Executed float32 4-dim command. */
  cmd: number[];
  /** Discrete action label (for the tabular lab). */
  a: number;
  r: number;
  cumR: number;
}

/** Map the tabular agent's discrete action back to a continuous command on thruster 0. */
function actionToCommand(a: number): number[] {
  const a0 = a === 0 ? 0.95 : a === 1 ? 0.5 : a === 3 ? -0.5 : 0;
  return [a0, 0, 0, 0];
}

export type RolloutMode = { mode: "behavior" } | { mode: "policy"; policy: (cell: number) => number };

/**
 * Roll out ONE real episode of the exact 13-dim stall-world system for playback.
 * "behavior" replays the spec's ε=0.25 random recorder; "policy" executes the
 * tabular agent's greedy action in the (p₁,v₁) cell at every step.
 */
export function rolloutThruster(opts: RolloutMode, seed: number): ThrusterFrame[] {
  const rng = mulberry32(seed * 104729 + 1);
  let x = resetRecorded(rng);
  const frames: ThrusterFrame[] = [];
  let cum = 0;
  for (let t = 0; t < HORIZON; t++) {
    let cmd: number[];
    let a: number;
    if (opts.mode === "behavior") {
      cmd = executedCommand(recordCommand(rng, 0.25));
      a = toAction(cmd[0]);
    } else {
      a = opts.policy(toCell(x));
      cmd = executedCommand(actionToCommand(a));
    }
    const { nx, r } = stepContinuous(x, cmd);
    cum += r;
    frames.push({ x, cmd, a, r: Math.round(r * 1e6) / 1e6, cumR: cum });
    x = nx;
  }
  return frames;
}

/* ---------------- Offline dataset: behavior recorder + discretization ---------------- */

// (p1, v1) projection grid
const W = 11;
const H = 9;
const P_LO = -2.4;
const P_HI = 2.4;
const V_LO = -1.6;
const V_HI = 1.6;

/** Everything an external simulator needs to map the full 13-dim state onto the tabular grid and decode actions. */
export const THRUSTER_PROJECTION = {
  stateDims: 13,
  commandDims: 4,
  xAxis: { stateIndex: 0, name: "p1 (link-1 position)", lo: P_LO, hi: P_HI, bins: W },
  yAxis: { stateIndex: 1, name: "v1 (link-1 velocity)", lo: V_LO, hi: V_HI, bins: H, inverted: true },
  cellFormula:
    "xi = clamp(floor((x[0] - xLo) / (xHi - xLo) * xBins), 0, xBins-1); yi = clamp(floor((yHi - x[1]) / (yHi - yLo) * yBins), 0, yBins-1); cell = yi * xBins + xi",
  actionCommands: [
    [0.95, 0, 0, 0],
    [0.5, 0, 0, 0],
    [0, 0, 0, 0],
    [-0.5, 0, 0, 0],
  ] as number[][],
  stallKnee: KNEE,
  stallSlope: STALL_SLOPE,
  horizon: HORIZON,
  specGamma: SPEC_GAMMA,
  note: "Commands are clipped to [-1, 1] and cast to float32 before the stall nonlinearity (|a| > knee → slope applied). Dynamics: x' = A·x + B·stall(cmd) in float64.",
};

function toCell(x: Float64Array): number {
  const xi = clamp(Math.floor(((x[0] - P_LO) / (P_HI - P_LO)) * W), 0, W - 1);
  const yi = clamp(Math.floor(((V_HI - x[1]) / (V_HI - V_LO)) * H), 0, H - 1);
  return yi * W + xi;
}

/** Discrete action label from the executed a0 command, teaching the stall knee. */
function toAction(a0: number): number {
  if (a0 > KNEE) return 0; // Burst — over the knee, thrust stalls
  if (a0 > 0.15) return 1; // Thrust +
  if (a0 < -0.15) return 3; // Reverse
  return 2; // Coast
}

/** Spec behavior recorder ("broad", ε = 0.25): state-independent random commands. */
function recordCommand(rng: () => number, epsilon: number): number[] {
  const a0 = rng() < epsilon ? 0.5 + rng() * 0.5 : -1 + rng() * 1.5;
  return [a0, -1 + rng() * 2, -1 + rng() * 2, -1 + rng() * 2];
}

function gaussian(rng: () => number): number {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

/** Spec "recorded" reset law: kick 0.5·N(0,1) on oscillators; current ~U[0.1,0.7]; wave amp ~U[0.1,0.4]. */
function resetRecorded(rng: () => number): Float64Array {
  const x = new Float64Array(13);
  for (let i = 0; i < 10; i++) x[i] = 0.5 * gaussian(rng);
  x[10] = 0.1 + rng() * 0.6;
  const amp = 0.1 + rng() * 0.3;
  const phase = rng() * 2 * Math.PI;
  x[11] = amp * Math.cos(phase);
  x[12] = amp * Math.sin(phase);
  return x;
}

/** Simulate the REAL stall-world system with the spec's recorder, project onto the (p1, v1) grid. */
export function generateThrusterDataset(seed: number, onProgress?: (pct: number, label: string) => void): Dataset {
  const rng = mulberry32(seed);
  onProgress?.(2, "Verifying fixture stall_1…");
  const verified = verifyFixture();
  const transitions: Transition[] = [];
  const episodes = 140;
  for (let e = 0; e < episodes; e++) {
    let x = resetRecorded(rng);
    for (let t = 0; t < HORIZON; t++) {
      const cmd = executedCommand(recordCommand(rng, 0.25));
      const { nx, r } = stepContinuous(x, cmd);
      transitions.push({ s: toCell(x), a: toAction(cmd[0]), r: Math.round(r * 1e6) / 1e6, s2: toCell(nx), done: false });
      x = nx;
    }
    if (e % 10 === 9) onProgress?.(5 + Math.round(((e + 1) / episodes) * 85), `Simulating 13-dim float64 physics — episode ${e + 1}/${episodes}`);
  }
  onProgress?.(93, "Aggregating transitions into empirical MDP…");

  const env: Env = {
    id: "thrusterchain-stall",
    name: "ThrusterChain v4 (stall)",
    tag: "13-dim → (p₁,v₁) · 11×9",
    description: `Your frozen ThrusterChain core: 5 coupled oscillators driven by current + rotating waves, with the real stall knee at |a|>${KNEE.toFixed(2)} (slope 1.5). Simulated in exact float64 with the spec's ε=0.25 random recorder, projected onto link-1 position × velocity. ${verified ? "Fixture stall_1 verified ✓." : "⚠ fixture check FAILED — do not trust dynamics."} Reward = −(state energy + 0.3·‖command‖²); Burst actions are rare in data (ε-region) and stalled.`,
    width: W,
    height: H,
    axisX: "link-1 position p₁",
    axisY: "link-1 velocity v₁ (top = +)",
    actionNames: ["Burst >knee", "Thrust +", "Coast", "Reverse"],
    start: toCell(new Float64Array(13)),
    blocked: new Set(),
    terminals: new Map(),
    groundTruth: "empirical",
    step: () => ({ s2: 0, r: 0, done: true }),
  };
  const ds = buildDataset(env, transitions, episodes);
  const rMin = ds.rewardMin;
  env.step = (s, a) => {
    const k = s * 4 + a;
    if (ds.counts[k] === 0) return { s2: s, r: rMin, done: false };
    return { s2: ds.modeNext[k], r: ds.rewardMean[k], done: false };
  };
  return ds;
}

/* ---------------- Physics test bench ---------------- */

export interface PhysicsOverrides {
  /** Stall knee |a| above which thrust degrades (spec: 0.8). */
  knee: number;
  /** Stall slope (spec: 1.5). */
  slope: number;
  /** Multiplier on the initial ocean current x[10] (spec: 1). */
  currentScale: number;
  /** Multiplier on the initial wave amplitude x[11..12] (spec: 1). */
  waveScale: number;
  /** Weight on the state-energy term of the reward (spec: 1). */
  energyWeight: number;
  /** Weight on the ‖command‖² effort term of the reward (spec: 0.3). */
  cmdWeight: number;
  /** Extra per-step penalty whenever thruster 1 exceeds the stall knee (spec: 0). */
  stallPenalty: number;
  /** Success criterion: an episode succeeds if final oscillator energy Σx[0..9]² is below this (does not affect reward). */
  successEnergy: number;
}

export const SPEC_PHYSICS: PhysicsOverrides = {
  knee: KNEE,
  slope: STALL_SLOPE,
  currentScale: 1,
  waveScale: 1,
  energyWeight: 1,
  cmdWeight: 0.3,
  stallPenalty: 0,
  successEnergy: 1,
};

export type BenchMode = { mode: "behavior" } | { mode: "policy"; policy: (cell: number) => number } | { mode: "coast" };

export interface BenchStats {
  episodes: number;
  meanReturn: number;
  stdReturn: number;
  minReturn: number;
  maxReturn: number;
  /** Fraction of steps where thruster 0 exceeded the stall knee. */
  stallRate: number;
  /** Fraction of episodes ending with final oscillator energy below the success threshold. */
  successRate: number;
  /** Mean |p1| over all steps (lower = link 1 better stabilized). */
  meanAbsP1: number;
  /** Mean oscillator energy Σx[0..9]² at the final step. */
  finalEnergy: number;
  /** Per-episode returns (for the distribution strip). */
  returns: number[];
}

function stepWith(x: Float64Array, cmd: number[], ph: PhysicsOverrides): { nx: Float64Array; r: number } {
  const th = cmd.map((a) => clamp(a, -ph.knee, ph.knee) - ph.slope * Math.sign(a) * Math.max(Math.abs(a) - ph.knee, 0));
  const nx = new Float64Array(13);
  for (let i = 0; i < 13; i++) {
    let v = 0;
    for (let j = 0; j < 13; j++) v += A[i][j] * x[j];
    v += B[i][0] * th[0] + B[i][1] * th[1] + B[i][2] * th[2] + B[i][3] * th[3];
    nx[i] = v;
  }
  let energy = 0;
  for (let j = 0; j < 10; j++) energy += x[j] * x[j];
  let effort = 0;
  for (let k = 0; k < 4; k++) effort += cmd[k] * cmd[k];
  let cost = ph.energyWeight * energy + ph.cmdWeight * effort;
  if (Math.abs(cmd[0]) > ph.knee) cost += ph.stallPenalty;
  return { nx, r: -cost };
}

/**
 * Evaluate a controller over many real 13-dim episodes of the stall-world physics,
 * optionally under perturbed physics (knee, slope, current, waves) to stress-test
 * whether the learned tabular policy transfers beyond the exact spec.
 * Same seeds are used for every mode so comparisons are paired.
 */
export function benchmarkThruster(opts: BenchMode, episodes: number, ph: PhysicsOverrides = SPEC_PHYSICS, baseSeed = 1000): BenchStats {
  const returns: number[] = [];
  let stalls = 0;
  let steps = 0;
  let absP1 = 0;
  let finalE = 0;
  let successes = 0;
  for (let e = 0; e < episodes; e++) {
    const rng = mulberry32((baseSeed + e) * 104729 + 1);
    let x = resetRecorded(rng);
    x[10] *= ph.currentScale;
    x[11] *= ph.waveScale;
    x[12] *= ph.waveScale;
    let cum = 0;
    for (let t = 0; t < HORIZON; t++) {
      let cmd: number[];
      if (opts.mode === "behavior") cmd = executedCommand(recordCommand(rng, 0.25));
      else if (opts.mode === "coast") cmd = [0, 0, 0, 0];
      else cmd = executedCommand(actionToCommand(opts.policy(toCell(x))));
      if (Math.abs(cmd[0]) > ph.knee) stalls++;
      absP1 += Math.abs(x[0]);
      steps++;
      const { nx, r } = stepWith(x, cmd, ph);
      cum += r;
      x = nx;
    }
    let en = 0;
    for (let j = 0; j < 10; j++) en += x[j] * x[j];
    finalE += en;
    if (en < ph.successEnergy) successes++;
    returns.push(cum);
  }
  const n = Math.max(1, returns.length);
  const mean = returns.reduce((a, b) => a + b, 0) / n;
  const variance = returns.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n;
  return {
    episodes,
    meanReturn: mean,
    stdReturn: Math.sqrt(variance),
    minReturn: returns.length ? Math.min(...returns) : 0,
    maxReturn: returns.length ? Math.max(...returns) : 0,
    stallRate: steps ? stalls / steps : 0,
    successRate: successes / n,
    meanAbsP1: steps ? absP1 / steps : 0,
    finalEnergy: finalE / n,
    returns,
  };
}
