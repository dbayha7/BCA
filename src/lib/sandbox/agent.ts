// Agent playground: templates, the obs/ctx objects handed to user `act` functions, rollouts and
// built-in baselines. `runUserGridAgent` / `runUserThrusterAgent` call user code and run ONLY in
// the sandbox worker; they return raw action/command sequences. The main thread validates those
// and REPLAYS them on the trusted simulators (`replayGridActions` / `replayThrusterCommands`), so
// every number shown in the UI is computed by trusted code.

import { generateDataset, greedy, mulberry32, qStar, type Env } from "../rl";
import { executedCommand, HORIZON, KNEE, STALL_SLOPE, recordCommand, resetRecorded, stepContinuous, toCell } from "../thrusterchain";
import { UserCodeFailure, describeValue, type GridPlaygroundEnv, type PlaygroundEnvId } from "./protocol";
import type { CodePreset } from "./reward";

export const GRID_HORIZON = 100;
export const PLAYGROUND_GAMMA = 0.95;
/** Seeds ctx.random() and the Random baseline, so a user policy using ctx.random() is paired with it. */
export const AGENT_SEED_BASE = 1000;
/** Same reset seeds as benchmarkThruster's default (baseSeed 1000), i.e. the physics bench. */
export const THRUSTER_BASE_SEED = 1000;
export const MAX_EPISODES = 200;

export const PLAYGROUND_ENVS: { id: PlaygroundEnvId; label: string; kind: "grid" | "thruster" }[] = [
  { id: "gridworld-random", label: "Gridworld-Random", kind: "grid" },
  { id: "halfcheetah-medium", label: "HalfCheetah-Medium", kind: "grid" },
  { id: "thrusterchain-stall", label: "ThrusterChain v4", kind: "thruster" },
];

const ENV_CACHE = new Map<GridPlaygroundEnv, Env>();

/** The real grid environment (same seeds as the lab; the env itself does not depend on the seed). */
export function playgroundGridEnv(id: GridPlaygroundEnv): Env {
  let env = ENV_CACHE.get(id);
  if (!env) {
    env = generateDataset(id, id === "gridworld-random" ? 7 : 11).env;
    ENV_CACHE.set(id, env);
  }
  return env;
}

/* ---------------- Output validation ---------------- */

export const isAction = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v) && v >= 0 && v <= 3;

export function checkAction(v: unknown, where: string): number {
  if (isAction(v)) return v;
  throw new UserCodeFailure("output", `act() must return an integer action 0–3, got ${describeValue(v)} at ${where}.`);
}

/** Four finite numbers (array or typed array). They are clipped/cast later by executedCommand. */
export function checkCommand(v: unknown, where: string): number[] {
  if (v === null || typeof v !== "object") {
    throw new UserCodeFailure("output", `act() must return 4 numbers (one command per thruster), got ${describeValue(v)} at ${where}.`);
  }
  const arr = v as ArrayLike<unknown>;
  if (arr.length !== 4) {
    throw new UserCodeFailure("output", `act() must return exactly 4 commands, got ${describeValue(v)} at ${where}.`);
  }
  const out: number[] = [];
  for (let i = 0; i < 4; i++) {
    const c = arr[i];
    if (typeof c !== "number" || !Number.isFinite(c)) {
      throw new UserCodeFailure("output", `command ${i} must be a finite number, got ${describeValue(c)} at ${where}.`);
    }
    out.push(c);
  }
  return out;
}

/* ---------------- Grid environments ---------------- */

export interface GridObs {
  readonly s: number;
  readonly x: number;
  readonly y: number;
  readonly t: number;
}

export interface GridAgentCtx {
  readonly width: number;
  readonly height: number;
  readonly goal: { x: number; y: number } | null;
  readonly start: { x: number; y: number };
  readonly actionNames: readonly string[];
  x(s: number): number;
  y(s: number): number;
  /** Cell index of (x, y), or −1 outside the grid. */
  cell(x: number, y: number): number;
  isGoal(s: number): boolean;
  isHazard(s: number): boolean;
  isBlocked(s: number): boolean;
  /** Seeded uniform [0, 1) generator, reset per episode. */
  random(): number;
}

export function makeGridAgentCtx(env: Env, episode: number): GridAgentCtx {
  const W = env.width;
  const H = env.height;
  const goalCells = [...env.terminals].filter(([, k]) => k === "goal").map(([s]) => s);
  const goal = goalCells.length ? Object.freeze({ x: Math.min(...goalCells) % W, y: Math.floor(Math.min(...goalCells) / W) }) : null;
  const rng = mulberry32(AGENT_SEED_BASE + episode);
  return Object.freeze({
    width: W,
    height: H,
    goal,
    start: Object.freeze({ x: env.start % W, y: Math.floor(env.start / W) }),
    actionNames: Object.freeze([...env.actionNames]),
    x: (s: number) => s % W,
    y: (s: number) => Math.floor(s / W),
    cell: (x: number, y: number) => (Number.isInteger(x) && Number.isInteger(y) && x >= 0 && y >= 0 && x < W && y < H ? y * W + x : -1),
    isGoal: (s: number) => env.terminals.get(s) === "goal",
    isHazard: (s: number) => env.terminals.get(s) === "hazard",
    isBlocked: (s: number) => env.blocked.has(s),
    random: () => rng(),
  });
}

export interface GridStep {
  s: number;
  a: number;
  r: number;
  s2: number;
  done: boolean;
  cumR: number;
}

export interface GridEpisode {
  steps: GridStep[];
  ret: number;
  hazards: number;
  goal: boolean;
}

/** One episode from env.start until termination or the horizon. `policy` must return a valid action. */
export function rolloutGrid(env: Env, policy: (obs: GridObs, t: number) => number, horizon = GRID_HORIZON): GridEpisode {
  const W = env.width;
  let s = env.start;
  let cum = 0;
  let hazards = 0;
  let goal = false;
  const steps: GridStep[] = [];
  for (let t = 0; t < horizon; t++) {
    const a = policy(Object.freeze({ s, x: s % W, y: Math.floor(s / W), t }), t);
    if (!isAction(a)) throw new Error(`Invalid action ${describeValue(a)} at step ${t}.`);
    const { s2, r, done } = env.step(s, a);
    cum += r;
    const kind = env.terminals.get(s2);
    if (done && kind === "hazard") hazards++;
    if (done && kind === "goal") goal = true;
    steps.push({ s, a, r, s2, done, cumR: cum });
    if (done) break;
    s = s2;
  }
  return { steps, ret: cum, hazards, goal };
}

export type GridActFn = (obs: GridObs, ctx: GridAgentCtx) => unknown;

/** Worker side: run the user's policy closed-loop and return each episode's action sequence. */
export function runUserGridAgent(envId: GridPlaygroundEnv, fn: GridActFn, episodes: number, horizon = GRID_HORIZON): number[][] {
  const env = playgroundGridEnv(envId);
  const out: number[][] = [];
  for (let e = 0; e < episodes; e++) {
    const ctx = makeGridAgentCtx(env, e);
    const ep = rolloutGrid(
      env,
      (obs, t) => {
        let v: unknown;
        try {
          v = fn(obs, ctx);
        } catch (err) {
          throw new UserCodeFailure("runtime", `act() threw at episode ${e + 1}, step ${t}`, err);
        }
        return checkAction(v, `episode ${e + 1}, step ${t}`);
      },
      horizon,
    );
    out.push(ep.steps.map((st) => st.a));
  }
  return out;
}

/** Main side: validate untrusted action sequences and replay them on the trusted env. */
export function replayGridActions(env: Env, actions: unknown, episodes: number, horizon = GRID_HORIZON): GridEpisode[] {
  if (!Array.isArray(actions) || actions.length !== episodes) throw new Error(`Sandbox returned ${describeValue(actions)}; expected ${episodes} episodes.`);
  return actions.map((seq: unknown, e) => {
    if (!Array.isArray(seq) || seq.length === 0 || seq.length > horizon) throw new Error(`Episode ${e + 1}: malformed action sequence.`);
    let k = 0;
    const ep = rolloutGrid(
      env,
      (_obs, t) => {
        if (k >= seq.length) throw new Error(`Episode ${e + 1}: sequence ended early at step ${t}.`);
        const a = seq[k++];
        if (!isAction(a)) throw new Error(`Episode ${e + 1}, step ${t}: invalid action ${describeValue(a)}.`);
        return a;
      },
      horizon,
    );
    if (k !== seq.length) throw new Error(`Episode ${e + 1}: ${seq.length - k} actions after the episode ended.`);
    return ep;
  });
}

export function gridRandomEpisodes(env: Env, episodes: number, horizon = GRID_HORIZON): GridEpisode[] {
  return Array.from({ length: episodes }, (_, e) => {
    const rng = mulberry32(AGENT_SEED_BASE + e);
    return rolloutGrid(env, () => Math.floor(rng() * 4), horizon);
  });
}

export function gridOptimalEpisodes(env: Env, episodes: number, horizon = GRID_HORIZON): GridEpisode[] {
  const Q = qStar(env, PLAYGROUND_GAMMA);
  return Array.from({ length: episodes }, () => rolloutGrid(env, (obs) => greedy(Q, obs.s), horizon));
}

/* ---------------- ThrusterChain ---------------- */

export interface ThrusterObs {
  /** Full 13-dim state: [p1, v1, …, p5, v5, current, wave cos, wave sin]. */
  readonly state: readonly number[];
  /** (p1, v1) grid cell the tabular lab would see. */
  readonly cell: number;
  readonly t: number;
}

export interface ThrusterAgentCtx {
  readonly stateDims: number;
  readonly commandDims: number;
  readonly knee: number;
  readonly stallSlope: number;
  readonly horizon: number;
  random(): number;
}

export function makeThrusterAgentCtx(episode: number): ThrusterAgentCtx {
  const rng = mulberry32(AGENT_SEED_BASE + episode);
  return Object.freeze({ stateDims: 13, commandDims: 4, knee: KNEE, stallSlope: STALL_SLOPE, horizon: HORIZON, random: () => rng() });
}

export interface ThrusterStep {
  /** State BEFORE the step. */
  x: Float64Array;
  /** Executed (clipped, float32-cast) command. */
  cmd: number[];
  r: number;
  cumR: number;
}

export interface ThrusterEpisode {
  steps: ThrusterStep[];
  ret: number;
  /** Steps where any thruster command exceeded the stall knee. */
  stallSteps: number;
  /** Oscillator energy Σ x[0..9]² after the last step. */
  finalEnergy: number;
}

export function thrusterEpisodeRng(e: number) {
  return mulberry32((THRUSTER_BASE_SEED + e) * 104729 + 1);
}

/**
 * One exact float64 episode with the same reset law and seeds as benchmarkThruster.
 * `policy` returns a requested command; it is clipped to [−1, 1] and cast by executedCommand.
 * `envRng` is the episode generator after reset (only the behavior recorder consumes it).
 */
export function rolloutThruster(e: number, policy: (obs: ThrusterObs, envRng: () => number) => number[]): ThrusterEpisode {
  const rng = thrusterEpisodeRng(e);
  let x = resetRecorded(rng);
  let cum = 0;
  let stallSteps = 0;
  const steps: ThrusterStep[] = [];
  for (let t = 0; t < HORIZON; t++) {
    const obs = Object.freeze({ state: Object.freeze(Array.from(x)), cell: toCell(x), t });
    const cmd = executedCommand(policy(obs, rng));
    if (cmd.some((c) => Math.abs(c) > KNEE)) stallSteps++;
    const { nx, r } = stepContinuous(x, cmd);
    cum += r;
    steps.push({ x, cmd, r, cumR: cum });
    x = nx;
  }
  let finalEnergy = 0;
  for (let j = 0; j < 10; j++) finalEnergy += x[j] * x[j];
  return { steps, ret: cum, stallSteps, finalEnergy };
}

export type ThrusterActFn = (obs: ThrusterObs, ctx: ThrusterAgentCtx) => unknown;

/** Worker side: run the user's controller and return the requested commands (flattened) per episode. */
export function runUserThrusterAgent(fn: ThrusterActFn, episodes: number): number[][] {
  const out: number[][] = [];
  for (let e = 0; e < episodes; e++) {
    const ctx = makeThrusterAgentCtx(e);
    const flat: number[] = [];
    rolloutThruster(e, (obs) => {
      let v: unknown;
      try {
        v = fn(obs, ctx);
      } catch (err) {
        throw new UserCodeFailure("runtime", `act() threw at episode ${e + 1}, step ${obs.t}`, err);
      }
      const req = checkCommand(v, `episode ${e + 1}, step ${obs.t}`);
      flat.push(...req);
      return req;
    });
    out.push(flat);
  }
  return out;
}

/** Main side: validate untrusted command sequences and re-simulate them with the exact physics. */
export function replayThrusterCommands(commands: unknown, episodes: number): ThrusterEpisode[] {
  if (!Array.isArray(commands) || commands.length !== episodes) throw new Error(`Sandbox returned ${describeValue(commands)}; expected ${episodes} episodes.`);
  return commands.map((seq: unknown, e) => {
    if (!Array.isArray(seq) || seq.length !== 4 * HORIZON) throw new Error(`Episode ${e + 1}: expected ${4 * HORIZON} command values.`);
    for (let i = 0; i < seq.length; i++) {
      if (typeof seq[i] !== "number" || !Number.isFinite(seq[i])) throw new Error(`Episode ${e + 1}: command value #${i} is not a finite number.`);
    }
    return rolloutThruster(e, (obs) => (seq as number[]).slice(obs.t * 4, obs.t * 4 + 4));
  });
}

export function thrusterCoastEpisodes(episodes: number): ThrusterEpisode[] {
  return Array.from({ length: episodes }, (_, e) => rolloutThruster(e, () => [0, 0, 0, 0]));
}

/** The spec's ε = 0.25 behavior recorder that produced the offline dataset. */
export function thrusterBehaviorEpisodes(episodes: number): ThrusterEpisode[] {
  return Array.from({ length: episodes }, (_, e) => rolloutThruster(e, (_obs, rng) => recordCommand(rng, 0.25)));
}

/* ---------------- Summaries ---------------- */

export interface PolicySummary {
  id: string;
  label: string;
  episodes: number;
  meanReturn: number;
  stdReturn: number;
  minReturn: number;
  maxReturn: number;
  meanSteps: number;
  /** Grid: episodes that ended in a hazard (HalfCheetah: a flip). */
  hazardEpisodes: number;
  /** Grid: episodes that reached the goal. */
  goalEpisodes: number;
  /** ThrusterChain: fraction of steps with any |command| above the stall knee. */
  stallRate: number;
  /** ThrusterChain: mean final oscillator energy. */
  finalEnergy: number;
  returns: number[];
}

function stats(returns: number[]) {
  const n = Math.max(1, returns.length);
  const mean = returns.reduce((a, b) => a + b, 0) / n;
  const variance = returns.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n;
  return {
    meanReturn: mean,
    stdReturn: Math.sqrt(variance),
    minReturn: returns.length ? Math.min(...returns) : 0,
    maxReturn: returns.length ? Math.max(...returns) : 0,
  };
}

export function summarizeGrid(id: string, label: string, eps: GridEpisode[]): PolicySummary {
  const returns = eps.map((e) => e.ret);
  return {
    id,
    label,
    episodes: eps.length,
    ...stats(returns),
    meanSteps: eps.reduce((a, e) => a + e.steps.length, 0) / Math.max(1, eps.length),
    hazardEpisodes: eps.filter((e) => e.hazards > 0).length,
    goalEpisodes: eps.filter((e) => e.goal).length,
    stallRate: 0,
    finalEnergy: 0,
    returns,
  };
}

export function summarizeThruster(id: string, label: string, eps: ThrusterEpisode[]): PolicySummary {
  const returns = eps.map((e) => e.ret);
  const steps = eps.reduce((a, e) => a + e.steps.length, 0);
  return {
    id,
    label,
    episodes: eps.length,
    ...stats(returns),
    meanSteps: steps / Math.max(1, eps.length),
    hazardEpisodes: 0,
    goalEpisodes: 0,
    stallRate: steps ? eps.reduce((a, e) => a + e.stallSteps, 0) / steps : 0,
    finalEnergy: eps.reduce((a, e) => a + e.finalEnergy, 0) / Math.max(1, eps.length),
    returns,
  };
}

/* ---------------- Templates ---------------- */

const RANDOM_TEMPLATE: CodePreset = {
  id: "random",
  name: "Uniform random",
  description: "Uses the seeded ctx.random(), so it matches the Random baseline exactly.",
  code: `// Uniform random actions from the seeded per-episode generator.
// Because ctx.random() is seeded like the Random baseline, this reproduces it exactly.
function act(obs, ctx) {
  return Math.floor(ctx.random() * 4);
}
`,
};

export const AGENT_TEMPLATES: Record<PlaygroundEnvId, CodePreset[]> = {
  "gridworld-random": [
    {
      id: "bfs",
      name: "Shortest safe path (BFS)",
      description: "Plans once with breadth-first search over the known map, then follows the plan.",
      code: `// Plan once with breadth-first search over the known map, then follow the plan.
// Moving into a blocked cell leaves you where you are; hazards and the goal end the episode.
const DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // Up, Right, Down, Left
let plan = null; // plan[cell] = best action, built on the first call

function buildPlan(ctx) {
  const n = ctx.width * ctx.height;
  const dist = new Array(n).fill(Infinity);
  const goal = ctx.cell(ctx.goal.x, ctx.goal.y);
  const queue = [goal];
  dist[goal] = 0;
  while (queue.length) {
    const c = queue.shift();
    for (const [dx, dy] of DIRS) {
      // p is a cell from which moving by (dx, dy) lands on c
      const p = ctx.cell(ctx.x(c) - dx, ctx.y(c) - dy);
      if (p < 0 || ctx.isBlocked(p) || ctx.isHazard(p) || ctx.isGoal(p)) continue;
      if (dist[p] === Infinity) {
        dist[p] = dist[c] + 1;
        queue.push(p);
      }
    }
  }
  const best = new Array(n).fill(0);
  for (let c = 0; c < n; c++) {
    let bestDist = Infinity;
    DIRS.forEach(([dx, dy], a) => {
      const next = ctx.cell(ctx.x(c) + dx, ctx.y(c) + dy);
      if (next >= 0 && !ctx.isBlocked(next) && dist[next] < bestDist) {
        bestDist = dist[next];
        best[c] = a;
      }
    });
  }
  return best;
}

function act(obs, ctx) {
  if (!ctx.goal) return Math.floor(ctx.random() * 4);
  if (!plan) plan = buildPlan(ctx);
  return plan[obs.s];
}
`,
    },
    RANDOM_TEMPLATE,
  ],
  "halfcheetah-medium": [
    {
      id: "level",
      name: "Level-pitch controller",
      description: "Accelerate while level; correct pitch whenever the nose rises or dips.",
      code: `// Accelerate while the body is level. Above speed 6, accelerating lifts the nose,
// so pitch down whenever the nose is up and pitch up whenever it dips.
// Actions: 0 Pitch up, 1 Accelerate, 2 Pitch down, 3 Brake.
function act(obs, ctx) {
  const level = ctx.start.y; // the episode starts level
  if (obs.y < level) return 2; // nose up   -> pitch down
  if (obs.y > level) return 0; // nose down -> pitch up
  return 1; // level -> accelerate
}
`,
    },
    RANDOM_TEMPLATE,
  ],
  "thrusterchain-stall": [
    {
      id: "damping",
      name: "Velocity damping",
      description: "Each thruster damps the mean velocity of the two links it drives and cancels the forcing, staying under the knee.",
      code: `// Thruster i pushes the velocities of links i+1 and i+2 (0.3 per unit of thrust).
// Damp their mean velocity, cancel the current + wave push, and stay under the stall knee.
// state = [p1, v1, p2, v2, p3, v3, p4, v4, p5, v5, current, wave cos, wave sin]
const GAIN = 1.5;

function act(obs, ctx) {
  const x = obs.state;
  const push = 0.3 * (x[10] + x[11]); // forcing added to every velocity each step
  const limit = 0.98 * ctx.knee; // beyond the knee thrust stalls and reverses
  const cmd = [];
  for (let i = 0; i < 4; i++) {
    const v = (x[2 * i + 1] + x[2 * i + 3]) / 2;
    const u = (-GAIN * v - push) / 0.6;
    cmd.push(Math.max(-limit, Math.min(limit, u)));
  }
  return cmd;
}
`,
    },
    {
      id: "coast",
      name: "Coast (all zero)",
      description: "Never fires — should match the Coast baseline.",
      code: `// Never fire a thruster. Matches the Coast baseline.
function act(obs, ctx) {
  return [0, 0, 0, 0];
}
`,
    },
  ],
};
