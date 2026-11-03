// Core Offline RL engine: datasets, tabular conservative Q-learning (CQL-style), evaluation.

/** The research goal/danger task on ThrusterChain, under its three data conditions (see thrusterchain-route.ts). */
export type RouteDatasetId = "route-full" | "route-honeypot" | "route-bottleneck";
export type DatasetId = "halfcheetah-medium" | "gridworld-random" | "thrusterchain-stall" | "custom" | RouteDatasetId;

export interface StepResult {
  s2: number;
  r: number;
  done: boolean;
}

export interface Env {
  id: DatasetId;
  name: string;
  tag: string;
  description: string;
  width: number;
  height: number;
  axisX: string;
  axisY: string;
  actionNames: [string, string, string, string];
  start: number;
  blocked: Set<number>;
  terminals: Map<number, "goal" | "hazard">;
  step: (s: number, a: number) => StepResult;
  groundTruth: "exact" | "empirical";
  /** Named cells drawn on the grid (e.g. the route task's A, B, C). */
  landmarks?: { cell: number; label: string; hint?: string }[];
  /** Reference transition model (when the env's truth is a separate simulation, not the training data). */
  model?: { counts: Float64Array; modeNext: Int32Array; rewardMean: Float64Array; terminalFrac: Float64Array; rewardMin: number };
}

export interface Transition {
  s: number;
  a: number;
  r: number;
  s2: number;
  done: boolean;
}

export interface Dataset {
  env: Env;
  transitions: Transition[];
  episodes: number;
  counts: Float64Array; // S*4 visit counts of (s,a)
  rewardMean: Float64Array; // S*4 mean reward
  modeNext: Int32Array; // S*4 most common s'
  visits: Float64Array; // S state visit counts
  rewardMin: number;
  rewardMax: number;
}

export interface Hyper {
  alpha: number;
  gamma: number;
  lr: number;
  batch: number;
}

export interface Metrics {
  iter: number;
  updates: number;
  estQ: number;
  trueQ: number;
  loss: number;
  tdLoss: number;
  cqlGap: number;
  oodRate: number;
  divergence: number;
}

export interface EvalResult {
  policy: Int8Array;
  vPi: Float64Array;
  estQ: number;
  trueQ: number;
  oodRate: number;
  divergence: number;
  oodStates: Set<number>;
}

export const DX = [0, 1, 0, -1];
export const DY = [-1, 0, 1, 0];
export const UPDATES_PER_STEP = 20;

export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/* ---------------- Environments ---------------- */

function makeGridworld(): Env {
  const W = 10;
  const H = 10;
  const idx = (x: number, y: number) => y * W + x;
  const blocked = new Set([idx(6, 1), idx(6, 2), idx(6, 3), idx(2, 6), idx(3, 6), idx(4, 6)]);
  const terminals = new Map<number, "goal" | "hazard">([
    [idx(9, 0), "goal"],
    [idx(4, 3), "hazard"],
    [idx(8, 5), "hazard"],
    [idx(1, 2), "hazard"],
    [idx(6, 8), "hazard"],
  ]);
  return {
    id: "gridworld-random",
    name: "Gridworld-Random",
    tag: "10×10 · discrete",
    description:
      "A 10×10 maze logged by a uniformly random behavior policy. Goal (+10) top-right, hazards (−5), −0.1 per step.",
    width: W,
    height: H,
    axisX: "x position",
    axisY: "y position",
    actionNames: ["Up", "Right", "Down", "Left"],
    start: idx(0, 9),
    blocked,
    terminals,
    groundTruth: "exact",
    step(s, a) {
      if (terminals.has(s)) return { s2: s, r: 0, done: true };
      const x = s % W;
      const y = Math.floor(s / W);
      let s2 = idx(clamp(x + DX[a], 0, W - 1), clamp(y + DY[a], 0, H - 1));
      if (blocked.has(s2)) s2 = s;
      const t = terminals.get(s2);
      if (t === "goal") return { s2, r: 10, done: true };
      if (t === "hazard") return { s2, r: -5, done: true };
      return { s2, r: -0.1, done: false };
    },
  };
}

function makeCheetah(): Env {
  const W = 12;
  const H = 9;
  const LEVEL = 4;
  const idx = (x: number, y: number) => y * W + x;
  const terminals = new Map<number, "goal" | "hazard">();
  for (let x = 0; x < W; x++) {
    terminals.set(idx(x, 0), "hazard");
    terminals.set(idx(x, H - 1), "hazard");
  }
  return {
    id: "halfcheetah-medium",
    name: "D4RL HalfCheetah-Medium",
    tag: "2D projection · 12×9",
    description:
      "Educational 2D projection of HalfCheetah-medium: forward velocity vs. body pitch, logged by a medium-skill policy. Accelerating above speed 6 lifts the nose; pitching past the edge flips the agent (−4).",
    width: W,
    height: H,
    axisX: "forward velocity →",
    axisY: "body pitch (center = level)",
    actionNames: ["Pitch up", "Accelerate", "Pitch down", "Brake"],
    start: idx(0, LEVEL),
    blocked: new Set(),
    terminals,
    groundTruth: "exact",
    step(s, a) {
      if (terminals.has(s)) return { s2: s, r: 0, done: true };
      const x = s % W;
      const y = Math.floor(s / W);
      let nx = x;
      let ny = y;
      if (a === 0) ny = y - 1;
      if (a === 1) {
        nx = x + 1;
        if (x >= 6) ny = y - 1;
      }
      if (a === 2) ny = y + 1;
      if (a === 3) nx = x - 1;
      nx = clamp(nx, 0, W - 1);
      ny = clamp(ny, 0, H - 1);
      const s2 = idx(nx, ny);
      if (ny === 0 || ny === H - 1) return { s2, r: -4, done: true };
      const r = 0.05 + nx / (W - 1) - (0.3 * Math.abs(ny - LEVEL)) / LEVEL;
      return { s2, r: Math.round(r * 1000) / 1000, done: false };
    },
  };
}

/* ---------------- Dataset construction ---------------- */

export function buildDataset(env: Env, transitions: Transition[], episodes: number): Dataset {
  const S = env.width * env.height;
  const counts = new Float64Array(S * 4);
  const rsum = new Float64Array(S * 4);
  const visits = new Float64Array(S);
  const nextMaps: Map<number, number>[] = Array.from({ length: S * 4 }, () => new Map());
  let rMin = Infinity;
  let rMax = -Infinity;
  for (const t of transitions) {
    const k = t.s * 4 + t.a;
    counts[k]++;
    rsum[k] += t.r;
    visits[t.s]++;
    nextMaps[k].set(t.s2, (nextMaps[k].get(t.s2) ?? 0) + 1);
    rMin = Math.min(rMin, t.r);
    rMax = Math.max(rMax, t.r);
  }
  const rewardMean = new Float64Array(S * 4);
  const modeNext = new Int32Array(S * 4).fill(-1);
  for (let k = 0; k < S * 4; k++) {
    if (counts[k] > 0) {
      rewardMean[k] = rsum[k] / counts[k];
      let best = -1;
      let bc = -1;
      nextMaps[k].forEach((c, s2) => {
        if (c > bc) {
          bc = c;
          best = s2;
        }
      });
      modeNext[k] = best;
    }
  }
  return {
    env,
    transitions,
    episodes,
    counts,
    rewardMean,
    modeNext,
    visits,
    rewardMin: isFinite(rMin) ? rMin : 0,
    rewardMax: isFinite(rMax) ? rMax : 0,
  };
}

export function generateDataset(id: Exclude<DatasetId, "custom" | RouteDatasetId>, seed: number): Dataset {
  const rng = mulberry32(seed);
  const env = id === "gridworld-random" ? makeGridworld() : makeCheetah();
  const transitions: Transition[] = [];
  const S = env.width * env.height;
  const free: number[] = [];
  for (let s = 0; s < S; s++) if (!env.blocked.has(s) && !env.terminals.has(s)) free.push(s);

  const episodes = id === "gridworld-random" ? 240 : 170;
  const horizon = id === "gridworld-random" ? 50 : 40;
  const behavior = id === "gridworld-random" ? [0.25, 0.25, 0.25, 0.25] : [0.15, 0.5, 0.2, 0.15];
  const sample = () => {
    const u = rng();
    let acc = 0;
    for (let a = 0; a < 4; a++) {
      acc += behavior[a];
      if (u < acc) return a;
    }
    return 3;
  };

  for (let e = 0; e < episodes; e++) {
    let s =
      id === "gridworld-random"
        ? rng() < 0.6
          ? env.start
          : free[Math.floor(rng() * free.length)]
        : env.start + env.width * (Math.floor(rng() * 3) - 1);
    for (let t = 0; t < horizon; t++) {
      const a = sample();
      const { s2, r, done } = env.step(s, a);
      transitions.push({ s, a, r, s2, done });
      if (done) break;
      s = s2;
    }
  }
  return buildDataset(env, transitions, episodes);
}

/* ---------------- Custom JSON datasets ---------------- */

interface RawTransition {
  s: [number, number];
  a: number;
  r: number;
  s2: [number, number];
  done?: boolean;
}

export function parseCustomDataset(raw: unknown, onProgress?: (pct: number, label: string) => void): Dataset {
  if (!raw || typeof raw !== "object") throw new Error("File is not a JSON object.");
  onProgress?.(15, "Validating grid dimensions…");
  const o = raw as Record<string, unknown>;
  const W = Number(o.width);
  const H = Number(o.height);
  if (!Number.isInteger(W) || !Number.isInteger(H) || W < 2 || H < 2 || W > 24 || H > 24)
    throw new Error("`width` and `height` must be integers between 2 and 24.");
  if (!Array.isArray(o.transitions) || o.transitions.length < 10)
    throw new Error("`transitions` must be an array with at least 10 entries.");
  const idx = (p: unknown, field: string, i: number) => {
    if (!Array.isArray(p) || p.length !== 2) throw new Error(`transitions[${i}].${field} must be [x, y].`);
    const [x, y] = p.map(Number);
    if (!Number.isInteger(x) || !Number.isInteger(y) || x < 0 || y < 0 || x >= W || y >= H)
      throw new Error(`transitions[${i}].${field} is outside the ${W}×${H} grid.`);
    return y * W + x;
  };
  const rawT = o.transitions as RawTransition[];
  const total = rawT.length;
  const transitions: Transition[] = rawT.map((t, i) => {
    const a = Number(t.a);
    if (![0, 1, 2, 3].includes(a)) throw new Error(`transitions[${i}].a must be 0–3 (up, right, down, left).`);
    const r = Number(t.r);
    if (!Number.isFinite(r)) throw new Error(`transitions[${i}].r must be a number.`);
    if (onProgress && total > 500 && i % Math.ceil(total / 20) === 0)
      onProgress(20 + Math.round((i / total) * 60), `Validating transitions — ${i.toLocaleString()} / ${total.toLocaleString()}`);
    return { s: idx(t.s, "s", i), a, r, s2: idx(t.s2, "s2", i), done: Boolean(t.done) };
  });
  onProgress?.(85, "Building empirical MDP & reward statistics…");

  const terminals = new Map<number, "goal" | "hazard">();
  transitions.forEach((t) => {
    if (t.done) terminals.set(t.s2, t.r > 0 ? "goal" : "hazard");
  });
  const startArr = Array.isArray(o.start) ? (o.start as number[]) : null;
  const start = startArr ? clamp(startArr[1], 0, H - 1) * W + clamp(startArr[0], 0, W - 1) : transitions[0].s;

  const env: Env = {
    id: "custom",
    name: typeof o.name === "string" && o.name.trim() ? o.name.trim().slice(0, 48) : "Custom dataset",
    tag: `${W}×${H} · uploaded`,
    description: "User-uploaded transitions. Ground truth uses the empirical MDP estimated from the data.",
    width: W,
    height: H,
    axisX: "x",
    axisY: "y",
    actionNames: ["Up", "Right", "Down", "Left"],
    start,
    blocked: new Set(),
    terminals,
    groundTruth: "empirical",
    step: () => ({ s2: 0, r: 0, done: true }),
  };
  const ds = buildDataset(env, transitions, Number(o.episodes) || 1);
  const rMin = ds.rewardMin;
  env.step = (s, a) => {
    if (terminals.has(s)) return { s2: s, r: 0, done: true };
    const k = s * 4 + a;
    if (ds.counts[k] === 0) return { s2: s, r: rMin, done: false };
    const s2 = ds.modeNext[k];
    return { s2, r: ds.rewardMean[k], done: terminals.has(s2) };
  };
  return ds;
}

export function sampleCustomJson(): string {
  const W = 6;
  const H = 4;
  const rng = mulberry32(42);
  const transitions: RawTransition[] = [];
  for (let e = 0; e < 60; e++) {
    let x = 0;
    let y = 3;
    for (let t = 0; t < 20; t++) {
      const a = Math.floor(rng() * 4);
      const nx = clamp(x + DX[a], 0, W - 1);
      const ny = clamp(y + DY[a], 0, H - 1);
      const goal = nx === 5 && ny === 0;
      const pit = nx === 3 && ny === 2;
      transitions.push({ s: [x, y], a, r: goal ? 5 : pit ? -3 : -0.05, s2: [nx, ny], done: goal || pit });
      if (goal || pit) break;
      x = nx;
      y = ny;
    }
  }
  return JSON.stringify({ name: "Corridor Demo", width: W, height: H, start: [0, 3], episodes: 60, transitions }, null, 2);
}

/* ---------------- Learning ---------------- */

export function createQ(ds: Dataset) {
  return new Float64Array(ds.env.width * ds.env.height * 4);
}

/** Tabular Conservative Q-Learning: TD update + alpha * (logsumexp Q(s,·) − Q(s,a_data)). */
export function trainUpdates(Q: Float64Array, ds: Dataset, h: Hyper, rng: () => number, n: number) {
  const T = ds.transitions;
  const delta = new Float64Array(Q.length);
  const hits = new Float64Array(Q.length / 4);
  let tdSum = 0;
  let gapSum = 0;
  for (let u = 0; u < n; u++) {
    delta.fill(0);
    hits.fill(0);
    for (let b = 0; b < h.batch; b++) {
      const t = T[Math.floor(rng() * T.length)];
      const base = t.s * 4;
      let maxNext = 0;
      if (!t.done) {
        const nb = t.s2 * 4;
        maxNext = Math.max(Q[nb], Q[nb + 1], Q[nb + 2], Q[nb + 3]);
      }
      const td = t.r + h.gamma * maxNext - Q[base + t.a];
      delta[base + t.a] += td;
      tdSum += td * td;
      const m = Math.max(Q[base], Q[base + 1], Q[base + 2], Q[base + 3]);
      let z = 0;
      for (let k = 0; k < 4; k++) z += Math.exp(Q[base + k] - m);
      gapSum += m + Math.log(z) - Q[base + t.a];
      if (h.alpha > 0) {
        for (let k = 0; k < 4; k++) delta[base + k] -= (h.alpha * Math.exp(Q[base + k] - m)) / z;
        delta[base + t.a] += h.alpha;
      }
      hits[t.s]++;
    }
    for (let s = 0; s < hits.length; s++) {
      if (hits[s] === 0) continue;
      for (let k = 0; k < 4; k++) Q[s * 4 + k] += (h.lr * delta[s * 4 + k]) / hits[s];
    }
  }
  const total = n * h.batch;
  return { tdLoss: tdSum / total, cqlGap: gapSum / total };
}

export function greedy(Q: Float64Array, s: number) {
  const b = s * 4;
  let best = 0;
  for (let k = 1; k < 4; k++) if (Q[b + k] > Q[b + best]) best = k;
  return best;
}

export function policyEvaluation(env: Env, policy: Int8Array, gamma: number) {
  const S = env.width * env.height;
  const V = new Float64Array(S);
  for (let it = 0; it < 500; it++) {
    let diff = 0;
    for (let s = 0; s < S; s++) {
      if (env.blocked.has(s) || env.terminals.has(s)) continue;
      const { s2, r, done } = env.step(s, policy[s]);
      const v = r + (done ? 0 : gamma * V[s2]);
      diff = Math.max(diff, Math.abs(v - V[s]));
      V[s] = v;
    }
    if (diff < 1e-7) break;
  }
  return V;
}

export function qStar(env: Env, gamma: number) {
  const S = env.width * env.height;
  const V = new Float64Array(S);
  const Q = new Float64Array(S * 4);
  for (let it = 0; it < 500; it++) {
    let diff = 0;
    for (let s = 0; s < S; s++) {
      if (env.blocked.has(s) || env.terminals.has(s)) continue;
      let best = -Infinity;
      for (let a = 0; a < 4; a++) {
        const { s2, r, done } = env.step(s, a);
        const q = r + (done ? 0 : gamma * V[s2]);
        Q[s * 4 + a] = q;
        if (q > best) best = q;
      }
      diff = Math.max(diff, Math.abs(best - V[s]));
      V[s] = best;
    }
    if (diff < 1e-7) break;
  }
  return Q;
}

export function evaluate(Q: Float64Array, ds: Dataset, gamma: number): EvalResult {
  const { env } = ds;
  const S = env.width * env.height;
  const policy = new Int8Array(S);
  for (let s = 0; s < S; s++) policy[s] = greedy(Q, s);
  const vPi = policyEvaluation(env, policy, gamma);
  let est = 0;
  let tru = 0;
  let n = 0;
  let ood = 0;
  let div = 0;
  const oodStates = new Set<number>();
  for (let s = 0; s < S; s++) {
    if (ds.visits[s] === 0 || env.terminals.has(s) || env.blocked.has(s)) continue;
    const a = policy[s];
    est += Q[s * 4 + a];
    tru += vPi[s];
    n++;
    const c = ds.counts[s * 4 + a];
    if (c === 0) {
      ood++;
      oodStates.add(s);
    }
    div += -Math.log((c + 0.05) / (ds.visits[s] + 0.2));
  }
  return {
    policy,
    vPi,
    estQ: n ? est / n : 0,
    trueQ: n ? tru / n : 0,
    oodRate: n ? ood / n : 0,
    divergence: n ? div / n : 0,
    oodStates,
  };
}
