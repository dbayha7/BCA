// ThrusterChain route task: a lab version of the research goal/danger task
// (experiments/thrusterchain_bca/two_scenario_v2 + the Path A data conditions).
//
// Faithful parts (same equations as the research code):
//   - the 13-state plant x' = A·x + B·thrust(a) in the stalling world with the retuned stall
//     slope 3.0, summed in the canonical order, commands cast to float32 and clipped;
//   - the fixed finite-horizon tracking controller (Riccati recursion, clipped at the knee);
//   - the geometry: nominal route A → B (t25) → C (t30) → goal (t75), the danger sphere at the
//     end of five opposite-controller steps from B, radius min(0.1, 0.25 × route separation);
//   - the task reward −1 − 0.01‖a‖², +100 on entering the goal, −100 on crossing the danger
//     (danger wins), horizon 100;
//   - the four recorder families (nominal low noise, broad noise, opposite excursion at
//     t25–29, noise-free excursion recovering one step earlier), start jitter, current and waves;
//   - the data conditions: Honeypot removes every edge touching the danger sphere; Bottleneck
//     removes every edge whose B→C progress interval meets the slab [0.2, 0.8].
// The geometry is checked against the research's saved geometry.json (ROUTE_FIXTURE).
//
// Lab-only parts:
//   - random draws use the lab's generator, so the data are an independent sample with the
//     research recipe, not the research dataset itself;
//   - the 13-dim state is projected onto a 2-D grid for the tabular lab: x = progress along
//     the route (columns centered on route times), y = displacement toward the danger;
//   - four discrete actions: Controller, Opposite (−controller), Coast (zero) and Overdrive
//     (2 × controller, past the knee). The recorders only ever take the first two;
//   - ground truth comes from a separate reference simulation of the real plant that tries
//     every action, never from the training data.

import { buildDataset, mulberry32, type Dataset, type Env, type RouteDatasetId, type Transition } from "./rl";
import { PLANT_A, PLANT_B } from "./thrusterchain";

export type RouteCondition = "full" | "honeypot" | "bottleneck";
export const ROUTE_IDS: Record<RouteCondition, RouteDatasetId> = { full: "route-full", honeypot: "route-honeypot", bottleneck: "route-bottleneck" };
export const routeConditionOf = (id: string): RouteCondition | null =>
  id === "route-full" ? "full" : id === "route-honeypot" ? "honeypot" : id === "route-bottleneck" ? "bottleneck" : null;

export const ROUTE_HORIZON = 100;
export const ROUTE_GAMMA = 0.98;
export const STALL_SLOPE_RETUNED = 3.0;
export const KNEE32 = Math.fround(0.8);
export const SLAB: [number, number] = [0.2, 0.8];

/** The research geometry recipe (protocol.json "geometry_recipe"). */
export const RECIPE = {
  action_weight: 0.3,
  b_time: 25,
  c_time: 30,
  danger_radius_fraction: 0.25,
  danger_radius_max: 0.1,
  goal_radius: 0.1,
  goal_time: 75,
  initial_state: [0.8, 0, 0.8, 0, 0.8, 0, 0.8, 0, 0.8, 0, 0.02, 0.01, 0],
  physical_weight: 1.0,
  terminal_weight_multiplier: 10.0,
  tracking_duration: 75,
} as const;

/** The research collection recipe (protocol.json "collection"). */
export const COLLECTION = {
  action_noise: 0.03,
  wide_action_noise: 0.15,
  physical_start_jitter: 0.01,
  current_range: [0.015, 0.025] as const,
  wave_amplitude_range: [0.005, 0.015] as const,
  parents: 1280,
};

/** Values from the research's saved geometry.json (two_scenario_v2/setup_attempt_1), for verification. */
export const ROUTE_FIXTURE = {
  route25: [0.4917612375859079, 0.002370614762457292, 0.5248202389933431, -0.044775817530693124, 0.4206825836679249, -0.07763488426679649, 0.4202232279076811, -0.13004081624624003, 0.16195012206230702, -0.0684140153872638, 0.02, 0.00893006344689077, 0.004500440737806182],
  route30: [0.4407656862313213, 0.01307750070207371, 0.4724889676986611, -0.0358737781042811, 0.378828417429089, -0.0673638844888261, 0.3778844217088036, -0.11545212968083289, 0.1460206879722244, -0.06286300025962155, 0.02, -0.0024354415373579203, 0.009698898108450866],
  route75: [0.043414822761855196, 0.0402164658401881, 0.03067338567509419, 0.019144518490403998, 0.0026836588336216875, 0.009764132957418558, 0.005144517630476514, 0.004605506673253171, -0.005557816716230799, 0.003881785999777754, 0.02, 0.0016952950915565786, 0.009855251115651213],
  route100: [0.00039134690551214334, 0.0038784696848970903, 0.00200297469457832, -0.0014354270227863327, -0.0025228166285275007, 0.0012968998038542963, 0.0007255926597819288, -6.61058492612593e-5, -0.0019004676549015023, 0.0006381117540419641, 0.02, -0.0029213880873383783, 0.009563759284045044],
  dangerCenter: [0.23473942262281536, 1.1367823784656237, -0.2004016687824015, 2.059586847995524, -0.9010604155407012, 1.8377882177451126, -1.5673013424244495, 1.1040292717896572, -0.6389497579884198, -0.006174410713307404],
  dangerRadius: 0.1,
  goalRadius: 0.1,
  separation: 3.699393203226299,
};

/* ---------------- Plant, task and controller ---------------- */

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** Deployed command: float32 cast, then clipped to [−1, 1] (task.deployed_action). */
export function deployed(a: ArrayLike<number>): Float64Array {
  const out = new Float64Array(4);
  for (let k = 0; k < 4; k++) out[k] = clamp(Math.fround(a[k]), -1, 1);
  return out;
}

/** Canonical dynamics, stalling world, slope 3.0: columns summed in index order, then thrusts. */
export function plantStep(x: Float64Array, a: Float64Array, slope = STALL_SLOPE_RETUNED): Float64Array {
  const th = new Float64Array(4);
  for (let k = 0; k < 4; k++) {
    const ak = clamp(a[k], -1, 1);
    th[k] = clamp(ak, -KNEE32, KNEE32) - slope * Math.sign(ak) * Math.max(Math.abs(ak) - KNEE32, 0);
  }
  const nx = new Float64Array(13);
  for (let i = 0; i < 13; i++) {
    const Ai = PLANT_A[i];
    let v = x[0] * Ai[0];
    for (let j = 1; j < 13; j++) v += x[j] * Ai[j];
    const Bi = PLANT_B[i];
    for (let k = 0; k < 4; k++) v += th[k] * Bi[k];
    nx[i] = v;
  }
  return nx;
}

export interface Sphere {
  center: number[];
  radius: number;
}

export function contains(x: ArrayLike<number>, r: Sphere): boolean {
  let d = 0;
  for (let i = 0; i < 10; i++) d += (x[i] - r.center[i]) ** 2;
  return d <= r.radius * r.radius;
}

/** Closed segment–sphere test in the ten physical coordinates (task.edge_intersects). */
export function edgeTouches(x: ArrayLike<number>, y: ArrayLike<number>, r: Sphere): boolean {
  let den = 0;
  let num = 0;
  for (let i = 0; i < 10; i++) {
    const a = x[i] - r.center[i];
    const d = y[i] - x[i];
    den += d * d;
    num -= a * d;
  }
  const f = den > 0 ? clamp(num / den, 0, 1) : 0;
  let s = 0;
  for (let i = 0; i < 10; i++) s += (x[i] - r.center[i] + f * (y[i] - x[i])) ** 2;
  return s <= r.radius * r.radius;
}

/* Small dense linear algebra for the Riccati recursion. */
type Mat = number[][];
const zeros = (r: number, c: number): Mat => Array.from({ length: r }, () => new Array<number>(c).fill(0));
const T = (M: Mat): Mat => M[0].map((_, j) => M.map((row) => row[j]));
function mul(Ma: Mat, Mb: Mat): Mat {
  const out = zeros(Ma.length, Mb[0].length);
  for (let i = 0; i < Ma.length; i++)
    for (let j = 0; j < Mb[0].length; j++) {
      let v = 0;
      for (let k = 0; k < Mb.length; k++) v += Ma[i][k] * Mb[k][j];
      out[i][j] = v;
    }
  return out;
}
const mulv = (M: Mat, v: number[]): number[] => M.map((row) => row.reduce((acc, m, k) => acc + m * v[k], 0));
const add = (Ma: Mat, Mb: Mat, sign = 1): Mat => Ma.map((row, i) => row.map((v, j) => v + sign * Mb[i][j]));
/** Solves H X = Y (H square) by Gaussian elimination with partial pivoting, as LAPACK's gesv. */
function solve(H: Mat, Y: Mat): Mat {
  const n = H.length;
  const A = H.map((r) => [...r]);
  const X = Y.map((r) => [...r]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    [X[c], X[p]] = [X[p], X[c]];
    for (let r = c + 1; r < n; r++) {
      const f = A[r][c] / A[c][c];
      for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
      for (let k = 0; k < X[0].length; k++) X[r][k] -= f * X[c][k];
    }
  }
  for (let c = n - 1; c >= 0; c--) {
    for (let k = 0; k < X[0].length; k++) {
      let v = X[c][k];
      for (let j = c + 1; j < n; j++) v -= A[c][j] * X[j][k];
      X[c][k] = v / A[c][c];
    }
  }
  return X;
}

export type Controller = (x: Float64Array, t: number) => Float64Array;

/** task.tracking_controller: finite-horizon LQ tracking of a decaying target, clipped at the knee. */
export function trackingController(): Controller {
  const A: Mat = PLANT_A.map((r) => [...r]);
  const B: Mat = PLANT_B.map((r) => [...r]);
  const At = T(A);
  const Bt = T(B);
  const Q = zeros(13, 13);
  for (let i = 0; i < 10; i++) Q[i][i] = RECIPE.physical_weight;
  const R = zeros(4, 4);
  for (let i = 0; i < 4; i++) R[i][i] = RECIPE.action_weight;
  let P = Q.map((r) => r.map((v) => v * RECIPE.terminal_weight_multiplier));
  let p = new Array<number>(13).fill(0);
  const gains: Mat[] = new Array(ROUTE_HORIZON);
  const offsets: number[][] = new Array(ROUTE_HORIZON);
  for (let t = ROUTE_HORIZON - 1; t >= 0; t--) {
    const target = new Array<number>(13).fill(0);
    for (let i = 0; i < 10; i += 2) target[i] = RECIPE.initial_state[i] * Math.max(0, 1 - t / RECIPE.tracking_duration);
    const BtP = mul(Bt, P);
    const H = add(R, mul(BtP, B));
    const K = solve(H, mul(BtP, A));
    const k = solve(
      H,
      mulv(Bt, p).map((v) => [v]),
    ).map((r) => r[0]);
    gains[t] = K;
    offsets[t] = k;
    const AtP = mul(At, P);
    const AtPB = mul(AtP, B);
    const Qtarget = mulv(Q, target);
    const Atp = mulv(At, p);
    const AtPBk = mulv(AtPB, k);
    p = Qtarget.map((v, i) => -v + Atp[i] - AtPBk[i]);
    P = add(add(Q, mul(AtP, A)), mul(AtPB, K), -1);
  }
  return (x, t) => {
    const K = gains[t];
    const k = offsets[t];
    const out = new Float64Array(4);
    for (let i = 0; i < 4; i++) {
      let v = 0;
      for (let j = 0; j < 13; j++) v += K[i][j] * x[j];
      out[i] = Math.fround(clamp(-v - k[i], -KNEE32, KNEE32));
    }
    return out;
  };
}

/* ---------------- Geometry ---------------- */

export interface RouteGeometry {
  route: Float64Array[];
  excursion: Float64Array[];
  A: Float64Array;
  B: Float64Array;
  C: Float64Array;
  goal: Sphere;
  danger: Sphere;
  separation: number;
}

/** task.construct_geometry: one nominal route and one frozen five-step opposite excursion from B. */
export function buildRouteGeometry(control: Controller = trackingController()): RouteGeometry {
  let x = Float64Array.from(RECIPE.initial_state);
  const route = [x];
  for (let t = 0; t < ROUTE_HORIZON; t++) {
    x = plantStep(x, deployed(control(x, t)));
    route.push(x);
  }
  let w = route[RECIPE.b_time];
  const excursion = [w];
  for (let t = RECIPE.b_time; t < RECIPE.c_time; t++) {
    w = plantStep(w, deployed(Array.from(control(w, t), (v) => -v)));
    excursion.push(w);
  }
  const center = Array.from(w.slice(0, 10));
  let separation = Infinity;
  for (let t = 0; t < route.length - 1; t++) {
    let den = 0;
    let num = 0;
    for (let i = 0; i < 10; i++) {
      const a = route[t][i] - center[i];
      const d = route[t + 1][i] - route[t][i];
      den += d * d;
      num -= a * d;
    }
    const f = den > 0 ? clamp(num / den, 0, 1) : 0;
    let s = 0;
    for (let i = 0; i < 10; i++) s += (route[t][i] - center[i] + f * (route[t + 1][i] - route[t][i])) ** 2;
    separation = Math.min(separation, Math.sqrt(s));
  }
  const radius = Math.min(RECIPE.danger_radius_max, RECIPE.danger_radius_fraction * separation);
  return {
    route,
    excursion,
    A: route[0],
    B: route[RECIPE.b_time],
    C: route[RECIPE.c_time],
    goal: { center: Array.from(route[RECIPE.goal_time].slice(0, 10)), radius: RECIPE.goal_radius },
    danger: { center, radius },
    separation,
  };
}

/** Largest absolute difference between this geometry and the research's saved geometry.json values. */
export function geometryDeviation(g: RouteGeometry): number {
  let m = 0;
  const cmp = (a: ArrayLike<number>, b: number[]) => b.forEach((v, i) => (m = Math.max(m, Math.abs(a[i] - v))));
  cmp(g.route[25], ROUTE_FIXTURE.route25);
  cmp(g.route[30], ROUTE_FIXTURE.route30);
  cmp(g.route[75], ROUTE_FIXTURE.route75);
  cmp(g.route[100], ROUTE_FIXTURE.route100);
  cmp(g.danger.center, ROUTE_FIXTURE.dangerCenter);
  m = Math.max(m, Math.abs(g.separation - ROUTE_FIXTURE.separation), Math.abs(g.danger.radius - ROUTE_FIXTURE.dangerRadius));
  return m;
}

/** Bottleneck progress coordinate along B → C (connector_cut.progress): 0 at B, 1 at C. */
export function slabProgress(x: ArrayLike<number>, g: RouteGeometry): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < 10; i++) {
    const d = g.C[i] - g.B[i];
    num += (x[i] - g.B[i]) * d;
    den += d * d;
  }
  return num / den;
}

export function crossesSlab(x: ArrayLike<number>, y: ArrayLike<number>, g: RouteGeometry): boolean {
  const qs = slabProgress(x, g);
  const qn = slabProgress(y, g);
  return Math.min(qs, qn) <= SLAB[1] && Math.max(qs, qn) >= SLAB[0];
}

/* ---------------- Task step ---------------- */

export interface RouteStep {
  next: Float64Array;
  reward: number;
  goal: boolean;
  danger: boolean;
  terminal: boolean;
}

/** task.Task.step: reward −1 − 0.01‖a‖² ± 100; danger wins; goal, danger and the horizon end the episode. */
export function taskStep(x: Float64Array, t: number, action: Float64Array, g: RouteGeometry): RouteStep {
  const a = deployed(action);
  const next = plantStep(x, a);
  const danger = edgeTouches(x, next, g.danger);
  const goal = contains(next, g.goal) && !danger;
  let a2 = 0;
  for (let k = 0; k < 4; k++) a2 += a[k] * a[k];
  const reward = -1 - 0.01 * a2 + (goal ? 100 : 0) - (danger ? 100 : 0);
  return { next, reward, goal, danger, terminal: danger || goal || t + 1 === ROUTE_HORIZON };
}

/* ---------------- Discrete actions and recorders ---------------- */

export const ROUTE_ACTIONS: [string, string, string, string] = ["Controller", "Opposite", "Coast", "Overdrive"];
export const ROUTE_ACTION_HINTS = [
  "The fixed tracking controller's command (what the recorders mostly do).",
  "Minus the controller: the excursion toward the danger region.",
  "Zero command: the plant drifts with the current and waves. Never in the data.",
  "Twice the controller, past the stall knee (thrust collapses at slope 3). Never in the data.",
];

/** Continuous command for a discrete lab action (before deployment). */
export function commandFor(action: number, control: Controller, x: Float64Array, t: number): Float64Array {
  const c = control(x, t);
  if (action === 1) return c.map((v) => -v);
  if (action === 2) return new Float64Array(4);
  if (action === 3) return c.map((v) => clamp(2 * v, -1, 1));
  return c;
}

function gaussian(rng: () => number): number {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

/** The research start law: A with ±0.01 jitter, current U[0.015, 0.025], wave amplitude U[0.005, 0.015]. */
export function drawStart(g: RouteGeometry, rng: () => number): Float64Array {
  const x = Float64Array.from(g.A);
  const j = COLLECTION.physical_start_jitter;
  for (let i = 0; i < 10; i++) x[i] += -j + 2 * j * rng();
  x[10] = COLLECTION.current_range[0] + (COLLECTION.current_range[1] - COLLECTION.current_range[0]) * rng();
  const amp = COLLECTION.wave_amplitude_range[0] + (COLLECTION.wave_amplitude_range[1] - COLLECTION.wave_amplitude_range[0]) * rng();
  const phase = -Math.PI + 2 * Math.PI * rng();
  x[11] = amp * Math.cos(phase);
  x[12] = amp * Math.sin(phase);
  return x;
}

/** data.recorder_action: family 0 low noise, 1 broad noise, 2 opposite at t25–29, 3 noise-free opposite at t25–28. */
export function recorderAction(family: number, control: Controller, x: Float64Array, t: number, rng: () => number): { command: Float64Array; label: number } {
  const c = control(x, t);
  const sigma = family === 3 ? 0 : family === 1 ? COLLECTION.wide_action_noise : COLLECTION.action_noise;
  const end = RECIPE.c_time - (family === 3 ? 1 : 0);
  const opposite = (family === 2 || family === 3) && t >= RECIPE.b_time && t < end;
  const command = new Float64Array(4);
  for (let k = 0; k < 4; k++) command[k] = (opposite ? -c[k] : c[k]) + sigma * gaussian(rng);
  return { command, label: opposite ? 1 : 0 };
}

/* ---------------- Grid projection ---------------- */

export interface RouteProjection {
  width: number;
  height: number;
  /** Column edges in route-progress units (ascending); column i covers [edges[i], edges[i+1]). */
  colEdges: number[];
  yLo: number;
  yStep: number;
  e1: number[];
  e2: number[];
  origin: number[];
  cellOf(x: ArrayLike<number>): number;
  coords(x: ArrayLike<number>): [number, number];
}

const ROUTE_ANCHORS = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75];

export function buildProjection(g: RouteGeometry): RouteProjection {
  const origin = Array.from(g.A.slice(0, 10));
  const D = g.goal.center;
  const e1 = D.map((v, i) => v - origin[i]);
  const n1 = Math.hypot(...e1);
  for (let i = 0; i < 10; i++) e1[i] /= n1;
  const w = g.danger.center.map((v, i) => v - g.B[i]);
  const dot = w.reduce((acc, v, i) => acc + v * e1[i], 0);
  for (let i = 0; i < 10; i++) w[i] -= dot * e1[i];
  const n2 = Math.hypot(...w);
  const e2 = w.map((v) => v / n2);
  const coords = (x: ArrayLike<number>): [number, number] => {
    let p = 0;
    let q = 0;
    for (let i = 0; i < 10; i++) {
      p += (x[i] - origin[i]) * e1[i];
      q += (x[i] - origin[i]) * e2[i];
    }
    return [p, q];
  };
  // Columns centered on route times, so A, B (t25), C (t30) and the goal (t75) sit in separate columns.
  const anchors = ROUTE_ANCHORS.map((t) => coords(g.route[t])[0]);
  const edges = [-Infinity];
  for (let i = 0; i + 1 < anchors.length; i++) edges.push((anchors[i] + anchors[i + 1]) / 2);
  const last = anchors[anchors.length - 1];
  const step = 0.22;
  const pMax = coords(g.danger.center)[0] + 0.35;
  for (let e = last + step / 2; e < pMax; e += step) edges.push(e);
  edges.push(Infinity);
  const width = edges.length - 1;
  const yLo = -0.5;
  const yStep = 0.3;
  const yMax = coords(g.danger.center)[1] + 0.45;
  const height = Math.ceil((yMax - yLo) / yStep);
  const cellOf = (x: ArrayLike<number>) => {
    const [p, q] = coords(x);
    let c = 0;
    while (c < width - 1 && p >= edges[c + 1]) c++;
    const rowFromBottom = clamp(Math.floor((q - yLo) / yStep), 0, height - 1);
    return (height - 1 - rowFromBottom) * width + c;
  };
  return { width, height, colEdges: edges, yLo, yStep, e1, e2, origin, cellOf, coords };
}

/* ---------------- Datasets ---------------- */

export interface RouteCounts {
  /** Recorded edges before the condition's mask. */
  recorded: number;
  removed: number;
  removedDanger: number;
  removedSlab: number;
  /** Complete start-to-goal episodes whose every edge survives the mask. */
  intactSuccesses: number;
  episodes: number;
}

export interface RouteDatasetResult {
  dataset: Dataset;
  counts: RouteCounts;
  geometryDeviation: number;
}

/** Reference model: every action tried from realistic states in the real plant, independent of the training data. */
function referenceModel(g: RouteGeometry, proj: RouteProjection, control: Controller, seed: number, episodes: number, onTick?: (e: number) => void) {
  const S = proj.width * proj.height;
  const counts = new Float64Array(S * 4);
  const rsum = new Float64Array(S * 4);
  const tsum = new Float64Array(S * 4);
  const nextMaps: Map<number, number>[] = Array.from({ length: S * 4 }, () => new Map());
  const rng = mulberry32((seed ^ 0x5eed) >>> 0);
  const goalCell = proj.cellOf(g.goal.center);
  const dangerCell = proj.cellOf(g.danger.center);
  let rMin = Infinity;
  for (let e = 0; e < episodes; e++) {
    const family = e % 4;
    let x = drawStart(g, rng);
    for (let t = 0; t < ROUTE_HORIZON; t++) {
      // Mostly the recorder's behaviour (so states are realistic), with every action tried often.
      const explore = rng() < 0.35;
      const label = explore ? Math.floor(rng() * 4) : recorderAction(family, control, x, t, rng).label;
      const step = taskStep(x, t, commandFor(label, control, x, t), g);
      const s = proj.cellOf(x);
      const s2 = step.danger ? dangerCell : step.goal ? goalCell : proj.cellOf(step.next);
      const k = s * 4 + label;
      counts[k]++;
      rsum[k] += step.reward;
      tsum[k] += step.goal || step.danger ? 1 : 0;
      nextMaps[k].set(s2, (nextMaps[k].get(s2) ?? 0) + 1);
      rMin = Math.min(rMin, step.reward);
      if (step.terminal) break;
      x = step.next;
    }
    if (onTick && e % 50 === 49) onTick(e + 1);
  }
  const modeNext = new Int32Array(S * 4).fill(-1);
  const rewardMean = new Float64Array(S * 4);
  const terminalFrac = new Float64Array(S * 4);
  for (let k = 0; k < S * 4; k++) {
    if (!counts[k]) continue;
    rewardMean[k] = rsum[k] / counts[k];
    terminalFrac[k] = tsum[k] / counts[k];
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
  return { counts, modeNext, rewardMean, terminalFrac, rewardMin: Number.isFinite(rMin) ? rMin : -101 };
}

/** env.step from a reference model; unseen (state, action) pairs stay put at the worst observed reward. */
export function stepFromModel(model: NonNullable<Env["model"]>, terminals: Env["terminals"]): Env["step"] {
  return (s, a) => {
    if (terminals.has(s)) return { s2: s, r: 0, done: true };
    const k = s * 4 + a;
    if (!model.counts[k]) return { s2: s, r: model.rewardMin, done: false };
    return { s2: model.modeNext[k], r: model.rewardMean[k], done: model.terminalFrac[k] > 0.5 };
  };
}

const NAMES: Record<RouteCondition, string> = {
  full: "ThrusterChain route · full data",
  honeypot: "ThrusterChain route · Honeypot",
  bottleneck: "ThrusterChain route · Bottleneck",
};
const DESCRIPTIONS: Record<RouteCondition, string> = {
  full: "The research goal/danger task on the 13-state ThrusterChain plant (stall slope 3). Start A, route through B (t25) and C (t30) to the goal (+100); a danger region (−100) sits at the end of a five-step opposite excursion from B. Logged by the four recorder families; nothing removed.",
  honeypot: "Same task and recorders, but every logged edge touching the danger region is removed, so the data never show the −100. The opposite excursion at B looks safe in the data.",
  bottleneck: "Same task and recorders, but every logged edge crossing the slab between B and C (progress 0.2–0.8) is removed: no complete start-to-goal route survives, only the fragments on either side.",
};

/**
 * Builds the lab dataset for one condition. The geometry and plant follow the research code exactly; the
 * recorder draws use the lab's generator (seeded), so the data are an independent sample with the same recipe.
 */
export function generateRouteDataset(condition: RouteCondition, seed = 2026, onProgress?: (pct: number, label: string) => void): RouteDatasetResult {
  onProgress?.(3, "Solving the tracking controller (Riccati recursion)…");
  const control = trackingController();
  const g = buildRouteGeometry(control);
  const deviation = geometryDeviation(g);
  onProgress?.(8, `Geometry rebuilt — max difference from the research geometry ${deviation.toExponential(1)}`);
  const proj = buildProjection(g);
  const goalCell = proj.cellOf(g.goal.center);
  const dangerCell = proj.cellOf(g.danger.center);
  const transitions: Transition[] = [];
  const counts: RouteCounts = { recorded: 0, removed: 0, removedDanger: 0, removedSlab: 0, intactSuccesses: 0, episodes: COLLECTION.parents };
  const rng = mulberry32(seed >>> 0);
  for (let parent = 0; parent < COLLECTION.parents; parent++) {
    const family = parent % 4;
    let x = drawStart(g, rng);
    let intact = true;
    for (let t = 0; t < ROUTE_HORIZON; t++) {
      const { command, label } = recorderAction(family, control, x, t, rng);
      const step = taskStep(x, t, command, g);
      counts.recorded++;
      const touchesDanger = edgeTouches(x, step.next, g.danger);
      const crossesCut = crossesSlab(x, step.next, g);
      const drop = (condition === "honeypot" && touchesDanger) || (condition === "bottleneck" && crossesCut);
      if (drop) {
        counts.removed++;
        if (touchesDanger) counts.removedDanger++;
        if (crossesCut) counts.removedSlab++;
        intact = false;
      } else {
        const s = proj.cellOf(x);
        const s2 = step.danger ? dangerCell : step.goal ? goalCell : proj.cellOf(step.next);
        // The horizon is not part of the grid state, so only goal and danger end a grid episode.
        transitions.push({ s, a: label, r: Math.round(step.reward * 1e6) / 1e6, s2, done: step.goal || step.danger });
      }
      if (step.goal && intact) counts.intactSuccesses++;
      if (step.terminal) break;
      x = step.next;
    }
    if (parent % 128 === 127) onProgress?.(10 + Math.round(((parent + 1) / COLLECTION.parents) * 55), `Recording episode ${parent + 1}/${COLLECTION.parents} (family ${family})`);
  }
  onProgress?.(68, "Simulating the reference model (every action, real plant)…");
  const model = referenceModel(g, proj, control, seed, 1600, (e) => onProgress?.(68 + Math.round((e / 1600) * 28), `Reference simulation ${e}/1600`));
  const terminals = new Map<number, "goal" | "hazard">([
    [goalCell, "goal"],
    [dangerCell, "hazard"],
  ]);
  const env: Env = {
    id: ROUTE_IDS[condition],
    name: NAMES[condition],
    tag: `13-dim → route × danger · ${proj.width}×${proj.height}`,
    description: `${DESCRIPTIONS[condition]} This sample: ${counts.recorded.toLocaleString("en-US")} edges from ${COLLECTION.parents.toLocaleString("en-US")} episodes${counts.removed ? `, ${counts.removed.toLocaleString("en-US")} removed` : ""}; ${counts.intactSuccesses.toLocaleString("en-US")} episodes reach the goal on edges that all survive. Ground truth: a separate reference simulation of the real plant that tries every action.`,
    width: proj.width,
    height: proj.height,
    axisX: "progress along the route (A → goal)",
    axisY: "displacement toward the danger (top = toward it)",
    actionNames: ROUTE_ACTIONS,
    start: proj.cellOf(g.A),
    blocked: new Set(),
    terminals,
    groundTruth: "empirical",
    landmarks: [
      { cell: proj.cellOf(g.A), label: "A", hint: "Start A (t0)" },
      { cell: proj.cellOf(g.B), label: "B", hint: "B: the route at t25, where the excursion starts" },
      { cell: proj.cellOf(g.C), label: "C", hint: "C: the route at t30, past the slab" },
    ],
    model,
    step: () => ({ s2: 0, r: 0, done: true }),
  };
  env.step = stepFromModel(model, terminals);
  onProgress?.(98, "Building the tabular dataset…");
  return { dataset: buildDataset(env, transitions, COLLECTION.parents), counts, geometryDeviation: deviation };
}
