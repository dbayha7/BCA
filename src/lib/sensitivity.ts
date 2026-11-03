// Win drivers / automated sensitivity analysis for a hyperparameter sweep.
// Pure and rule-based (no network, no model): main effects per parameter level,
// Spearman rank correlations, overestimation and OOD summaries, an lr × α
// heatmap of true return, and a short plain-language summary.
//
// Which runs count: finished runs and pruned runs. A pruned run enters with the
// true return at the checkpoint where it was pruned — the value that got it
// pruned — so poor regions of the grid are not silently dropped. Stopped,
// skipped and unfinished runs are left out (their numbers are not comparable).

export interface SensitivityRun {
  id: number;
  lr: number;
  alpha: number;
  status: string;
  trueQ: number;
  gap: number;
  oodRate: number;
}

export type ParamKey = "lr" | "alpha";

export const PARAM_LABEL: Record<ParamKey, string> = { lr: "Learning rate", alpha: "α (conservative weight)" };
const SHORT: Record<ParamKey, string> = { lr: "learning rate", alpha: "α" };

export interface LevelStat {
  value: number;
  n: number;
  meanTrueQ: number;
  minTrueQ: number;
  maxTrueQ: number;
  meanGap: number;
  meanOod: number;
}

export type EffectShape = "increasing" | "decreasing" | "peak" | "valley" | "flat" | "mixed";

export interface Correlations {
  trueQ: number | null;
  gap: number | null;
  ood: number | null;
}

export interface ParamSensitivity {
  key: ParamKey;
  label: string;
  levels: LevelStat[];
  /** Max − min of the level means: how far this parameter alone moves mean true return. */
  effectRange: number;
  bestLevel: number;
  shape: EffectShape;
  /** Spearman ρ of the parameter with true return, overestimation gap and OOD rate. */
  rho: Correlations;
}

export interface HeatCell {
  id: number;
  lr: number;
  alpha: number;
  trueQ: number;
  status: string;
}

export interface Heatmap {
  lrs: number[];
  alphas: number[];
  /** cells[alphaIndex][lrIndex], null where no comparable run exists. */
  cells: (HeatCell | null)[][];
  min: number;
  max: number;
}

export interface Insight {
  tone: "good" | "warn" | "info";
  text: string;
}

export interface SensitivityReport {
  n: number;
  pruned: number;
  excluded: number;
  params: ParamSensitivity[];
  heatmap: Heatmap;
  best: SensitivityRun;
  spread: number;
  overall: {
    meanGap: number;
    meanOod: number;
    /** Spearman ρ between a run's overestimation gap and its true return. */
    rhoGapReturn: number | null;
    /** Spearman ρ between a run's OOD rate and its true return. */
    rhoOodReturn: number | null;
    /** Runs within 10% of the spread from the best true return. */
    nearBest: number;
  };
  summary: Insight[];
}

/* ---------------- Statistics ---------------- */

/** 1-based ranks, ties get the average of the ranks they span. */
export function averageRanks(xs: number[]): number[] {
  const idx = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
  const ranks = new Array<number>(xs.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && xs[idx[j + 1]] === xs[idx[i]]) j++;
    const r = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[idx[k]] = r;
    i = j + 1;
  }
  return ranks;
}

export function pearson(x: number[], y: number[]): number | null {
  const n = x.length;
  if (n < 2 || y.length !== n) return null;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += x[i];
    my += y[i];
  }
  mx /= n;
  my /= n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  if (sxx <= 1e-15 || syy <= 1e-15) return null;
  return Math.max(-1, Math.min(1, sxy / Math.sqrt(sxx * syy)));
}

/** Spearman rank correlation (null with fewer than 3 points or a constant input). */
export function spearman(x: number[], y: number[]): number | null {
  if (x.length < 3 || x.length !== y.length) return null;
  if (x.some((v) => !Number.isFinite(v)) || y.some((v) => !Number.isFinite(v))) return null;
  return pearson(averageRanks(x), averageRanks(y));
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

export function strength(rho: number | null): "strong" | "moderate" | "weak" | "n/a" {
  if (rho == null) return "n/a";
  const a = Math.abs(rho);
  return a >= 0.6 ? "strong" : a >= 0.3 ? "moderate" : "weak";
}

/* ---------------- Analysis ---------------- */

function levelStats(runs: SensitivityRun[], key: ParamKey): LevelStat[] {
  const values = [...new Set(runs.map((r) => r[key]))].sort((a, b) => a - b);
  return values.map((value) => {
    const rs = runs.filter((r) => r[key] === value);
    const q = rs.map((r) => r.trueQ);
    return {
      value,
      n: rs.length,
      meanTrueQ: mean(q),
      minTrueQ: Math.min(...q),
      maxTrueQ: Math.max(...q),
      meanGap: mean(rs.map((r) => r.gap)),
      meanOod: mean(rs.map((r) => r.oodRate)),
    };
  });
}

function shapeOf(levels: LevelStat[], tol: number): EffectShape {
  const m = levels.map((l) => l.meanTrueQ);
  if (m.length < 2) return "flat";
  const range = Math.max(...m) - Math.min(...m);
  if (range <= tol) return "flat";
  let up = true;
  let down = true;
  for (let i = 1; i < m.length; i++) {
    if (m[i] < m[i - 1] - tol) up = false;
    if (m[i] > m[i - 1] + tol) down = false;
  }
  if (up && !down) return "increasing";
  if (down && !up) return "decreasing";
  const iMax = m.indexOf(Math.max(...m));
  const iMin = m.indexOf(Math.min(...m));
  const last = m.length - 1;
  if (m.length >= 3 && iMax > 0 && iMax < last && m[0] < m[iMax] - tol && m[last] < m[iMax] - tol) return "peak";
  if (m.length >= 3 && iMin > 0 && iMin < last && m[0] > m[iMin] + tol && m[last] > m[iMin] + tol) return "valley";
  return "mixed";
}

const fmtV = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
const fmtRho = (r: number | null) => (r == null ? "n/a" : `${r >= 0 ? "+" : "−"}${Math.abs(r).toFixed(2)}`);
const fmtLevel = (key: ParamKey, v: number) => (key === "lr" ? `lr ${v}` : `α=${v}`);
const pct = (v: number) => `${(v * 100).toFixed(0)}%`;

/** Analyse comparable (finished or pruned) runs. Returns null with fewer than 2. */
export function analyzeSweep(allRuns: SensitivityRun[]): SensitivityReport | null {
  const runs = allRuns.filter((r) => (r.status === "done" || r.status === "pruned") && Number.isFinite(r.trueQ));
  if (runs.length < 2) return null;
  const pruned = runs.filter((r) => r.status === "pruned").length;
  const tq = runs.map((r) => r.trueQ);
  const gaps = runs.map((r) => r.gap);
  const oods = runs.map((r) => r.oodRate);
  const spread = Math.max(...tq) - Math.min(...tq);
  const tol = Math.max(1e-9, spread * 0.05);

  const params: ParamSensitivity[] = (["lr", "alpha"] as ParamKey[]).map((key) => {
    const levels = levelStats(runs, key);
    const means = levels.map((l) => l.meanTrueQ);
    const xs = runs.map((r) => r[key]);
    return {
      key,
      label: PARAM_LABEL[key],
      levels,
      effectRange: Math.max(...means) - Math.min(...means),
      bestLevel: levels[means.indexOf(Math.max(...means))].value,
      shape: shapeOf(levels, tol),
      rho: { trueQ: spearman(xs, tq), gap: spearman(xs, gaps), ood: spearman(xs, oods) },
    };
  });

  // Heatmap over the full grid shape (from all runs), filled with comparable runs.
  const lrs = [...new Set(allRuns.map((r) => r.lr))].sort((a, b) => a - b);
  const alphas = [...new Set(allRuns.map((r) => r.alpha))].sort((a, b) => a - b);
  const cells: (HeatCell | null)[][] = alphas.map(() => lrs.map(() => null));
  for (const r of runs) {
    cells[alphas.indexOf(r.alpha)][lrs.indexOf(r.lr)] = { id: r.id, lr: r.lr, alpha: r.alpha, trueQ: r.trueQ, status: r.status };
  }

  const best = runs.reduce((a, b) => (b.trueQ > a.trueQ || (b.trueQ === a.trueQ && b.id < a.id) ? b : a));
  const nearBest = runs.filter((r) => r.trueQ >= best.trueQ - 0.1 * spread).length;
  const overall = {
    meanGap: mean(gaps),
    meanOod: mean(oods),
    rhoGapReturn: spearman(gaps, tq),
    rhoOodReturn: spearman(oods, tq),
    nearBest,
  };

  const report: SensitivityReport = {
    n: runs.length,
    pruned,
    excluded: allRuns.length - runs.length,
    params,
    heatmap: { lrs, alphas, cells, min: Math.min(...tq), max: Math.max(...tq) },
    best,
    spread,
    overall,
    summary: [],
  };
  report.summary = summarize(report, runs);
  return report;
}

/* ---------------- Plain-language summary ---------------- */

function summarize(rep: SensitivityReport, runs: SensitivityRun[]): Insight[] {
  const out: Insight[] = [];
  const { best, spread, n } = rep;
  const [lrP, aP] = rep.params;

  if (n < 4) out.push({ tone: "warn", text: `Only ${n} comparable runs — treat these drivers as tentative and widen the sweep.` });

  const plateau = rep.overall.nearBest;
  out.push({
    tone: "good",
    text:
      `Best run: lr ${best.lr}, α ${best.alpha} → true return ${fmtV(best.trueQ)}. ` +
      (spread < 1e-6
        ? "Every configuration scores the same."
        : plateau >= Math.max(3, Math.ceil(n / 4))
          ? `${plateau} of ${n} runs land within 10% of the spread from it — a broad plateau, so the choice is robust.`
          : `Only ${plateau} of ${n} runs land within 10% of the spread from it — a narrow optimum, so retune when the data changes.`),
  });

  if (spread < 0.05 * Math.max(1, Math.abs(best.trueQ)) || spread < 1e-6) {
    out.push({ tone: "info", text: `All configurations score within ${fmtV(spread)} of each other — neither parameter matters much in this range; widen the ranges to find a real driver.` });
  } else {
    const [big, small] = lrP.effectRange >= aP.effectRange ? [lrP, aP] : [aP, lrP];
    const ratio = big.effectRange / Math.max(1e-9, small.effectRange);
    if (ratio >= 1.5) {
      out.push({
        tone: "info",
        text: `${cap(SHORT[big.key])} is the main win driver: its level means span ${fmtV(big.effectRange)} in true return, versus ${fmtV(small.effectRange)} for ${SHORT[small.key]}.`,
      });
    } else {
      out.push({
        tone: "info",
        text: `Both parameters matter about equally: level means span ${fmtV(lrP.effectRange)} for learning rate and ${fmtV(aP.effectRange)} for α.`,
      });
    }
    for (const p of [big, small]) {
      const s = describeShape(p);
      if (s) out.push(s);
    }
  }

  // Interaction: does the best learning rate change with α?
  if (lrP.levels.length >= 2 && aP.levels.length >= 2) {
    const bestLrAt = aP.levels
      .map((lvl) => {
        const rs = runs.filter((r) => r.alpha === lvl.value);
        if (rs.length < 2) return null;
        const b = rs.reduce((a, c) => (c.trueQ > a.trueQ ? c : a));
        return { alpha: lvl.value, lr: b.lr };
      })
      .filter((x): x is { alpha: number; lr: number } => x != null);
    if (bestLrAt.length >= 2 && new Set(bestLrAt.map((b) => b.lr)).size > 1) {
      const first = bestLrAt[0];
      const last = bestLrAt[bestLrAt.length - 1];
      out.push({
        tone: "warn",
        text:
          first.lr !== last.lr
            ? `The best learning rate depends on α (interaction): lr ${first.lr} wins at α=${first.alpha}, lr ${last.lr} at α=${last.alpha} — tune them together, not one at a time.`
            : `The best learning rate shifts between α levels (interaction) — tune the two parameters together.`,
      });
    }
  }

  // Overestimation.
  const aGap = aP.rho.gap;
  if (aGap != null && aGap <= -0.3) {
    const lo = aP.levels[0];
    const hi = aP.levels[aP.levels.length - 1];
    out.push({
      tone: "good",
      text: `Raising α shrinks overestimation (ρ ${fmtRho(aGap)}): the mean gap between estimated and true value goes from ${signed(lo.meanGap)} at α=${lo.value} to ${signed(hi.meanGap)} at α=${hi.value}.`,
    });
  } else if (aGap != null && aGap >= 0.3) {
    out.push({ tone: "warn", text: `Overestimation grows with α here (ρ ${fmtRho(aGap)}) — unusual for CQL; check that α is not so large that the penalty dominates the TD error.` });
  }
  const gr = rep.overall.rhoGapReturn;
  if (gr != null && gr <= -0.3) {
    out.push({ tone: "warn", text: `Runs that overestimate more score lower true return (ρ ${fmtRho(gr)}), so ranking by the agent's own Q-estimate would pick the wrong configurations.` });
  }

  // Out-of-distribution actions.
  const aOod = aP.rho.ood;
  if (aOod != null && aOod <= -0.3) {
    const lo = aP.levels[0];
    const hi = aP.levels[aP.levels.length - 1];
    out.push({ tone: "good", text: `Higher α keeps the greedy policy on actions seen in the data: OOD rate falls from ${pct(lo.meanOod)} at α=${lo.value} to ${pct(hi.meanOod)} at α=${hi.value}.` });
  }
  const or = rep.overall.rhoOodReturn;
  if (or != null && or <= -0.3) {
    out.push({ tone: "warn", text: `Policies that pick more out-of-distribution actions score lower (ρ ${fmtRho(or)}).` });
  }

  if (rep.pruned > 0) {
    out.push({ tone: "info", text: `${rep.pruned} run${rep.pruned === 1 ? " was" : "s were"} pruned early and enter${rep.pruned === 1 ? "s" : ""} with the true return at the prune checkpoint, which can understate ${rep.pruned === 1 ? "it" : "them"} a little.` });
  }
  return out;
}

function describeShape(p: ParamSensitivity): Insight | null {
  const name = SHORT[p.key];
  const r = p.rho.trueQ;
  const st = strength(r);
  switch (p.shape) {
    case "increasing":
      return { tone: "info", text: `Higher ${name} tends to raise true return (Spearman ρ ${fmtRho(r)}, ${st}); best level ${fmtLevel(p.key, p.bestLevel)}.` };
    case "decreasing":
      return { tone: "info", text: `Lower ${name} tends to raise true return (Spearman ρ ${fmtRho(r)}, ${st}); best level ${fmtLevel(p.key, p.bestLevel)}.` };
    case "peak":
      return { tone: "info", text: `${cap(name)} has a sweet spot near ${fmtLevel(p.key, p.bestLevel)}: both lower and higher values do worse on average (ρ ${fmtRho(r)} misses this because it only measures monotone trends).` };
    case "valley":
      return { tone: "warn", text: `Middle values of ${name} do worst; the extremes score higher on average (best ${fmtLevel(p.key, p.bestLevel)}).` };
    case "flat":
      return { tone: "info", text: `${cap(name)} barely moves the mean true return in this range.` };
    default:
      return st === "strong" || st === "moderate"
        ? { tone: "info", text: `${cap(name)} has a ${st}, uneven effect (ρ ${fmtRho(r)}); best level ${fmtLevel(p.key, p.bestLevel)}.` }
        : { tone: "info", text: `${cap(name)} has no consistent direction (ρ ${fmtRho(r)}); best level ${fmtLevel(p.key, p.bestLevel)}.` };
  }
}

/** Capitalise ASCII words only — "α" must stay lower-case (its capital is the Latin-looking "Α"). */
const cap = (s: string) => (/^[a-z]/.test(s) ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const signed = (v: number) => `${v >= 0 ? "+" : "−"}${fmtV(Math.abs(v))}`;
