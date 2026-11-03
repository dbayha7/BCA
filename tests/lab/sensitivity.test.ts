import { test, expect } from "bun:test";
import { analyzeSweep, averageRanks, pearson, spearman, type SensitivityRun } from "../../src/lib/sensitivity";
import { buildRuns, buildSweepDataset, createInlineExecutor, DEFAULT_SWEEP, driveSweep, SweepScheduler } from "../../src/lib/sweep-core";

const LRS = [0.02, 0.05, 0.2, 0.5];
const ALPHAS = [0, 1, 2, 3];

/** Synthetic 4×4 grid where only α matters: trueQ = 2α, gap = 3 − α, OOD = 0.4 − 0.1α. */
function alphaOnlyGrid(): SensitivityRun[] {
  const runs: SensitivityRun[] = [];
  let id = 0;
  for (const alpha of ALPHAS)
    for (const lr of LRS) runs.push({ id: id++, lr, alpha, status: "done", trueQ: 2 * alpha, gap: 3 - alpha, oodRate: 0.4 - 0.1 * alpha });
  return runs;
}

test("average ranks handle ties", () => {
  expect(averageRanks([10, 20, 20, 30])).toEqual([1, 2.5, 2.5, 4]);
  expect(averageRanks([3, 1, 2])).toEqual([3, 1, 2]);
  expect(averageRanks([5, 5, 5])).toEqual([2, 2, 2]);
});

test("spearman: monotone transforms give ±1, constants and tiny samples give null", () => {
  const x = [1, 2, 3, 4, 5];
  expect(spearman(x, x.map((v) => v ** 3))).toBeCloseTo(1, 12);
  expect(spearman(x, x.map((v) => -Math.exp(v)))).toBeCloseTo(-1, 12);
  expect(spearman(x, [7, 7, 7, 7, 7])).toBeNull();
  expect(spearman([1, 2], [2, 1])).toBeNull();
  // Known value with ties: ranks x=[1,2,3,4], y=[1,2.5,2.5,4] → pearson of ranks.
  expect(spearman([1, 2, 3, 4], [10, 20, 20, 30])).toBeCloseTo(pearson([1, 2, 3, 4], [1, 2.5, 2.5, 4])!, 12);
  expect(pearson([1, 2, 3], [2, 4, 6])).toBeCloseTo(1, 12);
});

test("main effects, correlations and heatmap on a grid where only α matters", () => {
  const rep = analyzeSweep(alphaOnlyGrid())!;
  expect(rep).not.toBeNull();
  expect(rep.n).toBe(16);
  const [lr, alpha] = rep.params;
  expect(lr.key).toBe("lr");
  expect(alpha.key).toBe("alpha");

  // Main effect = mean true return per level.
  expect(alpha.levels.map((l) => l.meanTrueQ)).toEqual([0, 2, 4, 6]);
  expect(alpha.levels.every((l) => l.n === 4)).toBe(true);
  expect(lr.levels.map((l) => l.meanTrueQ)).toEqual([3, 3, 3, 3]);
  expect(alpha.effectRange).toBe(6);
  expect(lr.effectRange).toBe(0);
  expect(alpha.shape).toBe("increasing");
  expect(lr.shape).toBe("flat");
  expect(alpha.bestLevel).toBe(3);

  // Spearman: α perfectly ranks return (+), gap (−) and OOD (−); lr is uncorrelated with return.
  expect(alpha.rho.trueQ).toBeCloseTo(1, 12);
  expect(alpha.rho.gap).toBeCloseTo(-1, 12);
  expect(alpha.rho.ood).toBeCloseTo(-1, 12);
  expect(lr.rho.trueQ).toBeCloseTo(0, 12);

  // Heatmap rows = α levels, columns = lr levels.
  expect(rep.heatmap.alphas).toEqual(ALPHAS);
  expect(rep.heatmap.lrs).toEqual(LRS);
  expect(rep.heatmap.cells[3][0]!.trueQ).toBe(6);
  expect(rep.heatmap.cells[0][2]!.trueQ).toBe(0);
  expect(rep.heatmap.min).toBe(0);
  expect(rep.heatmap.max).toBe(6);

  // Overall: more overestimation ↔ lower return.
  expect(rep.overall.rhoGapReturn).toBeCloseTo(-1, 12);
  expect(rep.best.alpha).toBe(3);
  expect(rep.best.id).toBe(12); // ties broken by lowest run id

  const text = rep.summary.map((s) => s.text).join("\n");
  expect(text).toContain("α is the main win driver");
  expect(text).toContain("Raising α shrinks overestimation");
  expect(text).toContain("OOD rate falls");
  expect(text).toContain("Learning rate barely moves");
});

test("detects a learning-rate sweet spot and an lr × α interaction", () => {
  const runs: SensitivityRun[] = [];
  let id = 0;
  for (const alpha of [0, 3])
    for (const lr of LRS) {
      // Peak at lr 0.05 for α=0 and at lr 0.2 for α=3 (interaction); lr matters far more than α.
      const peak = alpha === 0 ? 0.05 : 0.2;
      const trueQ = 5 - 10 * Math.abs(Math.log(lr / peak)) + 0.1 * alpha;
      runs.push({ id: id++, lr, alpha, status: "done", trueQ, gap: 0.5, oodRate: 0.1 });
    }
  const rep = analyzeSweep(runs)!;
  const text = rep.summary.map((s) => s.text).join("\n");
  expect(rep.params[0].shape).toBe("peak");
  expect(text).toContain("Learning rate is the main win driver");
  expect(text).toContain("sweet spot");
  expect(text).toContain("depends on α (interaction): lr 0.05 wins at α=0, lr 0.2 at α=3");
});

test("only finished and pruned runs count; stopped, skipped and pending are excluded", () => {
  const runs = alphaOnlyGrid();
  runs[0].status = "pruned";
  runs[1].status = "stopped";
  runs[2].status = "skipped";
  runs[3].status = "pending";
  const rep = analyzeSweep(runs)!;
  expect(rep.n).toBe(13);
  expect(rep.excluded).toBe(3);
  expect(rep.pruned).toBe(1);
  expect(rep.heatmap.cells[0][1]).toBeNull();
  expect(rep.heatmap.cells[0][0]!.status).toBe("pruned");
  expect(rep.summary.some((s) => s.text.includes("1 run was pruned early"))).toBe(true);
  expect(analyzeSweep(runs.slice(0, 2))).toBeNull();
});

test("flat sweep is reported as having no real driver", () => {
  const runs = alphaOnlyGrid().map((r) => ({ ...r, trueQ: 4 + r.id * 1e-4 }));
  const rep = analyzeSweep(runs)!;
  expect(rep.summary.some((s) => s.text.includes("neither parameter matters much"))).toBe(true);
});

test("runs on a real sweep and produces a summary for every dataset", async () => {
  for (const datasetId of ["gridworld-random", "halfcheetah-medium", "thrusterchain-stall"] as const) {
    const settings = { ...DEFAULT_SWEEP, datasetId, lrSteps: 3, alphaSteps: 3, totalUpdates: 360 };
    const sch = new SweepScheduler(settings, { runs: buildRuns(settings) });
    await driveSweep(sch, createInlineExecutor((k) => buildSweepDataset(k)));
    const rep = analyzeSweep(sch.runs)!;
    expect(rep.n).toBe(9);
    expect(rep.summary.length).toBeGreaterThan(1);
    expect(rep.summary[0].text.length + rep.summary[1].text.length).toBeGreaterThan(20);
    for (const p of rep.params) for (const v of Object.values(p.rho)) if (v != null) expect(Math.abs(v)).toBeLessThanOrEqual(1);
  }
});
