import { expect, test } from "bun:test";
import { checkAlerts, DEFAULT_ALERT_RULES, EMPTY_MEMORY, type AlertMemory } from "../../src/lib/alerts";
import { buildModelCard } from "../../src/lib/model-card";
import type { Metrics } from "../../src/lib/rl";
import type { ExperimentRecord } from "../../src/lib/experiments";

const m = (iter: number, trueQ: number): Metrics => ({ iter, updates: iter * 20, estQ: trueQ, trueQ, loss: 1, tdLoss: 1, cqlGap: 0, oodRate: 0, divergence: 1 });

function feed(values: number[], rules = DEFAULT_ALERT_RULES) {
  let memory: AlertMemory = EMPTY_MEMORY;
  const fired: { at: number; kind: string; pause: boolean }[] = [];
  const hist: Metrics[] = [];
  values.forEach((v, i) => {
    hist.push(m(i + 1, v));
    const r = checkAlerts(hist, rules, memory);
    memory = r.memory;
    for (const a of r.alerts) fired.push({ at: i + 1, kind: a.kind, pause: a.pause });
  });
  return fired;
}

test("plateau fires once, plateauSteps after the last real improvement", () => {
  const values = [...Array.from({ length: 10 }, (_, i) => i), ...Array(60).fill(9.01)];
  const fired = feed(values, { ...DEFAULT_ALERT_RULES, plateauSteps: 40 });
  expect(fired).toEqual([{ at: 50, kind: "plateau", pause: false }]);
});

test("a later improvement re-arms the plateau alert; auto-pause is carried on the alert", () => {
  const values = [0, ...Array(5).fill(1), 3, ...Array(5).fill(3)];
  const fired = feed(values, { ...DEFAULT_ALERT_RULES, plateauSteps: 4, autoPause: true });
  expect(fired.map((f) => [f.at, f.kind, f.pause])).toEqual([
    [6, "plateau", true],
    [11, "plateau", true],
  ]);
});

test("target fires once; regression fires once per new best and re-arms after recovery", () => {
  const rules = { ...DEFAULT_ALERT_RULES, plateauSteps: 0, targetReturn: 5, regressionDrop: 2 };
  const fired = feed([1, 5, 6, 3.5, 3, 6, 6.5, 4], rules);
  expect(fired.map((f) => [f.at, f.kind])).toEqual([
    [2, "target"],
    [4, "regression"],
    [8, "regression"],
  ]);
});

test("model card states data, evaluation basis and limitations", () => {
  const run = {
    id: "x",
    name: "card run",
    createdAt: 0,
    updatedAt: 0,
    datasetId: "gridworld-random",
    datasetName: "Gridworld-Random",
    seed: 1234,
    rngState: 1,
    hyper: { alpha: 1, gamma: 0.95, lr: 0.3, batch: 64 },
    hyperLog: [{ atUpdates: 0, hyper: { alpha: 1, gamma: 0.95, lr: 0.3, batch: 64 } }],
    history: Array.from({ length: 10 }, (_, i) => m(i + 1, i)),
    q: new Float64Array(400),
    optimal: 3,
    status: "paused",
    tags: [],
    notes: "",
    anomalies: 0,
    guardEvents: [],
  } satisfies ExperimentRecord;
  const card = buildModelCard({
    run,
    dataset: { name: "Gridworld-Random", description: "maze", tag: "10×10", transitions: 8148, episodes: 240, groundTruth: "empirical" },
    updatesPerStep: 20,
    anomalies: [],
  });
  expect(card).toContain("# Model card: card run");
  expect(card).toContain("100 states × 4 actions");
  expect(card).toContain("empirical MDP");
  expect(card).toContain("Do not deploy to hardware");
  expect(card).not.toContain("\n\n\n");
});
