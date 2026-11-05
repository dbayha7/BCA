import { test, expect } from "bun:test";
import {
  auditEpisode,
  auditThruster,
  cloneLimits,
  DEFAULT_SAFETY_LIMITS,
  measureStress,
  N_CHANNELS,
  shieldBounds,
  shieldCommand,
  STRESS_CHANNELS,
  summarizeAudit,
  VIOLATION_BIT,
  VIOLATION_TYPES,
  type SafetyLimits,
} from "../../src/lib/safety";
import { benchmarkThruster, KNEE, SPEC_PHYSICS, type PhysicsOverrides } from "../../src/lib/thrusterchain";

const ROUGH: PhysicsOverrides = { ...SPEC_PHYSICS, knee: 0.6, slope: 2.5, currentScale: 1.8, waveScale: 2, stallPenalty: 1.5, cmdWeight: 0.5 };
// A fixed, state-dependent tabular policy over the 99 (p1, v1) cells that uses every action.
const policy = (cell: number) => (cell * 7 + 3) % 4;

test("unshielded audit returns equal benchmarkThruster returns (coast / behavior / policy, spec and perturbed physics)", () => {
  for (const ph of [SPEC_PHYSICS, ROUGH]) {
    for (const mode of [{ mode: "coast" as const }, { mode: "behavior" as const }, { mode: "policy" as const, policy }]) {
      const bench = benchmarkThruster(mode, 24, ph);
      const audit = auditThruster(mode, 24, ph, DEFAULT_SAFETY_LIMITS);
      expect(audit.returns).toEqual(bench.returns);
      expect(audit.meanReturn).toBe(bench.meanReturn);
      expect(audit.stdReturn).toBeCloseTo(bench.stdReturn, 12);
    }
  }
  // Custom base seed is honoured the same way.
  expect(auditThruster({ mode: "behavior" }, 5, SPEC_PHYSICS, DEFAULT_SAFETY_LIMITS, { baseSeed: 77 }).returns).toEqual(benchmarkThruster({ mode: "behavior" }, 5, SPEC_PHYSICS, 77).returns);
});

test("audit accounting is internally consistent", () => {
  const a = auditThruster({ mode: "behavior" }, 30, SPEC_PHYSICS, DEFAULT_SAFETY_LIMITS);
  expect(a.episodes).toBe(30);
  expect(a.rateByT.length).toBe(100);
  for (const ep of a.perEpisode) {
    let any = 0;
    for (let t = 0; t < 100; t++) if (ep.mask[t]) any++;
    expect(ep.anyViolations).toBe(any);
    for (const ty of VIOLATION_TYPES) {
      let c = 0;
      let first = -1;
      for (let t = 0; t < 100; t++)
        if (ep.mask[t] & VIOLATION_BIT[ty]) {
          c++;
          if (first < 0) first = t;
        }
      expect(ep.violations[ty]).toBe(c);
      expect(ep.firstByType[ty]).toBe(first);
      expect(ep.violations[ty]).toBeLessThanOrEqual(ep.anyViolations);
    }
    expect(ep.peakStress.length).toBe(N_CHANNELS);
  }
  // Mean of the per-timestep rate equals the step violation rate.
  const meanRate = a.rateByT.reduce((s, v) => s + v, 0) / a.rateByT.length;
  expect(meanRate).toBeCloseTo(a.stepViolationRate, 12);
  for (const ty of VIOLATION_TYPES) {
    const m = a.rateByTypeT[ty].reduce((s, v) => s + v, 0) / 100;
    expect(m).toBeCloseTo(a.stepViolationRateByType[ty], 12);
  }
  // The random recorder draws thrusters 2-4 uniformly in [-1, 1]: it trips the 0.9 rating and the knee.
  expect(a.stepViolationRateByType.command).toBeGreaterThan(0.1);
  expect(a.stepViolationRateByType.stall).toBeGreaterThan(a.stepViolationRateByType.command);
  // Summarizing in chunks gives the same answer as one call.
  const chunked = summarizeAudit([...a.perEpisode.slice(0, 10), ...a.perEpisode.slice(10)], { horizon: 100, baseSeed: 1000 });
  expect(chunked.rateByT).toEqual(a.rateByT);
  expect(chunked.peakStress).toEqual(a.peakStress);
});

test("coast never violates command or stall limits; its state violations match a direct check", () => {
  const a = auditThruster({ mode: "coast" }, 20, SPEC_PHYSICS, DEFAULT_SAFETY_LIMITS);
  expect(a.stepViolationRateByType.command).toBe(0);
  expect(a.stepViolationRateByType.stall).toBe(0);
  // Energy limit of +Infinity and huge state limits => no violations at all.
  const loose: SafetyLimits = { ...cloneLimits(DEFAULT_SAFETY_LIMITS), maxPosition: [1e9, 1e9, 1e9, 1e9, 1e9], maxVelocity: [1e9, 1e9, 1e9, 1e9, 1e9], maxEnergy: Infinity };
  const b = auditThruster({ mode: "coast" }, 20, SPEC_PHYSICS, loose);
  expect(b.stepViolationRate).toBe(0);
  expect(b.meanFirstViolation).toBeNull();
  expect(b.returns).toEqual(a.returns); // limits never change the physics
});

test("the shield removes command and stall violations and stays paired with the unshielded run", () => {
  const raw = auditThruster({ mode: "behavior" }, 20, SPEC_PHYSICS, DEFAULT_SAFETY_LIMITS);
  const sh = auditThruster({ mode: "behavior" }, 20, SPEC_PHYSICS, DEFAULT_SAFETY_LIMITS, { shield: true });
  expect(sh.shielded).toBe(true);
  expect(sh.stepViolationRateByType.command).toBe(0);
  expect(sh.stepViolationRateByType.stall).toBe(0);
  expect(sh.shieldInterventionRate).toBeGreaterThan(0);
  // Peak command stress never exceeds 1 under the shield.
  for (let c = 0; c < 5; c++) expect(sh.peakStress[c]).toBeLessThanOrEqual(1 + 1e-12);
  expect(raw.peakStress[4]).toBeGreaterThan(1);
  // t = 0 state is the reset state: same for both runs, so the same position/velocity/energy flags at t = 0.
  const stateBits = VIOLATION_BIT.position | VIOLATION_BIT.velocity | VIOLATION_BIT.energy;
  raw.perEpisode.forEach((ep, i) => expect(ep.mask[0] & stateBits).toBe(sh.perEpisode[i].mask[0] & stateBits));
  // Policy Burst (0.95) is clipped to the knee, so a policy that always bursts no longer stalls.
  const burst = auditEpisode({ mode: "policy", policy: () => 0 }, 0, SPEC_PHYSICS, DEFAULT_SAFETY_LIMITS, { shield: true });
  expect(burst.violations.stall).toBe(0);
  expect(burst.violations.command).toBe(0);
  expect(burst.shieldInterventions).toBe(100);
  const burstRaw = auditEpisode({ mode: "policy", policy: () => 0 }, 0, SPEC_PHYSICS, DEFAULT_SAFETY_LIMITS);
  expect(burstRaw.violations.stall).toBe(100);
  expect(burstRaw.violations.command).toBe(100);
});

test("measureStress and shieldCommand", () => {
  const x = new Float64Array(13);
  x[0] = 3; // p1 = 3 > 2
  x[3] = -1.5; // v2 exactly at its limit: allowed
  const { stress, mask } = measureStress(x, [0.95, 0, -0.5, 0], DEFAULT_SAFETY_LIMITS, KNEE);
  expect(STRESS_CHANNELS.length).toBe(N_CHANNELS);
  expect(stress[0]).toBeCloseTo(0.95 / 0.9, 12);
  expect(stress[4]).toBeCloseTo(0.95 / KNEE, 12);
  expect(stress[5]).toBeCloseTo(1.5, 12); // p1
  expect(stress[11]).toBeCloseTo(1, 12); // v2
  expect(stress[15]).toBeCloseTo((9 + 2.25) / 8, 12);
  expect(mask & VIOLATION_BIT.command).toBeTruthy();
  expect(mask & VIOLATION_BIT.stall).toBeTruthy();
  expect(mask & VIOLATION_BIT.position).toBeTruthy();
  expect(mask & VIOLATION_BIT.velocity).toBe(0);
  expect(mask & VIOLATION_BIT.energy).toBeTruthy();
  const noStall = { ...cloneLimits(DEFAULT_SAFETY_LIMITS), stallIsViolation: false };
  expect(measureStress(x, [0.85, 0, 0, 0], noStall).mask & VIOLATION_BIT.stall).toBe(0);
  expect(shieldBounds(DEFAULT_SAFETY_LIMITS)).toEqual([KNEE, KNEE, KNEE, KNEE]);
  expect(shieldBounds(noStall)).toEqual([0.9, 0.9, 0.9, 0.9]);
  const s = shieldCommand([0.95, -1, 0.3, 0], noStall);
  expect(s.changed).toBe(true);
  expect(s.cmd).toEqual([0.9, -0.9, 0.3, 0]);
  const same = [0.1, 0, 0, 0];
  expect(shieldCommand(same, noStall)).toEqual({ cmd: same, changed: false });
});
