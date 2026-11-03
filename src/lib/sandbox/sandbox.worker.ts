/// <reference lib="webworker" />
// Sandbox worker: the ONLY place where user-written reward/agent code is compiled and run.
// One worker per job; the main thread terminates it on completion, on timeout, or on cancel.
//
// Order of operations (each job):
//   1. module load: capture private references (postMessage, Function), register the message
//      listener, then remove network/storage/messaging globals (harden.ts) and report "ready";
//   2. re-run the static guard on the code (the main thread already did; this is belt and braces);
//   3. compile the code with `new Function` in strict mode — compiling does not execute it;
//   4. lock down runtime compilation and freeze built-ins, THEN run the user's top level;
//   5. run the job with trusted code calling the user function and validating every return value.
// This is defense in depth for a single-user browser app, not a hardened sandbox: see harden.ts.

import { runUserGridAgent, runUserThrusterAgent, GRID_HORIZON, MAX_EPISODES, type GridActFn, type ThrusterActFn } from "./agent";
import { checkUserCode } from "./guard";
import { freezeIntrinsics, lockDynamicCode, neuterGlobals } from "./harden";
import { UserCodeFailure, describeValue, type SandboxErrorKind, type SandboxJob, type SandboxRequest, type SandboxResponse } from "./protocol";
import { relabelRows, type RewardFn } from "./reward";

const scope = self as unknown as DedicatedWorkerGlobalScope;
const post = scope.postMessage.bind(scope) as (m: SandboxResponse, transfer?: Transferable[]) => void;
const RealFunction = Function;
const perf = performance;

/** Position of the first stack frame that belongs to code compiled by `new Function`. */
function compiledFrame(stack: string): { line: number; column: number } | null {
  const m = /<anonymous>:(\d+):(\d+)/.exec(stack) ?? /> Function:(\d+):(\d+)/.exec(stack);
  return m ? { line: Number(m[1]), column: Number(m[2]) } : null;
}

// Engines prepend a different header to `new Function` bodies; calibrate once with a probe whose
// `throw` sits on user line 1 (body line 2, right after "use strict").
let lineOffset: number | null = null;
try {
  RealFunction('"use strict";\nthrow new Error("probe");')();
} catch (err) {
  const pos = err instanceof Error && typeof err.stack === "string" ? compiledFrame(err.stack) : null;
  lineOffset = pos ? pos.line - 1 : null;
}

function userPosition(err: unknown): { line: number; column: number } | null {
  try {
    if (lineOffset === null || !(err instanceof Error) || typeof err.stack !== "string") return null;
    const pos = compiledFrame(err.stack);
    if (!pos) return null;
    const line = pos.line - lineOffset;
    return line >= 1 ? { line, column: pos.column } : null;
  } catch {
    return null;
  }
}

function errorText(err: unknown): string {
  try {
    if (err instanceof Error) return `${String(err.name)}: ${String(err.message)}`;
    return `threw ${describeValue(err)}`;
  } catch {
    return "threw an exception";
  }
}

function fail(id: number, kind: SandboxErrorKind, message: string, err?: unknown) {
  const pos = err === undefined ? null : userPosition(err);
  post({ type: "error", id, kind, message: pos ? `Line ${pos.line}: ${message}` : message, line: pos?.line, column: pos?.column });
}

const clampInt = (v: unknown, lo: number, hi: number, dflt: number) => (typeof v === "number" && Number.isFinite(v) ? Math.max(lo, Math.min(hi, Math.floor(v))) : dflt);

function runJob(id: number, job: SandboxJob) {
  const t0 = perf.now();
  const fnName = job.kind === "reward" ? "reward" : "act";
  const check = checkUserCode(job.code, { requireFunction: fnName });
  if (!check.ok) {
    post({ type: "error", id, kind: "guard", message: check.message, line: check.line, column: check.column });
    return;
  }

  let factory: () => unknown;
  try {
    factory = RealFunction(`"use strict";\n${job.code}\n;return typeof ${fnName} === "function" ? ${fnName} : undefined;`) as () => unknown;
  } catch (err) {
    fail(id, "syntax", errorText(err));
    return;
  }

  // No user code has executed yet. Lock down before running its top level.
  lockDynamicCode(scope as unknown as Record<string, unknown>);
  freezeIntrinsics();

  let fn: unknown;
  try {
    fn = factory();
  } catch (err) {
    fail(id, "runtime", `top-level code threw ${errorText(err)}`, err);
    return;
  }
  if (typeof fn !== "function") {
    post({ type: "error", id, kind: "output", message: `Define a function named \`${fnName}\`.` });
    return;
  }

  try {
    if (job.kind === "reward") {
      const rewards = relabelRows(job.rows, job.original, job.info, fn as RewardFn);
      post({ type: "result", id, result: { kind: "reward", rewards }, elapsedMs: perf.now() - t0 }, [rewards.buffer]);
    } else if (job.kind === "agent-grid") {
      const episodes = clampInt(job.episodes, 1, MAX_EPISODES, 1);
      const horizon = clampInt(job.horizon, 1, 1000, GRID_HORIZON);
      const actions = runUserGridAgent(job.envId, fn as GridActFn, episodes, horizon);
      post({ type: "result", id, result: { kind: "agent-grid", actions }, elapsedMs: perf.now() - t0 });
    } else {
      const episodes = clampInt(job.episodes, 1, MAX_EPISODES, 1);
      const commands = runUserThrusterAgent(fn as ThrusterActFn, episodes);
      post({ type: "result", id, result: { kind: "agent-thruster", commands }, elapsedMs: perf.now() - t0 });
    }
  } catch (err) {
    if (err instanceof UserCodeFailure) {
      if (err.kind === "runtime") fail(id, "runtime", `${err.message}: ${errorText(err.original)}`, err.original);
      else post({ type: "error", id, kind: "output", message: err.message });
    } else {
      fail(id, "runtime", errorText(err), err);
    }
  }
}

let used = false;
scope.addEventListener("message", (ev: MessageEvent<SandboxRequest>) => {
  const msg = ev.data;
  if (!msg || msg.type !== "run" || typeof msg.id !== "number") return;
  if (used) {
    post({ type: "error", id: msg.id, kind: "worker", message: "This sandbox worker already ran a job." });
    return;
  }
  used = true;
  runJob(msg.id, msg.job);
});

// Remove network, storage and messaging globals before any job (and therefore any user code) arrives.
const hardening = neuterGlobals(scope as unknown as Record<string, unknown>);
post({ type: "ready", hardening });
