// Main-thread client for the sandbox worker.
//
// - Runs the static guard first (layer 1) and never spawns a worker for rejected code.
// - Spawns a fresh module worker per job (layer 2 lives in the worker) and enforces a wall-clock
//   timeout from the main thread by TERMINATING the worker, so an infinite loop cannot block the
//   page or linger in the background.
// - Treats every reply as untrusted data: shapes and numbers are validated, and agent results are
//   replayed on the trusted simulators before anything is shown.
// User code NEVER runs on the main thread: if Web Workers are unavailable the job just fails.

import {
  GRID_HORIZON,
  playgroundGridEnv,
  replayGridActions,
  replayThrusterCommands,
  type GridEpisode,
  type ThrusterEpisode,
} from "./agent";
import { checkUserCode, locateSyntaxError } from "./guard";
import type { HardeningReport } from "./harden";
import type { GridPlaygroundEnv, SandboxErrorKind, SandboxJob, SandboxRequest, SandboxResponse, SandboxResultData } from "./protocol";
import type { RewardJobData } from "./reward";

export const REWARD_TIMEOUT_MS = 2000;
export const AGENT_TIMEOUT_MS = 3000;
const LOAD_TIMEOUT_MS = 10000;
const KINDS: SandboxErrorKind[] = ["guard", "syntax", "runtime", "output", "timeout", "worker", "aborted"];

export class SandboxError extends Error {
  kind: SandboxErrorKind;
  line?: number;
  column?: number;
  constructor(kind: SandboxErrorKind, message: string, line?: number, column?: number) {
    super(message);
    this.kind = kind;
    this.line = line;
    this.column = column;
  }
}

export interface SandboxOptions {
  timeoutMs?: number;
  signal?: AbortSignal;
}

interface RawRun {
  result: SandboxResultData;
  /** Time measured inside the worker (compile + run). */
  elapsedMs: number;
  /** Wall-clock time from posting the job to receiving the reply. */
  wallMs: number;
  hardening: HardeningReport;
}

let seq = 0;

const posInt = (v: unknown) => (typeof v === "number" && Number.isInteger(v) && v >= 1 ? v : undefined);
const strList = (v: unknown) => (Array.isArray(v) ? v.filter((x): x is string => typeof x === "string").slice(0, 64) : []);

function runSandboxJob(job: SandboxJob, opts: SandboxOptions, timeoutMs: number): Promise<RawRun> {
  const fnName = job.kind === "reward" ? "reward" : "act";
  const check = checkUserCode(job.code, { requireFunction: fnName });
  if (!check.ok) return Promise.reject(new SandboxError("guard", check.message, check.line, check.column));
  if (typeof window === "undefined" || typeof Worker === "undefined") {
    return Promise.reject(new SandboxError("worker", "Web Workers are unavailable, and user code is never run on the main thread."));
  }
  const { signal } = opts;
  return new Promise<RawRun>((resolve, reject) => {
    let worker: Worker;
    try {
      worker = new Worker(new URL("./sandbox.worker.ts", import.meta.url), { type: "module" });
    } catch {
      reject(new SandboxError("worker", "Could not start the sandbox worker."));
      return;
    }
    const id = ++seq;
    let hardening: HardeningReport | null = null;
    let started = 0;
    let settled = false;
    let timer: number;

    const finish = (err: SandboxError | null, value?: RawRun) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      worker.terminate();
      signal?.removeEventListener("abort", onAbort);
      if (err) reject(err);
      else resolve(value as RawRun);
    };
    const onAbort = () => finish(new SandboxError("aborted", "Cancelled."));

    timer = window.setTimeout(() => finish(new SandboxError("worker", `The sandbox worker did not start within ${LOAD_TIMEOUT_MS / 1000} s.`)), LOAD_TIMEOUT_MS);
    if (signal?.aborted) {
      onAbort();
      return;
    }
    signal?.addEventListener("abort", onAbort);

    worker.onmessage = (ev: MessageEvent<SandboxResponse>) => {
      const m = ev.data;
      if (!m || typeof m !== "object") return;
      if (m.type === "ready") {
        if (hardening) return;
        hardening = { removed: strList(m.hardening?.removed), failed: strList(m.hardening?.failed) };
        window.clearTimeout(timer);
        started = performance.now();
        // Wall-clock budget for compile + run, enforced here by terminating the worker.
        timer = window.setTimeout(
          () =>
            finish(
              new SandboxError(
                "timeout",
                `Stopped after ${(timeoutMs / 1000).toFixed(1)} s — the code did not finish in time (infinite loop?). The sandbox worker was terminated; the page stayed responsive.`,
              ),
            ),
          timeoutMs,
        );
        const req: SandboxRequest = { type: "run", id, job };
        worker.postMessage(req);
        return;
      }
      if (m.id !== id || !hardening) return;
      if (m.type === "result") {
        const elapsedMs = typeof m.elapsedMs === "number" && Number.isFinite(m.elapsedMs) ? m.elapsedMs : 0;
        finish(null, { result: m.result, elapsedMs, wallMs: performance.now() - started, hardening });
      } else if (m.type === "error") {
        const kind = KINDS.includes(m.kind) ? m.kind : "runtime";
        let message = typeof m.message === "string" ? m.message.slice(0, 600) : "The sandbox reported an error.";
        let line = posInt(m.line);
        let column = posInt(m.column);
        if (kind === "syntax" && line === undefined) {
          const loc = locateSyntaxError(job.code);
          if (loc) {
            line = loc.line;
            column = loc.column;
            message = `Line ${loc.line}: ${message} (${loc.hint})`;
          }
        }
        finish(new SandboxError(kind, message, line, column));
      }
    };
    worker.onerror = (ev) => {
      ev.preventDefault();
      finish(new SandboxError("worker", `The sandbox worker failed${ev.message ? `: ${ev.message}` : "."}`));
    };
    worker.onmessageerror = () => finish(new SandboxError("worker", "Could not read the sandbox worker's reply."));
  });
}

export interface RewardRun {
  rewards: Float64Array;
  elapsedMs: number;
  wallMs: number;
  hardening: HardeningReport;
}

/** Relabel every row of `data` with the user's `reward` function inside the sandbox. */
export async function runRewardSandbox(code: string, data: RewardJobData, opts: SandboxOptions = {}): Promise<RewardRun> {
  const run = await runSandboxJob({ kind: "reward", code, rows: data.rows, original: data.original, info: data.info }, opts, opts.timeoutMs ?? REWARD_TIMEOUT_MS);
  const res = run.result;
  if (res?.kind !== "reward" || !(res.rewards instanceof Float64Array) || res.rewards.length !== data.original.length) {
    throw new SandboxError("output", "The sandbox returned a malformed reward array.");
  }
  for (let i = 0; i < res.rewards.length; i++) {
    if (!Number.isFinite(res.rewards[i])) throw new SandboxError("output", `Reward #${i} is not a finite number.`);
  }
  return { rewards: res.rewards, elapsedMs: run.elapsedMs, wallMs: run.wallMs, hardening: run.hardening };
}

export interface AgentRun<E> {
  episodes: E[];
  elapsedMs: number;
  wallMs: number;
  hardening: HardeningReport;
}

/** Run the user's `act` on a grid env in the sandbox, then replay its actions on the trusted env. */
export async function runGridAgentSandbox(code: string, envId: GridPlaygroundEnv, episodes: number, opts: SandboxOptions = {}): Promise<AgentRun<GridEpisode>> {
  const run = await runSandboxJob({ kind: "agent-grid", code, envId, episodes, horizon: GRID_HORIZON }, opts, opts.timeoutMs ?? AGENT_TIMEOUT_MS);
  if (run.result?.kind !== "agent-grid") throw new SandboxError("output", "The sandbox returned an unexpected result.");
  let eps: GridEpisode[];
  try {
    eps = replayGridActions(playgroundGridEnv(envId), run.result.actions, episodes, GRID_HORIZON);
  } catch (e) {
    throw new SandboxError("output", `Rejected the sandbox's result: ${e instanceof Error ? e.message : "malformed data"}`);
  }
  return { episodes: eps, elapsedMs: run.elapsedMs, wallMs: run.wallMs, hardening: run.hardening };
}

/** Run the user's 4-dim controller in the sandbox, then re-simulate its commands with the exact physics. */
export async function runThrusterAgentSandbox(code: string, episodes: number, opts: SandboxOptions = {}): Promise<AgentRun<ThrusterEpisode>> {
  const run = await runSandboxJob({ kind: "agent-thruster", code, episodes }, opts, opts.timeoutMs ?? AGENT_TIMEOUT_MS);
  if (run.result?.kind !== "agent-thruster") throw new SandboxError("output", "The sandbox returned an unexpected result.");
  let eps: ThrusterEpisode[];
  try {
    eps = replayThrusterCommands(run.result.commands, episodes);
  } catch (e) {
    throw new SandboxError("output", `Rejected the sandbox's result: ${e instanceof Error ? e.message : "malformed data"}`);
  }
  return { episodes: eps, elapsedMs: run.elapsedMs, wallMs: run.wallMs, hardening: run.hardening };
}
