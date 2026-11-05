// Message protocol between the main thread (client.ts) and the sandbox worker (sandbox.worker.ts).
// Only plain data crosses the boundary (structured clone): user code, its return values and
// any functions stay inside the worker.

import type { HardeningReport } from "./harden";
import type { RewardEnvInfo } from "./reward";

export type GridPlaygroundEnv = "gridworld-random" | "halfcheetah-medium";
export type PlaygroundEnvId = GridPlaygroundEnv | "thrusterchain-stall";

export type SandboxJob =
  | { kind: "reward"; code: string; rows: Int32Array; original: Float64Array; info: RewardEnvInfo }
  | { kind: "agent-grid"; code: string; envId: GridPlaygroundEnv; episodes: number; horizon: number }
  | { kind: "agent-thruster"; code: string; episodes: number };

export interface SandboxRequest {
  type: "run";
  id: number;
  job: SandboxJob;
}

export type SandboxResultData =
  | { kind: "reward"; rewards: Float64Array }
  /** Action sequence per episode; the main thread replays them on the trusted env. */
  | { kind: "agent-grid"; actions: number[][] }
  /** Requested 4-dim commands per episode, flattened (4 × HORIZON); the main thread re-simulates. */
  | { kind: "agent-thruster"; commands: number[][] };

export type SandboxErrorKind = "guard" | "syntax" | "runtime" | "output" | "timeout" | "worker" | "aborted";

export type SandboxResponse =
  | { type: "ready"; hardening: HardeningReport }
  | { type: "result"; id: number; result: SandboxResultData; elapsedMs: number }
  | { type: "error"; id: number; kind: SandboxErrorKind; message: string; line?: number; column?: number };

/** Thrown by trusted worker code when user code throws ("runtime") or returns something invalid ("output"). */
export class UserCodeFailure extends Error {
  kind: "runtime" | "output";
  original: unknown;
  constructor(kind: "runtime" | "output", message: string, original?: unknown) {
    super(message);
    this.kind = kind;
    this.original = original;
  }
}

/** Short description of an untrusted value for error messages, without calling its toString/valueOf. */
export function describeValue(v: unknown): string {
  if (v === undefined) return "undefined";
  if (v === null) return "null";
  const t = typeof v;
  if (t === "number") return String(v);
  if (t === "string") {
    const s = v as string;
    return JSON.stringify(s.length > 24 ? `${s.slice(0, 24)}…` : s);
  }
  if (t === "boolean" || t === "bigint") return `${t} ${String(v)}`;
  if (t === "function") return "a function";
  if (t === "symbol") return "a symbol";
  if (Array.isArray(v)) return `an array of length ${(v as unknown[]).length}`;
  return "an object";
}
