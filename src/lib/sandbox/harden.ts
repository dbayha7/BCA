// Runtime hardening for the sandbox worker (layer 2 of 2). Called ONLY from sandbox.worker.ts,
// before user code is compiled (network/storage removal) and right after it is compiled
// (runtime-compilation lock-down + freezing built-ins). Never call the lock-down functions on
// the main thread: they deliberately break `Function`, `eval` and mutation of built-ins.
//
// Honest scope: this removes the obvious exfiltration and persistence channels from the worker
// and makes the usual "reach the Function constructor" tricks fail, so a typo or a pasted
// snippet cannot phone home or touch storage. It does not defend against CPU/memory exhaustion
// beyond the main thread's wall-clock timeout (a huge allocation can still stress the tab), and
// it is not a substitute for a real origin-isolated sandbox.

/** Globals removed from the worker before any user code is compiled. */
export const NEUTERED_GLOBALS = [
  // network
  "fetch",
  "XMLHttpRequest",
  "WebSocket",
  "WebSocketStream",
  "EventSource",
  "WebTransport",
  "RTCPeerConnection",
  "importScripts",
  "FontFace",
  "fonts",
  "Notification",
  // storage
  "indexedDB",
  "caches",
  "localStorage",
  "sessionStorage",
  // other threads / channels
  "BroadcastChannel",
  "Worker",
  "SharedWorker",
  "MessageChannel",
  // navigator carries sendBeacon (where present), storage (OPFS), locks, etc.
  "navigator",
  // string-evaluating timers
  "setTimeout",
  "setInterval",
  // the worker's own message channel (the trusted worker code keeps private references)
  "postMessage",
  "onmessage",
  "addEventListener",
  "removeEventListener",
  "dispatchEvent",
  "close",
] as const;

export interface HardeningReport {
  /** Names that existed in the worker scope and are now undefined. */
  removed: string[];
  /** Names that could not be overridden (should be empty; surfaced in the UI if not). */
  failed: string[];
}

/** Shadow each name on `g` with a non-writable `undefined`. Works on any object (tested with a fake global). */
export function neuterGlobals(g: Record<string, unknown>, names: readonly string[] = NEUTERED_GLOBALS): HardeningReport {
  const removed: string[] = [];
  const failed: string[] = [];
  const nav = g.navigator as Record<string, unknown> | undefined;
  if (nav && typeof nav === "object" && "sendBeacon" in nav) {
    try {
      Object.defineProperty(nav, "sendBeacon", { value: undefined, writable: false, configurable: false });
    } catch {
      /* navigator itself is removed below */
    }
  }
  for (const name of names) {
    const existed = name in g;
    try {
      Object.defineProperty(g, name, { value: undefined, writable: false, configurable: false, enumerable: false });
    } catch {
      try {
        g[name] = undefined;
      } catch {
        /* reported below */
      }
    }
    if (g[name] !== undefined) failed.push(name);
    else if (existed) removed.push(name);
  }
  return { removed, failed };
}

/**
 * Disable runtime compilation AFTER the user's code has been compiled: every function family's
 * `.constructor` (Function, AsyncFunction, GeneratorFunction, AsyncGeneratorFunction) and the
 * global `eval`/`Function` are replaced with a stub that throws.
 */
export function lockDynamicCode(g: Record<string, unknown>) {
  const stub = function blocked() {
    throw new TypeError("Runtime code compilation is disabled in the sandbox.");
  };
  const families = [
    Object.getPrototypeOf(function () {
      return 0;
    }),
    Object.getPrototypeOf(async function () {
      return 0;
    }),
    Object.getPrototypeOf(function* () {
      yield 0;
    }),
    Object.getPrototypeOf(async function* () {
      yield 0;
    }),
  ];
  for (const proto of families) {
    try {
      Object.defineProperty(proto, "constructor", { value: stub, writable: false, configurable: false });
    } catch {
      /* already locked */
    }
  }
  for (const name of ["eval", "Function"]) {
    try {
      Object.defineProperty(g, name, { value: stub, writable: false, configurable: false, enumerable: false });
    } catch {
      /* already locked */
    }
  }
}

// Captured at module load: after lockDynamicCode the global `Function` name is a stub.
const FUNCTION_PROTO: object = Object.getPrototypeOf(function () {
  return 0;
});

/**
 * Freeze the built-ins the trusted rollout/relabel code depends on, so user code cannot
 * monkey-patch e.g. Math.max or Array.prototype.push to corrupt results computed in the worker.
 * (The main thread re-validates and, for agents, re-simulates everything anyway.)
 */
export function freezeIntrinsics() {
  const typed = [Float64Array, Float32Array, Int32Array, Int16Array, Int8Array, Uint32Array, Uint16Array, Uint8Array, Uint8ClampedArray];
  const typedArrayCtor = Object.getPrototypeOf(Int8Array);
  const arrayIter = Object.getPrototypeOf([][Symbol.iterator]());
  const targets: unknown[] = [
    Object,
    Object.prototype,
    Array,
    Array.prototype,
    arrayIter,
    Object.getPrototypeOf(arrayIter),
    Object.getPrototypeOf(new Map()[Symbol.iterator]()),
    Object.getPrototypeOf(new Set()[Symbol.iterator]()),
    FUNCTION_PROTO,
    Math,
    JSON,
    Reflect,
    Number,
    Number.prototype,
    String,
    String.prototype,
    Boolean,
    Boolean.prototype,
    Symbol,
    Symbol.prototype,
    Map,
    Map.prototype,
    Set,
    Set.prototype,
    WeakMap,
    WeakMap.prototype,
    Promise,
    Promise.prototype,
    Error,
    ArrayBuffer,
    ArrayBuffer.prototype,
    typedArrayCtor,
    typedArrayCtor.prototype,
    ...typed,
    ...typed.map((t) => t.prototype),
  ];
  for (const t of targets) {
    try {
      Object.freeze(t);
    } catch {
      /* ignore exotic objects */
    }
  }
}
