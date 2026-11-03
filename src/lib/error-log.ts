// Lab error log: every caught crash (panel render errors, training/physics guard trips,
// worker failures, uncaught exceptions) is recorded here, shown in the UI and saved locally,
// so a failure leaves evidence instead of a blank panel.

export type ErrorSource = "panel" | "training" | "physics" | "worker" | "storage" | "uncaught";

export interface ErrorEntry {
  id: string;
  at: number;
  source: ErrorSource;
  message: string;
  /** Where it happened: panel title, dataset, update count… */
  context?: string;
  stack?: string;
}

type Listener = (e: ErrorEntry) => void;
const listeners = new Set<Listener>();
const recent: ErrorEntry[] = [];

function uid(): string {
  try {
    return crypto.randomUUID();
  } catch {
    return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }
}

export function recordError(source: ErrorSource, error: unknown, context?: string): ErrorEntry {
  const err = error instanceof Error ? error : null;
  const entry: ErrorEntry = {
    id: uid(),
    at: Date.now(),
    source,
    message: err ? err.message : typeof error === "string" ? error : "Unknown error",
    context,
    stack: err?.stack?.split("\n").slice(0, 8).join("\n"),
  };
  recent.unshift(entry);
  if (recent.length > 100) recent.length = 100;
  for (const l of listeners) {
    try {
      l(entry);
    } catch {
      /* a broken listener must not hide the error */
    }
  }
  return entry;
}

export function recentErrors(): ErrorEntry[] {
  return [...recent];
}

export function onError(l: Listener): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

/** Third-party noise that is not a lab failure (the hosting platform's analytics beacon). */
export function isExternalNoise(message: string, source?: string): boolean {
  const text = `${message} ${source ?? ""}`.toLowerCase();
  return text.includes("shipper-api") || text.includes("tracker.js") || text.includes("/.shipper/") || text.includes("resizeobserver loop");
}

let installed = false;

/** Catches anything that escapes React and the guards. Safe to call more than once. */
export function installGlobalErrorHandlers(): void {
  if (installed || typeof window === "undefined") return;
  installed = true;
  window.addEventListener("error", (ev) => {
    const msg = ev.message || (ev.error instanceof Error ? ev.error.message : "Script error");
    if (isExternalNoise(msg, ev.filename)) return;
    recordError("uncaught", ev.error ?? msg, ev.filename ? `${ev.filename}:${ev.lineno}` : undefined);
  });
  window.addEventListener("unhandledrejection", (ev) => {
    const reason = ev.reason;
    const msg = reason instanceof Error ? reason.message : String(reason);
    if (isExternalNoise(msg, reason instanceof Error ? reason.stack : undefined)) return;
    recordError("uncaught", reason, "unhandled promise rejection");
  });
}
