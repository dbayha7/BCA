// Training alerts: notice when a run has stopped paying for its compute (plateau), hit a target,
// or regressed — and optionally pause auto-train so the browser stops spending cycles on it.

import type { Metrics } from "./rl";

export interface AlertRules {
  /** Alert when the best true return has not improved by more than `plateauEps` for this many steps (0 = off). */
  plateauSteps: number;
  plateauEps: number;
  /** Alert once when the true return first reaches this value (null = off). */
  targetReturn: number | null;
  /** Alert when the true return falls this far below the run's best (0 = off). */
  regressionDrop: number;
  /** Pause auto-train when a plateau or target alert fires. */
  autoPause: boolean;
  /** Also show a system notification (asks the browser for permission). */
  notify: boolean;
}

export const DEFAULT_ALERT_RULES: AlertRules = {
  plateauSteps: 40,
  plateauEps: 0.05,
  targetReturn: null,
  regressionDrop: 0,
  autoPause: false,
  notify: false,
};

export type AlertKind = "plateau" | "target" | "regression";

export interface TrainingAlert {
  kind: AlertKind;
  atUpdates: number;
  message: string;
  /** Whether this alert asks auto-train to pause. */
  pause: boolean;
}

/** What has already fired in this run, so each condition alerts once until it resets. */
export interface AlertMemory {
  plateauBestIter: number | null;
  targetHit: boolean;
  regressionFrom: number | null;
}

export const EMPTY_MEMORY: AlertMemory = { plateauBestIter: null, targetHit: false, regressionFrom: null };

/**
 * Checks the newest step against the rules. Pure: returns the alerts to raise and the updated memory.
 * Plateau = no step in the last `plateauSteps` beat the earlier best by more than `plateauEps`.
 */
export function checkAlerts(history: Metrics[], rules: AlertRules, memory: AlertMemory): { alerts: TrainingAlert[]; memory: AlertMemory } {
  const alerts: TrainingAlert[] = [];
  const next = { ...memory };
  const last = history[history.length - 1];
  if (!last) return { alerts, memory: next };

  let bestIdx = 0;
  for (let i = 1; i < history.length; i++) if (history[i].trueQ > history[bestIdx].trueQ + rules.plateauEps) bestIdx = i;
  const best = history[bestIdx];

  if (rules.plateauSteps > 0 && history.length - 1 - bestIdx >= rules.plateauSteps && next.plateauBestIter !== best.iter) {
    next.plateauBestIter = best.iter;
    alerts.push({
      kind: "plateau",
      atUpdates: last.updates,
      pause: rules.autoPause,
      message: `True return has not improved by more than ${rules.plateauEps} for ${history.length - 1 - bestIdx} steps (best ${best.trueQ.toFixed(2)} at ${best.updates.toLocaleString()} updates). More training here is unlikely to help.`,
    });
  }

  if (rules.targetReturn !== null && !next.targetHit && last.trueQ >= rules.targetReturn) {
    next.targetHit = true;
    alerts.push({
      kind: "target",
      atUpdates: last.updates,
      pause: rules.autoPause,
      message: `Target reached: true return ${last.trueQ.toFixed(2)} ≥ ${rules.targetReturn} after ${last.updates.toLocaleString()} updates.`,
    });
  }

  let runBest = -Infinity;
  for (const m of history) runBest = Math.max(runBest, m.trueQ);
  if (rules.regressionDrop > 0 && runBest - last.trueQ >= rules.regressionDrop) {
    if (next.regressionFrom !== runBest) {
      next.regressionFrom = runBest;
      alerts.push({
        kind: "regression",
        atUpdates: last.updates,
        pause: false,
        message: `True return fell to ${last.trueQ.toFixed(2)}, ${(runBest - last.trueQ).toFixed(2)} below this run's best of ${runBest.toFixed(2)}. Consider restoring a checkpoint.`,
      });
    }
  } else if (next.regressionFrom !== null && runBest - last.trueQ < rules.regressionDrop / 2) {
    next.regressionFrom = null;
  }

  return { alerts, memory: next };
}

/** Shows a system notification when allowed; never throws. */
export function notifySystem(title: string, body: string): void {
  try {
    if (typeof Notification === "undefined" || Notification.permission !== "granted") return;
    new Notification(title, { body, tag: "offline-lab-training" });
  } catch {
    /* some browsers only allow notifications from a service worker */
  }
}

export async function requestNotifyPermission(): Promise<NotificationPermission | "unsupported"> {
  try {
    if (typeof Notification === "undefined") return "unsupported";
    if (Notification.permission === "granted" || Notification.permission === "denied") return Notification.permission;
    return await Notification.requestPermission();
  } catch {
    return "unsupported";
  }
}
