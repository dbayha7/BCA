import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useTheme } from "@/components/theme-provider";
import { LineChart, Legend } from "@/components/lab/charts";
import { StateGrid, heatColor, rewardColor, type HeatMode } from "@/components/lab/grid";
import {
  createQ,
  evaluate,
  generateDataset,
  mulberry32,
  parseCustomDataset,
  qStar,
  sampleCustomJson,
  trainUpdates,
  UPDATES_PER_STEP,
  type Dataset,
  type DatasetId,
  type Hyper,
  type Metrics,
} from "@/lib/rl";
import { generateThrusterInWorker, parseCustomInWorker } from "@/lib/worker-client";
import { Progress } from "@/components/ui/progress";
import { buildPolicyExport, downloadPolicyExport, type ExportTrigger } from "@/lib/export-policy";
import { ExportPanel } from "@/components/lab/export-panel";
import { CheckpointHistory, type Checkpoint } from "@/components/lab/checkpoint-history";
import { EpisodePlayer } from "@/components/lab/episode-player";
import { ExperimentLog } from "@/components/lab/experiment-log";
import { LicensePanel } from "@/components/lab/license-panel";
import { CollaborationPanel } from "@/components/lab/collaboration-panel";
import { buildLicenseBlock, isExpired, signLicenseBlock, DEFAULT_LICENSE, type LicenseConfig, type SigningKey } from "@/lib/licensing";
import { buildLogEntries } from "@/components/lab/experiment-log";
import { PanelErrorBoundary } from "@/components/lab/error-boundary";
import { StorageStatus, type SaveState } from "@/components/lab/storage-status";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { resumableRng, type ResumableRng } from "@/lib/rng";
import { checkpointsStore, experimentsStore, kv, storageSupported, uploadsStore } from "@/lib/store";
import { defaultExperimentName, hyperEqual, parseTags, type ExperimentRecord, type ExperimentStatus } from "@/lib/experiments";
import { checkQ, metricsSane } from "@/lib/guards";
import { recordError, recentErrors } from "@/lib/error-log";
import { auditMarkdown, buildAuditLog, downloadText, metricsCsv, slugify } from "@/lib/log-export";
import { checkAlerts, DEFAULT_ALERT_RULES, EMPTY_MEMORY, notifySystem, requestNotifyPermission, type AlertRules } from "@/lib/alerts";
import { buildModelCard } from "@/lib/model-card";
import { usePersistentState } from "@/lib/use-persistent-state";
import { Switch } from "@/components/ui/switch";

export const Route = createFileRoute("/")({
  component: LabPage,
  // `?run=<id>` opens a saved run (used by the Experiments page).
  validateSearch: (s: Record<string, unknown>): { run?: string } => (typeof s.run === "string" ? { run: s.run } : {}),
});

const DEFAULT_HYPER: Hyper = { alpha: 1.0, gamma: 0.95, lr: 0.3, batch: 64 };
/** Seed of every run's minibatch stream (unchanged from before saving existed). */
const TRAIN_SEED = 1234;
const MAX_CHECKPOINTS = 20;

function newId(): string {
  try {
    return crypto.randomUUID();
  } catch {
    return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }
}

async function sha256Hex(text: string): Promise<string> {
  try {
    const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
  } catch {
    // Non-secure contexts lack crypto.subtle; a length+prefix key still separates uploads.
    return `n${text.length}-${text.slice(0, 64).replace(/\W/g, "")}`;
  }
}

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <svg width="26" height="26" viewBox="0 0 26 26" fill="none" className="text-primary" aria-hidden>
        {/* A frozen "replay buffer" stack with a learning curve escaping it */}
        <rect x="2.5" y="15" width="12" height="3" rx="1" stroke="currentColor" strokeWidth="1.6" />
        <rect x="2.5" y="20" width="12" height="3" rx="1" stroke="currentColor" strokeWidth="1.6" opacity="0.55" />
        <path d="M3 11.5 C 8 11.5, 10 4, 16 4 S 22 6, 23.5 3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        <circle cx="23.5" cy="3" r="1.6" fill="currentColor" />
        <path d="M18 14v9M18 23l-2.5-2.5M18 23l2.5-2.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" opacity="0.55" />
      </svg>
      <div className="flex items-baseline gap-1 leading-none">
        <span className="font-display text-[15px] font-bold tracking-tight">Offline</span>
        <span className="font-display text-[15px] font-light tracking-wide text-muted-foreground">Learning Lab</span>
      </div>
    </div>
  );
}

function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const dark = theme === "dark" || (theme === "system" && typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  return (
    <button
      onClick={() => setTheme(dark ? "light" : "dark")}
      className="flex h-8 w-8 items-center justify-center rounded-md border border-border bg-card text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {dark ? (
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
        </svg>
      ) : (
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
        </svg>
      )}
    </button>
  );
}

function Panel({ title, right, children, className = "", delay = 0 }: { title: string; right?: React.ReactNode; children: React.ReactNode; className?: string; delay?: number }) {
  return (
    <section className={`lab-rise rounded-xl border border-border bg-card shadow-sm ${className}`} style={{ animationDelay: `${delay}ms` }}>
      <header className="flex items-center justify-between gap-2 border-b border-border px-4 py-2.5">
        <h2 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{title}</h2>
        {right}
      </header>
      <div className="p-4">
        <PanelErrorBoundary name={title}>{children}</PanelErrorBoundary>
      </div>
    </section>
  );
}

function HyperSlider({ label, hint, value, min, max, step, onChange, format }: { label: string; hint: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; format: (v: number) => string }) {
  return (
    <div className="space-y-2.5">
      <div className="flex items-baseline justify-between gap-2">
        <div>
          <div className="text-[13px] font-medium">{label}</div>
          <div className="text-[11px] text-muted-foreground">{hint}</div>
        </div>
        <span className="rounded-md border border-border bg-surface px-2 py-0.5 font-mono text-xs tabular-nums">{format(value)}</span>
      </div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={(v) => onChange(v[0])} aria-label={label} />
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: "warn" | "ok" | "bad" }) {
  const c = tone === "warn" ? "text-warning" : tone === "bad" ? "text-destructive" : tone === "ok" ? "text-success" : "text-foreground";
  return (
    <div className="rounded-lg border border-border bg-surface/60 px-3 py-2">
      <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className={`mt-0.5 font-mono text-[15px] font-medium tabular-nums ${c}`}>{value}</div>
    </div>
  );
}

function RunMetaEditor({
  info,
  updates,
  computeMs,
  onChange,
}: {
  info: { id: string; name: string; tags: string[]; notes: string; status: ExperimentStatus };
  updates: number;
  computeMs: number;
  onChange: (patch: Partial<Pick<ExperimentRecord, "name" | "tags" | "notes">>) => void;
}) {
  const [name, setName] = useState(info.name);
  const [tags, setTags] = useState(info.tags.join(", "));
  const [notes, setNotes] = useState(info.notes);
  // The run is named automatically at its first step; follow that unless the user is typing.
  useEffect(() => setName((n) => (n === "" ? info.name : n)), [info.name]);
  const field = "w-full rounded-md border border-border bg-surface/60 px-2.5 py-1.5 text-[12px] outline-none transition-colors focus:border-ring focus-visible:ring-2 focus-visible:ring-ring/40";
  return (
    <div className="space-y-2.5">
      <label className="block">
        <span className="text-[11px] text-muted-foreground">Name</span>
        <input
          className={`${field} mt-1`}
          value={name}
          placeholder="Named at the first training step"
          maxLength={120}
          onChange={(e) => setName(e.target.value)}
          onBlur={() => name.trim() && name.trim() !== info.name && onChange({ name: name.trim() })}
        />
      </label>
      <label className="block">
        <span className="text-[11px] text-muted-foreground">Tags (comma-separated)</span>
        <input
          className={`${field} mt-1`}
          value={tags}
          placeholder="baseline, high-alpha"
          onChange={(e) => setTags(e.target.value)}
          onBlur={() => onChange({ tags: parseTags(tags) })}
        />
      </label>
      <label className="block">
        <span className="text-[11px] text-muted-foreground">Notes</span>
        <textarea
          className={`${field} mt-1 min-h-[56px] resize-y`}
          value={notes}
          maxLength={4000}
          placeholder="What are you testing?"
          onChange={(e) => setNotes(e.target.value)}
          onBlur={() => notes !== info.notes && onChange({ notes })}
        />
      </label>
      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
        <span className="font-mono tabular-nums" title="Time this run has spent computing updates and evaluations">
          {updates.toLocaleString()} upd · {(computeMs / 1000).toFixed(1)} s{computeMs > 0 ? ` · ${Math.round(updates / (computeMs / 1000)).toLocaleString()}/s` : ""}
        </span>
        <Link to="/experiments" className="font-medium text-primary hover:underline">
          All experiments →
        </Link>
      </div>
    </div>
  );
}

function AlertsEditor({ rules, onChange }: { rules: AlertRules; onChange: (r: AlertRules) => void }) {
  const num = "w-20 rounded-md border border-border bg-surface/60 px-2 py-1 text-right font-mono text-[12px] tabular-nums outline-none focus:border-ring";
  const set = (patch: Partial<AlertRules>) => onChange({ ...rules, ...patch });
  return (
    <div className="space-y-2.5 text-[12px]">
      <label className="flex items-center justify-between gap-2">
        <span>
          Plateau after <span className="text-muted-foreground">(steps, 0 = off)</span>
        </span>
        <input className={num} type="number" min={0} max={2000} step={5} value={rules.plateauSteps} onChange={(e) => set({ plateauSteps: Math.max(0, Math.round(Number(e.target.value) || 0)) })} />
      </label>
      <label className="flex items-center justify-between gap-2">
        <span>
          Target true return <span className="text-muted-foreground">(blank = off)</span>
        </span>
        <input
          className={num}
          type="number"
          step={0.1}
          value={rules.targetReturn ?? ""}
          onChange={(e) => set({ targetReturn: e.target.value === "" ? null : Number(e.target.value) })}
        />
      </label>
      <label className="flex items-center justify-between gap-2">
        <span>
          Regression drop <span className="text-muted-foreground">(0 = off)</span>
        </span>
        <input className={num} type="number" min={0} step={0.1} value={rules.regressionDrop} onChange={(e) => set({ regressionDrop: Math.max(0, Number(e.target.value) || 0) })} />
      </label>
      <div className="flex items-center justify-between gap-2">
        <span>Pause auto-train on plateau or target</span>
        <Switch checked={rules.autoPause} onCheckedChange={(v) => set({ autoPause: v })} aria-label="Pause auto-train on plateau or target" />
      </div>
      <div className="flex items-center justify-between gap-2">
        <span>System notification when the tab is hidden</span>
        <Switch
          checked={rules.notify}
          aria-label="System notification when the tab is hidden"
          onCheckedChange={(v) => {
            if (!v) return set({ notify: false });
            void requestNotifyPermission().then((p) => {
              if (p === "granted") set({ notify: true });
              else toast.message(p === "unsupported" ? "This browser does not support notifications here." : "Notifications are blocked for this site in your browser settings.");
            });
          }}
        />
      </div>
      <p className="text-[11px] leading-relaxed text-muted-foreground">Pausing when the true return stops improving saves compute: further steps rarely help once a run has plateaued.</p>
    </div>
  );
}

function LabPage() {
  const search = Route.useSearch();
  const navigate = useNavigate();
  const [datasetId, setDatasetId] = useState<DatasetId>("gridworld-random");
  const [customDs, setCustomDs] = useState<Dataset | null>(null);
  const [hyper, setHyper] = useState<Hyper>(DEFAULT_HYPER);
  const [view, setView] = useState<"dataset" | "policy" | "watch">("dataset");
  const [heat, setHeat] = useState<HeatMode>("estimate");
  const [history, setHistory] = useState<Metrics[]>([]);
  const [version, setVersion] = useState(0);
  const [auto, setAuto] = useState(false);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [activeCheckpointId, setActiveCheckpointId] = useState<string | null>(null);
  const [thrusterDs, setThrusterDs] = useState<Dataset | null>(null);
  const [working, setWorking] = useState<{ pct: number; label: string } | null>(null);
  const [saveState, setSaveState] = useState<SaveState>(() => (storageSupported() ? { kind: "idle" } : { kind: "unavailable" }));
  const [runInfo, setRunInfo] = useState<{ id: string; name: string; tags: string[]; notes: string; status: ExperimentStatus } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [alertRules, setAlertRules] = usePersistentState<AlertRules>("alerts:rules", DEFAULT_ALERT_RULES);
  const alertRulesRef = useRef(alertRules);
  alertRulesRef.current = alertRules;
  const alertMemoryRef = useRef(EMPTY_MEMORY);

  // ThrusterChain's 13-dim float64 simulation runs in a Web Worker so the UI never blocks.
  useEffect(() => {
    if (datasetId !== "thrusterchain-stall" || thrusterDs) return;
    let cancelled = false;
    setWorking({ pct: 0, label: "Starting background worker…" });
    generateThrusterInWorker(7, (pct, label) => {
      if (!cancelled) setWorking({ pct, label });
    })
      .then((d) => {
        if (cancelled) return;
        setThrusterDs(d);
        setWorking(null);
      })
      .catch((e) => {
        if (cancelled) return;
        setWorking(null);
        recordError("worker", e, "ThrusterChain dataset simulation");
        toast.error(e instanceof Error ? e.message : "ThrusterChain simulation failed.");
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, thrusterDs]);

  const ds = useMemo<Dataset | null>(() => {
    if (datasetId === "custom") return customDs;
    if (datasetId === "thrusterchain-stall") return thrusterDs;
    return generateDataset(datasetId, datasetId === "gridworld-random" ? 7 : 11);
  }, [datasetId, customDs, thrusterDs]);

  const qRef = useRef<Float64Array>(new Float64Array(0));
  const rngRef = useRef<ResumableRng>(resumableRng(TRAIN_SEED));

  /* ---------- Runs: every training run is recorded and saved locally as it trains ---------- */
  const runRef = useRef<ExperimentRecord | null>(null);
  /** Full metrics history of the current run (the charts show the latest 200 steps). */
  const fullHistoryRef = useRef<Metrics[]>([]);
  const pendingRestoreRef = useRef<ExperimentRecord | null>(null);
  const uploadKeyRef = useRef<string | undefined>(undefined);
  const hydratedRef = useRef(false);
  const saveTimer = useRef<number | null>(null);
  const dsRef = useRef(ds);
  dsRef.current = ds;
  const hyperRef = useRef(hyper);
  hyperRef.current = hyper;

  const publishRun = useCallback((rec: ExperimentRecord | null) => {
    setRunInfo(rec ? { id: rec.id, name: rec.name, tags: rec.tags, notes: rec.notes, status: rec.status } : null);
  }, []);

  /** A consistent copy of the current run, or null before its first training step. */
  const snapshotRun = useCallback((): ExperimentRecord | null => {
    const rec = runRef.current;
    const d = dsRef.current;
    if (!rec || !d || fullHistoryRef.current.length === 0) return null;
    const hist = [...fullHistoryRef.current];
    // Use the run's own hyperparameters and optimum: the page may already show a different dataset.
    return {
      ...rec,
      hyper: { ...rec.hyper },
      hyperLog: rec.hyperLog.map((h) => ({ ...h, hyper: { ...h.hyper } })),
      guardEvents: [...rec.guardEvents],
      history: hist,
      q: new Float64Array(qRef.current),
      rngState: rngRef.current.state(),
      anomalies: buildLogEntries(hist, d, rec.hyper, rec.optimal).filter((e) => e.kind === "anomaly").length,
      updatedAt: Date.now(),
    };
  }, []);

  const persist = useCallback((snap: ExperimentRecord) => {
    setSaveState({ kind: "saving" });
    experimentsStore
      .put(snap)
      .then(() => kv.set("currentExperimentId", snap.id))
      .then(() => setSaveState({ kind: "saved", at: Date.now() }))
      .catch((e) => {
        const message = e instanceof Error ? e.message : "Local storage write failed.";
        setSaveState(storageSupported() ? { kind: "failed", message } : { kind: "unavailable" });
        recordError("storage", e, `saving run “${snap.name}”`);
      });
  }, []);

  /** Saves at most once a second while training (trailing edge), so auto-train stays fast. */
  const scheduleSave = useCallback(() => {
    if (!hydratedRef.current || saveTimer.current !== null) return;
    saveTimer.current = window.setTimeout(() => {
      saveTimer.current = null;
      const snap = snapshotRun();
      if (snap) persist(snap);
    }, 1000);
  }, [snapshotRun, persist]);

  const flushSave = useCallback(() => {
    if (saveTimer.current !== null) {
      window.clearTimeout(saveTimer.current);
      saveTimer.current = null;
    }
    const snap = snapshotRun();
    if (snap) persist(snap);
  }, [snapshotRun, persist]);

  useEffect(() => {
    const onHide = () => {
      if (document.visibilityState === "hidden") flushSave();
    };
    window.addEventListener("pagehide", flushSave);
    document.addEventListener("visibilitychange", onHide);
    return () => {
      window.removeEventListener("pagehide", flushSave);
      document.removeEventListener("visibilitychange", onHide);
      // Leaving the lab for another page (Experiments, Sweep…) saves the run first.
      flushSave();
    };
  }, [flushSave]);

  const makeRun = useCallback((d: Dataset, h: Hyper): ExperimentRecord => {
    const now = Date.now();
    return {
      id: newId(),
      name: "",
      createdAt: now,
      updatedAt: now,
      datasetId: d.env.id,
      datasetName: d.env.name,
      uploadKey: d.env.id === "custom" ? uploadKeyRef.current : undefined,
      transitions: d.transitions.length,
      seed: TRAIN_SEED,
      rngState: TRAIN_SEED,
      hyper: { ...h },
      hyperLog: [{ atUpdates: 0, hyper: { ...h } }],
      history: [],
      q: new Float64Array(0),
      optimal: 0,
      status: "paused",
      tags: [],
      notes: "",
      anomalies: 0,
      guardEvents: [],
    };
  }, []);

  /** Starts a new run on the current dataset; the previous run stays saved in Experiments. */
  const reset = useCallback(() => {
    if (!ds) return;
    flushSave();
    qRef.current = createQ(ds);
    rngRef.current = resumableRng(TRAIN_SEED);
    fullHistoryRef.current = [];
    const rec = makeRun(ds, hyperRef.current);
    runRef.current = rec;
    alertMemoryRef.current = EMPTY_MEMORY;
    publishRun(rec);
    setHistory([]);
    setVersion((v) => v + 1);
    setAuto(false);
    setActiveCheckpointId(null);
  }, [ds, flushSave, makeRun, publishRun]);

  const applyRestore = useCallback(
    (rec: ExperimentRecord, d: Dataset): boolean => {
      if (rec.q.length !== d.env.width * d.env.height * 4) {
        recordError("storage", `Saved Q-table for “${rec.name}” does not fit ${d.env.name}`, "restoring a run");
        toast.error(`Could not reopen “${rec.name || "run"}”: its saved Q-table does not match ${d.env.name}.`);
        return false;
      }
      qRef.current = new Float64Array(rec.q);
      rngRef.current = resumableRng(rec.rngState);
      fullHistoryRef.current = [...rec.history];
      // Re-arm alerts as if the reopened run had been watched all along (no replay of old alerts).
      alertMemoryRef.current = checkAlerts(rec.history, alertRulesRef.current, EMPTY_MEMORY).memory;
      runRef.current = { ...rec, status: rec.status === "diverged" ? "diverged" : "paused" };
      publishRun(runRef.current);
      setHyper({ ...rec.hyper });
      setHistory(rec.history.slice(-200));
      setVersion((v) => v + 1);
      setAuto(false);
      setActiveCheckpointId(null);
      const last = rec.history[rec.history.length - 1];
      toast.success(`Reopened “${rec.name}” at ${(last?.updates ?? 0).toLocaleString()} updates`);
      return true;
    },
    [publishRun],
  );

  // A new dataset either receives the run waiting to be reopened on it, or starts a new run.
  useEffect(() => {
    if (!ds) return;
    const pending = pendingRestoreRef.current;
    if (pending && pending.datasetId === ds.env.id) {
      pendingRestoreRef.current = null;
      if (applyRestore(pending, ds)) return;
    }
    reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ds]);

  /** Reopens a saved run: switches to its dataset (re-parsing a saved upload) and restores Q, stream and history. */
  const openRun = useCallback(
    async (rec: ExperimentRecord) => {
      if (rec.datasetId === "custom") {
        const up = rec.uploadKey ? await uploadsStore.get(rec.uploadKey) : undefined;
        if (!up) {
          toast.error(`The uploaded dataset for “${rec.name}” is not saved in this browser, so the run cannot be reopened.`);
          return;
        }
        setWorking({ pct: 2, label: `Reopening ${up.name}…` });
        try {
          const parsed = await parseCustomInWorker(up.text, (pct, label) => setWorking({ pct, label }));
          uploadKeyRef.current = rec.uploadKey;
          pendingRestoreRef.current = rec;
          setCustomDs(parsed);
          setDatasetId("custom");
        } catch (e) {
          recordError("worker", e, `re-parsing ${up.name}`);
          toast.error(e instanceof Error ? e.message : "Could not re-parse the saved upload.");
        } finally {
          setWorking(null);
        }
        return;
      }
      const d = dsRef.current;
      if (d && d.env.id === rec.datasetId) applyRestore(rec, d);
      else {
        pendingRestoreRef.current = rec;
        setDatasetId(rec.datasetId);
      }
    },
    [applyRestore],
  );

  // Load saved checkpoints and reopen the last run (or the one named in ?run=).
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const cps = await checkpointsStore.all();
        if (alive) setCheckpoints(cps.slice(0, MAX_CHECKPOINTS));
        const id = search.run ?? (await kv.get<string>("currentExperimentId"));
        if (!alive || !id) return;
        const rec = await experimentsStore.get(id);
        if (!alive) return;
        if (!rec) {
          if (search.run) toast.error("That run is no longer saved in this browser.");
          return;
        }
        // Don't overwrite a run the user already started training while storage was loading.
        if (fullHistoryRef.current.length > 0 && !search.run) return;
        await openRun(rec);
      } catch (e) {
        if (!storageSupported()) setSaveState({ kind: "unavailable" });
        recordError("storage", e, "loading saved lab state");
      } finally {
        if (alive) {
          hydratedRef.current = true;
          if (search.run) void navigate({ to: "/", search: {}, replace: true });
        }
      }
    })();
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Record every hyperparameter change against the update count it took effect at.
  useEffect(() => {
    const rec = runRef.current;
    if (!rec) return;
    const lastH = rec.hyperLog[rec.hyperLog.length - 1]?.hyper;
    if (lastH && hyperEqual(lastH, hyper)) return;
    const at = fullHistoryRef.current[fullHistoryRef.current.length - 1]?.updates ?? 0;
    // Several slider nudges between two training steps are one change, effective at `at`.
    const prior = rec.hyperLog.length > 1 && rec.hyperLog[rec.hyperLog.length - 1].atUpdates === at ? rec.hyperLog.slice(0, -1) : rec.hyperLog;
    rec.hyperLog = at === 0 ? [{ atUpdates: 0, hyper: { ...hyper } }] : [...prior, { atUpdates: at, hyper: { ...hyper } }];
    rec.hyper = { ...hyper };
    scheduleSave();
  }, [hyper, scheduleSave]);

  useEffect(() => {
    const rec = runRef.current;
    if (!rec || rec.status === "diverged") return;
    rec.status = auto ? "training" : "paused";
    publishRun(rec);
    scheduleSave();
  }, [auto, publishRun, scheduleSave]);

  const updateRunMeta = (patch: Partial<Pick<ExperimentRecord, "name" | "tags" | "notes">>) => {
    const rec = runRef.current;
    if (!rec) return;
    Object.assign(rec, patch);
    publishRun(rec);
    scheduleSave();
  };

  const evalRes = useMemo(() => {
    if (!ds || qRef.current.length !== ds.env.width * ds.env.height * 4) return null;
    return evaluate(qRef.current, ds, hyper.gamma);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ds, version, hyper.gamma]);

  const optimal = useMemo(() => {
    if (!ds) return 0;
    const Qs = qStar(ds.env, hyper.gamma);
    const b = ds.env.start * 4;
    return Math.max(Qs[b], Qs[b + 1], Qs[b + 2], Qs[b + 3]);
  }, [ds, hyper.gamma]);

  const step = useCallback(() => {
    if (!ds) return;
    // Training guard: keep the last finite state so a diverged step can be rolled back.
    const backupQ = new Float64Array(qRef.current);
    const backupRng = rngRef.current.state();
    const rollback = (reason: string) => {
      qRef.current = backupQ;
      rngRef.current = resumableRng(backupRng);
      setAuto(false);
      const at = fullHistoryRef.current[fullHistoryRef.current.length - 1]?.updates ?? 0;
      const message = `Training diverged after ${at.toLocaleString()} updates: ${reason}. Rolled back to the last finite Q-table and paused — try a lower learning rate or α.`;
      const rec = runRef.current;
      if (rec) {
        rec.guardEvents = [...rec.guardEvents, { atUpdates: at, message, at: Date.now() }];
        rec.status = "diverged";
        publishRun(rec);
      }
      recordError("training", message, `${ds.env.name} · α ${hyper.alpha} · γ ${hyper.gamma} · lr ${hyper.lr} · batch ${hyper.batch}`);
      toast.error(message);
      setVersion((v) => v + 1);
      scheduleSave();
    };
    let tdLoss: number;
    let cqlGap: number;
    let r: ReturnType<typeof evaluate>;
    const t0 = performance.now();
    try {
      ({ tdLoss, cqlGap } = trainUpdates(qRef.current, ds, hyper, rngRef.current, UPDATES_PER_STEP));
      const qc = checkQ(qRef.current, ds.env.width);
      if (!qc.ok) {
        rollback(qc.reason ?? "non-finite Q-value");
        return;
      }
      r = evaluate(qRef.current, ds, hyper.gamma);
    } catch (e) {
      rollback(e instanceof Error ? e.message : "training step threw");
      return;
    }
    const prev = fullHistoryRef.current[fullHistoryRef.current.length - 1];
    const iter = (prev?.iter ?? 0) + 1;
    const m: Metrics = {
      iter,
      updates: iter * UPDATES_PER_STEP,
      estQ: r.estQ,
      trueQ: r.trueQ,
      loss: tdLoss + hyper.alpha * cqlGap,
      tdLoss,
      cqlGap,
      oodRate: r.oodRate,
      divergence: r.divergence,
    };
    if (!metricsSane(m)) {
      rollback("a training metric became non-finite");
      return;
    }
    fullHistoryRef.current.push(m);
    setHistory((h) => [...h.slice(-199), m]);
    setVersion((v) => v + 1);
    const rec = runRef.current;
    if (rec) rec.computeMs = (rec.computeMs ?? 0) + (performance.now() - t0);
    // Training alerts: plateau, target, regression (optionally pausing auto-train).
    const checked = checkAlerts(fullHistoryRef.current, alertRulesRef.current, alertMemoryRef.current);
    alertMemoryRef.current = checked.memory;
    for (const a of checked.alerts) {
      if (a.pause) setAuto(false);
      const text = a.pause ? `${a.message} Auto-train paused.` : a.message;
      if (a.kind === "target") toast.success(text);
      else if (a.kind === "regression") toast.warning(text);
      else toast.message(text);
      if (alertRulesRef.current.notify && document.visibilityState === "hidden") notifySystem("Offline Learning Lab", text);
    }
    if (rec) {
      if (!rec.name) rec.name = defaultExperimentName(ds.env.name, hyper, rec.createdAt);
      rec.optimal = optimal;
      if (rec.status === "diverged") rec.status = "paused";
      if (iter === 1 || rec.status !== runInfo?.status) publishRun(rec);
    }
    scheduleSave();
  }, [ds, hyper, optimal, scheduleSave, publishRun, runInfo?.status]);

  useEffect(() => {
    if (!auto) return;
    const id = window.setInterval(step, 140);
    return () => window.clearInterval(id);
  }, [auto, step]);

  const onUpload = async (f: File) => {
    try {
      setWorking({ pct: 2, label: `Reading ${f.name}…` });
      const text = await f.text();
      // Heavy parse/validation/aggregation runs in the Web Worker; UI only renders progress.
      const parsed = await parseCustomInWorker(text, (pct, label) => setWorking({ pct, label }));
      setWorking(null);
      // Keep the raw upload so runs on it can be reopened after a reload.
      const key = `upload:${await sha256Hex(text)}`;
      uploadKeyRef.current = key;
      uploadsStore.put(key, { name: f.name, text, size: text.length, savedAt: Date.now() }).catch((e) => {
        recordError("storage", e, `saving upload ${f.name}`);
        toast.message(`${f.name} could not be saved locally (${e instanceof Error ? e.message : "storage error"}); runs on it will not reopen after a reload.`);
      });
      setCustomDs(parsed);
      setDatasetId("custom");
      toast.success(`Loaded ${parsed.transitions.length.toLocaleString()} transitions from ${f.name}`);
    } catch (e) {
      setWorking(null);
      recordError("worker", e, `parsing upload ${f.name}`);
      toast.error(e instanceof Error ? e.message : "Could not parse that file.");
    }
  };

  const downloadSample = () => {
    const blob = new Blob([sampleCustomJson()], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "offline-dataset-sample.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const saveCheckpoint = () => {
    if (!ds || !evalRes || history.length === 0) return;
    const last = history[history.length - 1];
    const n = checkpoints.filter((c) => c.datasetId === ds.env.id).length + 1;
    const cp: Checkpoint = {
      id: newId(),
      label: `Checkpoint ${n} — ${ds.env.name}`,
      createdAt: Date.now(),
      datasetId: ds.env.id,
      datasetName: ds.env.name,
      hyper: { ...hyper },
      iterations: last.iter,
      updates: last.updates,
      estQ: last.estQ,
      trueQ: last.trueQ,
      oodRate: last.oodRate,
      loss: last.loss,
      q: new Float64Array(qRef.current),
      history: [...fullHistoryRef.current],
      rngState: rngRef.current.state(),
      experimentId: runRef.current?.id,
    };
    const next = [cp, ...checkpoints];
    const dropped = next.slice(MAX_CHECKPOINTS);
    setCheckpoints(next.slice(0, MAX_CHECKPOINTS));
    setActiveCheckpointId(cp.id);
    checkpointsStore
      .put(cp)
      .then(() => Promise.all(dropped.map((d) => checkpointsStore.delete(d.id))))
      .catch((e) => {
        recordError("storage", e, `saving ${cp.label}`);
        toast.message("Checkpoint kept for this session only — it could not be saved locally.");
      });
    toast.success(`Saved checkpoint at ${last.updates.toLocaleString()} updates`);
  };

  const restoreCheckpoint = (c: Checkpoint) => {
    if (!ds || c.datasetId !== ds.env.id || c.q.length !== ds.env.width * ds.env.height * 4) {
      toast.error(`Switch to ${c.datasetName} to restore this checkpoint.`);
      return;
    }
    setAuto(false);
    qRef.current = new Float64Array(c.q);
    if (c.rngState !== undefined) rngRef.current = resumableRng(c.rngState);
    fullHistoryRef.current = [...c.history];
    setHyper({ ...c.hyper });
    setHistory(c.history.slice(-200));
    setActiveCheckpointId(c.id);
    setVersion((v) => v + 1);
    scheduleSave();
    toast.success(`Restored "${c.label}" — training continues from ${c.updates.toLocaleString()} updates`);
  };

  const deleteCheckpoint = (id: string) => {
    setCheckpoints((cs) => cs.filter((c) => c.id !== id));
    if (activeCheckpointId === id) setActiveCheckpointId(null);
    checkpointsStore.delete(id).catch((e) => recordError("storage", e, "deleting a checkpoint"));
  };

  /** Audit log of the current run: every metric step, log entry, hyperparameter change and guard/crash event. */
  const exportAudit = (format: "json" | "csv" | "md") => {
    const rec = snapshotRun();
    if (!rec || !ds) {
      toast.error("Train at least one step to export the run log.");
      return;
    }
    const entries = buildLogEntries(rec.history, ds, rec.hyper, optimal);
    const audit = buildAuditLog({ ...rec, transitions: ds.transitions.length }, entries, recentErrors(), UPDATES_PER_STEP);
    const base = `audit-${slugify(rec.name)}-${rec.history[rec.history.length - 1].updates}u`;
    if (format === "json") downloadText(`${base}.json`, JSON.stringify(audit, null, 2), "application/json");
    else if (format === "csv") downloadText(`${base}.csv`, metricsCsv(rec.history), "text/csv");
    else downloadText(`${base}.md`, auditMarkdown(audit), "text/markdown");
    toast.success(`Run log exported (${format.toUpperCase()})`);
  };

  const exportModelCard = () => {
    const rec = snapshotRun();
    if (!rec || !ds) {
      toast.error("Train at least one step to generate a model card.");
      return;
    }
    const entries = buildLogEntries(rec.history, ds, rec.hyper, optimal);
    const card = buildModelCard({
      run: rec,
      dataset: { name: ds.env.name, description: ds.env.description, tag: ds.env.tag, transitions: ds.transitions.length, episodes: ds.episodes, groundTruth: ds.env.groundTruth },
      updatesPerStep: UPDATES_PER_STEP,
      anomalies: entries.filter((e) => e.kind === "anomaly").map((e) => `@ ${e.atUpdates.toLocaleString()} updates: ${e.text}`),
      license,
      oodRate: evalRes?.oodRate,
    });
    downloadText(`model-card-${slugify(rec.name)}.md`, card, "text/markdown");
    toast.success("Model card downloaded");
  };

  const [license, setLicense] = useState<LicenseConfig>(DEFAULT_LICENSE);
  const [signingKey, setSigningKey] = useState<SigningKey | null>(null);

  const exportRef = useRef({ ds, hyper, history, evalRes, license, signingKey });
  exportRef.current = { ds, hyper, history, evalRes, license, signingKey };
  const exportModel = useCallback(async (trigger: ExportTrigger) => {
    const { ds: d, hyper: h, evalRes: ev, license: lic, signingKey: key } = exportRef.current;
    const hist = fullHistoryRef.current;
    if (!d || hist.length === 0) return;
    if (isExpired(lic)) {
      toast.error("License has expired — extend or clear the expiry date to export.");
      return;
    }
    if (lic.exportLock && lic.accessKey.length >= 4) {
      const entered = window.prompt("This export is access-controlled. Enter the access key to download:");
      if (entered === null) return;
      if (entered !== lic.accessKey) {
        toast.error("Wrong access key — export blocked.");
        return;
      }
    }
    const exp = buildPolicyExport(qRef.current, d, h, hist, ev, TRAIN_SEED, UPDATES_PER_STEP, trigger);
    const checksum = exp.reconstruction.checksum.value;
    let block = buildLicenseBlock(lic, checksum);
    if (key) {
      try {
        block = await signLicenseBlock(block, checksum, key);
      } catch {
        toast.error("Signing failed — export cancelled. Reload your key backup and try again.");
        return;
      }
    }
    (exp as unknown as Record<string, unknown>).license = block;
    downloadPolicyExport(exp);
    toast.success(
      trigger.kind === "manual"
        ? `Exported policy after ${exp.training.totalUpdates.toLocaleString()} updates`
        : `Threshold ${trigger.threshold.toFixed(2)} reached — exported model at ${exp.training.totalUpdates.toLocaleString()} updates`,
    );
  }, []);
  const exportPolicy = () => void exportModel({ kind: "manual" });

  const last = history[history.length - 1];
  const xs = history.map((h) => h.updates);
  const gap = last ? last.estQ - last.trueQ : 0;

  return (
    <div className="flex min-h-screen flex-col bg-background">
      {/* Top bar */}
      <header className="lab-rise sticky top-0 z-40 border-b border-border bg-background/85 backdrop-blur-md">
        <div className="flex h-14 items-center justify-between gap-4 px-4 lg:px-6">
          <div className="flex items-center gap-5">
            <Logo />
            <div className="hidden items-center gap-1.5 rounded-md border border-border bg-card px-2.5 py-1 text-[11px] text-muted-foreground md:flex">
              <span className="h-1.5 w-1.5 rounded-full bg-success" />
              Offline · no environment interaction
            </div>
          </div>
          <div className="flex items-center gap-2">
            <StorageStatus state={saveState} onNewRun={reset} />
            <Link
              to="/experiments"
              className="flex items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 3h6M10 3v6L4.5 19a1.5 1.5 0 0 0 1.3 2h12.4a1.5 1.5 0 0 0 1.3-2L14 9V3" />
                <path d="M7 15h10" />
              </svg>
              Experiments
            </Link>
            <Link
              to="/sweep"
              className="flex items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3" />
                <path d="M1 14h6M9 8h6M17 16h6" />
              </svg>
              Sweep
            </Link>
            <Link
              to="/compare"
              className="flex items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="7" height="18" rx="1" />
                <rect x="14" y="3" width="7" height="18" rx="1" />
              </svg>
              Compare runs
            </Link>
            {ds && (
              <div className="hidden items-center gap-3 font-mono text-[11px] tabular-nums text-muted-foreground sm:flex">
                <span>{ds.transitions.length.toLocaleString()} transitions</span>
                <span className="text-border">|</span>
                <span>{(last?.updates ?? 0).toLocaleString()} updates</span>
              </div>
            )}
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="grid flex-1 grid-cols-1 gap-4 p-4 lg:grid-cols-[280px_minmax(0,1fr)] lg:p-5 xl:grid-cols-[288px_minmax(0,1fr)_360px]">
        {/* LEFT: configuration */}
        <aside className="space-y-4">
          <Panel title="Dataset" delay={60}>
            <div className="space-y-3">
              <Select
                value={datasetId}
                onValueChange={(v) => {
                  if (v === "custom" && !customDs) {
                    fileRef.current?.click();
                    return;
                  }
                  setDatasetId(v as DatasetId);
                }}
              >
                <SelectTrigger className="w-full px-3.5" aria-label="Dataset">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="z-50">
                  <SelectItem value="halfcheetah-medium">D4RL HalfCheetah-Medium</SelectItem>
                  <SelectItem value="gridworld-random">Gridworld-Random</SelectItem>
                  <SelectItem value="thrusterchain-stall">ThrusterChain v4 (stall)</SelectItem>
                  <SelectItem value="custom">Custom Upload (JSON)</SelectItem>
                </SelectContent>
              </Select>
              {ds && (
                <div className="rounded-lg border border-border bg-surface/60 p-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-[13px] font-medium">{ds.env.name}</span>
                    <span className="shrink-0 rounded border border-border bg-card px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">{ds.env.tag}</span>
                  </div>
                  <p className="mt-1.5 text-[11.5px] leading-relaxed text-muted-foreground">{ds.env.description}</p>
                  <div className="mt-2.5 grid grid-cols-2 gap-2 font-mono text-[11px] tabular-nums">
                    <div><span className="text-muted-foreground">episodes</span> {ds.episodes}</div>
                    <div><span className="text-muted-foreground">r∈</span> [{ds.rewardMin.toFixed(1)}, {ds.rewardMax.toFixed(1)}]</div>
                  </div>
                </div>
              )}
              <div className="flex gap-2">
                <button onClick={() => fileRef.current?.click()} className="flex-1 rounded-md border border-border bg-card px-3 py-2 text-xs font-medium transition-colors duration-200 hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                  Upload JSON
                </button>
                <button onClick={downloadSample} className="flex-1 rounded-md border border-border bg-card px-3 py-2 text-xs font-medium text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                  Sample format
                </button>
              </div>
              <input
                ref={fileRef}
                type="file"
                accept="application/json,.json"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onUpload(f);
                  e.target.value = "";
                }}
              />
            </div>
          </Panel>

          <Panel
            title="Hyperparameters"
            delay={110}
            right={
              <button onClick={() => setHyper(DEFAULT_HYPER)} className="rounded px-1.5 py-0.5 text-[11px] text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                Reset
              </button>
            }
          >
            <div className="space-y-5">
              <HyperSlider label="Alpha (α)" hint="Conservative penalty weight" value={hyper.alpha} min={0} max={5} step={0.1} format={(v) => v.toFixed(1)} onChange={(alpha) => setHyper((h) => ({ ...h, alpha }))} />
              <HyperSlider label="Discount (γ)" hint="Weight on future reward" value={hyper.gamma} min={0.5} max={0.99} step={0.01} format={(v) => v.toFixed(2)} onChange={(gamma) => setHyper((h) => ({ ...h, gamma }))} />
              <HyperSlider label="Learning rate" hint="Step size per update" value={hyper.lr} min={0.01} max={1} step={0.01} format={(v) => v.toFixed(2)} onChange={(lr) => setHyper((h) => ({ ...h, lr }))} />
              <HyperSlider label="Batch size" hint="Transitions per update" value={hyper.batch} min={8} max={512} step={8} format={(v) => String(v)} onChange={(batch) => setHyper((h) => ({ ...h, batch }))} />
            </div>
            <div className={`mt-5 rounded-lg border px-3 py-2.5 text-[11.5px] leading-relaxed ${hyper.alpha === 0 ? "border-warning/40 bg-warning/10 text-foreground" : "border-border bg-surface/60 text-muted-foreground"}`}>
              {hyper.alpha === 0
                ? "α = 0 is plain offline Q-learning. Watch the estimate drift above the ground truth on unseen actions."
                : `CQL loss = TD error + ${hyper.alpha.toFixed(1)} · (logsumexp Q(s,·) − Q(s, a_data)). Higher α pushes down actions the data never tried.`}
            </div>
          </Panel>

          <Panel
            title="Current run"
            delay={130}
            right={
              <span className={`font-mono text-[10px] ${runInfo?.status === "diverged" ? "text-destructive" : runInfo?.status === "training" ? "text-primary" : "text-muted-foreground"}`}>
                {runInfo?.status ?? "—"}
              </span>
            }
          >
            {runInfo ? (
              <RunMetaEditor key={runInfo.id} info={runInfo} updates={last?.updates ?? 0} computeMs={runRef.current?.computeMs ?? 0} onChange={updateRunMeta} />
            ) : (
              <p className="text-[11.5px] text-muted-foreground">Pick a dataset to start a run.</p>
            )}
          </Panel>

          <Panel title="Training alerts" delay={150} right={<span className="font-mono text-[10px] text-muted-foreground">{alertRules.autoPause ? "auto-pause on" : "notify only"}</span>}>
            <AlertsEditor rules={alertRules} onChange={setAlertRules} />
          </Panel>
        </aside>

        {/* CENTER: visual workspace */}
        <section className="lab-rise min-w-0 rounded-xl border border-border bg-card shadow-sm" style={{ animationDelay: "160ms" }}>
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
            <div role="tablist" aria-label="Workspace view" className="flex rounded-lg border border-border bg-surface p-0.5">
              {(["dataset", "policy", "watch"] as const).map((v) => (
                <button
                  key={v}
                  role="tab"
                  aria-selected={view === v}
                  onClick={() => setView(v)}
                  className={`rounded-md px-3.5 py-1.5 text-[13px] font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${view === v ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
                >
                  {v === "dataset" ? "Dataset Visualization" : v === "policy" ? "Policy Evaluation" : "Watch Episodes"}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={exportPolicy}
                disabled={!ds || history.length === 0}
                title={history.length === 0 ? "Train at least one step to export a policy" : "Download trained policy weights + config (JSON)"}
                className="flex items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1.5 text-[13px] font-medium text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground disabled:pointer-events-none disabled:opacity-45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3v12M7 10l5 5 5-5M4 21h16" /></svg>
                Export policy
              </button>
              <button
                onClick={saveCheckpoint}
                disabled={!ds || history.length === 0}
                title={history.length === 0 ? "Train at least one step to save a checkpoint" : "Snapshot the current Q-table + metrics"}
                className="flex items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1.5 text-[13px] font-medium text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground disabled:pointer-events-none disabled:opacity-45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" /><path d="M17 21v-8H7v8M7 3v5h8" /></svg>
                Save checkpoint
              </button>
              <button
                onClick={reset}
                disabled={!ds || history.length === 0}
                title="Start a new run from Q = 0. This run stays saved in Experiments."
                className="rounded-md border border-border bg-card px-3 py-1.5 text-[13px] font-medium text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground disabled:pointer-events-none disabled:opacity-45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                New run
              </button>
              <button onClick={() => setAuto((a) => !a)} disabled={!ds} className={`rounded-md border px-3 py-1.5 text-[13px] font-medium transition-colors duration-200 disabled:opacity-45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${auto ? "border-primary bg-accent text-accent-foreground" : "border-border bg-card text-muted-foreground hover:bg-surface hover:text-foreground"}`}>
                {auto ? "Pause" : "Auto-train"}
              </button>
              <button onClick={step} disabled={!ds} className="flex items-center gap-2 rounded-md bg-primary px-4 py-1.5 text-[13px] font-semibold text-primary-foreground shadow-sm transition-all duration-200 hover:bg-primary/90 disabled:opacity-45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-card">
                <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><path d="M2 1.5v9l6-4.5zM9 1.5h1.5v9H9z" /></svg>
                Step Training
              </button>
            </div>
          </div>

          <div className="p-4">
            <PanelErrorBoundary name="Workspace" resetKey={`${ds?.env.id ?? ""}:${view}`}>
            {working ? (
              <div className="flex min-h-[420px] flex-col items-center justify-center rounded-lg border border-dashed border-border bg-surface/50 text-center lab-grid-bg" role="status" aria-live="polite">
                <div className="w-full max-w-sm px-6">
                  <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-lg border border-border bg-card text-primary">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" className="animate-spin"><path d="M21 12a9 9 0 1 1-6.2-8.56" /></svg>
                  </div>
                  <div className="text-sm font-medium">Processing in background worker</div>
                  <p className="mt-1 min-h-[16px] text-xs text-muted-foreground">{working.label}</p>
                  <div className="mt-4 flex items-center gap-3">
                    <Progress value={working.pct} className="h-1.5" aria-label="Import progress" />
                    <span className="shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">{Math.round(working.pct)}%</span>
                  </div>
                  <p className="mt-3 text-[11px] text-muted-foreground">The UI stays responsive — parsing and physics run off the main thread.</p>
                </div>
              </div>
            ) : !ds || !evalRes ? (
              <div className="flex min-h-[420px] flex-col items-center justify-center rounded-lg border border-dashed border-border bg-surface/50 text-center lab-grid-bg">
                <div className="max-w-xs px-6">
                  <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-lg border border-border bg-card text-muted-foreground">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M12 16V4M7 9l5-5 5 5M4 20h16" /></svg>
                  </div>
                  <div className="text-sm font-medium">No custom dataset loaded</div>
                  <p className="mt-1 text-xs text-muted-foreground">Upload a JSON file of (s, a, r, s′) transitions on a grid, or switch to a built-in dataset.</p>
                  <button onClick={() => fileRef.current?.click()} className="mt-4 rounded-md bg-primary px-4 py-2 text-xs font-semibold text-primary-foreground hover:bg-primary/90">
                    Upload JSON
                  </button>
                </div>
              </div>
            ) : (
              <>
                <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h1 className="font-display text-lg font-semibold tracking-tight">
                      {view === "dataset" ? "Logged transitions (S, A, R, S′)" : view === "policy" ? "Learned value function Q(s, a)" : "Episode playback"}
                    </h1>
                    <p className="text-xs text-muted-foreground">
                      {view === "dataset"
                        ? "Arrows are the transitions the behavior policy recorded. Color = mean reward, thickness = frequency."
                        : view === "policy"
                          ? `Heatmap of max_a Q(s,a) · arrows show the greedy policy · dashed outline = action never seen in data.`
                          : "Fresh rollouts of the real environment — watch the trained policy act, or replay the behavior recorder that produced the dataset."}
                    </p>
                  </div>
                  {view === "policy" && (
                    <div className="flex rounded-md border border-border bg-surface p-0.5 text-[11px]">
                      {(["estimate", "truth", "error"] as const).map((m) => (
                        <button key={m} onClick={() => setHeat(m)} className={`rounded px-2.5 py-1 font-medium transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${heat === m ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>
                          {m === "estimate" ? "Q estimate" : m === "truth" ? "Ground truth V^π" : "Error"}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {view === "watch" ? (
                  <EpisodePlayer ds={ds} Q={qRef.current} version={version} trained={history.length > 0} />
                ) : (
                  <StateGrid ds={ds} mode={view} Q={qRef.current} evalRes={evalRes} heat={heat} version={version} />
                )}

                <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px] text-muted-foreground">
                  {view === "watch" ? (
                    <span>
                      Playback simulates the exact float64 stall-world system live — it is NOT part of training. The agent still learns only from the frozen dataset; this view just lets you see what its policy would do if deployed.
                    </span>
                  ) : view === "dataset" ? (
                    <>
                      <div className="flex items-center gap-2">
                        <span>reward</span>
                        <span className="font-mono tabular-nums">{ds.rewardMin.toFixed(1)}</span>
                        <div className="h-2 w-28 rounded-full" style={{ background: `linear-gradient(90deg, ${rewardColor(ds.rewardMin, ds.rewardMin, ds.rewardMax)}, ${rewardColor(0, ds.rewardMin, ds.rewardMax)}, ${rewardColor(ds.rewardMax, ds.rewardMin, ds.rewardMax)})` }} />
                        <span className="font-mono tabular-nums">{ds.rewardMax.toFixed(1)}</span>
                      </div>
                      <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm bg-primary/30" /> state density</span>
                      <span className="flex items-center gap-1.5"><span className="font-mono font-semibold text-foreground">G</span> goal</span>
                      <span className="flex items-center gap-1.5"><span className="font-mono font-semibold text-foreground">×</span> hazard / failure</span>
                    </>
                  ) : (
                    <>
                      <div className="flex items-center gap-2">
                        <span>low</span>
                        <div className="h-2 w-28 rounded-full" style={{ background: `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((t) => heatColor(t)).join(",")})` }} />
                        <span>high</span>
                      </div>
                      <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm border-[1.5px] border-dashed border-warning" /> out-of-distribution action</span>
                      {history.length === 0 && <span className="text-foreground">Q starts at 0 — press Step Training.</span>}
                    </>
                  )}
                </div>
              </>
            )}
            </PanelErrorBoundary>
          </div>
        </section>

        {/* RIGHT: performance dashboard */}
        <aside className="space-y-4 lg:col-span-2 xl:col-span-1">
          <Panel title="Estimated Q vs. Ground Truth" delay={210} right={<span className="font-mono text-[10px] text-muted-foreground">avg over data states</span>}>
            <LineChart
              xs={xs}
              empty="Press Step Training to start plotting value estimates."
              series={[
                { name: "Estimated Q", color: "var(--chart-1)", values: history.map((h) => h.estQ) },
                { name: "Ground truth", color: "var(--chart-2)", values: history.map((h) => h.trueQ), dashed: true },
              ]}
            />
            <div className="mt-2.5">
              <Legend
                items={[
                  { name: "Estimated Q", color: "var(--chart-1)", value: last?.estQ.toFixed(2) },
                  { name: "Ground truth V^π", color: "var(--chart-2)", value: last?.trueQ.toFixed(2), dashed: true },
                ]}
              />
            </div>
          </Panel>

          <Panel title="Policy Loss" delay={250} right={<span className="font-mono text-[10px] text-muted-foreground">TD + α·CQL</span>}>
            <LineChart
              xs={xs}
              height={120}
              empty="Loss appears after the first training step."
              series={[
                { name: "Total", color: "var(--chart-1)", values: history.map((h) => h.loss) },
                { name: "TD", color: "var(--chart-3)", values: history.map((h) => h.tdLoss), dashed: true },
              ]}
            />
            <div className="mt-2.5">
              <Legend items={[{ name: "Total loss", color: "var(--chart-1)", value: last?.loss.toFixed(3) }, { name: "TD error²", color: "var(--chart-3)", value: last?.tdLoss.toFixed(3), dashed: true }]} />
            </div>
          </Panel>

          <Panel title="Distributional Shift" delay={290}>
            <div className="grid grid-cols-2 gap-2">
              <Stat label="Overestimation" value={last ? `${gap >= 0 ? "+" : ""}${gap.toFixed(2)}` : "—"} tone={last ? (gap > 1 ? "bad" : gap > 0.25 ? "warn" : "ok") : undefined} />
              <Stat label="OOD actions" value={last ? `${(last.oodRate * 100).toFixed(1)}%` : "—"} tone={last ? (last.oodRate > 0.15 ? "warn" : "ok") : undefined} />
              <Stat label="Policy ↔ data NLL" value={last ? last.divergence.toFixed(2) : "—"} />
              <Stat label="Optimal V*(start)" value={optimal.toFixed(2)} />
            </div>
            <div className="mt-3">
              <LineChart
                xs={xs}
                height={110}
                empty="Shift trend appears after the first training step."
                series={[
                  { name: "Overestimation", color: "var(--chart-4)", values: history.map((h) => h.estQ - h.trueQ) },
                  { name: "OOD %", color: "var(--chart-2)", values: history.map((h) => h.oodRate * 10), dashed: true },
                ]}
              />
              <div className="mb-3 mt-2">
                <Legend items={[{ name: "Overestimation", color: "var(--chart-4)", value: last ? gap.toFixed(2) : undefined }, { name: "OOD rate (x10)", color: "var(--chart-2)", value: last ? (last.oodRate * 10).toFixed(2) : undefined, dashed: true }]} />
              </div>
              <div className="mb-1.5 flex items-center justify-between text-[11px] text-muted-foreground">
                <span>Greedy actions outside data support</span>
                <span className="font-mono tabular-nums text-foreground">{evalRes ? evalRes.oodStates.size : 0} states</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-surface">
                <div className="h-full rounded-full bg-warning transition-all duration-300" style={{ width: `${Math.min(100, (last?.oodRate ?? 0) * 100)}%` }} />
              </div>
              <p className="mt-3 text-[11.5px] leading-relaxed text-muted-foreground">
                {!last
                  ? "Train to measure how far the learned policy drifts from the behavior data."
                  : gap > 0.5 && last.oodRate > 0.05
                    ? "The critic is optimistic about actions the dataset never tried — classic extrapolation error. Raise α to stay closer to the data."
                    : "Estimates stay close to the true return of the learned policy: the conservative penalty is keeping the policy in-distribution."}
              </p>
            </div>
          </Panel>

          <Panel title="Model Export" delay={310} right={<span className="font-mono text-[10px] text-muted-foreground">reward-gated</span>}>
            <ExportPanel ds={ds} history={history} optimal={optimal} onExport={exportModel} />
          </Panel>

          <Panel title="Licensing & Access Control" delay={315} right={<span className="font-mono text-[10px] text-muted-foreground">{license.exportLock && license.accessKey.length >= 4 ? "locked" : "open"}{signingKey ? " · signed" : ""}</span>}>
            <LicensePanel cfg={license} onChange={setLicense} signingKey={signingKey} onSigningKeyChange={setSigningKey} />
          </Panel>

          <Panel
            title="Experiment Log"
            delay={320}
            right={
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button
                    type="button"
                    disabled={history.length === 0}
                    className="rounded border border-border bg-card px-2 py-0.5 text-[10.5px] font-medium text-muted-foreground transition-colors hover:bg-surface hover:text-foreground disabled:opacity-45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    title={history.length === 0 ? "Train at least one step to export the run log" : "Download the full run log for auditing"}
                  >
                    Export log
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-60">
                  <DropdownMenuItem onSelect={() => exportAudit("json")}>Full audit (JSON) — every step, log, change, error</DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => exportAudit("csv")}>Metrics per step (CSV)</DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => exportAudit("md")}>Readable report (Markdown)</DropdownMenuItem>
                  <DropdownMenuItem onSelect={exportModelCard}>Model card (Markdown) — what the policy is and where not to trust it</DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            }
          >
            <ExperimentLog history={history} ds={ds} hyper={hyper} optimal={optimal} />
          </Panel>

          <Panel title="Researcher Collaboration" delay={325} right={<span className="font-mono text-[10px] text-muted-foreground">threads · share</span>}>
            <CollaborationPanel history={history} ds={ds} hyper={hyper} optimal={optimal} />
          </Panel>

          <Panel title="Checkpoint History" delay={330} right={<span className="font-mono text-[10px] text-muted-foreground">{checkpoints.length}/20 saved</span>}>
            <CheckpointHistory
              checkpoints={checkpoints}
              activeId={activeCheckpointId}
              currentDatasetId={ds?.env.id ?? ""}
              currentQLength={ds ? ds.env.width * ds.env.height * 4 : 0}
              onRestore={restoreCheckpoint}
              onDelete={deleteCheckpoint}
            />
          </Panel>
        </aside>
      </main>
    </div>
  );
}
