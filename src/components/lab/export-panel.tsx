import { useEffect, useMemo, useRef, useState } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import type { Dataset, Metrics } from "@/lib/rl";
import { estimateExportKB, type ExportTrigger } from "@/lib/export-policy";

type MetricKey = "trueReturn" | "estimatedQ";

const METRIC_LABEL: Record<MetricKey, string> = {
  trueReturn: "True return V^π",
  estimatedQ: "Estimated Q",
};

const pick = (h: Metrics, m: MetricKey) => (m === "trueReturn" ? h.trueQ : h.estQ);

function defaultThreshold(optimal: number) {
  return Math.round(Math.max(0, optimal * 0.5) * 10) / 10;
}

export function ExportPanel({
  ds,
  history,
  optimal,
  onExport,
}: {
  ds: Dataset | null;
  history: Metrics[];
  optimal: number;
  onExport: (trigger: ExportTrigger) => void;
}) {
  const [metric, setMetric] = useState<MetricKey>("trueReturn");
  const [threshold, setThreshold] = useState(() => defaultThreshold(optimal));
  const [draft, setDraft] = useState(() => String(defaultThreshold(optimal)));
  const [auto, setAuto] = useState(false);
  const [exportedKey, setExportedKey] = useState<string | null>(null);
  const dsId = ds?.env.id ?? "";

  // New dataset → sensible default threshold for its reward scale.
  const lastDs = useRef(dsId);
  useEffect(() => {
    if (lastDs.current === dsId) return;
    lastDs.current = dsId;
    const t = defaultThreshold(optimal);
    setThreshold(t);
    setDraft(String(t));
    setExportedKey(null);
  }, [dsId, optimal]);

  // Reset Q clears the auto-export latch.
  useEffect(() => {
    if (history.length === 0) setExportedKey(null);
  }, [history.length]);

  const last = history[history.length - 1];
  const current = last ? pick(last, metric) : null;
  const best = useMemo(() => (history.length ? Math.max(...history.map((h) => pick(h, metric))) : null), [history, metric]);
  const crossing = useMemo(() => history.find((h) => pick(h, metric) >= threshold) ?? null, [history, metric, threshold]);
  const reached = current !== null && current >= threshold;
  const key = `${dsId}|${metric}|${threshold}`;

  useEffect(() => {
    if (!auto || !reached || !last || exportedKey === key) return;
    setExportedKey(key);
    onExport({
      kind: "threshold-auto",
      metric,
      threshold,
      reachedAtUpdates: crossing?.updates ?? last.updates,
      valueAtExport: pick(last, metric),
    });
  }, [auto, reached, last, exportedKey, key, metric, threshold, crossing, onExport]);

  const commitDraft = () => {
    const v = Number(draft);
    if (Number.isFinite(v)) {
      setThreshold(v);
      setExportedKey(null);
    } else setDraft(String(threshold));
  };

  const progress = (() => {
    if (current === null) return 0;
    if (reached) return 100;
    const floor = Math.min(0, ...history.map((h) => pick(h, metric)));
    const span = threshold - floor || 1;
    return Math.max(2, Math.min(99, ((current - floor) / span) * 100));
  })();

  const exportNow = () => {
    if (!last) return;
    onExport(
      reached
        ? { kind: "threshold-manual", metric, threshold, reachedAtUpdates: crossing?.updates ?? last.updates, valueAtExport: pick(last, metric) }
        : { kind: "manual" },
    );
  };

  if (!ds) {
    return <p className="text-[11.5px] text-muted-foreground">Load a dataset to configure model export.</p>;
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="mb-1.5 block text-[11px] text-muted-foreground">Gate metric</label>
          <Select value={metric} onValueChange={(v) => { setMetric(v as MetricKey); setExportedKey(null); }}>
            <SelectTrigger className="h-9 px-3 text-[12.5px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="trueReturn">True return V^π</SelectItem>
              <SelectItem value="estimatedQ">Estimated Q</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <label htmlFor="export-threshold" className="mb-1.5 block text-[11px] text-muted-foreground">Reward threshold ≥</label>
          <input
            id="export-threshold"
            type="number"
            step={0.1}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commitDraft}
            onKeyDown={(e) => e.key === "Enter" && commitDraft()}
            className="h-9 w-full rounded-md border border-input bg-background px-3 font-mono text-[12.5px] tabular-nums text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
        </div>
      </div>

      <div>
        <div className="mb-1.5 flex items-center justify-between text-[11px]">
          <span className="text-muted-foreground">{METRIC_LABEL[metric]}</span>
          <span className="font-mono tabular-nums text-foreground">
            {current === null ? "—" : current.toFixed(2)} <span className="text-muted-foreground">/ {threshold.toFixed(2)}</span>
          </span>
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-surface">
          <div className={`h-full rounded-full transition-all duration-300 ${reached ? "bg-success" : "bg-primary"}`} style={{ width: `${progress}%` }} />
        </div>
        <div className="mt-1.5 flex items-center justify-between text-[10.5px] text-muted-foreground">
          <span>best {best === null ? "—" : best.toFixed(2)}</span>
          <span>
            {crossing ? `first crossed at ${crossing.updates.toLocaleString()} upd` : history.length ? "not reached yet" : "train to begin"}
          </span>
        </div>
      </div>

      <label className="flex items-center justify-between gap-3 rounded-lg border border-border bg-surface/60 px-3 py-2.5">
        <span className="text-[11.5px] leading-snug text-foreground">
          Auto-export when reached
          <span className="block text-[10.5px] text-muted-foreground">Downloads once per threshold, mid-training</span>
        </span>
        <Switch checked={auto} onCheckedChange={(v) => { setAuto(v); setExportedKey(null); }} aria-label="Auto-export when threshold is reached" />
      </label>

      <button
        onClick={exportNow}
        disabled={!last || !reached}
        className="flex w-full items-center justify-center gap-2 rounded-md bg-primary px-5 py-2.5 text-[13px] font-semibold text-primary-foreground shadow-sm transition-all duration-200 hover:bg-primary/90 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3v12M7 10l5 5 5-5M4 21h16" /></svg>
        {reached ? "Export model" : "Locked until threshold"}
      </button>

      <p className="text-[10.5px] leading-relaxed text-muted-foreground">
        {exportedKey === key && auto ? "✓ Auto-exported at this threshold. " : ""}
        JSON · ~{estimateExportKB(ds)} KB · Q-table, greedy policy, V^π, state/action encoding,
        {ds.env.id === "thrusterchain-stall" ? " 13-dim → grid projection + thruster commands," : " env rules,"} empirical MDP, hyperparameters & checksum.
      </p>
    </div>
  );
}
