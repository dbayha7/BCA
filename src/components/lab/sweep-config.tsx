// Sweep configuration panel: define learning-rate and conservative-weight
// ranges, training budget, and early-stopping criteria.

import { DEFAULT_SWEEP, type SweepDatasetId, type SweepSettings } from "@/lib/sweep";

const DATASETS: { id: SweepDatasetId; name: string }[] = [
  { id: "gridworld-random", name: "Gridworld-Random (10×10)" },
  { id: "halfcheetah-medium", name: "HalfCheetah-Medium (12×9)" },
  { id: "thrusterchain-stall", name: "ThrusterChain-Stall (11×9)" },
];

function Num({
  label,
  value,
  min,
  max,
  step,
  onChange,
  disabled,
  mono = true,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  disabled?: boolean;
  mono?: boolean;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-[11px] font-medium text-muted-foreground">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v)) onChange(Math.max(min, Math.min(max, v)));
        }}
        className={`h-9 rounded-md border border-input bg-card px-3 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50 ${mono ? "font-mono tabular-nums" : ""}`}
      />
    </label>
  );
}

export function SweepConfig({
  settings,
  onChange,
  disabled,
}: {
  settings: SweepSettings;
  onChange: (s: SweepSettings) => void;
  disabled: boolean;
}) {
  const set = (patch: Partial<SweepSettings>) => onChange({ ...settings, ...patch });
  const combos = settings.lrSteps * settings.alphaSteps;

  return (
    <div className="space-y-5">
      <div className="space-y-1.5">
        <span className="text-[11px] font-medium text-muted-foreground">Dataset</span>
        <select
          value={settings.datasetId}
          disabled={disabled}
          onChange={(e) => set({ datasetId: e.target.value as SweepDatasetId })}
          className="flex h-9 w-full items-center justify-between gap-2 rounded-md border border-input bg-card px-3.5 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
        >
          {DATASETS.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}
            </option>
          ))}
        </select>
      </div>

      <fieldset className="space-y-3 rounded-lg border border-border bg-card/50 p-3.5">
        <legend className="px-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          Learning rate range (log-spaced)
        </legend>
        <div className="grid grid-cols-3 gap-2.5">
          <Num label="Min" value={settings.lrMin} min={0.001} max={settings.lrMax} step={0.01} onChange={(v) => set({ lrMin: v })} disabled={disabled} />
          <Num label="Max" value={settings.lrMax} min={settings.lrMin} max={1} step={0.01} onChange={(v) => set({ lrMax: v })} disabled={disabled} />
          <Num label="Steps" value={settings.lrSteps} min={1} max={8} step={1} onChange={(v) => set({ lrSteps: Math.round(v) })} disabled={disabled} />
        </div>
      </fieldset>

      <fieldset className="space-y-3 rounded-lg border border-border bg-card/50 p-3.5">
        <legend className="px-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          Conservative weight α range (linear)
        </legend>
        <div className="grid grid-cols-3 gap-2.5">
          <Num label="Min" value={settings.alphaMin} min={0} max={settings.alphaMax} step={0.25} onChange={(v) => set({ alphaMin: v })} disabled={disabled} />
          <Num label="Max" value={settings.alphaMax} min={settings.alphaMin} max={10} step={0.25} onChange={(v) => set({ alphaMax: v })} disabled={disabled} />
          <Num label="Steps" value={settings.alphaSteps} min={1} max={8} step={1} onChange={(v) => set({ alphaSteps: Math.round(v) })} disabled={disabled} />
        </div>
      </fieldset>

      <div className="grid grid-cols-2 gap-2.5">
        <Num label="Updates per run" value={settings.totalUpdates} min={200} max={6000} step={100} onChange={(v) => set({ totalUpdates: Math.round(v) })} disabled={disabled} />
        <Num label="Discount γ" value={settings.gamma} min={0.5} max={0.999} step={0.01} onChange={(v) => set({ gamma: v })} disabled={disabled} />
      </div>

      <fieldset className="space-y-3 rounded-lg border border-border bg-card/50 p-3.5">
        <legend className="px-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Early stopping</legend>
        <label className="flex items-center gap-2.5 text-sm text-foreground">
          <input
            type="checkbox"
            checked={settings.earlyStop}
            disabled={disabled}
            onChange={(e) => set({ earlyStop: e.target.checked })}
            className="h-4 w-4 accent-[var(--primary)]"
          />
          Prune poorly performing runs automatically
        </label>
        <div className="grid grid-cols-2 gap-2.5">
          <Num
            label="Grace period (% of run)"
            value={Math.round(settings.pruneAfterFrac * 100)}
            min={10}
            max={90}
            step={5}
            onChange={(v) => set({ pruneAfterFrac: v / 100 })}
            disabled={disabled || !settings.earlyStop}
          />
          <Num
            label="Prune gap (true return)"
            value={settings.pruneGap}
            min={0.5}
            max={20}
            step={0.5}
            onChange={(v) => set({ pruneGap: v })}
            disabled={disabled || !settings.earlyStop}
          />
        </div>
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          After the grace period, a run is stopped early if its true return trails the best run seen at the same training checkpoint by more than the prune gap.
        </p>
      </fieldset>

      <div className="flex items-center justify-between rounded-md border border-border bg-muted px-3 py-2 text-[12px]">
        <span className="text-muted-foreground">Configurations</span>
        <span className="font-mono font-semibold tabular-nums text-foreground">
          {settings.lrSteps} × {settings.alphaSteps} = {combos}
        </span>
      </div>

      <button
        type="button"
        disabled={disabled}
        onClick={() => onChange({ ...DEFAULT_SWEEP })}
        className="w-full rounded-md border border-border bg-card px-5 py-2 text-xs font-medium text-muted-foreground transition-colors duration-200 hover:text-foreground disabled:opacity-50"
      >
        Reset to defaults
      </button>
    </div>
  );
}
