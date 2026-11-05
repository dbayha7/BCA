// Checkpoint history log: saved Q-table snapshots with metrics + restore/delete actions.

import type { Hyper, Metrics } from "@/lib/rl";
import { VirtualList } from "@/components/lab/virtual-list";

export interface Checkpoint {
  id: string;
  label: string;
  createdAt: number;
  datasetId: string;
  datasetName: string;
  hyper: Hyper;
  iterations: number;
  updates: number;
  estQ: number;
  trueQ: number;
  oodRate: number;
  loss: number;
  /** Deep copy of the Q-table at save time. */
  q: Float64Array;
  /** Metrics history up to the checkpoint, so charts restore too. */
  history: Metrics[];
  /** Minibatch stream state at save time, so training resumes the exact same stream. */
  rngState?: number;
  /** Run the checkpoint was taken from. */
  experimentId?: string;
}

function timeAgo(ts: number) {
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (s < 10) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  return `${h}h ago`;
}

export function CheckpointHistory({
  checkpoints,
  activeId,
  currentDatasetId,
  currentQLength,
  onRestore,
  onDelete,
}: {
  checkpoints: Checkpoint[];
  activeId: string | null;
  currentDatasetId: string;
  currentQLength: number;
  onRestore: (c: Checkpoint) => void;
  onDelete: (id: string) => void;
}) {
  if (checkpoints.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border bg-surface/50 px-4 py-6 text-center">
        <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-md border border-border bg-card text-muted-foreground">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
            <path d="M17 21v-8H7v8M7 3v5h8" />
          </svg>
        </div>
        <div className="text-xs font-medium">No checkpoints yet</div>
        <p className="mt-1 max-w-[220px] text-[11px] leading-relaxed text-muted-foreground">
          Press <span className="font-medium text-foreground">Save checkpoint</span> during training to snapshot the Q-table, then restore it here any time.
        </p>
      </div>
    );
  }

  const ROW = 128;
  return (
    <VirtualList
      items={checkpoints}
      rowHeight={ROW}
      height={Math.min(checkpoints.length, 3) * ROW}
      scrollKey={`ckpt-log-${currentDatasetId}`}
      ariaLabel="Checkpoint history"
      className="-mx-1 px-1"
      renderRow={(c) => {
        const compatible = c.datasetId === currentDatasetId && c.q.length === currentQLength;
        const active = c.id === activeId;
        const gap = c.estQ - c.trueQ;
        return (
          <div
            className={`mb-2 rounded-lg border px-3 py-2.5 transition-colors duration-200 ${active ? "border-primary/60 bg-accent/50" : "border-border bg-surface/60"}`}
          >
            <div className="flex items-center justify-between gap-2">
              <div className="min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="truncate text-[12.5px] font-medium">{c.label}</span>
                  {active && <span className="shrink-0 rounded border border-primary/40 bg-accent px-1 py-px text-[9px] font-semibold uppercase tracking-wide text-accent-foreground">active</span>}
                </div>
                <div className="mt-0.5 truncate font-mono text-[10px] tabular-nums text-muted-foreground">
                  {c.datasetName} · {c.updates.toLocaleString()} updates · {timeAgo(c.createdAt)}
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-1.5">
                <button
                  onClick={() => onRestore(c)}
                  disabled={!compatible || active}
                  title={
                    active
                      ? "This checkpoint is currently loaded"
                      : compatible
                        ? "Load this Q-table into the active engine"
                        : `Switch to ${c.datasetName} to restore this checkpoint`
                  }
                  className="rounded-md border border-border bg-card px-2.5 py-1 text-[11px] font-medium transition-colors duration-200 hover:bg-surface disabled:pointer-events-none disabled:opacity-45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  Restore
                </button>
                <button
                  onClick={() => onDelete(c.id)}
                  aria-label={`Delete checkpoint ${c.label}`}
                  className="flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground transition-colors duration-200 hover:bg-destructive/10 hover:text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                  </svg>
                </button>
              </div>
            </div>
            <div className="mt-2 grid grid-cols-4 gap-1.5 font-mono text-[10px] tabular-nums">
              <div className="rounded border border-border bg-card px-1.5 py-1">
                <div className="text-[9px] uppercase tracking-wider text-muted-foreground">Est Q</div>
                <div className="mt-px">{c.estQ.toFixed(2)}</div>
              </div>
              <div className="rounded border border-border bg-card px-1.5 py-1">
                <div className="text-[9px] uppercase tracking-wider text-muted-foreground">True</div>
                <div className="mt-px">{c.trueQ.toFixed(2)}</div>
              </div>
              <div className="rounded border border-border bg-card px-1.5 py-1">
                <div className="text-[9px] uppercase tracking-wider text-muted-foreground">Gap</div>
                <div className={`mt-px ${gap > 1 ? "text-destructive" : gap > 0.25 ? "text-warning" : "text-success"}`}>
                  {gap >= 0 ? "+" : ""}
                  {gap.toFixed(2)}
                </div>
              </div>
              <div className="rounded border border-border bg-card px-1.5 py-1">
                <div className="text-[9px] uppercase tracking-wider text-muted-foreground">OOD</div>
                <div className="mt-px">{(c.oodRate * 100).toFixed(1)}%</div>
              </div>
            </div>
            <div className="mt-1.5 truncate font-mono text-[10px] tabular-nums text-muted-foreground">
              α {c.hyper.alpha.toFixed(1)} · γ {c.hyper.gamma.toFixed(2)} · lr {c.hyper.lr.toFixed(2)} · batch {c.hyper.batch}
            </div>
          </div>
        );
      }}
    />
  );
}
