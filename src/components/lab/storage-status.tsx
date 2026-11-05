// Header indicator for local saving, with the storage controls behind it: how much space the
// lab uses, asking the browser to keep the data, the error log, and clearing everything.

import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { clearAllLabData, errorsStore, requestPersistence, storageInfo, type StorageInfo } from "@/lib/store";
import { onError, recentErrors, type ErrorEntry } from "@/lib/error-log";

export type SaveState = { kind: "idle" } | { kind: "saving" } | { kind: "saved"; at: number } | { kind: "failed"; message: string } | { kind: "unavailable" };

function kb(n?: number) {
  if (n === undefined) return "—";
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function clock(ts: number) {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function StorageStatus({ state, onNewRun }: { state: SaveState; onNewRun: () => void }) {
  const [info, setInfo] = useState<StorageInfo | null>(null);
  const [errors, setErrors] = useState<ErrorEntry[]>(() => recentErrors());
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let alive = true;
    errorsStore
      .all()
      .then((saved) => {
        if (!alive) return;
        setErrors((cur) => {
          const ids = new Set(cur.map((e) => e.id));
          return [...cur, ...saved.filter((e) => !ids.has(e.id))].sort((a, b) => b.at - a.at).slice(0, 100);
        });
      })
      .catch(() => undefined);
    const off = onError((e) => setErrors((cur) => [e, ...cur].slice(0, 100)));
    return () => {
      alive = false;
      off();
    };
  }, []);

  useEffect(() => {
    if (open) void storageInfo().then(setInfo);
  }, [open, state]);

  const dot =
    state.kind === "saved" ? "bg-success" : state.kind === "saving" ? "bg-primary animate-pulse" : state.kind === "failed" || state.kind === "unavailable" ? "bg-destructive" : "bg-muted-foreground";
  const label =
    state.kind === "saved"
      ? `Saved ${clock(state.at)}`
      : state.kind === "saving"
        ? "Saving…"
        : state.kind === "failed"
          ? "Not saved"
          : state.kind === "unavailable"
            ? "Saving off"
            : "Local saving";
  const recentCount = errors.filter((e) => Date.now() - e.at < 24 * 3600 * 1000).length;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="flex items-center gap-1.5 rounded-md border border-border bg-card px-2.5 py-1.5 text-[11px] font-medium text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label={`Local saving: ${label}. Open storage settings`}
        >
          <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
          <span className="hidden tabular-nums sm:inline">{label}</span>
          {recentCount > 0 && <span className="rounded bg-destructive/15 px-1 font-mono text-[10px] text-destructive">{recentCount}</span>}
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[340px] p-0">
        <div className="border-b border-border px-4 py-3">
          <div className="text-[13px] font-semibold">Saved in this browser</div>
          <p className="mt-1 text-[11.5px] leading-relaxed text-muted-foreground">
            Runs, full training history, checkpoints, comments and uploads save automatically to this browser's local storage. Nothing is uploaded.
          </p>
          {state.kind === "failed" && <p className="mt-2 text-[11.5px] text-destructive">Last save failed: {state.message}</p>}
          {state.kind === "unavailable" && <p className="mt-2 text-[11.5px] text-destructive">This browser blocks local storage for the lab (private window or blocked site data). Work is not being saved.</p>}
          <div className="mt-2.5 grid grid-cols-2 gap-2 font-mono text-[11px] tabular-nums">
            <div><span className="text-muted-foreground">used</span> {kb(info?.usage)}</div>
            <div><span className="text-muted-foreground">quota</span> {kb(info?.quota)}</div>
            <div className="col-span-2"><span className="text-muted-foreground">protected from eviction</span> {info?.persisted === undefined ? "—" : info.persisted ? "yes" : "no"}</div>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <button type="button" onClick={onNewRun} className="rounded-md bg-primary px-2.5 py-1 text-[11px] font-semibold text-primary-foreground hover:bg-primary/90">
              Start a new run
            </button>
            {info?.persisted === false && (
              <button
                type="button"
                onClick={() =>
                  void requestPersistence().then((ok) => {
                    toast[ok ? "success" : "message"](ok ? "The browser will keep the lab's data." : "The browser declined; it may still clear data under storage pressure.");
                    void storageInfo().then(setInfo);
                  })
                }
                className="rounded-md border border-border bg-card px-2.5 py-1 text-[11px] font-medium hover:bg-surface"
              >
                Keep data permanently
              </button>
            )}
            <button
              type="button"
              onClick={() => {
                if (!window.confirm("Delete every saved run, checkpoint, comment, upload and error from this browser? This cannot be undone.")) return;
                void clearAllLabData()
                  .then(() => {
                    toast.success("All lab data cleared. Reloading…");
                    setTimeout(() => window.location.reload(), 600);
                  })
                  .catch((e) => toast.error(e instanceof Error ? e.message : "Could not clear lab data."));
              }}
              className="rounded-md border border-destructive/40 bg-card px-2.5 py-1 text-[11px] font-medium text-destructive hover:bg-destructive/10"
            >
              Clear all lab data
            </button>
          </div>
        </div>
        <div className="px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="text-[12px] font-semibold">Error log</div>
            {errors.length > 0 && (
              <button
                type="button"
                onClick={() => void errorsStore.clear().then(() => setErrors([])).catch(() => setErrors([]))}
                className="text-[11px] text-muted-foreground hover:text-foreground"
              >
                Clear
              </button>
            )}
          </div>
          {errors.length === 0 ? (
            <p className="mt-1.5 text-[11.5px] text-muted-foreground">No crashes or guard trips recorded.</p>
          ) : (
            <ul className="mt-2 max-h-56 space-y-1.5 overflow-y-auto pr-1">
              {errors.slice(0, 40).map((e) => (
                <li key={e.id} className="rounded-md border border-border bg-surface/60 px-2.5 py-1.5">
                  <div className="flex items-center justify-between gap-2 font-mono text-[10px] text-muted-foreground">
                    <span className="uppercase tracking-wider">{e.source}</span>
                    <span>{new Date(e.at).toLocaleString()}</span>
                  </div>
                  <div className="mt-0.5 break-words text-[11.5px]">{e.message}</div>
                  {e.context && <div className="mt-0.5 break-words text-[10.5px] text-muted-foreground">{e.context}</div>}
                </li>
              ))}
            </ul>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
