// Researcher collaboration: comment threads on flagged training anomalies/milestones,
// plus shareable experiment bundles (results + log + discussion) that teammates can
// import to review a run and merge their comments back in.

import { useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { buildLogEntries, type LogEntry } from "@/components/lab/experiment-log";
import {
  buildShareBundle,
  bundleToMarkdown,
  initialsOf,
  isDiscussable,
  mergeThreads,
  parseShareBundle,
  relTime,
  threadFromEntry,
  threadKeyOf,
  toneOf,
  type SharedRun,
  type Thread,
  type Threads,
} from "@/lib/collab";
import type { Dataset, Hyper, Metrics } from "@/lib/rl";

const inputCls =
  "h-9 w-full rounded-md border border-input bg-background px-3 text-[12.5px] text-foreground placeholder:text-muted-foreground/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";
const btn =
  "inline-flex h-8 items-center justify-center gap-1.5 rounded-md border border-border bg-card px-3 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50";
const primaryBtn =
  "inline-flex h-8 items-center justify-center gap-1.5 rounded-md bg-primary px-3 text-[11px] font-semibold text-primary-foreground shadow-sm transition-all hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50";

type Tab = "discuss" | "share" | "inbox";
type Filter = "all" | "open" | "resolved";

function Avatar({ name }: { name: string }) {
  return (
    <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[9.5px] font-bold ${toneOf(name)}`} aria-hidden>
      {initialsOf(name)}
    </span>
  );
}

export function CollaborationPanel({ history, ds, hyper, optimal }: { history: Metrics[]; ds: Dataset | null; hyper: Hyper; optimal: number }) {
  const [tab, setTab] = useState<Tab>("discuss");
  const [me, setMe] = useState("");
  const [threads, setThreads] = useState<Threads>({});
  const [inbox, setInbox] = useState<SharedRun[]>([]);

  const entries = useMemo(() => buildLogEntries(history, ds, hyper, optimal), [history, ds, hyper, optimal]);
  const flagged = useMemo(() => entries.filter(isDiscussable), [entries]);
  const datasetId = ds?.env.id ?? "";
  const datasetName = ds?.env.name ?? "";

  const runThreads = Object.values(threads).filter((t) => t.datasetId === datasetId);
  const commentCount = runThreads.reduce((n, t) => n + t.comments.length, 0);
  const openCount = runThreads.filter((t) => t.status === "open" && t.comments.length > 0).length;

  const addComment = (e: LogEntry, text: string) => {
    if (!ds) return;
    const key = threadKeyOf(e, datasetId);
    const now = new Date().toISOString();
    setThreads((ts) => {
      const t = ts[key] ?? threadFromEntry(e, datasetId, datasetName);
      return { ...ts, [key]: { ...t, updatedAt: now, comments: [...t.comments, { id: crypto.randomUUID(), author: me.trim() || "You", text: text.trim(), createdAt: now }] } };
    });
  };

  const toggleResolved = (e: LogEntry) => {
    if (!ds) return;
    const key = threadKeyOf(e, datasetId);
    setThreads((ts) => {
      const t = ts[key] ?? threadFromEntry(e, datasetId, datasetName);
      return { ...ts, [key]: { ...t, status: t.status === "open" ? "resolved" : "open", updatedAt: new Date().toISOString() } };
    });
  };

  const deleteComment = (key: string, id: string) =>
    setThreads((ts) => (ts[key] ? { ...ts, [key]: { ...ts[key], comments: ts[key].comments.filter((c) => c.id !== id), updatedAt: new Date().toISOString() } } : ts));

  const importBundle = (raw: string, filename: string) => {
    try {
      const { run } = parseShareBundle(raw);
      const { threads: merged, newComments } = mergeThreads(threads, run.threads);
      setThreads(merged);
      setInbox((xs) => [run, ...xs.filter((x) => x.id !== run.id)].slice(0, 12));
      toast.success(`Imported ${run.sharedBy}'s run from ${filename}${newComments ? ` · merged ${newComments} new comment${newComments === 1 ? "" : "s"}` : ""}`);
      setTab("inbox");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not read that share file.");
    }
  };

  const TABS: { id: Tab; label: string; count?: number }[] = [
    { id: "discuss", label: "Discuss", count: openCount || undefined },
    { id: "share", label: "Share run" },
    { id: "inbox", label: "Team runs", count: inbox.length || undefined },
  ];

  return (
    <div className="space-y-3">
      <div className="flex items-end gap-2">
        <div className="flex-1">
          <label htmlFor="collab-me" className="mb-1.5 block text-[11px] text-muted-foreground">Your name (shown on comments & shares)</label>
          <input id="collab-me" value={me} maxLength={60} onChange={(e) => setMe(e.target.value)} placeholder="e.g. Priya Raman" className={inputCls} />
        </div>
        {me.trim() && <Avatar name={me} />}
      </div>

      <div role="tablist" aria-label="Collaboration" className="grid grid-cols-3 gap-1 rounded-lg border border-border bg-surface/60 p-1">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={`flex h-7 items-center justify-center gap-1.5 rounded-md px-2 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${tab === t.id ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
          >
            {t.label}
            {t.count ? <span className="rounded-full bg-primary px-1.5 text-[9.5px] font-bold leading-4 text-primary-foreground">{t.count}</span> : null}
          </button>
        ))}
      </div>

      {tab === "discuss" && (
        <DiscussTab
          flagged={flagged}
          threads={threads}
          datasetId={datasetId}
          commentCount={commentCount}
          me={me}
          onAdd={addComment}
          onToggle={toggleResolved}
          onDelete={deleteComment}
        />
      )}
      {tab === "share" && (
        <ShareTab
          disabled={!ds || history.length === 0}
          flaggedCount={flagged.length}
          commentCount={commentCount}
          build={(note) =>
            buildShareBundle({
              sharedBy: me,
              note,
              datasetId,
              datasetName,
              transitions: ds?.transitions.length ?? 0,
              hyper,
              history,
              optimal,
              log: entries,
              threads,
            })
          }
          onImport={importBundle}
        />
      )}
      {tab === "inbox" && <InboxTab runs={inbox} threads={threads} currentDatasetId={datasetId} onImport={importBundle} onRemove={(id) => setInbox((xs) => xs.filter((x) => x.id !== id))} />}
    </div>
  );
}

function DiscussTab({
  flagged,
  threads,
  datasetId,
  commentCount,
  me,
  onAdd,
  onToggle,
  onDelete,
}: {
  flagged: LogEntry[];
  threads: Threads;
  datasetId: string;
  commentCount: number;
  me: string;
  onAdd: (e: LogEntry, text: string) => void;
  onToggle: (e: LogEntry) => void;
  onDelete: (key: string, id: string) => void;
}) {
  const [filter, setFilter] = useState<Filter>("all");
  const [openKey, setOpenKey] = useState<string | null>(null);

  if (flagged.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border px-4 py-6 text-center text-[11.5px] leading-relaxed text-muted-foreground">
        No anomalies or milestones to discuss yet. Train with <span className="font-medium text-foreground">Step Training</span> — every event the experiment log flags gets its own comment thread here.
      </div>
    );
  }

  const rows = flagged.filter((e) => {
    const t = threads[threadKeyOf(e, datasetId)];
    if (filter === "open") return !t || t.status === "open";
    if (filter === "resolved") return t?.status === "resolved";
    return true;
  });

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2 text-[10.5px] text-muted-foreground">
        <span>
          {flagged.length} flagged event{flagged.length === 1 ? "" : "s"} · {commentCount} comment{commentCount === 1 ? "" : "s"}
        </span>
        <div className="flex gap-1">
          {(["all", "open", "resolved"] as Filter[]).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`rounded-md px-2 py-0.5 capitalize transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${filter === f ? "bg-accent font-semibold text-accent-foreground" : "hover:text-foreground"}`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {rows.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border px-3 py-4 text-center text-[11px] text-muted-foreground">No {filter} threads.</p>
      ) : (
        <ul className="max-h-[420px] space-y-2 overflow-y-auto pr-1">
          {rows.map((e) => {
            const key = threadKeyOf(e, datasetId);
            return (
              <ThreadRow
                key={key}
                entry={e}
                thread={threads[key]}
                expanded={openKey === key}
                me={me}
                onExpand={() => setOpenKey((k) => (k === key ? null : key))}
                onAdd={(text) => onAdd(e, text)}
                onToggle={() => onToggle(e)}
                onDelete={(id) => onDelete(key, id)}
              />
            );
          })}
        </ul>
      )}
    </div>
  );
}

function ThreadRow({
  entry,
  thread,
  expanded,
  me,
  onExpand,
  onAdd,
  onToggle,
  onDelete,
}: {
  entry: LogEntry;
  thread: Thread | undefined;
  expanded: boolean;
  me: string;
  onExpand: () => void;
  onAdd: (text: string) => void;
  onToggle: () => void;
  onDelete: (id: string) => void;
}) {
  const [draft, setDraft] = useState("");
  const comments = thread?.comments ?? [];
  const resolved = thread?.status === "resolved";
  const anomaly = entry.kind === "anomaly";
  const tone = entry.severity === "bad" ? "border-l-destructive" : entry.severity === "good" ? "border-l-success" : "border-l-warning";

  const submit = () => {
    if (!draft.trim()) return;
    onAdd(draft);
    setDraft("");
  };

  return (
    <li className={`rounded-lg border border-l-[3px] border-border bg-surface/40 ${tone} ${resolved ? "opacity-75" : ""}`}>
      <button onClick={onExpand} aria-expanded={expanded} className="w-full rounded-lg px-3 py-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
        <div className="flex items-center gap-2 text-[10px]">
          <span className={`rounded px-1.5 py-0.5 font-semibold uppercase tracking-wider text-foreground ${anomaly ? "bg-warning/20" : "bg-success/20"}`}>{anomaly ? "Anomaly" : "Milestone"}</span>
          <span className="font-mono tabular-nums text-muted-foreground">@ {entry.atUpdates.toLocaleString()}</span>
          <span className="ml-auto flex items-center gap-1.5 text-muted-foreground">
            {resolved && <span className="rounded-full border border-success/40 bg-success/10 px-2 py-px font-semibold text-success">resolved</span>}
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z" />
            </svg>
            <span className="tabular-nums">{comments.length}</span>
          </span>
        </div>
        <p className={`mt-1 text-[11px] leading-snug text-foreground ${expanded ? "" : "line-clamp-2"}`}>{entry.text}</p>
      </button>

      {expanded && (
        <div className="space-y-2 border-t border-border px-3 py-2.5">
          {comments.length === 0 ? (
            <p className="text-[10.5px] text-muted-foreground">No comments yet — note a hypothesis, a fix you tried, or tag a teammate.</p>
          ) : (
            <ul className="space-y-2">
              {comments.map((c) => (
                <li key={c.id} className="flex gap-2">
                  <Avatar name={c.author} />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-baseline gap-2 text-[10.5px]">
                      <span className="font-semibold text-foreground">{c.author}</span>
                      <span className="text-muted-foreground">{relTime(c.createdAt)}</span>
                      {(c.author === (me.trim() || "You")) && (
                        <button onClick={() => onDelete(c.id)} className="ml-auto text-muted-foreground hover:text-destructive focus-visible:outline-none focus-visible:underline">
                          Delete
                        </button>
                      )}
                    </div>
                    <p className="whitespace-pre-wrap break-words text-[11.5px] leading-snug text-foreground">{c.text}</p>
                  </div>
                </li>
              ))}
            </ul>
          )}
          <div className="space-y-1.5">
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
              }}
              rows={2}
              maxLength={2000}
              placeholder={`Comment as ${me.trim() || "You"}… (⌘/Ctrl + Enter to post)`}
              aria-label="Add a comment"
              className="w-full resize-none rounded-md border border-input bg-background px-3 py-2 text-[12px] text-foreground placeholder:text-muted-foreground/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
            <div className="flex items-center justify-between gap-2">
              <button onClick={onToggle} className={btn}>{resolved ? "Reopen thread" : "Mark resolved"}</button>
              <button onClick={submit} disabled={!draft.trim()} className={primaryBtn}>Post comment</button>
            </div>
          </div>
        </div>
      )}
    </li>
  );
}

function DropZone({ onImport, label }: { onImport: (raw: string, name: string) => void; label: string }) {
  const ref = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const read = (f: File | undefined) => {
    if (!f) return;
    f.text().then((t) => onImport(t, f.name)).catch(() => toast.error("Could not read that file."));
  };
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        read(e.dataTransfer.files[0]);
      }}
      className={`rounded-lg border border-dashed px-3 py-3 text-center transition-colors ${over ? "border-primary bg-primary/5" : "border-border"}`}
    >
      <p className="text-[11px] text-muted-foreground">{label}</p>
      <button onClick={() => ref.current?.click()} className={`${btn} mt-2`}>Choose share file</button>
      <input
        ref={ref}
        type="file"
        accept="application/json,.json"
        className="hidden"
        onChange={(e) => {
          read(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
    </div>
  );
}

function ShareTab({
  disabled,
  flaggedCount,
  commentCount,
  build,
  onImport,
}: {
  disabled: boolean;
  flaggedCount: number;
  commentCount: number;
  build: (note: string) => ReturnType<typeof buildShareBundle>;
  onImport: (raw: string, name: string) => void;
}) {
  const [note, setNote] = useState("");

  const download = () => {
    const b = build(note);
    const url = URL.createObjectURL(new Blob([JSON.stringify(b, null, 2)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `experiment-${b.run.dataset.id}-${b.run.final?.updates ?? 0}u.share.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Share file downloaded — send it to a teammate to review and comment.");
  };

  const copyReport = () => {
    const md = bundleToMarkdown(build(note).run);
    navigator.clipboard
      ?.writeText(md)
      .then(() => toast.success("Markdown report copied — paste it into Slack, a PR or your lab notes."))
      .catch(() => toast.error("Clipboard unavailable in this browser."));
  };

  return (
    <div className="space-y-3">
      <div>
        <label htmlFor="collab-note" className="mb-1.5 block text-[11px] text-muted-foreground">
          Note to your team <span className="text-muted-foreground/70">(optional)</span>
        </label>
        <textarea
          id="collab-note"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          rows={2}
          maxLength={1000}
          placeholder="e.g. α=1.0 looks stable on this split — can someone check the loss spike around 4k updates?"
          className="w-full resize-none rounded-md border border-input bg-background px-3 py-2 text-[12.5px] text-foreground placeholder:text-muted-foreground/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
      </div>
      <div className="rounded-lg border border-border bg-surface/60 px-3 py-2 text-[10.5px] leading-relaxed text-muted-foreground">
        The share file bundles hyperparameters, final metrics, the Q-estimate vs. ground-truth curve, the full experiment log ({flaggedCount} flagged event{flaggedCount === 1 ? "" : "s"}) and {commentCount} comment{commentCount === 1 ? "" : "s"}. It contains no Q-table weights — use Model Export for those.
      </div>
      <div className="grid grid-cols-2 gap-2">
        <button onClick={download} disabled={disabled} className={primaryBtn}>Download share file</button>
        <button onClick={copyReport} disabled={disabled} className={btn}>Copy markdown report</button>
      </div>
      {disabled && <p className="text-[10.5px] text-muted-foreground">Train at least one step to share results.</p>}
      <DropZone onImport={onImport} label="Got a share file from a teammate? Drop it here to review their run and merge their comments." />
    </div>
  );
}

function MiniCurve({ run }: { run: SharedRun }) {
  const pts = run.curve;
  if (pts.length < 2) return null;
  const W = 260;
  const H = 54;
  const vals = pts.flatMap((p) => [p.est, p.tru]).concat(run.optimal);
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const span = hi - lo || 1;
  const x = (i: number) => (i / (pts.length - 1)) * W;
  const y = (v: number) => H - 3 - ((v - lo) / span) * (H - 6);
  const path = (k: "est" | "tru") => pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[k]).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-14 w-full" role="img" aria-label="Estimated Q versus true return over training">
      <line x1="0" x2={W} y1={y(run.optimal)} y2={y(run.optimal)} className="stroke-muted-foreground/50" strokeDasharray="3 3" strokeWidth="1" />
      <path d={path("est")} fill="none" className="stroke-warning" strokeWidth="1.6" />
      <path d={path("tru")} fill="none" className="stroke-primary" strokeWidth="1.8" />
    </svg>
  );
}

function InboxTab({
  runs,
  threads,
  currentDatasetId,
  onImport,
  onRemove,
}: {
  runs: SharedRun[];
  threads: Threads;
  currentDatasetId: string;
  onImport: (raw: string, name: string) => void;
  onRemove: (id: string) => void;
}) {
  const [openId, setOpenId] = useState<string | null>(null);
  return (
    <div className="space-y-2.5">
      {runs.length === 0 ? (
        <div className="rounded-lg border border-dashed border-border px-4 py-5 text-center text-[11.5px] leading-relaxed text-muted-foreground">
          No team runs yet. When a teammate sends you a <span className="font-mono text-foreground">.share.json</span> file, import it to compare their results and see their comments on each anomaly.
        </div>
      ) : (
        <ul className="space-y-2">
          {runs.map((r) => {
            const f = r.final;
            const gap = f ? f.estQ - f.trueQ : 0;
            const flagged = r.log.filter((e) => e.kind !== "summary");
            const expanded = openId === r.id;
            const comments = r.threads.reduce((n, t) => n + t.comments.length, 0);
            return (
              <li key={r.id} className="rounded-lg border border-border bg-surface/40 p-3">
                <div className="flex items-start gap-2">
                  <Avatar name={r.sharedBy} />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-baseline gap-2 text-[11px]">
                      <span className="font-semibold text-foreground">{r.sharedBy}</span>
                      <span className="text-muted-foreground">{relTime(r.sharedAt)}</span>
                    </div>
                    <div className="truncate text-[11px] text-muted-foreground">{r.label}</div>
                  </div>
                  <button onClick={() => onRemove(r.id)} className="text-[10.5px] text-muted-foreground hover:text-destructive focus-visible:outline-none focus-visible:underline" aria-label={`Remove ${r.sharedBy}'s run`}>
                    Remove
                  </button>
                </div>
                {r.note && <p className="mt-2 rounded-md border-l-2 border-primary/60 bg-card px-2.5 py-1.5 text-[11px] italic leading-snug text-foreground">{r.note}</p>}
                <div className="mt-2 grid grid-cols-3 gap-1.5 text-center">
                  <div className="rounded-md border border-border bg-card px-1.5 py-1">
                    <div className="text-[9px] uppercase tracking-wider text-muted-foreground">True return</div>
                    <div className="font-mono text-[12px] tabular-nums text-foreground">{f ? f.trueQ.toFixed(2) : "—"}</div>
                  </div>
                  <div className="rounded-md border border-border bg-card px-1.5 py-1">
                    <div className="text-[9px] uppercase tracking-wider text-muted-foreground">Est. gap</div>
                    <div className={`font-mono text-[12px] tabular-nums ${gap > 1 ? "text-destructive" : "text-foreground"}`}>{f ? `${gap >= 0 ? "+" : ""}${gap.toFixed(2)}` : "—"}</div>
                  </div>
                  <div className="rounded-md border border-border bg-card px-1.5 py-1">
                    <div className="text-[9px] uppercase tracking-wider text-muted-foreground">α / γ</div>
                    <div className="font-mono text-[12px] tabular-nums text-foreground">
                      {r.hyper.alpha.toFixed(1)} / {r.hyper.gamma.toFixed(2)}
                    </div>
                  </div>
                </div>
                <MiniCurve run={r} />
                <div className="flex items-center justify-between gap-2 text-[10.5px] text-muted-foreground">
                  <span>
                    {flagged.length} flagged · {comments} comment{comments === 1 ? "" : "s"}
                    {r.dataset.id === currentDatasetId ? " · merged into Discuss" : ` · switch to ${r.dataset.name} to reply`}
                  </span>
                  <button onClick={() => setOpenId(expanded ? null : r.id)} className="font-medium text-primary hover:underline focus-visible:outline-none focus-visible:underline" aria-expanded={expanded}>
                    {expanded ? "Hide events" : "View events"}
                  </button>
                </div>
                {expanded && (
                  <ul className="mt-2 max-h-56 space-y-1.5 overflow-y-auto pr-1">
                    {flagged.length === 0 && <li className="text-[10.5px] text-muted-foreground">This run had no flagged events.</li>}
                    {[...flagged].reverse().map((e, i) => {
                      const t = threads[threadKeyOf(e, r.dataset.id)];
                      return (
                        <li key={i} className="rounded-md border border-border bg-card px-2.5 py-1.5">
                          <div className="flex items-center gap-2 text-[9.5px] text-muted-foreground">
                            <span className="font-semibold uppercase tracking-wider text-foreground">{e.kind}</span>
                            <span className="font-mono">@ {e.atUpdates.toLocaleString()}</span>
                            {t?.status === "resolved" && <span className="font-semibold text-success">resolved</span>}
                          </div>
                          <p className="line-clamp-2 text-[10.5px] leading-snug text-foreground">{e.text}</p>
                          {t?.comments.map((c) => (
                            <p key={c.id} className="mt-1 text-[10.5px] text-muted-foreground">
                              <span className="font-semibold text-foreground">{c.author}:</span> {c.text}
                            </p>
                          ))}
                        </li>
                      );
                    })}
                  </ul>
                )}
              </li>
            );
          })}
        </ul>
      )}
      <DropZone onImport={onImport} label="Import another teammate's share file" />
    </div>
  );
}
