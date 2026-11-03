// Dependency-free code editor: a monospace <textarea> with a line-number gutter, Tab/Shift+Tab
// indentation, Enter auto-indent, an error-line highlight and a read-only mode.
// Keyboard note: Tab is captured for indentation; press Esc then Tab to move focus onward.

import { useCallback, useId, useLayoutEffect, useMemo, useRef } from "react";

const LINE_H = 20;
const PAD_Y = 8;
const INDENT = "  ";

interface CodeEditorProps {
  value: string;
  onChange?: (value: string) => void;
  readOnly?: boolean;
  /** 1-based line to highlight as an error. */
  errorLine?: number | null;
  /** Height in px (the editor scrolls inside). */
  height?: number;
  ariaLabel: string;
  className?: string;
}

/** Insert text at the selection, keeping the browser's undo stack when possible. */
function insertText(ta: HTMLTextAreaElement, text: string, onChange?: (v: string) => void) {
  ta.focus();
  let ok = false;
  try {
    // Deprecated but still the only way to keep native undo for programmatic edits.
    ok = document.execCommand("insertText", false, text);
  } catch {
    ok = false;
  }
  if (!ok) {
    ta.setRangeText(text, ta.selectionStart, ta.selectionEnd, "end");
    onChange?.(ta.value);
  }
}

export function CodeEditor({ value, onChange, readOnly = false, errorLine = null, height = 300, ariaLabel, className = "" }: CodeEditorProps) {
  const taRef = useRef<HTMLTextAreaElement>(null);
  const gutterRef = useRef<HTMLDivElement>(null);
  const markRef = useRef<HTMLDivElement>(null);
  const escRef = useRef(false);
  const hintId = useId();

  const lineCount = useMemo(() => value.split("\n").length, [value]);
  const gutterCh = Math.max(2, String(lineCount).length);

  const syncScroll = useCallback(() => {
    const ta = taRef.current;
    if (!ta) return;
    if (gutterRef.current) gutterRef.current.style.transform = `translateY(${-ta.scrollTop}px)`;
    if (markRef.current) markRef.current.style.transform = `translateY(${-ta.scrollTop}px)`;
  }, []);

  useLayoutEffect(() => syncScroll(), [syncScroll, value, errorLine]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (readOnly) return;
    const ta = e.currentTarget;
    if (e.key === "Escape") {
      escRef.current = true;
      return;
    }
    if (e.key === "Tab" && !e.ctrlKey && !e.metaKey && !e.altKey) {
      if (escRef.current) {
        escRef.current = false;
        return; // let focus move on
      }
      e.preventDefault();
      const { selectionStart: start, selectionEnd: end } = ta;
      const v = ta.value;
      const multiLine = v.slice(start, end).includes("\n");
      if (!multiLine && !e.shiftKey) {
        insertText(ta, INDENT, onChange);
        return;
      }
      // (De)indent every line touched by the selection.
      const lineStart = v.lastIndexOf("\n", start - 1) + 1;
      const endAdj = end > start && v[end - 1] === "\n" ? end - 1 : end;
      const lineEnd = v.indexOf("\n", endAdj);
      const blockEnd = lineEnd === -1 ? v.length : lineEnd;
      const lines = v.slice(lineStart, blockEnd).split("\n");
      const next = lines.map((l) => (e.shiftKey ? l.replace(/^ {1,2}/, "") : INDENT + l)).join("\n");
      ta.setSelectionRange(lineStart, blockEnd);
      insertText(ta, next, onChange);
      ta.setSelectionRange(lineStart, lineStart + next.length);
      return;
    }
    escRef.current = false;
    if (e.key === "Enter" && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
      e.preventDefault();
      const v = ta.value;
      const start = ta.selectionStart;
      const lineStart = v.lastIndexOf("\n", start - 1) + 1;
      const indent = /^[ \t]*/.exec(v.slice(lineStart, start))?.[0] ?? "";
      const before = v.slice(lineStart, start).trimEnd();
      const opens = /[{[(]$/.test(before);
      insertText(ta, `\n${indent}${opens ? INDENT : ""}`, onChange);
    }
  };

  return (
    <div
      className={`relative flex overflow-hidden rounded-lg border font-mono text-[12.5px] transition-colors focus-within:ring-2 focus-within:ring-ring ${
        errorLine ? "border-destructive/50" : "border-border"
      } ${readOnly ? "bg-surface/70" : "bg-surface/30"} ${className}`}
      style={{ height }}
    >
      <div aria-hidden className="relative shrink-0 select-none overflow-hidden border-r border-border bg-surface/70 text-right text-muted-foreground" style={{ width: `calc(${gutterCh}ch + 20px)` }}>
        <div ref={gutterRef} style={{ paddingTop: PAD_Y }}>
          {Array.from({ length: lineCount }, (_, i) => (
            <div key={i} className={`pr-2.5 tabular-nums ${errorLine === i + 1 ? "font-semibold text-destructive" : ""}`} style={{ height: LINE_H, lineHeight: `${LINE_H}px` }}>
              {i + 1}
            </div>
          ))}
        </div>
      </div>
      <div className="relative min-w-0 flex-1 overflow-hidden">
        {errorLine ? (
          <div ref={markRef} aria-hidden className="pointer-events-none absolute inset-x-0 top-0">
            <div className="absolute inset-x-0 border-y border-destructive/30 bg-destructive/10" style={{ top: PAD_Y + (errorLine - 1) * LINE_H, height: LINE_H }} />
          </div>
        ) : null}
        <textarea
          ref={taRef}
          value={value}
          onChange={(e) => onChange?.(e.target.value)}
          onScroll={syncScroll}
          onKeyDown={onKeyDown}
          readOnly={readOnly}
          aria-readonly={readOnly || undefined}
          aria-label={ariaLabel}
          aria-invalid={errorLine ? true : undefined}
          aria-describedby={readOnly ? undefined : hintId}
          spellCheck={false}
          autoCapitalize="off"
          autoComplete="off"
          autoCorrect="off"
          wrap="off"
          className={`absolute inset-0 h-full w-full resize-none overflow-auto whitespace-pre bg-transparent px-3 text-foreground caret-primary outline-none selection:bg-primary/25 ${readOnly ? "cursor-default text-muted-foreground" : ""}`}
          style={{ lineHeight: `${LINE_H}px`, paddingTop: PAD_Y, paddingBottom: PAD_Y, tabSize: 2 }}
        />
      </div>
      {!readOnly && (
        <span id={hintId} className="sr-only">
          Tab inserts two spaces and Shift+Tab removes them. Press Escape, then Tab, to move focus out of the editor.
        </span>
      )}
    </div>
  );
}

/** One-line description of where the code runs and what the latest run reported. */
export function SandboxNote({ limitMs, removed, failed }: { limitMs: number; removed?: number; failed?: string[] }) {
  return (
    <p className="text-[10.5px] leading-relaxed text-muted-foreground">
      Runs in a dedicated Web Worker, never on the page: network &amp; storage APIs removed{removed ? ` (${removed} globals)` : ""}, runtime compilation disabled,{" "}
      {(limitMs / 1000).toFixed(0)} s wall-clock limit. Defense in depth for a single-user lab, not a hardened sandbox.
      {failed && failed.length > 0 && <span className="text-destructive"> Could not remove: {failed.join(", ")}.</span>}
    </p>
  );
}

/** Error panel used under both editors. */
export function CodeError({ title, message }: { title: string; message: string }) {
  return (
    <div role="alert" className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-[11.5px] leading-relaxed">
      <div className="font-semibold text-destructive">{title}</div>
      <div className="mt-0.5 whitespace-pre-wrap break-words font-mono text-[11px] text-foreground">{message}</div>
    </div>
  );
}
