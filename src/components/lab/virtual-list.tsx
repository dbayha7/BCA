// Windowed (virtualized) list: renders only the visible rows + overscan.
// Scroll positions are preserved per `scrollKey` in a module-level store, so
// switching between runs/datasets and coming back restores where you were.

import { useLayoutEffect, useRef, useState, type ReactNode } from "react";

const scrollMemory = new Map<string, number>();

export function VirtualList<T>({
  items,
  rowHeight,
  height,
  scrollKey,
  overscan = 8,
  renderRow,
  className = "",
  empty,
  ariaLabel,
}: {
  items: T[];
  /** Fixed row height in px — required for O(1) windowing math. */
  rowHeight: number;
  /** Viewport height in px. */
  height: number;
  /** Stable key identifying this logical list (run id + dataset). Scroll position is saved/restored per key. */
  scrollKey: string;
  overscan?: number;
  renderRow: (item: T, index: number) => ReactNode;
  className?: string;
  empty?: ReactNode;
  ariaLabel?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);

  // Restore saved scroll position whenever the logical list changes.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const max = Math.max(0, items.length * rowHeight - height);
    const saved = Math.min(scrollMemory.get(scrollKey) ?? 0, max);
    el.scrollTop = saved;
    setScrollTop(saved);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scrollKey]);

  if (items.length === 0) {
    return (
      <div className={`flex items-center justify-center rounded-md border border-dashed border-border bg-surface/50 text-[11px] text-muted-foreground ${className}`} style={{ height }}>
        {empty ?? "No rows"}
      </div>
    );
  }

  const total = items.length * rowHeight;
  const start = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const end = Math.min(items.length, Math.ceil((scrollTop + height) / rowHeight) + overscan);
  const slice = items.slice(start, end);

  return (
    <div
      ref={ref}
      role="list"
      aria-label={ariaLabel}
      className={`overflow-y-auto overscroll-contain ${className}`}
      style={{ height }}
      onScroll={(e) => {
        const top = e.currentTarget.scrollTop;
        setScrollTop(top);
        scrollMemory.set(scrollKey, top);
      }}
    >
      <div style={{ height: total, position: "relative" }}>
        {slice.map((item, i) => {
          const index = start + i;
          return (
            <div key={index} role="listitem" style={{ position: "absolute", top: index * rowHeight, left: 0, right: 0, height: rowHeight }}>
              {renderRow(item, index)}
            </div>
          );
        })}
      </div>
    </div>
  );
}
