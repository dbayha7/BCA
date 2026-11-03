// Contains a crash to the panel it happened in: the rest of the lab keeps running, the
// error is logged with its panel name, and the panel can be retried without a reload.

import { Component, type ErrorInfo, type ReactNode } from "react";
import { recordError } from "@/lib/error-log";

interface Props {
  /** Panel name shown in the fallback and the error log. */
  name: string;
  children: ReactNode;
  /** Changing this value clears the error (e.g. the dataset id). */
  resetKey?: unknown;
}

interface State {
  error: Error | null;
  resetKey?: unknown;
}

export class PanelErrorBoundary extends Component<Props, State> {
  state: State = { error: null, resetKey: this.props.resetKey };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    if (props.resetKey !== state.resetKey) return { error: null, resetKey: props.resetKey };
    return null;
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    recordError("panel", error, `${this.props.name}${info.componentStack ? ` — ${info.componentStack.split("\n").filter(Boolean)[0]?.trim()}` : ""}`);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div role="alert" className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-[12px]">
        <div className="font-semibold text-foreground">{this.props.name} stopped working</div>
        <p className="mt-1 break-words font-mono text-[11px] text-muted-foreground">{error.message}</p>
        <p className="mt-1 text-[11px] text-muted-foreground">The rest of the lab is unaffected and your run is still saved. The error is in the lab's error log.</p>
        <div className="mt-2 flex gap-2">
          <button
            type="button"
            onClick={() => this.setState({ error: null })}
            className="rounded-md border border-border bg-card px-2.5 py-1 text-[11px] font-medium hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            Try again
          </button>
          <button
            type="button"
            onClick={() => void navigator.clipboard?.writeText(`${this.props.name}: ${error.message}\n${error.stack ?? ""}`).catch(() => undefined)}
            className="rounded-md border border-border bg-card px-2.5 py-1 text-[11px] font-medium text-muted-foreground hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            Copy details
          </button>
        </div>
      </div>
    );
  }
}
