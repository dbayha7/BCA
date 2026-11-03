import { createFileRoute, Link } from "@tanstack/react-router";
import { useTheme } from "@/components/theme-provider";
import { ComparisonView } from "@/components/lab/comparison-view";

export const Route = createFileRoute("/compare")({ component: ComparePage });

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

function ComparePage() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="sticky top-0 z-40 border-b border-border bg-background/85 backdrop-blur-md">
        <div className="flex h-14 items-center justify-between gap-4 px-4 lg:px-6">
          <div className="flex items-center gap-4">
            <Link
              to="/"
              className="flex items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors duration-200 hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
              Back to lab
            </Link>
            <div>
              <h1 className="font-display text-[15px] font-bold tracking-tight leading-none">Experiment comparison</h1>
              <p className="mt-0.5 text-[11px] text-muted-foreground">Two runs, trained side by side from the same fixed seed</p>
            </div>
          </div>
          <ThemeToggle />
        </div>
      </header>
      <main className="mx-auto w-full max-w-6xl flex-1 p-4 lg:p-6">
        <ComparisonView />
      </main>
    </div>
  );
}
