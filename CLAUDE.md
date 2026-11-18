# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
bun run dev      # Start development server
bun run build    # TypeScript check + Vite build
bun run lint     # ESLint
bun test         # Web app tests in tests/lab
bun run preview  # Preview production build
```

CI (`.github/workflows/checks.yml`) runs on every PR. Run the same checks before you say a change is done:

```bash
bun run lint && bun test && bun run build
python3 .github/scripts/check_claude_md.py   # Session-start instructions stay under 200 lines
python3 .github/scripts/check_frozen.py      # No edit, move or delete under .github/frozen-paths.txt
python3 -m unittest discover -s .github/scripts
```

## Fixing a repeat mistake

Everything this file loads is in context for every session, and longer files reduce adherence, so CI
keeps it under 200 lines. For each repeat mistake, use the strongest fix that works:

1. Make it impossible: a type, a schema, an API that can't be misused.
2. Catch it automatically: a lint rule or a CI check.
3. Make the agent prove it: a test it runs before it says "done".
4. Write a rule here, with its scope and reason, only when 1 to 3 can't.
5. Keep a person on it, for judgment calls.

If code can check it, code does it. When a check replaces a rule, delete the rule.

## Architecture

This is a React + TypeScript + Vite template using:

- **TanStack Router** - File-based routing with auto code splitting. Routes live in `src/routes/`, and `src/routeTree.gen.ts` is auto-generated.
- **TanStack Query** - Data fetching/caching via QueryClientProvider in main.tsx
- **Tailwind CSS v4** - Configured via `@tailwindcss/vite` plugin
- **shadcn/ui** - Radix-based components in `src/components/ui/`

### Key Files

- `src/main.tsx` - App entry: sets up router, query client, theme provider, and toaster
- `src/routes/__root.tsx` - Root layout wrapping all routes
- `src/lib/utils.ts` - `cn()` helper for merging Tailwind classes
- `plugins/vite-plugin-shipper-ids.ts` - Dev-only plugin that adds `data-shipper-id` attributes to JSX elements for tracking

### Path Alias

Use `@/` to import from `src/` (ESLint rejects `../` imports inside `src/`):
```tsx
import { Button } from "@/components/ui/button";
```

### Routing

Create routes by adding files to `src/routes/`. The TanStack Router plugin auto-generates the route tree on save.

```tsx
// src/routes/about.tsx → /about
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/about")({
  component: About,
});

function About() {
  return <div>About page</div>;
}
```

```tsx
// src/routes/users/$userId.tsx → /users/:userId
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/users/$userId")({
  component: UserDetail,
});

function UserDetail() {
  const { userId } = Route.useParams();
  return <div>User {userId}</div>;
}
```
