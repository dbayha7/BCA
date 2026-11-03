# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
bun run dev      # Start development server
bun run build    # TypeScript check + Vite build
bun run lint     # ESLint
bun run preview  # Preview production build
```

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

Use `@/` to import from `src/`:
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
