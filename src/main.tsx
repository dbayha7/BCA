import { StrictMode } from 'react'
import ReactDOM from 'react-dom/client'
import { RouterProvider, createRouter } from '@tanstack/react-router'
import './index.css'
import { ThemeProvider } from '@/components/theme-provider'
import { Toaster } from '@/components/ui/sonner'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { installGlobalErrorHandlers, onError, recordError } from '@/lib/error-log'
import { errorsStore } from '@/lib/store'

// Import the generated route tree
import { routeTree } from './routeTree.gen'

// Crash guard: anything that escapes the panels is logged and kept locally.
installGlobalErrorHandlers()
onError((e) => {
  errorsStore.add(e).catch(() => undefined)
})

function RouteCrash({ error, reset }: { error: unknown; reset: () => void }) {
  const message = error instanceof Error ? error.message : String(error)
  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-6">
      <div role="alert" className="max-w-md rounded-xl border border-destructive/40 bg-card p-5 text-sm shadow-sm">
        <div className="font-semibold">This page hit an error</div>
        <p className="mt-1 break-words font-mono text-xs text-muted-foreground">{message}</p>
        <p className="mt-2 text-xs text-muted-foreground">Your runs are saved in this browser. Try again, or reload the lab.</p>
        <div className="mt-3 flex gap-2">
          <button className="rounded-md bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground" onClick={reset}>Try again</button>
          <button className="rounded-md border border-border px-3 py-1.5 text-xs" onClick={() => window.location.assign('/')}>Reload the lab</button>
        </div>
      </div>
    </div>
  )
}

// Create a new router instance
const router = createRouter({
  routeTree,
  defaultErrorComponent: ({ error, reset }) => <RouteCrash error={error} reset={reset} />,
  defaultOnCatch: (error) => {
    recordError('uncaught', error, 'route render')
  },
})

const queryClient = new QueryClient()

// Register the router instance for type safety
declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}

// Render the app
const rootElement = document.getElementById('root')!
if (!rootElement.innerHTML) {
  const root = ReactDOM.createRoot(rootElement)
  root.render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
        <RouterProvider router={router} />
        <Toaster />
      </ThemeProvider>
      </QueryClientProvider>
    </StrictMode>,
  )
}