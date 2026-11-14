import type { ReactNode } from 'react'

interface PageShellProps {
  children: ReactNode
}

export function PageShell({ children }: PageShellProps) {
  return (
    <main id="main-content" tabIndex={-1} className="relative min-h-screen outline-none">
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 py-6">
        {children}
      </div>
    </main>
  )
}
