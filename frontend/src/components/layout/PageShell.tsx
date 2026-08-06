import type { ReactNode } from 'react'

interface PageShellProps {
  children: ReactNode
}

export function PageShell({ children }: PageShellProps) {
  return (
    <main className="relative min-h-screen">
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 py-6">
        {children}
      </div>
    </main>
  )
}
