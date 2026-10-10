import type { ReactNode } from 'react'

/** A titled card. One shared look for settings groups, charts and tables. */
export function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-xl overflow-hidden border border-white/[0.07] bg-panel-gradient shadow-card">
      <div className="px-5 py-3 border-b border-white/5">
        <h2 className="text-sm font-semibold text-fg">{title}</h2>
      </div>
      <div className="p-5">{children}</div>
    </section>
  )
}
