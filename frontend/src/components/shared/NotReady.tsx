import type { ReactNode } from 'react'
import { Hourglass } from 'lucide-react'
import { cn, FOCUS_RING } from '@/lib/utils'

/** Button looks for the actions inside a NotReady notice. */
export const NOT_READY_PRIMARY = cn(
  'inline-flex items-center rounded-lg border border-tone/[0.35] bg-tone/15 px-3 py-1.5 text-xs font-semibold text-tone transition-colors hover:bg-tone/25',
  FOCUS_RING,
)
export const NOT_READY_SECONDARY = cn(
  'inline-flex items-center rounded-lg border border-white/10 px-3 py-1.5 text-xs font-semibold text-fg-muted transition-colors hover:border-white/20 hover:text-fg',
  FOCUS_RING,
)

interface NotReadyProps {
  /** What is not published yet, in plain words ("Week 11 projections are not ready yet"). */
  title: string
  /** One or two sentences: why it is empty and what happens next. */
  children?: ReactNode
  /** The next step: buttons or links (use NOT_READY_PRIMARY / NOT_READY_SECONDARY). */
  actions?: ReactNode
}

/**
 * "Nothing published yet." The server answered and said there is no data (a 404,
 * or an empty set where the data should be). That is a normal state, not a
 * failure: calm info tone, no "something went wrong", and always a next step.
 * A failed request (5xx, no connection) is an ErrorBanner instead.
 */
export function NotReady({ title, children, actions }: NotReadyProps) {
  return (
    <div role="status" className="tone-info flex items-start gap-4 rounded-xl border border-tone/[0.22] bg-tone/[0.05] p-5">
      <span className="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-lg border border-tone/30 bg-tone/[0.14]">
        <Hourglass className="size-5 text-tone" aria-hidden="true" />
      </span>
      <div className="flex min-w-0 flex-1 flex-col items-start gap-2">
        <p className="text-sm font-semibold text-tone">{title}</p>
        {children && <p className="text-xs leading-relaxed text-fg-muted">{children}</p>}
        {actions && <div className="mt-1 flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
    </div>
  )
}
