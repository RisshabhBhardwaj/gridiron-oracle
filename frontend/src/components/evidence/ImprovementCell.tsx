import { ArrowDown, ArrowUp, Minus } from 'lucide-react'
import { formatPct } from '@/lib/evidence'
import { cn } from '@/lib/utils'

/**
 * "How much better than the simple method". Colour is never the only signal:
 * an arrow, a sign and a hidden word say the same thing.
 */
export function ImprovementCell({ pct, className }: { pct: number | null; className?: string }) {
  if (pct === null) return <span className="text-fg-muted">n/a</span>
  const better = pct > 0.05
  const worse = pct < -0.05
  const Icon = better ? ArrowUp : worse ? ArrowDown : Minus
  return (
    <span
      className={cn(
        'inline-flex items-center justify-end gap-1 font-semibold',
        better ? 'text-positive' : worse ? 'text-negative' : 'text-fg-muted',
        className,
      )}
    >
      <Icon className="size-3.5" aria-hidden="true" />
      <span aria-hidden="true">{better ? '+' : worse ? '-' : ''}{formatPct(pct)}</span>
      <span className="sr-only">
        {better ? `${formatPct(pct)} better` : worse ? `${formatPct(pct)} worse` : 'about the same'}
      </span>
    </span>
  )
}
