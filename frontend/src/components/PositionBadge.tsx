import clsx from 'clsx'
import { positionTone, toneClass } from '@/lib/tokens'

/** Small coloured label for WR / RB / TE / QB. One colour per position everywhere. */
export function PositionBadge({ position, className }: { position: string | null | undefined; className?: string }) {
  return (
    <span className={clsx('pos-badge', toneClass(positionTone(position)), className)}>
      {position ?? '—'}
    </span>
  )
}
