import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { toneClass, type Tone } from '@/lib/tokens'

interface ProjectionCardProps {
  label: string
  value: number | null
  unit?: string
  tone?: Tone
  subtext?: string
  /** Words to show instead of "—" when there is no value (for example "No line yet"). */
  emptyText?: string
}

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true
}

/**
 * Counts up to `target` as decoration. The number must reach the screen even when
 * animation frames do not run (a reduced-motion setting, a backgrounded tab), so
 * in those cases it is shown at once instead of sitting at a confident "0.0".
 */
function useCountUp(target: number, duration = 600): number {
  const [current, setCurrent] = useState(() => (prefersReducedMotion() ? target : 0))
  useEffect(() => {
    if (prefersReducedMotion() || document.hidden) {
      setCurrent(target)
      return
    }
    const start = performance.now()
    let frame = 0
    const step = (now: number) => {
      const elapsed = now - start
      const progress = Math.min(elapsed / duration, 1)
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3)
      setCurrent(target * eased)
      if (progress < 1) frame = requestAnimationFrame(step)
    }
    frame = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frame)
  }, [target, duration])
  return current
}

export function ProjectionCard({ label, value, unit, tone = 'info', subtext, emptyText }: ProjectionCardProps) {
  const animated = useCountUp(value ?? 0)
  const display = value == null ? '—' : animated.toFixed(1)

  return (
    <div
      className={clsx(
        toneClass(tone),
        'relative rounded-xl p-4 flex flex-col gap-2 overflow-hidden transition-all duration-300 cursor-default',
        'border border-tone/20 bg-panel-gradient shadow-card',
      )}
    >
      {/* Corner glow */}
      <div className="corner-glow absolute top-0 right-0 w-16 h-16 rounded-bl-full opacity-40" />

      <span className="section-label">{label}</span>
      <div className="flex items-baseline gap-1.5">
        {value == null && emptyText ? (
          <span className="font-display text-3xl leading-none text-tone">{emptyText}</span>
        ) : (
          <span className="glow-text font-display text-4xl leading-none text-tone transition-all">
            {value == null ? '—' : display}
          </span>
        )}
        {unit && (
          <span className="text-xs font-medium text-tone/50">
            {unit}
          </span>
        )}
      </div>
      {subtext && <span className="text-xs text-fg-muted">{subtext}</span>}
    </div>
  )
}
