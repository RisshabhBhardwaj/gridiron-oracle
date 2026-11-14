import { toneClass } from '@/lib/tokens'

interface PlayerHeroProps {
  /** What the number is, from `weekFigureLabel` ("Week 5 · Receiving Yards"), the same words search uses. */
  label: string
  value: number
  /** Bad week (1 in 10) and great week (1 in 10), or null when the model gave no range. */
  low: number | null
  high: number | null
  unit?: string
}

/** Where the projection sits between the bad-week and great-week marks, 0 to 100. */
function markerPercent(value: number, low: number, high: number): number {
  const span = high - low
  if (span <= 0) return 50
  return Math.min(100, Math.max(0, ((value - low) / span) * 100))
}

/**
 * The one number on the page and the range it sits in. Nothing else on the page
 * repeats the number, so a visitor who reads only this knows what to expect.
 */
export function PlayerHero({ label, value, low, high, unit }: PlayerHeroProps) {
  const hasRange = low != null && high != null

  return (
    <section
      aria-label={label}
      className={`${toneClass('info')} rounded-xl border border-tone/25 bg-panel-gradient p-5 shadow-card sm:p-6`}
    >
      <p className="section-label">{label}</p>
      <p className="mt-2 flex items-baseline gap-2">
        <span className="glow-text font-display text-6xl leading-none text-tone sm:text-7xl">{value.toFixed(1)}</span>
        {unit && <span className="text-sm font-medium text-fg-muted">{unit}</span>}
      </p>
      <p className="mt-1 text-xs text-fg-muted">Projection</p>

      {hasRange ? (
        <div className="mt-6 flex flex-col gap-2">
          <div
            role="img"
            aria-label={`Projection ${value.toFixed(1)}${unit ? ` ${unit}` : ''}, in a range from a bad week of ${low.toFixed(1)} to a great week of ${high.toFixed(1)}. The same numbers are written below the bar.`}
            className="relative h-3 rounded-full bg-surface-overlay/80"
          >
            <div className="animate-bar-fill absolute inset-0 rounded-full bg-range-bar" />
            <div
              className="absolute top-1/2 z-10 h-5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-sm bg-fg shadow-[0_0_8px_rgb(var(--fg)/0.6)]"
              style={{ left: `${markerPercent(value, low, high)}%` }}
            />
          </div>
          <div className="flex items-start justify-between gap-4">
            <div className={toneClass('negative')}>
              <p className="text-xs text-fg-muted">Bad week</p>
              <p className="font-display text-2xl leading-tight text-tone">{low.toFixed(1)}</p>
            </div>
            <div className={`${toneClass('positive')} text-right`}>
              <p className="text-xs text-fg-muted">Great week</p>
              <p className="font-display text-2xl leading-tight text-tone">{high.toFixed(1)}</p>
            </div>
          </div>
          <p className="text-xs leading-relaxed text-fg-muted">
            About 1 week in 10 comes in under the bad-week number, and about 1 in 10 comes in over the great-week number.
          </p>
        </div>
      ) : (
        <p className="mt-6 text-sm text-fg-muted">Range not available for this player.</p>
      )}
    </section>
  )
}
