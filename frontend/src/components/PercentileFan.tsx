import clsx from 'clsx'
import { toneClass, type Tone } from '@/lib/tokens'

interface PercentileFanProps {
  p10: number | null
  p50: number
  p90: number | null
  label?: string
}

function EdgeBox({ tone, name, value, note }: { tone: Tone; name: string; value: number; note: string }) {
  return (
    <div className={clsx(toneClass(tone), 'rounded-xl p-4 flex flex-col gap-1 text-center border border-tone/20 bg-tone/[0.08]')}>
      <span className="text-[10px] font-semibold uppercase tracking-widest text-tone/70">{name}</span>
      <span className="glow-text font-display text-3xl text-tone">{value.toFixed(1)}</span>
      <span className="text-[10px] text-tone/50">{note}</span>
    </div>
  )
}

export function PercentileFan({ p10, p50, p90, label }: PercentileFanProps) {
  if (p10 == null || p90 == null) {
    return (
      <div className="flex flex-col gap-2">
        {label && <span className="section-label">{label}</span>}
        <p className="text-sm text-fg-muted">Range not available for this player.</p>
        <p className="font-display text-3xl text-fg">{p50.toFixed(1)}</p>
      </div>
    )
  }
  const range      = p90 - p10
  const totalWidth = range > 0 ? range : 1
  const medianPct  = range > 0 ? ((p50 - p10) / range) * 100 : 50

  return (
    <div className="flex flex-col gap-5">
      {label && (
        <span className="section-label">{label}</span>
      )}

      {/* Three stat boxes */}
      <div className="grid grid-cols-3 gap-3">
        <EdgeBox tone="negative" name="Floor" value={p10} note="p10" />

        {/* Projection */}
        <div
          className={clsx(
            toneClass('info'),
            'relative overflow-hidden rounded-xl p-4 flex flex-col gap-1 text-center',
            'border border-tone/30 bg-tone/[0.08] shadow-[0_0_20px_rgb(var(--tone)/0.1)]',
          )}
        >
          <div className="absolute inset-0 opacity-20 bg-[radial-gradient(circle_at_50%_0%,rgb(var(--tone)/0.4),transparent_70%)]" />
          <span className="relative z-10 text-[10px] font-semibold uppercase tracking-widest text-tone/80">
            Projection
          </span>
          <span className="glow-text relative z-10 font-display text-4xl text-tone">
            {p50.toFixed(1)}
          </span>
          <span className="relative z-10 text-[10px] text-tone/50">p50</span>
        </div>

        <EdgeBox tone="positive" name="Ceiling" value={p90} note="p90" />
      </div>

      {/* Gradient range bar */}
      <div className="flex flex-col gap-2">
        <div className="flex justify-between text-xs text-fg-muted">
          <span>Floor (p10)</span>
          <span className="font-medium text-fg">Range: {totalWidth.toFixed(1)} yds</span>
          <span>Ceiling (p90)</span>
        </div>

        <div className="relative h-3 rounded-full overflow-visible bg-surface-overlay/80">
          {/* Gradient fill */}
          <div className="absolute inset-0 rounded-full animate-bar-fill bg-range-bar" />

          {/* Median marker */}
          <div
            className="absolute top-1/2 -translate-y-1/2 w-3 h-5 rounded-sm z-10 bg-fg shadow-[0_0_8px_rgb(var(--fg)/0.6),0_2px_8px_rgb(var(--shadow)/0.4)]"
            style={{ left: `calc(${medianPct}% - 6px)` }}
            title={`Median: ${p50.toFixed(1)}`}
          />
        </div>

        {/* Percentile labels */}
        <div className="relative h-4">
          <span className="absolute left-0 text-[10px] font-bold text-negative">
            {p10.toFixed(0)}
          </span>
          <span
            className="absolute text-[10px] font-bold -translate-x-1/2 text-info"
            style={{ left: `${medianPct}%` }}
          >
            {p50.toFixed(0)}
          </span>
          <span className="absolute right-0 text-[10px] font-bold text-positive">
            {p90.toFixed(0)}
          </span>
        </div>
      </div>
    </div>
  )
}
