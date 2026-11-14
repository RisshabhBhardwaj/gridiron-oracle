import { toneClass } from '@/lib/tokens'

interface SimulationFigureProps {
  /** What the number is, in plain words ("Rest of season · Receiving Yards"). */
  label: string
  /** How it was made, from `provenanceLabel` ("Preseason simulation, not updated for games played"). */
  provenance: string
  value: number
  low: number | null
  high: number | null
  unit?: string
}

/**
 * A simulated figure, shown only when the visitor asks for it. It is not the page's
 * number: the provenance sits on the figure itself so it cannot be read as a forecast.
 */
export function SimulationFigure({ label, provenance, value, low, high, unit }: SimulationFigureProps) {
  const range = low != null && high != null ? `${low.toFixed(1)} to ${high.toFixed(1)}` : null
  return (
    <div className={`${toneClass('muted')} rounded-xl border border-dashed border-tone/40 bg-tone/[0.04] p-4`}>
      <p className="section-label">{label}</p>
      <p className="mt-1 flex items-baseline gap-1.5">
        <span className="font-display text-3xl leading-none text-fg">{value.toFixed(1)}</span>
        {unit && <span className="text-xs text-fg-muted">{unit}</span>}
      </p>
      {range && <p className="mt-1 text-xs text-fg-muted">Range {range}</p>}
      <p className="mt-2 text-xs font-medium text-fg-muted">{provenance}</p>
    </div>
  )
}
