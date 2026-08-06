interface PercentileFanProps {
  p10: number
  p50: number
  p90: number
  label?: string
}

export function PercentileFan({ p10, p50, p90, label }: PercentileFanProps) {
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
        {/* Floor */}
        <div
          className="rounded-xl p-4 flex flex-col gap-1 text-center"
          style={{
            background: 'rgba(255,59,92,0.08)',
            border: '1px solid rgba(255,59,92,0.2)',
          }}
        >
          <span className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: 'rgba(255,59,92,0.7)' }}>
            Floor
          </span>
          <span className="font-display text-3xl" style={{ color: '#FF3B5C', textShadow: '0 0 16px rgba(255,59,92,0.5)' }}>
            {p10.toFixed(1)}
          </span>
          <span className="text-[10px]" style={{ color: 'rgba(255,59,92,0.5)' }}>p10</span>
        </div>

        {/* Projection */}
        <div
          className="rounded-xl p-4 flex flex-col gap-1 text-center relative overflow-hidden"
          style={{
            background: 'rgba(0,194,255,0.08)',
            border: '1px solid rgba(0,194,255,0.3)',
            boxShadow: '0 0 20px rgba(0,194,255,0.1)',
          }}
        >
          <div
            className="absolute inset-0 opacity-20"
            style={{ background: 'radial-gradient(circle at 50% 0%, rgba(0,194,255,0.4), transparent 70%)' }}
          />
          <span className="text-[10px] font-semibold uppercase tracking-widest relative z-10" style={{ color: 'rgba(0,194,255,0.8)' }}>
            Projection
          </span>
          <span
            className="font-display text-4xl relative z-10"
            style={{ color: '#00C2FF', textShadow: '0 0 20px rgba(0,194,255,0.6)' }}
          >
            {p50.toFixed(1)}
          </span>
          <span className="text-[10px] relative z-10" style={{ color: 'rgba(0,194,255,0.5)' }}>p50</span>
        </div>

        {/* Ceiling */}
        <div
          className="rounded-xl p-4 flex flex-col gap-1 text-center"
          style={{
            background: 'rgba(0,255,163,0.08)',
            border: '1px solid rgba(0,255,163,0.2)',
          }}
        >
          <span className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: 'rgba(0,255,163,0.7)' }}>
            Ceiling
          </span>
          <span className="font-display text-3xl" style={{ color: '#00FFA3', textShadow: '0 0 16px rgba(0,255,163,0.5)' }}>
            {p90.toFixed(1)}
          </span>
          <span className="text-[10px]" style={{ color: 'rgba(0,255,163,0.5)' }}>p90</span>
        </div>
      </div>

      {/* Gradient range bar */}
      <div className="flex flex-col gap-2">
        <div className="flex justify-between text-xs text-oracle-muted">
          <span>Floor (p10)</span>
          <span className="font-medium text-oracle-white">Range: {totalWidth.toFixed(1)} yds</span>
          <span>Ceiling (p90)</span>
        </div>

        <div
          className="relative h-3 rounded-full overflow-visible"
          style={{
            background: 'rgba(26,47,78,0.8)',
          }}
        >
          {/* Gradient fill */}
          <div
            className="absolute inset-0 rounded-full animate-bar-fill"
            style={{
              background: 'linear-gradient(90deg, rgba(255,59,92,0.7) 0%, rgba(0,194,255,0.8) 50%, rgba(0,255,163,0.7) 100%)',
            }}
          />

          {/* Median marker */}
          <div
            className="absolute top-1/2 -translate-y-1/2 w-3 h-5 rounded-sm z-10"
            style={{
              left: `calc(${medianPct}% - 6px)`,
              background: '#ffffff',
              boxShadow: '0 0 8px rgba(255,255,255,0.6), 0 2px 8px rgba(0,0,0,0.4)',
            }}
            title={`Median: ${p50.toFixed(1)}`}
          />
        </div>

        {/* Percentile labels */}
        <div
          className="relative h-4"
          style={{ position: 'relative' }}
        >
          <span
            className="absolute left-0 text-[10px] font-bold"
            style={{ color: '#FF3B5C' }}
          >
            {p10.toFixed(0)}
          </span>
          <span
            className="absolute text-[10px] font-bold -translate-x-1/2"
            style={{ left: `${medianPct}%`, color: '#00C2FF' }}
          >
            {p50.toFixed(0)}
          </span>
          <span
            className="absolute right-0 text-[10px] font-bold"
            style={{ color: '#00FFA3' }}
          >
            {p90.toFixed(0)}
          </span>
        </div>
      </div>
    </div>
  )
}
