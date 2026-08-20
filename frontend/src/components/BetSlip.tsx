/**
 * BetSlip.tsx — floating slide-out parlay builder panel.
 * Appears from the right side of the screen.
 */
import { useBetSlip, parlayOdds, decimalToAmerican, calcEdge, type BetSlipLeg } from '@/context/BetSlipContext'
import { STAT_LABELS } from '@/lib/constants'

const POS_COLORS: Record<string, string> = {
  WR: '#00C2FF', RB: '#00FFA3', TE: '#A855F7', QB: '#FFB800',
}

interface LegCardProps {
  leg: BetSlipLeg
  onRemove: () => void
}

function LegCard({ leg, onRemove }: LegCardProps) {
  const posColor  = POS_COLORS[leg.position] ?? '#607B9B'
  const edge      = leg.book_line != null ? calcEdge(leg.our_projection, leg.book_line) : null
  const hasEdge   = edge !== null && edge > 0
  const lean      = leg.book_line == null ? null : leg.our_projection >= leg.book_line ? 'Over' : 'Under'
  const statLabel = STAT_LABELS[leg.stat] ?? leg.stat

  return (
    <div
      className="rounded-xl p-3 flex flex-col gap-2"
      style={{
        background: 'rgba(13,27,48,0.8)',
        border: `1px solid ${posColor}25`,
      }}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span
              className="text-[10px] font-bold px-1 py-0.5 rounded"
              style={{ background: `${posColor}20`, color: posColor, border: `1px solid ${posColor}30` }}
            >
              {leg.position}
            </span>
            <p className="text-xs font-semibold text-oracle-white truncate">{leg.player_name}</p>
          </div>
          <p className="text-[10px] text-oracle-muted mt-0.5">{leg.team ?? 'UNK'} · W{leg.week} · {statLabel}</p>
        </div>
        <button
          onClick={onRemove}
          className="w-5 h-5 flex items-center justify-center rounded-md flex-shrink-0 transition-colors"
          style={{ color: '#607B9B' }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.color = '#FF3B5C' }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.color = '#607B9B' }}
          aria-label={`Remove ${leg.player_name}`}
        >
          ×
        </button>
      </div>

      {/* Projections vs line */}
      <div className="grid grid-cols-3 gap-1.5 text-center">
        <div
          className="rounded-lg py-1.5 px-1"
          style={{ background: 'rgba(5,11,24,0.6)' }}
        >
          <p className="text-[9px] text-oracle-muted mb-0.5">MODEL</p>
          <p className="font-display text-base" style={{ color: posColor }}>
            {leg.our_projection.toFixed(1)}
          </p>
        </div>
        <div
          className="rounded-lg py-1.5 px-1"
          style={{ background: 'rgba(5,11,24,0.6)' }}
        >
          <p className="text-[9px] text-oracle-muted mb-0.5">LINE</p>
          <p className="font-display text-base text-oracle-white">
            {leg.book_line != null ? leg.book_line.toFixed(1) : '—'}
          </p>
        </div>
        <div
          className="rounded-lg py-1.5 px-1"
          style={{
            background: hasEdge ? 'rgba(0,255,163,0.08)' : edge !== null && edge < 0 ? 'rgba(255,59,92,0.08)' : 'rgba(5,11,24,0.6)',
          }}
        >
          <p className="text-[9px] text-oracle-muted mb-0.5">EDGE</p>
          <p
            className="font-display text-base"
            style={{ color: hasEdge ? '#00FFA3' : edge !== null && edge < 0 ? '#FF3B5C' : '#607B9B' }}
          >
            {edge !== null ? `${edge > 0 ? '+' : ''}${edge.toFixed(1)}%` : '—'}
          </p>
        </div>
      </div>

      <div className="flex items-center justify-between gap-2 text-[10px]">
        <span className="text-oracle-muted">
          Lean: <span style={{ color: hasEdge ? '#00FFA3' : edge !== null ? '#FFB800' : '#607B9B' }} className="font-semibold">{lean ?? 'Waiting for line'}</span>
        </span>
        {leg.book_line != null && (
          <span className="text-oracle-muted">
            {Math.abs(leg.our_projection - leg.book_line).toFixed(1)} from line
          </span>
        )}
      </div>

      {/* Floor/ceiling range */}
      <div className="flex items-center gap-2 text-[10px]">
        <span style={{ color: '#FF3B5C' }}>{leg.floor == null ? '—' : leg.floor.toFixed(0)}</span>
        <div className="flex-1 h-1 rounded-full" style={{ background: 'rgba(26,47,78,0.8)' }}>
          <div
            className="h-full rounded-full"
            style={{ background: `linear-gradient(90deg, rgba(255,59,92,0.6), ${posColor}80, rgba(0,255,163,0.6))` }}
          />
        </div>
        <span style={{ color: '#00FFA3' }}>{leg.ceiling == null ? '—' : leg.ceiling.toFixed(0)}</span>
      </div>
    </div>
  )
}

export function BetSlip() {
  const { state, removeLeg, clearSlip, setOpen } = useBetSlip()
  const { legs, isOpen } = state

  const nLegs     = legs.length
  const combined  = nLegs > 0 ? parlayOdds(nLegs) : 0
  const american  = nLegs > 0 ? decimalToAmerican(combined) : null
  const missingLines = legs.filter((leg) => leg.book_line == null).length
  const edgedLegs = legs.filter(
    (leg) => leg.book_line != null && calcEdge(leg.our_projection, leg.book_line) > 0,
  )
  const bestEdge = edgedLegs.length > 0
    ? Math.max(...edgedLegs.map((leg) => calcEdge(leg.our_projection, leg.book_line!)))
    : null

  function copySlip() {
    const text = legs.map((leg) => {
      const statLabel = STAT_LABELS[leg.stat] ?? leg.stat
      const side = leg.book_line == null ? 'Proj' : leg.our_projection >= leg.book_line ? 'Over' : 'Under'
      const line = leg.book_line ?? leg.our_projection
      return `${leg.player_name} ${statLabel} ${side} ${line.toFixed(1)} (proj ${leg.our_projection.toFixed(1)})`
    }).join('\n')

    const header = nLegs >= 2 && american ? `${nLegs}-Leg Parlay (${american})` : `${nLegs}-Leg Slip`
    void navigator.clipboard.writeText(`${header}\n${text}`)
  }

  return (
    <>
      {/* Panel */}
      <div
        className="fixed right-0 top-16 bottom-0 z-[150] flex flex-col"
        style={{
          width: '360px',
          transform: isOpen ? 'translateX(0)' : 'translateX(100%)',
          transition: 'transform 0.3s cubic-bezier(0.22, 1, 0.36, 1)',
          background: 'rgba(5,11,24,0.97)',
          borderLeft: '1px solid rgba(0,194,255,0.15)',
          boxShadow: isOpen ? '-8px 0 40px rgba(0,0,0,0.6), -2px 0 0 rgba(0,194,255,0.08)' : 'none',
          backdropFilter: 'blur(20px)',
        }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-4 py-4 flex-shrink-0"
          style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}
        >
          <div className="flex items-center gap-2">
            <div
              className="w-7 h-7 rounded-lg flex items-center justify-center"
              style={{ background: 'rgba(0,255,163,0.12)', border: '1px solid rgba(0,255,163,0.25)' }}
            >
              <span className="text-sm">🎯</span>
            </div>
            <div>
              <h2 className="text-sm font-bold text-oracle-white">Bet Slip</h2>
              <p className="text-[10px] text-oracle-muted">{nLegs} leg{nLegs !== 1 ? 's' : ''}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {nLegs > 0 && (
              <button
                onClick={clearSlip}
                className="text-[10px] font-medium px-2.5 py-1 rounded-lg transition-colors"
                style={{ color: '#607B9B', border: '1px solid rgba(255,255,255,0.06)' }}
                onMouseEnter={(e) => {
                  const el = e.currentTarget as HTMLButtonElement
                  el.style.color = '#FF3B5C'
                  el.style.borderColor = 'rgba(255,59,92,0.3)'
                }}
                onMouseLeave={(e) => {
                  const el = e.currentTarget as HTMLButtonElement
                  el.style.color = '#607B9B'
                  el.style.borderColor = 'rgba(255,255,255,0.06)'
                }}
              >
                Clear all
              </button>
            )}
            <button
              onClick={() => setOpen(false)}
              className="w-7 h-7 flex items-center justify-center rounded-lg text-oracle-muted hover:text-oracle-white transition-colors"
              style={{ border: '1px solid rgba(255,255,255,0.06)' }}
              aria-label="Close bet slip"
            >
              ×
            </button>
          </div>
        </div>

        {/* Legs */}
        <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-2">
          {nLegs === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-6">
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center"
                style={{ background: 'rgba(0,194,255,0.06)', border: '1px solid rgba(0,194,255,0.15)' }}
              >
                <span className="text-3xl">➕</span>
              </div>
              <div>
                <p className="text-sm font-semibold text-oracle-white">Slip is empty</p>
                <p className="text-xs text-oracle-muted mt-1">
                  Click <span className="text-oracle-blue font-bold">+</span> on any player card to add a leg
                </p>
                <p className="text-[11px] text-oracle-muted mt-2">
                  Legs persist for this browser session so you can build a slip across dashboard and player-detail views.
                </p>
              </div>
            </div>
          ) : (
            legs.map((leg) => (
              <LegCard
                key={leg.id}
                leg={leg}
                onRemove={() => removeLeg(leg.id)}
              />
            ))
          )}
        </div>

        {/* Parlay summary footer */}
        {nLegs >= 1 && (
          <div
            className="flex-shrink-0 p-4 flex flex-col gap-3"
            style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}
          >
            {nLegs >= 2 && (
              <div
                className="rounded-xl p-4 flex flex-col gap-1"
                style={{
                  background: 'rgba(0,255,163,0.06)',
                  border: '1px solid rgba(0,255,163,0.2)',
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-semibold text-oracle-muted uppercase tracking-widest">
                    {nLegs}-Leg Parlay Odds
                  </span>
                  <span
                    className="font-display text-2xl"
                    style={{ color: '#00FFA3', textShadow: '0 0 16px rgba(0,255,163,0.5)' }}
                  >
                    {american}
                  </span>
                </div>
                <div className="flex items-center justify-between text-[10px] text-oracle-muted mt-1">
                  <span>Decimal: {combined.toFixed(2)}×</span>
                  <span>At -110 per leg</span>
                </div>
              </div>
            )}

            {/* Legs with positive edge */}
            {(() => {
              if (edgedLegs.length === 0 && missingLines === 0) return null
              return (
                <div className="flex flex-col gap-2">
                  {edgedLegs.length > 0 && (
                    <div
                      className="text-[10px] px-3 py-2 rounded-lg"
                      style={{ background: 'rgba(0,194,255,0.06)', border: '1px solid rgba(0,194,255,0.15)', color: '#00C2FF' }}
                    >
                      ⚡ {edgedLegs.length} leg{edgedLegs.length !== 1 ? 's' : ''} beating the book line
                      {bestEdge !== null ? ` · best edge +${bestEdge.toFixed(1)}%` : ''}
                    </div>
                  )}
                  {missingLines > 0 && (
                    <div
                      className="text-[10px] px-3 py-2 rounded-lg"
                      style={{ background: 'rgba(255,184,0,0.06)', border: '1px solid rgba(255,184,0,0.15)', color: '#FFB800' }}
                    >
                      {missingLines} leg{missingLines !== 1 ? 's' : ''} still missing sportsbook lines for edge calculation
                    </div>
                  )}
                </div>
              )
            })()}

            <button
              onClick={copySlip}
              className="w-full py-2.5 rounded-xl text-xs font-semibold transition-all duration-150"
              style={{
                background: 'rgba(0,194,255,0.1)',
                border: '1px solid rgba(0,194,255,0.3)',
                color: '#00C2FF',
              }}
            >
              {nLegs >= 2 ? 'Copy Parlay to Clipboard' : 'Copy Slip to Clipboard'}
            </button>
          </div>
        )}
      </div>
    </>
  )
}
