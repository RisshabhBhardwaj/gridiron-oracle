import { Link } from 'react-router-dom'

interface NoForecastBannerProps {
  /** Optional detail message from the API — falls back to a generic copy if absent. */
  message?: string
}

/**
 * Informational (not error) state for /predict's honest-empty 404: no approved
 * weekly projection exists yet for this player/week/season. Distinct from
 * ErrorBanner — calm blue/cyan tone, no "something went wrong" framing — because
 * this is an expected product state, not a failure.
 */
export function NoForecastBanner({ message }: NoForecastBannerProps) {
  return (
    <div
      className="relative rounded-xl p-5 flex items-start gap-4 overflow-hidden"
      style={{
        background: 'linear-gradient(135deg, rgba(89,166,255,0.08) 0%, rgba(89,166,255,0.02) 100%)',
        border: '1px solid rgba(89,166,255,0.22)',
        boxShadow: '0 4px 24px rgba(89,166,255,0.06)',
      }}
    >
      {/* Glow edge */}
      <div
        className="absolute left-0 top-0 bottom-0 w-0.5 rounded-l-xl"
        style={{ background: 'linear-gradient(180deg, #59A6FF, rgba(89,166,255,0.15))' }}
      />

      {/* Icon */}
      <div
        className="flex-shrink-0 flex items-center justify-center w-9 h-9 rounded-lg mt-0.5"
        style={{ background: 'rgba(89,166,255,0.14)', border: '1px solid rgba(89,166,255,0.3)' }}
      >
        <svg className="w-5 h-5" style={{ color: '#59A6FF' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
        </svg>
      </div>

      {/* Text */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold" style={{ color: '#59A6FF' }}>
          2026 in-season projections are not yet available
        </p>
        <p className="text-xs mt-1 leading-relaxed" style={{ color: 'rgba(89,166,255,0.75)' }}>
          {message ??
            'The weekly forecast surface for this player/week is not live yet. The 2026 draft board — powered by season-long draft projections with real data — is live today.'}
        </p>
        <Link
          to="/draft"
          className="inline-flex items-center gap-1.5 mt-3 text-xs font-semibold px-3 py-1.5 rounded-lg transition-all duration-150 hover:brightness-110 active:scale-[0.98]"
          style={{
            background: 'rgba(89,166,255,0.15)',
            border: '1px solid rgba(89,166,255,0.35)',
            color: '#59A6FF',
          }}
        >
          Go to Draft Board
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
          </svg>
        </Link>
      </div>
    </div>
  )
}
