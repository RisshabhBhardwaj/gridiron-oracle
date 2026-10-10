import { Link } from 'react-router-dom'
import { useSearch } from '@/context/SearchContext'

interface NoForecastBannerProps {
  /** Season of the request that came back empty (from the page address, or CURRENT_SEASON). */
  season: number
  week: number
  /** Shown when we have it, so the message names who it is about. */
  playerName?: string
}

/**
 * Informational (not error) state for /predict's honest-empty 404: no approved
 * weekly projection exists yet for this player, week and season. It is an
 * expected product state, not a failure, so it uses the calm info tone and no
 * "something went wrong" wording.
 *
 * The words are written here, not taken from the API. The API's own message
 * echoes the request (player='...' week=5) and uses internal terms.
 */
export function NoForecastBanner({ season, week, playerName }: NoForecastBannerProps) {
  const { open: openSearch } = useSearch()
  const who = playerName ? playerName : 'this player'

  return (
    <div
      role="status"
      className="tone-info relative rounded-xl p-5 flex items-start gap-4 overflow-hidden border border-tone/[0.22] bg-gradient-to-br from-tone/[0.08] to-tone/[0.02] shadow-[0_4px_24px_rgb(var(--tone)/0.06)]"
    >
      {/* Glow edge */}
      <div className="absolute left-0 top-0 bottom-0 w-0.5 rounded-l-xl bg-gradient-to-b from-tone to-tone/15" />

      {/* Icon */}
      <div className="flex-shrink-0 flex items-center justify-center w-9 h-9 rounded-lg mt-0.5 bg-tone/[0.14] border border-tone/30">
        <svg className="w-5 h-5 text-tone" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
        </svg>
      </div>

      {/* Text */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-tone">
          No forecast yet for Week {week}, {season}
        </p>
        <p className="text-xs mt-1 leading-relaxed text-fg-muted">
          We do not have a Week {week} projection for {who} in {season} yet.
          Season-long projections are ready on the draft board.
        </p>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Link
            to="/draft"
            className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg transition-all duration-150 hover:brightness-110 active:scale-[0.98] bg-tone/15 border border-tone/[0.35] text-tone"
          >
            Open the draft board
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
            </svg>
          </Link>
          <button
            type="button"
            onClick={openSearch}
            className="inline-flex items-center text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors border border-white/10 text-fg-muted hover:text-fg hover:border-white/20"
          >
            Search for another player
          </button>
        </div>
      </div>
    </div>
  )
}
