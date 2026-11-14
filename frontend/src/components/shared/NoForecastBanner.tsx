import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, CircleHelp } from 'lucide-react'
import { useSearch } from '@/context/SearchContext'

interface NoForecastBannerProps {
  /** Season of the request that came back empty (from the page address, or CURRENT_SEASON). */
  season: number
  week: number
  /** Shown when we have it, so the message names who it is about. */
  playerName?: string
  /**
   * 'player' (default): one player has no forecast. 'week': the whole week has no
   * real forecast, so the page says so once instead of a number on every card.
   */
  scope?: 'player' | 'week'
  /** Week scope only: switches the page to its season-long view. */
  onOpenSeasonLong?: () => void
  /** Extra actions after the standard ones (for example the opt-in simulation toggle). */
  children?: ReactNode
}

/**
 * Informational (not error) state for /predict's honest-empty 404: no approved
 * weekly projection exists yet for this player, week and season. It is an
 * expected product state, not a failure, so it uses the calm info tone and no
 * "something went wrong" wording.
 *
 * Also used at page level (`scope="week"`) by the Weekly tab when the served
 * numbers are the preseason simulation, not a forecast for that week.
 *
 * The words are written here, not taken from the API. The API's own message
 * echoes the request (player='...' week=5) and uses internal terms.
 */
export function NoForecastBanner({ season, week, playerName, scope = 'player', onOpenSeasonLong, children }: NoForecastBannerProps) {
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
        <CircleHelp className="size-5 text-tone" aria-hidden="true" />
      </div>

      {/* Text */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-tone">
          No forecast yet for Week {week}, {season}
        </p>
        <p className="text-xs mt-1 leading-relaxed text-fg-muted">
          {scope === 'week'
            ? `We do not have a real Week ${week} forecast yet. The numbers we do have are a preseason simulation that does not know about games already played, so we do not show them as a Week ${week} forecast.`
            : `We do not have a Week ${week} projection for ${who} in ${season} yet. Season-long projections are ready on the draft board.`}
        </p>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {scope === 'week' && onOpenSeasonLong && (
            <button
              type="button"
              onClick={onOpenSeasonLong}
              className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg transition-all duration-150 hover:brightness-110 active:scale-[0.98] bg-tone/15 border border-tone/[0.35] text-tone"
            >
              See season-long projections
              <ArrowRight className="size-3.5" aria-hidden="true" />
            </button>
          )}
          <Link
            to="/draft"
            className={
              scope === 'week'
                ? 'inline-flex items-center text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors border border-white/10 text-fg-muted hover:text-fg hover:border-white/20'
                : 'inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg transition-all duration-150 hover:brightness-110 active:scale-[0.98] bg-tone/15 border border-tone/[0.35] text-tone'
            }
          >
            Open the draft board
            {scope === 'player' && <ArrowRight className="size-3.5" aria-hidden="true" />}
          </Link>
          {scope === 'player' && (
            <button
              type="button"
              onClick={openSearch}
              className="inline-flex items-center text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors border border-white/10 text-fg-muted hover:text-fg hover:border-white/20"
            >
              Search for another player
            </button>
          )}
          {children}
        </div>
      </div>
    </div>
  )
}
