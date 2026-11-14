import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Search } from 'lucide-react'
import { useCurrentSeason } from '@/hooks/useCurrentSeason'
import { useSeasonProjections } from '@/hooks/useSeasonProjections'
import { useSearch } from '@/context/SearchContext'
import type { SearchablePlayer } from '@/context/SearchContext'
import { CURRENT_SEASON, CURRENT_WEEK, POSITIONS } from '@/lib/constants'
import { usePageTitle } from '@/hooks/usePageTitle'
import { PAGE_NAMES } from '@/lib/page-names'

// Labels must track NavBar's NAV_LINKS — /projections and /season were
// repointed at CurrentSeason and SeasonReview, and the hero kept advertising
// the old destinations.
const QUICK_LINKS = [
  { to: '/projections', label: PAGE_NAMES.projections, meta: 'Weekly boards, team wins, and game forecasts' },
  { to: '/season', label: PAGE_NAMES.season, meta: 'How our projections compared with reality' },
  { to: '/backtest', label: PAGE_NAMES.backtest, meta: 'Past results, season by season' },
]

export function Home() {
  const { data: currentSeasonData, isPlaceholderData, isError: currentFailed } = useCurrentSeason()
  usePageTitle()
  const week = currentSeasonData?.week ?? CURRENT_WEEK
  const season = currentSeasonData?.season ?? CURRENT_SEASON
  // Home loads the season list only to feed the search box; nothing on this page shows it,
  // so a failure here is silent (search has its own unavailable state).
  const seasonResolved = !isPlaceholderData && !currentFailed

  const { setPlayers, setDefaults, open } = useSearch()

  const { data } = useSeasonProjections({
    season,
    startWeek: 1,
    positions: [...POSITIONS],
    enabled: seasonResolved,
  })

  useEffect(() => {
    setDefaults({ week, season, stat: 'fantasy_ppr' })
  }, [season, setDefaults, week])

  useEffect(() => {
    if (!data?.projections) return
    const searchablePlayers: SearchablePlayer[] = data.projections
      .filter((p) => p.fantasy_ppr != null)
      .map((p) => ({
        player_id: p.player_id,
        player_name: p.player_name,
        position: p.position,
        team: p.team,
        stat: 'fantasy_ppr',
        // A season total is not what the player page shows (a week forecast or none), so search shows no number.
        projection: null,
      }))
    setPlayers(searchablePlayers)
  }, [data?.projections, setPlayers])

  return (
    <div className="min-h-[calc(100vh-53px)] flex flex-col justify-center py-12 sm:py-16">
      <section className="mx-auto flex w-full max-w-4xl flex-col items-center text-center">
        <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-line bg-surface-raised px-3 py-1 text-[11px] font-mono uppercase tracking-[0.14em] text-fg-muted">
          <span className="h-1.5 w-1.5 rounded-full bg-accent shadow-[0_0_8px_rgb(var(--accent)/0.8)]" />
          WK {week} · {season}
        </div>

        <h1 className="max-w-3xl font-display text-[4.25rem] leading-[0.9] tracking-normal text-fg sm:text-[6.5rem]">
          Gridiron Oracle
        </h1>
        <p className="mt-5 max-w-2xl text-base leading-7 text-fg-muted sm:text-lg">
          Search a player, open the projection, then move into weekly boards, season simulations, and backtest results from the nav.
        </p>

        {/* Search lives in the shared dialog (Cmd+K): one combobox for the
            whole site instead of a second hand-built one on this page. */}
        <div className="relative mt-9 w-full max-w-2xl">
          <button
            type="button"
            onClick={open}
            aria-label="Search players, teams, or positions"
            className="flex w-full items-center gap-3 rounded-lg border border-line bg-surface-raised px-4 py-3 transition-colors hover:border-accent/40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            <Search className="size-5 flex-shrink-0 text-fg-muted" aria-hidden="true" />
            <span className="min-w-0 flex-1 text-left text-base text-fg-subtle">
              Search players, teams, or positions
            </span>
            <span className="hidden rounded-md border border-line px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-fg-muted sm:block">
              Cmd K
            </span>
          </button>
        </div>

        <div className="mt-8 grid w-full max-w-3xl grid-cols-1 gap-3 sm:grid-cols-3">
          {QUICK_LINKS.map((link) => (
            <Link
              key={link.to}
              to={link.to}
              className="rounded-lg border border-line bg-surface-raised px-4 py-4 text-left transition-colors hover:border-accent/40 hover:bg-surface-overlay"
            >
              <span className="block text-sm font-semibold text-fg">{link.label}</span>
              <span className="mt-1 block text-xs text-fg-muted">{link.meta}</span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  )
}
