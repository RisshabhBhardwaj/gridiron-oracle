import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import clsx from 'clsx'
import { useCurrentSeason } from '@/hooks/useCurrentSeason'
import { useWeekProjections } from '@/hooks/useWeekProjections'
import { useSearch } from '@/context/SearchContext'
import { CURRENT_SEASON, CURRENT_WEEK, POSITIONS, STAT_LABELS } from '@/lib/constants'

const POS_COLORS: Record<string, string> = {
  WR: '#59a6ff', RB: '#42d889', TE: '#b789ff', QB: '#ffb84d',
}

const QUICK_LINKS = [
  { to: '/projections', label: 'All Projections', meta: 'Weekly player board' },
  { to: '/season', label: 'Season Projections', meta: 'Rest-of-season totals' },
  { to: '/backtest', label: 'Backtesting', meta: 'Accuracy and calibration' },
]

export function Home() {
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const [focused, setFocused] = useState(false)
  const { data: currentSeasonData } = useCurrentSeason()
  const week = currentSeasonData?.week ?? CURRENT_WEEK
  const season = currentSeasonData?.season ?? CURRENT_SEASON

  const {
    query, setQuery, results, setPlayers, setDefaults, open,
  } = useSearch()

  const { data, isLoading } = useWeekProjections({
    week,
    season,
    stat: 'receiving_yards',
    positions: [...POSITIONS],
  })

  useEffect(() => {
    setDefaults({ week, season, stat: 'receiving_yards' })
  }, [season, setDefaults, week])

  useEffect(() => {
    if (data?.projections) setPlayers(data.projections)
  }, [data?.projections, setPlayers])

  const visibleResults = useMemo(() => results.slice(0, 6), [results])
  const showResults = focused && visibleResults.length > 0

  function selectPlayer(playerId: string, name: string, stat?: string) {
    navigate(`/player/${playerId}?week=${week}&season=${season}&stat=${stat ?? 'receiving_yards'}&name=${encodeURIComponent(name)}`)
    setQuery('')
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' && visibleResults[0]) {
      e.preventDefault()
      selectPlayer(visibleResults[0].player_id, visibleResults[0].player_name, visibleResults[0].stat)
    }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault()
      open()
    }
  }

  return (
    <div className="min-h-[calc(100vh-53px)] flex flex-col justify-center py-12 sm:py-16">
      <section className="mx-auto flex w-full max-w-4xl flex-col items-center text-center">
        <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-oracle-border bg-oracle-card px-3 py-1 text-[11px] font-mono uppercase tracking-[0.14em] text-oracle-muted">
          <span className="h-1.5 w-1.5 rounded-full bg-oracle-green shadow-[0_0_8px_rgba(66,216,137,0.8)]" />
          WK {week} · {season}
        </div>

        <h1 className="max-w-3xl font-display text-[4.25rem] leading-[0.9] tracking-normal text-oracle-white sm:text-[6.5rem]">
          Gridiron Oracle
        </h1>
        <p className="mt-5 max-w-2xl text-base leading-7 text-oracle-muted sm:text-lg">
          Search a player, open the projection, then move into weekly boards, season simulations, and backtest results from the nav.
        </p>

        <div className="relative mt-9 w-full max-w-2xl">
          <div
            className={clsx(
              'flex items-center gap-3 rounded-lg border bg-oracle-card px-4 py-3 transition-colors',
              focused ? 'border-oracle-green/50 shadow-[0_0_0_3px_rgba(66,216,137,0.08)]' : 'border-oracle-border',
            )}
          >
            <svg className="h-5 w-5 flex-shrink-0 text-oracle-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M21 21l-5.2-5.2m1.7-5.1a6.8 6.8 0 11-13.6 0 6.8 6.8 0 0113.6 0z" />
            </svg>
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => setFocused(true)}
              onBlur={() => window.setTimeout(() => setFocused(false), 120)}
              onKeyDown={handleKeyDown}
              placeholder="Search players, teams, or positions"
              className="min-w-0 flex-1 bg-transparent text-left text-base font-medium text-oracle-white outline-none placeholder:text-oracle-subtle"
            />
            <button
              onClick={open}
              className="hidden rounded-md border border-oracle-border px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-oracle-muted transition-colors hover:border-oracle-green/40 hover:text-oracle-green sm:block"
              type="button"
            >
              Cmd K
            </button>
          </div>

          {showResults && (
            <div className="absolute left-0 right-0 top-[calc(100%+8px)] z-20 overflow-hidden rounded-lg border border-oracle-border bg-[#111111] text-left shadow-2xl">
              {visibleResults.map((player) => {
                const posColor = POS_COLORS[player.position] ?? '#77859a'
                return (
                  <button
                    key={player.player_id}
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => selectPlayer(player.player_id, player.player_name, player.stat)}
                    className="flex w-full items-center gap-3 border-b border-oracle-border/70 px-4 py-3 text-left transition-colors last:border-b-0 hover:bg-oracle-surface"
                    type="button"
                  >
                    <span className="rounded border px-1.5 py-0.5 font-mono text-[10px] font-bold" style={{ color: posColor, borderColor: `${posColor}55`, background: `${posColor}18` }}>
                      {player.position}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-semibold text-oracle-white">{player.player_name}</span>
                      <span className="block text-xs text-oracle-muted">{player.team ?? 'UNK'} · {STAT_LABELS[player.stat] ?? player.stat}</span>
                    </span>
                    <span className="font-mono text-sm font-semibold" style={{ color: posColor }}>
                      {player.projection.toFixed(1)}
                    </span>
                  </button>
                )
              })}
            </div>
          )}

          {!isLoading && focused && data && visibleResults.length === 0 && (
            <div className="absolute left-0 right-0 top-[calc(100%+8px)] z-20 rounded-lg border border-oracle-border bg-[#111111] px-4 py-5 text-sm text-oracle-muted shadow-2xl">
              No matching players found.
            </div>
          )}
        </div>

        <div className="mt-8 grid w-full max-w-3xl grid-cols-1 gap-3 sm:grid-cols-3">
          {QUICK_LINKS.map((link) => (
            <Link
              key={link.to}
              to={link.to}
              className="rounded-lg border border-oracle-border bg-oracle-card px-4 py-4 text-left transition-colors hover:border-oracle-green/40 hover:bg-oracle-surface"
            >
              <span className="block text-sm font-semibold text-oracle-white">{link.label}</span>
              <span className="mt-1 block text-xs text-oracle-muted">{link.meta}</span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  )
}
