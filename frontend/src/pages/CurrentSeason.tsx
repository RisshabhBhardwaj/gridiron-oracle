import { useState, useMemo, useEffect, useCallback } from 'react'
import clsx from 'clsx'
import { useNavigate, Link } from 'react-router-dom'
import { useCurrentSeason } from '@/hooks/useCurrentSeason'
import { useSeasonProjections } from '@/hooks/useSeasonProjections'
import { useSeasonWeekProjections } from '@/hooks/useSeasonWeekProjections'
import { useTeamWins } from '@/hooks/useTeamWins'
import { useTeamGames } from '@/hooks/useTeamGames'
import { useSearch } from '@/context/SearchContext'
import { SkeletonCard } from '@/components/shared/LoadingSpinner'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { DataStalenessWarning } from '@/components/shared/DataStalenessWarning'
import { CURRENT_SEASON, POSITIONS, SEASON_SIM_STATS, STAT_LABELS, type SeasonSimStat } from '@/lib/constants'
import type { SeasonPlayerProjection } from '@/types/api'
import { PositionBadge } from '@/components/PositionBadge'
import { positionTone, toneClass } from '@/lib/tokens'

type Tab = 'weekly' | 'season' | 'games'

interface PlayerCardProps {
  row: SeasonPlayerProjection
  onClick: () => void
  statKey: SeasonSimStat
}

function PlayerCard({ row, onClick, statKey }: PlayerCardProps) {
  const statData = row[statKey]
  const projection = statData?.mean ?? 0
  const floor = statData?.p10 ?? null
  const ceiling = statData?.p90 ?? null

  // The simulator's weekly distributions are zero-inflated: a player who is
  // inactive in a minority of paths gets a point mass at 0 that drags the mean
  // below p10 (or, for a stat he rarely records at all, above p90). That is a
  // real property of the model, not a bug — but a headline number printed
  // outside the interval directly beneath it reads as broken, so say why.
  const meanOutsideBand =
    (floor != null && projection < floor) || (ceiling != null && projection > ceiling)
  const bandTitle = meanOutsideBand
    ? 'The mean falls outside the middle 80% because this stat is zero-inflated: ' +
      'some simulated paths have the player inactive or not recording the stat at all.'
    : undefined

  if (!statData) return null

  return (
    <div
      className={clsx(
        toneClass(positionTone(row.position)),
        'relative rounded-xl overflow-hidden flex flex-col gap-3 p-4 group cursor-pointer transition-all duration-200 hover:-translate-y-0.5',
        'bg-panel-gradient border border-white/[0.06] shadow-[0_2px_12px_rgb(var(--shadow)/0.3)]',
      )}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter') onClick() }}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-fg truncate leading-tight">{row.player_name}</p>
          <p className="text-[11px] text-fg-muted mt-0.5">{row.team ?? 'UNK'}</p>
        </div>
        <PositionBadge position={row.position} />
      </div>

      <div className="flex items-baseline gap-1.5 mt-1">
        <span className="glow-text font-display text-4xl leading-none text-tone">
          {projection.toFixed(1)}
        </span>
        <span className="text-xs text-fg-muted">{STAT_LABELS[statKey] || statKey}</span>
      </div>

      <div className="flex flex-col gap-1 mt-1">
        <div className="h-1.5 rounded-full overflow-hidden bg-surface-overlay/80">
          <div className="h-full w-full rounded-full bg-gradient-to-r from-negative/60 via-tone/50 to-positive/60" />
        </div>
        <div className="flex justify-between font-mono text-[10px] text-fg-muted" title={bandTitle}>
          <span>{floor != null ? `${floor.toFixed(1)} (p10)` : '—'}</span>
          {meanOutsideBand && (
            <span className="text-fg-muted/70" aria-label="mean outside middle 80 percent">
              zero-inflated
            </span>
          )}
          <span>{ceiling != null ? `${ceiling.toFixed(1)} (p90)` : '—'}</span>
        </div>
      </div>
    </div>
  )
}

export function CurrentSeason() {
  const navigate = useNavigate()
  // isPlaceholderData is true while /season/current is still in flight, during
  // which `season` is the offline fallback. Firing the child queries against it
  // produced a full duplicate request set for the wrong season on every mount,
  // including a /team-games 404 in the console.
  const { data: seasonMeta, isPlaceholderData } = useCurrentSeason()
  const seasonResolved = !isPlaceholderData
  const season = seasonMeta?.season ?? CURRENT_SEASON
  const currentWeek = seasonMeta?.week ?? 1

  const [activeTab, setActiveTab] = useState<Tab>('weekly')
  const [selectedWeek, setSelectedWeek] = useState<number>(currentWeek)
  const [startWeek, setStartWeek] = useState<number>(1)
  const [statKey, setStatKey] = useState<SeasonSimStat>('fantasy_ppr')
  const [selectedPositions, setSelectedPositions] = useState<string[]>(['WR', 'RB', 'TE', 'QB'])

  useEffect(() => {
    if (seasonMeta?.week) {
      setSelectedWeek(seasonMeta.week)
    }
  }, [seasonMeta?.week])

  // Projections queries
  const weeklyQuery = useSeasonWeekProjections({
    season,
    week: selectedWeek,
    startWeek,
    positions: selectedPositions,
    enabled: seasonResolved,
  })

  const seasonQuery = useSeasonProjections({
    season,
    startWeek,
    positions: selectedPositions,
    enabled: seasonResolved,
  })

  const teamWinsQuery = useTeamWins({
    season,
    startWeek,
    enabled: seasonResolved,
  })

  const teamGamesQuery = useTeamGames({
    season,
    week: selectedWeek,
    enabled: seasonResolved,
  })

  const { setPlayers, setDefaults } = useSearch()

  // Feed search context
  useEffect(() => {
    const activeList = activeTab === 'weekly' ? weeklyQuery.data?.projections : seasonQuery.data?.projections
    if (!activeList) return
    const searchable = activeList.map((p) => ({
      player_id: p.player_id,
      player_name: p.player_name,
      position: p.position,
      team: p.team,
      stat: statKey,
      projection: p[statKey]?.mean ?? 0,
    }))
    setPlayers(searchable)
    setDefaults({ season, week: selectedWeek, stat: statKey })
  }, [activeTab, weeklyQuery.data, seasonQuery.data, statKey, season, selectedWeek, setPlayers, setDefaults])

  const togglePosition = useCallback((pos: string) => {
    setSelectedPositions((prev) =>
      prev.includes(pos)
        ? prev.length === 1 ? prev : prev.filter((p) => p !== pos)
        : [...prev, pos],
    )
  }, [])

  const playersList = useMemo(() => {
    const source = activeTab === 'weekly' ? weeklyQuery.data?.projections : seasonQuery.data?.projections
    if (!source) return []
    return [...source].sort((a, b) => {
      const aVal = a[statKey]?.mean ?? 0
      const bVal = b[statKey]?.mean ?? 0
      return bVal - aVal
    })
  }, [activeTab, weeklyQuery.data, seasonQuery.data, statKey])

  const freshness = activeTab === 'weekly' ? weeklyQuery.data?.data_freshness : seasonQuery.data?.data_freshness

  return (
    <div className="flex flex-col gap-6 p-4 sm:p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-line/60 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold font-display tracking-wider text-fg">
              Current Season
            </h1>
            <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-accent/10 text-accent border border-accent/20">
              {season} Season
            </span>
          </div>
          <p className="text-xs text-fg-muted mt-1">
            Simulated rest-of-season weekly & aggregate projections, team win totals, and matchup forecasts.
          </p>
        </div>

        {/* Tab switcher */}
        <div className="flex items-center rounded-lg bg-surface-overlay p-1 border border-line">
          <button
            onClick={() => setActiveTab('weekly')}
            className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-colors ${
              activeTab === 'weekly'
                ? 'bg-accent text-accent-fg shadow'
                : 'text-fg-muted hover:text-fg'
            }`}
          >
            Weekly
          </button>
          <button
            onClick={() => setActiveTab('season')}
            className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-colors ${
              activeTab === 'season'
                ? 'bg-accent text-accent-fg shadow'
                : 'text-fg-muted hover:text-fg'
            }`}
          >
            Season-Long
          </button>
          <button
            onClick={() => setActiveTab('games')}
            className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-colors ${
              activeTab === 'games'
                ? 'bg-accent text-accent-fg shadow'
                : 'text-fg-muted hover:text-fg'
            }`}
          >
            Games
          </button>
        </div>
      </div>

      {freshness && <DataStalenessWarning dataFreshness={freshness} />}

      {/* Control bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-surface-raised/40 p-4 rounded-xl border border-line/40">
        <div className="flex flex-wrap items-center gap-3">
          {(activeTab === 'weekly' || activeTab === 'games') && (
            <div className="flex items-center gap-2">
              <label className="text-xs font-medium text-fg-muted">Week:</label>
              <select
                aria-label="Select week"
                value={selectedWeek}
                onChange={(e) => setSelectedWeek(Number(e.target.value))}
                className="oracle-select bg-surface-overlay border border-line text-xs rounded-lg px-2.5 py-1.5 text-fg"
              >
                {Array.from({ length: 18 }, (_, i) => i + 1).map((w) => (
                  <option key={w} value={w}>
                    Week {w}
                  </option>
                ))}
              </select>
            </div>
          )}

          {activeTab === 'season' && (
            <div className="flex items-center gap-2">
              <label className="text-xs font-medium text-fg-muted">Start Week:</label>
              <select
                aria-label="Select start week"
                value={startWeek}
                onChange={(e) => setStartWeek(Number(e.target.value))}
                className="oracle-select bg-surface-overlay border border-line text-xs rounded-lg px-2.5 py-1.5 text-fg"
              >
                {Array.from({ length: 18 }, (_, i) => i + 1).map((w) => (
                  <option key={w} value={w}>
                    From Week {w}
                  </option>
                ))}
              </select>
            </div>
          )}

          {activeTab !== 'games' && (
            <div className="flex items-center gap-2">
              <label className="text-xs font-medium text-fg-muted">Stat:</label>
              <select
                aria-label="Select stat"
                value={statKey}
                onChange={(e) => setStatKey(e.target.value as SeasonSimStat)}
                className="oracle-select bg-surface-overlay border border-line text-xs rounded-lg px-2.5 py-1.5 text-fg"
              >
                {SEASON_SIM_STATS.map((s) => (
                  <option key={s} value={s}>
                    {STAT_LABELS[s] || s}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {activeTab !== 'games' && (
          <div className="flex items-center gap-1.5">
            {POSITIONS.map((pos) => {
              const active = selectedPositions.includes(pos)
              return (
                <button
                  key={pos}
                  onClick={() => togglePosition(pos)}
                  aria-pressed={active}
                  className={clsx(
                    toneClass(positionTone(pos)),
                    'px-2.5 py-1 text-xs font-semibold rounded-md border transition-colors',
                    active
                      ? 'bg-tone text-accent-fg border-transparent'
                      : 'border-line text-fg-muted hover:border-line-strong',
                  )}
                >
                  {pos}
                </button>
              )
            })}
          </div>
        )}
      </div>

      {/* Main Tab Content */}
      {activeTab === 'games' ? (
        <div className="flex flex-col gap-6">
          {/* Team Wins Header */}
          <div className="flex flex-col gap-3">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-fg-muted">
              Projected Season Win Totals
            </h2>
            {teamWinsQuery.isLoading ? (
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-2">
                {Array.from({ length: 8 }).map((_, i) => (
                  <div key={i} className="h-16 rounded-lg bg-surface-overlay animate-pulse" />
                ))}
              </div>
            ) : teamWinsQuery.error ? (
              <ErrorBanner error={teamWinsQuery.error} />
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-2">
                {teamWinsQuery.data?.teams.slice(0, 16).map((t) => (
                  <div
                    key={t.team}
                    className="p-2.5 rounded-lg bg-surface-raised/60 border border-line/50 flex flex-col items-center"
                  >
                    <span className="font-bold text-xs text-fg">{t.team}</span>
                    <span className="font-display text-xl text-accent mt-0.5">
                      {t.wins_mean.toFixed(1)}
                    </span>
                    <span className="text-[10px] text-fg-muted font-mono">
                      {t.wins_p10 != null && t.wins_p90 != null ? `${t.wins_p10.toFixed(0)}–${t.wins_p90.toFixed(0)}` : 'wins'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Weekly Matchup Games */}
          <div className="flex flex-col gap-3">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-fg-muted">
              Week {selectedWeek} Matchup Projections
            </h2>
            {teamGamesQuery.isLoading ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Array.from({ length: 4 }).map((_, i) => (
                  <div key={i} className="h-28 rounded-xl bg-surface-overlay animate-pulse" />
                ))}
              </div>
            ) : teamGamesQuery.error ? (
              <ErrorBanner error={teamGamesQuery.error} />
            ) : (teamGamesQuery.data || []).length === 0 ? (
              <div className="text-center p-8 text-fg-muted border border-dashed border-line rounded-xl">
                No games scheduled or predicted for Week {selectedWeek}.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {teamGamesQuery.data?.map((game) => (
                  <Link
                    key={game.game_id}
                    to={`/projections/game/${game.game_id}`}
                    className="p-4 rounded-xl bg-surface-raised/40 border border-line/50 hover:border-accent/40 hover:bg-surface-raised/80 transition-all flex flex-col gap-3 group"
                  >
                    <div className="flex items-center justify-between text-xs text-fg-muted">
                      <span>Week {game.week}</span>
                      <span className="font-mono text-accent group-hover:underline flex items-center gap-1">
                        Drive Sim →
                      </span>
                    </div>

                    <div className="flex items-center justify-between gap-4">
                      {/* Away Team */}
                      <div className="flex-1 flex flex-col">
                        <span className="text-base font-bold text-fg">{game.away_team}</span>
                        <span className="text-xs text-fg-muted">Away</span>
                        <span className="text-xl font-display text-fg mt-1">
                          {game.away_points != null ? game.away_points.toFixed(1) : '—'} pts
                        </span>
                      </div>

                      <div className="flex flex-col items-center justify-center px-3">
                        <span className="text-xs font-mono text-fg-muted">VS</span>
                        {game.home_win_probability != null && (
                          <span className="text-[10px] font-mono text-accent mt-1">
                            {(game.home_win_probability * 100).toFixed(0)}% Home Win
                          </span>
                        )}
                      </div>

                      {/* Home Team */}
                      <div className="flex-1 flex flex-col items-end">
                        <span className="text-base font-bold text-fg">{game.home_team}</span>
                        <span className="text-xs text-fg-muted">Home</span>
                        <span className="text-xl font-display text-fg mt-1">
                          {game.home_points != null ? game.home_points.toFixed(1) : '—'} pts
                        </span>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Player Projections Grid */
        <div>
          {(activeTab === 'weekly' ? weeklyQuery.isLoading : seasonQuery.isLoading) ? (
            <SkeletonCard count={8} />
          ) : (activeTab === 'weekly' ? weeklyQuery.error : seasonQuery.error) ? (
            <ErrorBanner error={(activeTab === 'weekly' ? weeklyQuery.error : seasonQuery.error)!} />
          ) : playersList.length === 0 ? (
            <div className="text-center p-12 text-fg-muted border border-dashed border-line rounded-xl">
              No player projections found matching the active filters.
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {playersList.map((player) => (
                <PlayerCard
                  key={player.player_id}
                  row={player}
                  statKey={statKey}
                  onClick={() => navigate(`/player/${player.player_id}`)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
