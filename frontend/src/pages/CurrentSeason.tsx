import { useState, useMemo, useEffect, useCallback } from 'react'
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

type Tab = 'weekly' | 'season' | 'games'

const POS_CLASS: Record<string, string> = {
  WR: 'pos-wr',
  RB: 'pos-rb',
  TE: 'pos-te',
  QB: 'pos-qb',
}
const POS_COLORS: Record<string, string> = {
  WR: '#00C2FF',
  RB: '#00FFA3',
  TE: '#A855F7',
  QB: '#FFB800',
}

interface PlayerCardProps {
  row: SeasonPlayerProjection
  onClick: () => void
  statKey: SeasonSimStat
}

function PlayerCard({ row, onClick, statKey }: PlayerCardProps) {
  const posColor = POS_COLORS[row.position] ?? '#607B9B'
  const statData = row[statKey]
  const projection = statData?.mean ?? 0
  const floor = statData?.p10 ?? null
  const ceiling = statData?.p90 ?? null

  if (!statData) return null

  return (
    <div
      className="relative rounded-xl overflow-hidden flex flex-col gap-3 p-4 group cursor-pointer transition-all duration-200 hover:-translate-y-0.5"
      style={{
        background: 'linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.015) 100%)',
        border: '1px solid rgba(255,255,255,0.06)',
        boxShadow: '0 2px 12px rgba(0,0,0,0.3)',
      }}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter') onClick() }}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-oracle-white truncate leading-tight">{row.player_name}</p>
          <p className="text-[11px] text-oracle-muted mt-0.5">{row.team ?? 'UNK'}</p>
        </div>
        <span className={POS_CLASS[row.position] ?? 'pos-badge text-oracle-muted border-oracle-border'}>
          {row.position}
        </span>
      </div>

      <div className="flex items-baseline gap-1.5 mt-1">
        <span
          className="font-display text-4xl leading-none"
          style={{ color: posColor, textShadow: `0 0 16px ${posColor}60` }}
        >
          {projection.toFixed(1)}
        </span>
        <span className="text-xs text-oracle-muted">{STAT_LABELS[statKey] || statKey}</span>
      </div>

      <div className="flex flex-col gap-1 mt-1">
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(26,47,78,0.8)' }}>
          <div
            className="h-full rounded-full"
            style={{ width: '100%', background: `linear-gradient(90deg, rgba(255,59,92,0.6) 0%, ${posColor}80 50%, rgba(0,255,163,0.6) 100%)` }}
          />
        </div>
        <div className="flex justify-between font-mono text-[10px] text-oracle-muted">
          <span>{floor != null ? `${floor.toFixed(1)} (p10)` : '—'}</span>
          <span>{ceiling != null ? `${ceiling.toFixed(1)} (p90)` : '—'}</span>
        </div>
      </div>
    </div>
  )
}

export function CurrentSeason() {
  const navigate = useNavigate()
  const { data: seasonMeta } = useCurrentSeason()
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
  })

  const seasonQuery = useSeasonProjections({
    season,
    startWeek,
    positions: selectedPositions,
  })

  const teamWinsQuery = useTeamWins({
    season,
    startWeek,
  })

  const teamGamesQuery = useTeamGames({
    season,
    week: selectedWeek,
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
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-oracle-border/60 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold font-display tracking-wider text-oracle-white">
              Current Season
            </h1>
            <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-oracle-green/10 text-oracle-green border border-oracle-green/20">
              {season} Season
            </span>
          </div>
          <p className="text-xs text-oracle-muted mt-1">
            Simulated rest-of-season weekly & aggregate projections, team win totals, and matchup forecasts.
          </p>
        </div>

        {/* Tab switcher */}
        <div className="flex items-center rounded-lg bg-oracle-surface p-1 border border-oracle-border">
          <button
            onClick={() => setActiveTab('weekly')}
            className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-colors ${
              activeTab === 'weekly'
                ? 'bg-oracle-green text-oracle-dark shadow'
                : 'text-oracle-muted hover:text-oracle-white'
            }`}
          >
            Weekly
          </button>
          <button
            onClick={() => setActiveTab('season')}
            className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-colors ${
              activeTab === 'season'
                ? 'bg-oracle-green text-oracle-dark shadow'
                : 'text-oracle-muted hover:text-oracle-white'
            }`}
          >
            Season-Long
          </button>
          <button
            onClick={() => setActiveTab('games')}
            className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-colors ${
              activeTab === 'games'
                ? 'bg-oracle-green text-oracle-dark shadow'
                : 'text-oracle-muted hover:text-oracle-white'
            }`}
          >
            Games
          </button>
        </div>
      </div>

      {freshness && <DataStalenessWarning dataFreshness={freshness} />}

      {/* Control bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-oracle-card/40 p-4 rounded-xl border border-oracle-border/40">
        <div className="flex flex-wrap items-center gap-3">
          {(activeTab === 'weekly' || activeTab === 'games') && (
            <div className="flex items-center gap-2">
              <label className="text-xs font-medium text-oracle-muted">Week:</label>
              <select
                aria-label="Select week"
                value={selectedWeek}
                onChange={(e) => setSelectedWeek(Number(e.target.value))}
                className="oracle-select bg-oracle-surface border border-oracle-border text-xs rounded-lg px-2.5 py-1.5 text-oracle-white"
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
              <label className="text-xs font-medium text-oracle-muted">Start Week:</label>
              <select
                aria-label="Select start week"
                value={startWeek}
                onChange={(e) => setStartWeek(Number(e.target.value))}
                className="oracle-select bg-oracle-surface border border-oracle-border text-xs rounded-lg px-2.5 py-1.5 text-oracle-white"
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
              <label className="text-xs font-medium text-oracle-muted">Stat:</label>
              <select
                aria-label="Select stat"
                value={statKey}
                onChange={(e) => setStatKey(e.target.value as SeasonSimStat)}
                className="oracle-select bg-oracle-surface border border-oracle-border text-xs rounded-lg px-2.5 py-1.5 text-oracle-white"
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
                  className={`px-2.5 py-1 text-xs font-semibold rounded-md border transition-colors ${
                    active
                      ? `${POS_CLASS[pos]} text-oracle-dark border-transparent`
                      : 'border-oracle-border text-oracle-muted hover:border-oracle-border-bright'
                  }`}
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
            <h2 className="text-sm font-semibold uppercase tracking-wider text-oracle-muted">
              Projected Season Win Totals
            </h2>
            {teamWinsQuery.isLoading ? (
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-2">
                {Array.from({ length: 8 }).map((_, i) => (
                  <div key={i} className="h-16 rounded-lg bg-oracle-surface animate-pulse" />
                ))}
              </div>
            ) : teamWinsQuery.error ? (
              <ErrorBanner error={teamWinsQuery.error} />
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-2">
                {teamWinsQuery.data?.teams.slice(0, 16).map((t) => (
                  <div
                    key={t.team}
                    className="p-2.5 rounded-lg bg-oracle-card/60 border border-oracle-border/50 flex flex-col items-center"
                  >
                    <span className="font-bold text-xs text-oracle-white">{t.team}</span>
                    <span className="font-display text-xl text-oracle-green mt-0.5">
                      {t.wins_mean.toFixed(1)}
                    </span>
                    <span className="text-[10px] text-oracle-muted font-mono">
                      {t.wins_p10 != null && t.wins_p90 != null ? `${t.wins_p10.toFixed(0)}–${t.wins_p90.toFixed(0)}` : 'wins'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Weekly Matchup Games */}
          <div className="flex flex-col gap-3">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-oracle-muted">
              Week {selectedWeek} Matchup Projections
            </h2>
            {teamGamesQuery.isLoading ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Array.from({ length: 4 }).map((_, i) => (
                  <div key={i} className="h-28 rounded-xl bg-oracle-surface animate-pulse" />
                ))}
              </div>
            ) : teamGamesQuery.error ? (
              <ErrorBanner error={teamGamesQuery.error} />
            ) : (teamGamesQuery.data || []).length === 0 ? (
              <div className="text-center p-8 text-oracle-muted border border-dashed border-oracle-border rounded-xl">
                No games scheduled or predicted for Week {selectedWeek}.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {teamGamesQuery.data?.map((game) => (
                  <Link
                    key={game.game_id}
                    to={`/projections/game/${game.game_id}`}
                    className="p-4 rounded-xl bg-oracle-card/40 border border-oracle-border/50 hover:border-oracle-green/40 hover:bg-oracle-card/80 transition-all flex flex-col gap-3 group"
                  >
                    <div className="flex items-center justify-between text-xs text-oracle-muted">
                      <span>Week {game.week}</span>
                      <span className="font-mono text-oracle-green group-hover:underline flex items-center gap-1">
                        Drive Sim →
                      </span>
                    </div>

                    <div className="flex items-center justify-between gap-4">
                      {/* Away Team */}
                      <div className="flex-1 flex flex-col">
                        <span className="text-base font-bold text-oracle-white">{game.away_team}</span>
                        <span className="text-xs text-oracle-muted">Away</span>
                        <span className="text-xl font-display text-oracle-white mt-1">
                          {game.away_points != null ? game.away_points.toFixed(1) : '—'} pts
                        </span>
                      </div>

                      <div className="flex flex-col items-center justify-center px-3">
                        <span className="text-xs font-mono text-oracle-muted">VS</span>
                        {game.home_win_probability != null && (
                          <span className="text-[10px] font-mono text-oracle-green mt-1">
                            {(game.home_win_probability * 100).toFixed(0)}% Home Win
                          </span>
                        )}
                      </div>

                      {/* Home Team */}
                      <div className="flex-1 flex flex-col items-end">
                        <span className="text-base font-bold text-oracle-white">{game.home_team}</span>
                        <span className="text-xs text-oracle-muted">Home</span>
                        <span className="text-xl font-display text-oracle-white mt-1">
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
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {Array.from({ length: 8 }).map((_, i) => (
                <SkeletonCard key={i} />
              ))}
            </div>
          ) : (activeTab === 'weekly' ? weeklyQuery.error : seasonQuery.error) ? (
            <ErrorBanner error={(activeTab === 'weekly' ? weeklyQuery.error : seasonQuery.error)!} />
          ) : playersList.length === 0 ? (
            <div className="text-center p-12 text-oracle-muted border border-dashed border-oracle-border rounded-xl">
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
