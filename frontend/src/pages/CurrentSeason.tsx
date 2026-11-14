import { useState, useMemo, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useCurrentSeason } from '@/hooks/useCurrentSeason'
import { useSeasonProjections } from '@/hooks/useSeasonProjections'
import { useSeasonWeekProjections } from '@/hooks/useSeasonWeekProjections'
import { useTeamWins } from '@/hooks/useTeamWins'
import { useTeamGames } from '@/hooks/useTeamGames'
import { useSearch } from '@/context/SearchContext'
import { CardGridSkeleton, LoadingRegion, Skeleton, TableSkeleton } from '@/components/shared/Skeletons'
import { NotReady, NOT_READY_PRIMARY, NOT_READY_SECONDARY } from '@/components/shared/NotReady'
import { usePageTitle } from '@/hooks/usePageTitle'
import { PAGE_NAMES } from '@/lib/page-names'
import { AVAILABILITY_UNKNOWN_NOTE, isAvailabilityUnknown } from '@/lib/availability'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { DataStalenessWarning } from '@/components/shared/DataStalenessWarning'
import { NoForecastBanner } from '@/components/shared/NoForecastBanner'
import { isForecastRow, needsSimulationNotice, noticeLabel, toSearchablePlayers } from '@/lib/forecast-provenance'
import { ProjectionList } from '@/components/ProjectionList'
import { useProjectionView } from '@/hooks/useProjectionView'
import { CURRENT_SEASON, POSITIONS, SEASON_SIM_STATS, STAT_LABELS, type SeasonSimStat } from '@/lib/constants'
import { positionTone } from '@/lib/tokens'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'

type Tab = 'weekly' | 'season' | 'games'

const WEEK_ITEMS = Array.from({ length: 18 }, (_, i) => ({
  value: String(i + 1),
  label: `Week ${i + 1}`,
}))
const START_WEEK_ITEMS = Array.from({ length: 18 }, (_, i) => ({
  value: String(i + 1),
  label: `From Week ${i + 1}`,
}))
const STAT_ITEMS = SEASON_SIM_STATS.map((s) => ({ value: s, label: STAT_LABELS[s] || s }))

export function CurrentSeason() {
  const [listView, setListView] = useProjectionView()
  // isPlaceholderData is true while /season/current is still in flight, during
  // which `season` is the offline fallback. Firing the child queries against it
  // produced a full duplicate request set for the wrong season on every mount,
  // including a /team-games 404 in the console.
  const { data: seasonMeta, isPlaceholderData, isError: currentFailed, error: currentError, refetch: refetchCurrent } = useCurrentSeason()
  // Until /season/current answers, `season` and `week` are fallbacks. Nothing that depends
  // on them (week picker, season badge, lists) shows a value, so no wrong "Week 1" flashes.
  const seasonResolved = !isPlaceholderData && !currentFailed
  usePageTitle(PAGE_NAMES.projections)
  const season = seasonMeta?.season ?? CURRENT_SEASON
  const currentWeek = seasonMeta?.week ?? 1

  const [activeTab, setActiveTab] = useState<Tab>('weekly')
  // The week the user picked, or null to follow the current week. Deriving the
  // week (instead of copying seasonMeta.week into state from an effect) means the
  // first request after /season/current resolves already asks for the right week.
  // The copy left one render on the offline fallback week, so the page requested
  // that wrong week, discarded it, and flashed a skeleton over the real cards.
  const [pickedWeek, setPickedWeek] = useState<number | null>(null)
  const selectedWeek = pickedWeek ?? currentWeek
  const [startWeek, setStartWeek] = useState<number>(1)
  const [statKey, setStatKey] = useState<SeasonSimStat>('fantasy_ppr')
  const [selectedPositions, setSelectedPositions] = useState<string[]>(['WR', 'RB', 'TE', 'QB'])
  // Simulated numbers are hidden on the Weekly tab until the visitor asks for it.
  const [showSimulation, setShowSimulation] = useState(false)

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
    // Search opens the player page, which is a weekly page. So search always shows
    // the weekly figure (or none), whichever tab is open, never a season total.
    const weeklyList = weeklyQuery.data?.projections
    if (!weeklyList) return
    setPlayers(toSearchablePlayers(weeklyList, statKey))
    setDefaults({ season, week: selectedWeek, stat: statKey })
  }, [weeklyQuery.data, statKey, season, selectedWeek, setPlayers, setDefaults])

  const playersList = useMemo(() => {
    const source = activeTab === 'weekly' ? weeklyQuery.data?.projections : seasonQuery.data?.projections
    return source ?? []
  }, [activeTab, weeklyQuery.data, seasonQuery.data])

  const weeklyRows = weeklyQuery.data?.projections
  const weeklyNeedsNotice = activeTab === 'weekly' && needsSimulationNotice(weeklyRows ?? [])
  // Rows shown on the Weekly tab: real forecasts always, simulated rows only on request.
  const visibleList = useMemo(
    () => (activeTab === 'weekly' && !showSimulation ? playersList.filter(isForecastRow) : playersList),
    [activeTab, showSimulation, playersList],
  )
  const simulationShown = weeklyNeedsNotice && showSimulation

  const freshness = activeTab === 'weekly' ? weeklyQuery.data?.data_freshness : seasonQuery.data?.data_freshness

  const listQuery = activeTab === 'weekly' ? weeklyQuery : seasonQuery
  // isPending (not isLoading) is true for a query that is waiting to start, so the
  // empty state can never show before the first answer.
  const listPending = !seasonResolved || listQuery.isPending
  const allPositions = selectedPositions.length >= POSITIONS.length
  const anyAvailabilityUnknown =
    activeTab === 'weekly' && visibleList.some((r) => isAvailabilityUnknown(r.availability_status))

  if (currentFailed) {
    return (
      <div className="flex flex-col gap-6">
        <h1 className="text-2xl font-bold font-display tracking-wider text-fg">{PAGE_NAMES.projections}</h1>
        <ErrorBanner error={currentError} onRetry={() => void refetchCurrent()} />
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-line/60 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold font-display tracking-wider text-fg">
              {PAGE_NAMES.projections}
            </h1>
            {seasonResolved ? (
              <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-accent/10 text-accent border border-accent/20">
                {season} Season
              </span>
            ) : (
              <Skeleton className="h-6 w-24 rounded-full" />
            )}
          </div>
          <p className="text-xs text-fg-muted mt-1">
            Rest-of-season projections, team win totals, and matchup forecasts.
          </p>
        </div>

        {/* Tab switcher */}
        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as Tab)}>
          <TabsList aria-label="Season view">
            <TabsTrigger value="weekly">Weekly</TabsTrigger>
            <TabsTrigger value="season">Season-Long</TabsTrigger>
            <TabsTrigger value="games">Games</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {freshness && <DataStalenessWarning dataFreshness={freshness} />}

      {/* Control bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-surface-raised/40 p-4 rounded-xl border border-line/40">
        <div className="flex flex-wrap items-center gap-3">
          {(activeTab === 'weekly' || activeTab === 'games') && (
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-fg-muted" id="week-label">Week:</span>
              {seasonResolved ? (
                <Select
                  items={WEEK_ITEMS}
                  value={String(selectedWeek)}
                  onValueChange={(v) => { if (v) setPickedWeek(Number(v)) }}
                >
                  <SelectTrigger aria-labelledby="week-label" className="h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {WEEK_ITEMS.map((item) => (
                      <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              ) : (
                <Skeleton className="h-8 w-24 rounded-md" />
              )}
            </div>
          )}

          {activeTab === 'season' && (
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-fg-muted" id="start-week-label">Start Week:</span>
              <Select
                items={START_WEEK_ITEMS}
                value={String(startWeek)}
                onValueChange={(v) => { if (v) setStartWeek(Number(v)) }}
              >
                <SelectTrigger aria-labelledby="start-week-label" className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {START_WEEK_ITEMS.map((item) => (
                    <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {activeTab !== 'games' && (
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-fg-muted" id="stat-label">Stat:</span>
              <Select
                items={STAT_ITEMS}
                value={statKey}
                onValueChange={(v) => { if (v) setStatKey(v as SeasonSimStat) }}
              >
                <SelectTrigger aria-labelledby="stat-label" className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {STAT_ITEMS.map((item) => (
                    <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>

        {activeTab !== 'games' && (
          <ToggleGroup
            multiple
            aria-label="Positions"
            value={selectedPositions}
            onValueChange={(next) => { if (next.length > 0) setSelectedPositions(next) }}
          >
            {POSITIONS.map((pos) => (
              <ToggleGroupItem key={pos} value={pos} tone={positionTone(pos)}>
                {pos}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
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
            {!seasonResolved || teamWinsQuery.isPending ? (
              <LoadingRegion label="Loading win totals">
                <CardGridSkeleton compact count={8} />
              </LoadingRegion>
            ) : teamWinsQuery.error ? (
              <ErrorBanner error={teamWinsQuery.error} onRetry={() => void teamWinsQuery.refetch()} />
            ) : !teamWinsQuery.data || teamWinsQuery.data.teams.length === 0 ? (
              <NotReady
                title="Season win totals aren't ready yet"
                actions={
                  <>
                    <button type="button" onClick={() => void teamWinsQuery.refetch()} className={NOT_READY_PRIMARY}>
                      Check again
                    </button>
                    <button type="button" onClick={() => setActiveTab('season')} className={NOT_READY_SECONDARY}>
                      See season-long projections
                    </button>
                  </>
                }
              >
                Team win totals appear here once they are published.
              </NotReady>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-2">
                {teamWinsQuery.data.teams.slice(0, 16).map((t) => (
                  <div
                    key={t.team}
                    className="p-2.5 rounded-lg bg-surface-raised/60 border border-line/50 flex flex-col items-center"
                  >
                    <span className="font-bold text-xs text-fg">{t.team}</span>
                    <span className="font-display text-xl text-accent mt-0.5">
                      {t.wins_mean.toFixed(1)}
                    </span>
                    <span className="text-xs text-fg-muted font-mono">
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
            {!seasonResolved || teamGamesQuery.isPending ? (
              <LoadingRegion label="Loading matchups">
                <CardGridSkeleton wide count={4} />
              </LoadingRegion>
            ) : teamGamesQuery.error ? (
              <ErrorBanner error={teamGamesQuery.error} onRetry={() => void teamGamesQuery.refetch()} />
            ) : teamGamesQuery.data === null ? (
              <NotReady
                title={`Matchup forecasts for Week ${selectedWeek} aren't ready yet`}
                actions={
                  <button type="button" onClick={() => setActiveTab('season')} className={NOT_READY_PRIMARY}>
                    See season-long projections
                  </button>
                }
              >
                Season win totals are above. Season-long player projections are ready now.
              </NotReady>
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
                          <span className="text-xs font-mono text-accent mt-1">
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
        <div className="flex flex-col gap-4">
          {weeklyNeedsNotice && (
            <NoForecastBanner
              scope="week"
              season={season}
              week={selectedWeek}
              onOpenSeasonLong={() => setActiveTab('season')}
            >
              <button
                type="button"
                aria-expanded={showSimulation}
                onClick={() => setShowSimulation((v) => !v)}
                className="inline-flex items-center text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors border border-white/10 text-fg-muted hover:text-fg hover:border-white/20"
              >
                {showSimulation ? 'Hide these numbers' : 'Show these numbers anyway'}
              </button>
            </NoForecastBanner>
          )}
          {anyAvailabilityUnknown && (
            <p className="text-xs text-fg-muted">{AVAILABILITY_UNKNOWN_NOTE}</p>
          )}
          {simulationShown && (
            <h2 className="text-sm font-semibold uppercase tracking-wider text-fg-muted">
              {noticeLabel(weeklyRows ?? [])}
            </h2>
          )}
          {listPending ? (
            <LoadingRegion label="Loading player projections">
              {listView === 'table' ? <TableSkeleton rows={10} columns={5} /> : <CardGridSkeleton count={8} />}
            </LoadingRegion>
          ) : listQuery.error ? (
            <ErrorBanner error={listQuery.error} onRetry={() => void listQuery.refetch()} />
          ) : visibleList.length === 0 ? (
            // On the Weekly tab with only simulated rows the banner above is the answer.
            weeklyNeedsNotice ? null : allPositions ? (
              <NotReady
                title={
                  activeTab === 'weekly'
                    ? `Week ${selectedWeek} projections aren't ready yet`
                    : `${season} season-long projections aren't ready yet`
                }
                actions={
                  <>
                    <button type="button" onClick={() => void listQuery.refetch()} className={NOT_READY_PRIMARY}>
                      Check again
                    </button>
                    {activeTab === 'weekly' && (
                      <button type="button" onClick={() => setActiveTab('season')} className={NOT_READY_SECONDARY}>
                        See season-long projections
                      </button>
                    )}
                  </>
                }
              >
                Projections show up here as soon as they are published.
              </NotReady>
            ) : (
              <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-line p-10 text-center">
                <p className="text-sm text-fg-muted">
                  No {selectedPositions.join(', ')} players to show for this view.
                </p>
                <button type="button" onClick={() => setSelectedPositions([...POSITIONS])} className={NOT_READY_SECONDARY}>
                  Show all positions
                </button>
              </div>
            )
          ) : (
            <ProjectionList
              key={activeTab}
              rows={visibleList}
              statKey={statKey}
              scope={activeTab === 'weekly' ? 'weekly' : 'season'}
              season={season}
              week={selectedWeek}
              view={listView}
              onViewChange={setListView}
              onResetPositions={allPositions ? undefined : () => setSelectedPositions([...POSITIONS])}
            />
          )}
        </div>
      )}
    </div>
  )
}
