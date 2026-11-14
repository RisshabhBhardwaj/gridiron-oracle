import { useEffect, useState } from 'react'
import { ChevronLeft } from 'lucide-react'
import { useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { usePredict } from '@/hooks/usePredict'
import { useOdds, findPlayerLine } from '@/hooks/useOdds'
import { useCurrentSeason } from '@/hooks/useCurrentSeason'
import { useSeasonProjections } from '@/hooks/useSeasonProjections'
import { ExplainPanel } from '@/components/ExplainPanel'
import { PlayerDetailSkeleton } from '@/components/PlayerDetailSkeleton'
import { PlayerHero } from '@/components/PlayerHero'
import { PositionBadge } from '@/components/PositionBadge'
import { ProjectionCard } from '@/components/ProjectionCard'
import { SimulationFigure } from '@/components/SimulationFigure'
import { WhatIfStudio } from '@/components/WhatIfStudio'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { NoForecastBanner } from '@/components/shared/NoForecastBanner'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useSearch } from '@/context/SearchContext'
import { calcEdge } from '@/lib/edge'
import { provenanceLabel, weekFigureLabel } from '@/lib/forecast-provenance'
import { isCurrentOrLater, isLiveLineWeek, isNoForecastError, parseWeek, repeatsHeroFigure, weekFigure } from '@/lib/player-page'
import { CURRENT_SEASON, CURRENT_WEEK, LAST_FANTASY_WEEK, STATS, STAT_LABELS, statUnit } from '@/lib/constants'
import { toneClass } from '@/lib/tokens'
import { FOCUS_RING } from '@/lib/utils'
import { usePageTitle } from '@/hooks/usePageTitle'
import { AVAILABILITY_UNKNOWN_NOTE, isAvailabilityUnknown } from '@/lib/availability'
import type { SeasonPlayerProjection, SeasonStatProjection } from '@/types/api'

const STAT_ITEMS = STATS.map((s) => ({ value: s, label: STAT_LABELS[s] ?? s }))

function parseSeason(raw: string | null, fallback: number): number {
  if (raw === null) return fallback
  const n = Number(raw)
  return Number.isInteger(n) && n >= 2000 && n <= 2100 ? n : fallback
}

export function PlayerDetail() {
  const navigate = useNavigate()
  const location = useLocation()
  const { player_id } = useParams<{ player_id: string }>()
  const [searchParams, setSearchParams] = useSearchParams()
  const { setDefaults, players: searchPlayers } = useSearch()
  const [showSimulation, setShowSimulation] = useState(false)

  // A link without a week or season means "now". Hold every request until the real
  // current week is known, so the page never asks for a placeholder week first.
  const { data: current, isPlaceholderData: currentIsPlaceholder, isError: currentFailed, error: currentError, refetch: refetchCurrent } = useCurrentSeason()
  const needsCurrent = parseWeek(searchParams.get('week'), -1) === -1 || parseSeason(searchParams.get('season'), -1) === -1
  // If the current week cannot be learned, ask for nothing rather than a guessed week.
  const currentUnknown = needsCurrent && currentFailed
  const waitingForCurrent = needsCurrent && (currentIsPlaceholder || currentFailed)
  const now = { season: current?.season ?? CURRENT_SEASON, week: current?.week ?? CURRENT_WEEK }

  const week = parseWeek(searchParams.get('week'), now.week)
  const season = parseSeason(searchParams.get('season'), now.season)
  const stat = searchParams.get('stat') ?? 'receiving_yards'
  const name = searchParams.get('name') ?? ''
  const viewed = { season, week }
  usePageTitle(name || 'Player')

  const predict = usePredict({ player: name, week, season, stat, enabled: name.length > 0 && !waitingForCurrent })
  const figure = weekFigure(predict.data)

  // One season query serves two jobs: who this is (team, position) when /predict has no
  // answer, and the opt-in rest-of-season simulation. Unfiltered, so it needs no position.
  const rosStartWeek = Math.min(week + 1, LAST_FANTASY_WEEK)
  const seasonQuery = useSeasonProjections({
    season,
    startWeek: rosStartWeek,
    positions: [],
    enabled: name.length > 0 && !waitingForCurrent,
  })
  const seasonRow: SeasonPlayerProjection | undefined = seasonQuery.data?.projections.find((p) => p.player_id === player_id)
  const searchRow = searchPlayers.find((p) => p.player_id === player_id)
  const position = predict.data?.position ?? seasonRow?.position ?? searchRow?.position ?? ''
  const team = seasonRow?.team ?? searchRow?.team ?? null

  const rosStat = (seasonRow?.[stat as 'fantasy_ppr'] ?? null) as SeasonStatProjection | null
  const offerRosSimulation = rosStat != null && isCurrentOrLater(viewed, now)

  // A row that is served but is not an allow-listed forecast: never the Week N number.
  // Its figure stays behind the opt-in, labelled for what it is.
  const servedNonForecast = predict.data && figure.kind === 'none' ? predict.data : null
  const simulatedWeek = servedNonForecast?.percentiles?.p50 ?? null
  const hasSimulation = simulatedWeek != null || offerRosSimulation

  // The sportsbook only prices the week being played now. A line next to a past week
  // would be a different game's line, so it is only looked up for the live week.
  const liveWeek = isLiveLineWeek(viewed, now)
  const odds = useOdds({ playerNames: [name], stat, enabled: name.length > 0 && figure.kind === 'forecast' && liveWeek })
  // Without a server-side key useOdds returns generated lines (source 'mock'). A generated
  // number is not a market price, so only live lines are shown.
  const liveOdds = odds.data?.source === 'live' ? odds.data : null
  const bookLine = liveOdds && figure.kind === 'forecast' ? findPlayerLine(liveOdds.player_props, name) : null
  const edge = bookLine != null && figure.kind === 'forecast' ? calcEdge(figure.value, bookLine) : null

  useEffect(() => {
    setDefaults({ week, season, stat })
  }, [season, setDefaults, stat, week])

  function handleStatChange(next: string) {
    setSearchParams((prev) => {
      const params = new URLSearchParams(prev)
      params.set('stat', next)
      return params
    }, { replace: true })
  }

  // Back goes to where the visitor came from; opened straight from a link, it goes Home.
  function handleBack() {
    if (location.key === 'default') navigate('/')
    else navigate(-1)
  }

  if (!name) {
    return (
      <div className="flex flex-col gap-4">
        <button onClick={() => navigate('/')} className={`self-start rounded text-xs font-medium text-fg-muted transition-colors hover:text-fg ${FOCUS_RING}`}>
          Back to Home
        </button>
        <div className="rounded-xl border border-line bg-surface-raised p-6">
          <p className="text-sm font-semibold text-fg">No player selected</p>
          <p className="mt-1 text-xs text-fg-muted">
            This page is opened from search, which supplies the player name in the URL.
            Search for a player from the home page to see their projection.
          </p>
        </div>
      </div>
    )
  }

  const loading = (needsCurrent && currentIsPlaceholder) || predict.isLoading
  const noForecast = !loading && (isNoForecastError(predict.error) || (predict.data != null && figure.kind === 'none'))
  const failed = !loading && (currentUnknown || (predict.isError && !isNoForecastError(predict.error)))
  const forecast = !loading && figure.kind === 'forecast' && predict.data ? { figure, data: predict.data } : null

  const unit = statUnit(stat)
  const fantasyValue = forecast?.data.projection.ppr_points ?? null
  const showFantasy = forecast != null && fantasyValue != null && !repeatsHeroFigure(stat, forecast.figure.value, fantasyValue)
  const boom = forecast?.data.boom_probability ?? null
  const bust = forecast?.data.bust_probability ?? null
  const hasTiles = showFantasy || boom != null || bust != null
  const fantasyLabel =
    forecast?.data.fantasy_scoring === 'half_ppr' ? 'Fantasy points, half-PPR'
    : forecast?.data.fantasy_scoring === 'standard' ? 'Fantasy points, standard'
    : 'Fantasy points, PPR'
  const played = !isCurrentOrLater(viewed, now)

  return (
    <div className="flex flex-col gap-6">
      {/* Header: who this is is on screen in every state. */}
      <div>
        <button
          onClick={handleBack}
          className={`group mb-4 flex items-center gap-2 rounded text-xs font-medium text-fg-muted transition-colors hover:text-fg ${FOCUS_RING}`}
        >
          <ChevronLeft className="size-3.5 transition-transform group-hover:-translate-x-0.5" aria-hidden="true" />
          Back
        </button>

        <div className="flex flex-wrap items-end justify-between gap-4">
          <div className="min-w-0">
            <h1 className="break-words font-display text-4xl leading-tight tracking-wide text-fg sm:text-5xl">
              {name.toUpperCase()}
            </h1>
            <p className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-fg-muted">
              {position && <PositionBadge position={position} />}
              {/* Availability chips (Questionable / Doubtful, decision 0012 D3) are not built.
                  Only the "Unknown" note exists: nothing for any other status. */}
              {!loading && isAvailabilityUnknown(predict.data?.availability_status) && (
                <span>{AVAILABILITY_UNKNOWN_NOTE}</span>
              )}
              {team && <span className="font-medium text-fg">{team}</span>}
              <span>Week {week} · {season}{played ? ' · already played' : ''}</span>
            </p>
          </div>

          {/* 15 options, so a dropdown rather than a wall of buttons */}
          <div className="flex w-full flex-col gap-1.5 sm:w-64">
            <span className="section-label" aria-hidden="true">Stat</span>
            <Select items={STAT_ITEMS} value={stat} onValueChange={(next) => { if (next) handleStatChange(next) }}>
              <SelectTrigger aria-label="Stat" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {STAT_ITEMS.map((item) => (
                  <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {loading && <PlayerDetailSkeleton />}

      {failed && (currentUnknown ? (
        <ErrorBanner error={currentError} onRetry={() => void refetchCurrent()} />
      ) : (
        predict.error && <ErrorBanner error={predict.error} onRetry={() => void predict.refetch()} />
      ))}

      {noForecast && (
        <NoForecastBanner season={season} week={week} playerName={name}>
          {hasSimulation && !showSimulation && (
            <button
              type="button"
              onClick={() => setShowSimulation(true)}
              className="inline-flex items-center rounded-lg border border-white/10 px-3 py-1.5 text-xs font-semibold text-fg-muted transition-colors hover:border-white/20 hover:text-fg"
            >
              Show the preseason simulation
            </button>
          )}
        </NoForecastBanner>
      )}

      {forecast && (
        <div className="flex animate-fade-in flex-col gap-5">
          {forecast.data.weighting === 'user' && (
            <p role="note" className="text-xs leading-relaxed text-fg-muted">
              These numbers reflect your saved settings. They are a what-if and have not been checked against past results.
            </p>
          )}

          <div className={hasTiles ? 'grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]' : undefined}>
            <PlayerHero
              label={weekFigureLabel(week, stat)}
              value={forecast.figure.value}
              low={forecast.figure.low}
              high={forecast.figure.high}
              unit={unit}
            />
            {hasTiles && (
              <div className="grid grid-cols-1 content-start gap-3 sm:grid-cols-3 lg:grid-cols-1">
                {showFantasy && <ProjectionCard label={fantasyLabel} value={fantasyValue} unit="pts" tone="highlight" />}
                {boom != null && <ProjectionCard label="Boom chance" value={boom * 100} unit="%" tone="positive" subtext="Chance of a great week" />}
                {bust != null && <ProjectionCard label="Bust chance" value={bust * 100} unit="%" tone="negative" subtext="Chance of a bad week" />}
              </div>
            )}
          </div>

          {bookLine != null && edge != null && (
            <div className={`${toneClass(edge > 0 ? 'positive' : 'negative')} flex flex-wrap items-baseline justify-between gap-2 rounded-xl border border-tone/30 bg-tone/[0.06] px-4 py-3`}>
              <p className="text-sm text-fg">
                Sportsbook line <span className="font-display text-xl text-tone">{bookLine.toFixed(1)}</span>
                {unit && <span className="ml-1 text-xs text-fg-muted">{unit}</span>}
              </p>
              <p className="text-xs font-semibold text-tone">
                Our projection is {Math.abs(edge).toFixed(1)}% {edge > 0 ? 'above' : 'below'} the line
              </p>
            </div>
          )}

          {offerRosSimulation && !showSimulation && (
            <button
              type="button"
              onClick={() => setShowSimulation(true)}
              className="self-start rounded-lg border border-white/10 px-3 py-1.5 text-xs font-semibold text-fg-muted transition-colors hover:border-white/20 hover:text-fg"
            >
              Show the preseason simulation
            </button>
          )}
        </div>
      )}

      {showSimulation && hasSimulation && (
        <div className="grid gap-3 sm:grid-cols-2">
          {simulatedWeek != null && servedNonForecast && (
            <SimulationFigure
              label={`${STAT_LABELS[stat] ?? stat}, simulated`}
              provenance={provenanceLabel(servedNonForecast.interval_method)}
              value={simulatedWeek}
              low={servedNonForecast.percentiles.p10 ?? null}
              high={servedNonForecast.percentiles.p90 ?? null}
              unit={unit}
            />
          )}
          {offerRosSimulation && rosStat && seasonRow && (
            <SimulationFigure
              label={`Rest of season · ${STAT_LABELS[stat] ?? stat}`}
              provenance={provenanceLabel(seasonRow.interval_method)}
              value={rosStat.mean}
              low={rosStat.p10}
              high={rosStat.p90}
              unit={unit}
            />
          )}
        </div>
      )}

      {forecast && player_id && (
        <>
          <ExplainPanel playerId={player_id} week={week} season={season} stat={stat} />
          <WhatIfStudio playerId={player_id} week={week} season={season} stat={stat} />
        </>
      )}
    </div>
  )
}
