import { useEffect } from 'react'
import clsx from 'clsx'
import { useParams, useSearchParams, useNavigate } from 'react-router-dom'
import { usePredict } from '@/hooks/usePredict'
import { useExplain } from '@/hooks/useExplain'
import { ProjectionCard } from '@/components/ProjectionCard'
import { SHAPChart } from '@/components/SHAPChart'
import { PercentileFan } from '@/components/PercentileFan'
import { WhatIfStudio } from '@/components/WhatIfStudio'
import { useOdds, findPlayerLine } from '@/hooks/useOdds'
import { calcEdge } from '@/lib/edge'
import { LoadingSpinner } from '@/components/shared/LoadingSpinner'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { NoForecastBanner } from '@/components/shared/NoForecastBanner'
import { DataStalenessWarning } from '@/components/shared/DataStalenessWarning'
import { useSeasonProjections } from '@/hooks/useSeasonProjections'
import { STATS, STAT_LABELS, CURRENT_SEASON, LAST_FANTASY_WEEK } from '@/lib/constants'
import { useSearch } from '@/context/SearchContext'
import { ApiError } from '@/lib/api-client'
import { Panel } from '@/components/shared/Panel'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { PositionBadge } from '@/components/PositionBadge'
import { positionTone, toneClass } from '@/lib/tokens'

const STAT_ITEMS = STATS.map((s) => ({ value: s, label: STAT_LABELS[s] ?? s }))

export function PlayerDetail() {
  const navigate = useNavigate()
  const { player_id } = useParams<{ player_id: string }>()
  const [searchParams, setSearchParams] = useSearchParams()
  const { setDefaults } = useSearch()

  const week   = Number(searchParams.get('week')   ?? '1')
  const season = Number(searchParams.get('season') ?? String(CURRENT_SEASON))
  const stat   = searchParams.get('stat')   ?? 'receiving_yards'
  const name   = searchParams.get('name')   ?? ''

  const { data: predictData, isLoading: predictLoading, isError: predictError, error: predictErr, refetch: predictRefetch } =
    usePredict({ player: name, week, season, stat, enabled: name.length > 0 })

  const { data: explainData, isLoading: explainLoading } =
    useExplain({ player_id: player_id ?? '', week, season, stat })

  const { data: oddsData } = useOdds({ playerNames: [name], stat, enabled: name.length > 0 })

  // Clamp: week 18 + 1 would ask for start_week=19, which the endpoint rejects, and
  // the rest-of-season card would silently never appear.
  const rosStartWeek = Math.min(week + 1, LAST_FANTASY_WEEK)
  const { data: seasonData } = useSeasonProjections({ season, startWeek: rosStartWeek, positions: predictData?.position ? [predictData.position] : [] })
  const rosStatKey = stat as keyof import('@/types/api').SeasonPlayerProjection
  const rosProjectionData = seasonData?.projections?.find((p) => p.player_id === player_id)?.[rosStatKey] as import('@/types/api').SeasonStatProjection | undefined
  const rosProjection = rosProjectionData?.mean ?? null

  function handleStatChange(newStat: string) {
    setSearchParams((prev) => { const next = new URLSearchParams(prev); next.set('stat', newStat); return next })
  }

  const projection = predictData?.percentiles.p50 ?? 0
  const floor      = predictData?.percentiles.p10 ?? null
  const ceiling    = predictData?.percentiles.p90 ?? null
  const position   = predictData?.position ?? ''
  const tone       = toneClass(positionTone(position))

  // Without a server-side key, useOdds returns generated lines (source 'mock').
  // A generated number is not a market price, and an edge computed from it would
  // be invented, so only live lines are shown.
  const liveOdds   = oddsData?.source === 'live' ? oddsData : null
  const bookLine   = liveOdds ? findPlayerLine(liveOdds.player_props, name) : null
  const edge       = bookLine != null ? calcEdge(projection, bookLine) : null
  const edgePos    = edge !== null && edge > 0

  useEffect(() => {
    setDefaults({ week, season, stat })
  }, [season, setDefaults, stat, week])

  if (!name) {
    return (
      <div className="flex flex-col gap-4">
        <button
          onClick={() => navigate('/')}
          className="self-start text-xs font-medium text-fg-muted hover:text-fg transition-colors"
        >
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

  return (
    <div className="flex flex-col gap-6">

      {/* Header */}
      <div>
        {/* Back button */}
        <button
          onClick={() => navigate(-1)}
          className="flex items-center gap-2 text-fg-muted hover:text-fg text-xs font-medium mb-4 transition-colors group"
        >
          <svg className="w-3.5 h-3.5 transition-transform group-hover:-translate-x-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back
        </button>

        <div className="flex flex-wrap items-start justify-between gap-4">
          {/* Player title block */}
          <div className="flex items-center gap-4">
            {/* Position circle */}
            {position && (
              <div className={clsx(tone, 'w-14 h-14 rounded-xl flex items-center justify-center flex-shrink-0 bg-tone/[0.08] border border-tone/25 shadow-[0_0_20px_rgb(var(--tone)/0.12)]')}>
                <PositionBadge position={position} />
              </div>
            )}
            <div className={tone}>
              <h1 className="font-display text-5xl leading-tight tracking-wide bg-gradient-to-br from-tone to-tone/60 bg-clip-text text-transparent">
                {(name || 'PLAYER').toUpperCase()}
              </h1>
              {predictData && (
                <p className="text-xs text-fg-muted mt-0.5">
                  Week {week} · {season}
                </p>
              )}
            </div>
          </div>

          {/* Stat selector: 15 options, so a dropdown rather than a wall of buttons */}
          <div className="flex w-full flex-col gap-1.5 sm:w-64">
            <span className="section-label" aria-hidden="true">Stat</span>
            <Select
              items={STAT_ITEMS}
              value={stat}
              onValueChange={(next) => { if (next) handleStatChange(next) }}
            >
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

      {/* Staleness */}
      {predictData && <DataStalenessWarning dataFreshness={predictData.data_freshness} />}

      {/* Loading */}
      {predictLoading && (
        <div className="flex justify-center py-16">
          <LoadingSpinner size="lg" />
        </div>
      )}

      {/* Error — distinguish the honest "no forecast yet" 404 from a real failure */}
      {predictError && predictErr && (
        predictErr instanceof ApiError && predictErr.status === 404 && predictErr.structuredDetail?.forecast_available === false ? (
          <NoForecastBanner season={season} week={week} playerName={name} />
        ) : (
          <ErrorBanner error={predictErr} onRetry={() => void predictRefetch()} />
        )
      )}

      {/* Content */}
      {predictData && (
        <div className="flex flex-col gap-5 animate-fade-in">

          {/* Metric cards */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <ProjectionCard label="WK Proj"  value={projection} unit="yds" tone="info" />
            {rosProjection !== null ? (
              <ProjectionCard label="ROS Proj" value={rosProjection} unit="yds" tone="highlight" />
            ) : null}
            
            {/* Book Line Card (if available) */}
            {bookLine !== null ? (
              <div
                className={clsx(
                  toneClass(edgePos ? 'positive' : 'negative'),
                  'rounded-xl p-4 flex flex-col gap-2 relative overflow-hidden bg-tone/[0.06] border border-tone/30 shadow-card',
                )}
              >
                <div className="corner-glow absolute top-0 right-0 w-16 h-16 rounded-bl-full opacity-30" />
                <span className="section-label">Sportsbook line</span>
                <div className="flex items-baseline gap-1.5">
                  <span className="font-display text-4xl leading-none text-fg">
                    {bookLine.toFixed(1)}
                  </span>
                  <span className="text-xs font-medium text-fg/50">yds</span>
                </div>
                <span className="text-xs font-bold text-tone">
                  {edgePos ? '+' : ''}{edge?.toFixed(1)}% edge
                </span>
              </div>
            ) : (
              <ProjectionCard
                label="Sportsbook line"
                value={null}
                emptyText="No line yet"
                subtext="Compare the range below with the line at your sportsbook."
                tone="muted"
              />
            )}

            <ProjectionCard label="Floor (p10)" value={floor}      unit="yds" tone="negative" />
            <ProjectionCard label="Ceiling (p90)" value={ceiling}  unit="yds" tone="positive" />
            {/* Pass null straight through. ProjectionCard shows a dash for it. Turning it
                into 0 would turn "the model said nothing" into "0% bust risk". */}
            <ProjectionCard
              label="Boom %"
              value={predictData.boom_probability == null ? null : predictData.boom_probability * 100}
              unit="%"
              tone="positive"
              subtext={predictData.boom_probability == null ? 'Not available for this stat' : undefined}
            />
            <ProjectionCard
              label="Bust %"
              value={predictData.bust_probability == null ? null : predictData.bust_probability * 100}
              unit="%"
              tone="negative"
              subtext={predictData.bust_probability == null ? 'Not available for this stat' : undefined}
            />
            <ProjectionCard
              label="Fantasy PPR"
              value={predictData.projection.ppr_points}
              unit="pts"
              tone="highlight"
              subtext={predictData.projection.ppr_points == null ? 'Not available for this stat' : undefined}
            />
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Panel title="Confidence Interval">
              <PercentileFan p10={floor} p50={projection} p90={ceiling} label={STAT_LABELS[stat] ?? stat} />
            </Panel>

            <Panel title="Factor Attributions (SHAP)">
              {explainLoading && (
                <div className="flex justify-center py-8">
                  <LoadingSpinner size="md" />
                </div>
              )}
              {!explainLoading && explainData && explainData.top_factors.length > 0 && (
                <div className="flex flex-col gap-3">
                  <SHAPChart factors={explainData.top_factors} height={280} />
                  <div className="flex flex-wrap items-center gap-2 text-[11px] text-fg-muted">
                    <span className="px-2.5 py-1 rounded-full bg-white/[0.04] border border-white/[0.08]">
                      Attribution: {explainData.attribution_source.replaceAll('_', ' ')}
                    </span>
                    {explainData.attribution_source !== 'real_shap' && (
                      <span>
                        Explanation uses fallback attribution logic rather than a fully loaded SHAP artifact.
                      </span>
                    )}
                  </div>
                </div>
              )}
              {!explainLoading && explainData && explainData.top_factors.length === 0 && (
                <p className="text-fg-muted text-sm py-8 text-center">
                  No SHAP factors available for this player / week.
                </p>
              )}
              {/* The explain endpoint answers 503 when no breakdown is published for this
                  stat. Without this the panel was a titled box with nothing in it. */}
              {!explainLoading && !explainData && (
                <p className="text-fg-muted text-sm py-8 text-center">
                  We cannot show what drives this projection for this stat yet.
                  Try another stat above.
                </p>
              )}
            </Panel>
          </div>

          {/* Kalman estimates */}
          {(predictData.kalman_ability_estimate != null || predictData.confidence_score != null) && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {predictData.kalman_ability_estimate != null && (
                <ProjectionCard
                  label="Kalman Ability"
                  value={predictData.kalman_ability_estimate}
                  tone="info"
                />
              )}
              {predictData.kalman_uncertainty != null && (
                <ProjectionCard
                  label="Kalman σ"
                  value={predictData.kalman_uncertainty}
                  tone="muted"
                />
              )}
              {predictData.confidence_score != null && (
                <ProjectionCard
                  label="Confidence"
                  value={predictData.confidence_score * 100}
                  unit="%"
                  tone="warning"
                />
              )}
            </div>
          )}

          {/* What-If Studio */}
          {player_id && (
            <>
              {liveOdds && (
                <div className="flex flex-wrap items-center gap-2 text-[11px] text-fg-muted">
                  <span className="px-2.5 py-1 rounded-full bg-white/[0.04] border border-white/[0.08]">
                    {liveOdds.provider} live line
                  </span>
                  {bookLine !== null && (
                    <span>
                      Current line {bookLine.toFixed(1)} vs model {projection.toFixed(1)} ({edgePos ? '+' : ''}{edge?.toFixed(1)}% edge)
                    </span>
                  )}
                </div>
              )}
              <WhatIfStudio
                playerId={player_id}
                week={week}
                season={season}
                stat={stat}
                baseProjection={projection}
              />
            </>
          )}
        </div>
      )}
    </div>
  )
}
