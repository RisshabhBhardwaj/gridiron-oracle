import { useEffect } from 'react'
import { useParams, useSearchParams, useNavigate } from 'react-router-dom'
import { usePredict } from '@/hooks/usePredict'
import { useExplain } from '@/hooks/useExplain'
import { ProjectionCard } from '@/components/ProjectionCard'
import { SHAPChart } from '@/components/SHAPChart'
import { PercentileFan } from '@/components/PercentileFan'
import { WhatIfStudio } from '@/components/WhatIfStudio'
import { AddToBetSlipButton } from '@/components/AddToBetSlipButton'
import { useOdds, findPlayerLine } from '@/hooks/useOdds'
import { calcEdge, useBetSlip } from '@/context/BetSlipContext'
import { LoadingSpinner } from '@/components/shared/LoadingSpinner'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { NoForecastBanner } from '@/components/shared/NoForecastBanner'
import { DataStalenessWarning } from '@/components/shared/DataStalenessWarning'
import { useSeasonProjections } from '@/hooks/useSeasonProjections'
import { STATS, STAT_LABELS, CURRENT_SEASON } from '@/lib/constants'
import { useSearch } from '@/context/SearchContext'
import { ApiError } from '@/lib/api-client'

const POS_COLORS: Record<string, string> = {
  WR: '#00C2FF', RB: '#00FFA3', TE: '#A855F7', QB: '#FFB800',
}
const POS_CLASS: Record<string, string> = {
  WR: 'pos-wr', RB: 'pos-rb', TE: 'pos-te', QB: 'pos-qb',
}

function GlassPanel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{
        background: 'linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%)',
        border: '1px solid rgba(255,255,255,0.07)',
        boxShadow: '0 4px 24px rgba(0,0,0,0.4)',
      }}
    >
      <div
        className="px-5 py-3 flex items-center gap-2"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}
      >
        <h3 className="text-sm font-semibold text-oracle-white">{title}</h3>
      </div>
      <div className="p-5">{children}</div>
    </div>
  )
}

export function PlayerDetail() {
  const navigate = useNavigate()
  const { player_id } = useParams<{ player_id: string }>()
  const [searchParams, setSearchParams] = useSearchParams()
  const { setDefaults } = useSearch()
  const { setBookLine } = useBetSlip()

  const week   = Number(searchParams.get('week')   ?? '1')
  const season = Number(searchParams.get('season') ?? String(CURRENT_SEASON))
  const stat   = searchParams.get('stat')   ?? 'receiving_yards'
  const name   = searchParams.get('name')   ?? ''

  const { data: predictData, isLoading: predictLoading, isError: predictError, error: predictErr, refetch: predictRefetch } =
    usePredict({ player: name, week, season, stat, enabled: name.length > 0 })

  const { data: explainData, isLoading: explainLoading } =
    useExplain({ player_id: player_id ?? '', week, season, stat })

  const { data: oddsData } = useOdds({ playerNames: [name], stat, enabled: name.length > 0 })

  const { data: seasonData } = useSeasonProjections({ season, startWeek: week + 1, positions: predictData?.position ? [predictData.position] : [] })
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
  const posColor   = POS_COLORS[position] ?? '#607B9B'

  const bookLine   = oddsData ? findPlayerLine(oddsData.player_props, name) : null
  const edge       = bookLine != null ? calcEdge(projection, bookLine) : null
  const edgePos    = edge !== null && edge > 0
  const activeSlipId = player_id ? `${player_id}-${stat}-${week}` : null

  useEffect(() => {
    setDefaults({ week, season, stat })
  }, [season, setDefaults, stat, week])

  useEffect(() => {
    if (activeSlipId && bookLine != null) {
      setBookLine(activeSlipId, bookLine)
    }
  }, [activeSlipId, bookLine, setBookLine])

  if (!name) {
    return (
      <div className="flex flex-col gap-4">
        <button
          onClick={() => navigate('/')}
          className="self-start text-xs font-medium text-oracle-muted hover:text-oracle-white transition-colors"
        >
          Back to Home
        </button>
        <div className="rounded-xl border border-oracle-border bg-oracle-card p-6">
          <p className="text-sm font-semibold text-oracle-white">No player selected</p>
          <p className="mt-1 text-xs text-oracle-muted">
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
          className="flex items-center gap-2 text-oracle-muted hover:text-oracle-white text-xs font-medium mb-4 transition-colors group"
        >
          <svg className="w-3.5 h-3.5 transition-transform group-hover:-translate-x-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back to Dashboard
        </button>

        <div className="flex flex-wrap items-start justify-between gap-4">
          {/* Player title block */}
          <div className="flex items-center gap-4">
            {/* Position circle */}
            {position && (
              <div
                className="w-14 h-14 rounded-xl flex items-center justify-center flex-shrink-0"
                style={{
                  background: `${posColor}15`,
                  border: `1px solid ${posColor}40`,
                  boxShadow: `0 0 20px ${posColor}20`,
                }}
              >
                <span className={POS_CLASS[position] ?? 'text-oracle-muted text-sm font-bold'}>{position}</span>
              </div>
            )}
            <div>
              <h1
                className="font-display text-5xl leading-tight tracking-wide"
                style={{
                  background: `linear-gradient(135deg, ${posColor} 0%, ${posColor}99 100%)`,
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                {(name || 'PLAYER').toUpperCase()}
              </h1>
              {predictData && (
                <p className="text-xs text-oracle-muted mt-0.5">
                  Week {week} · {season} · {predictData.player_id}
                </p>
              )}
            </div>
            {predictData && (
              <AddToBetSlipButton
                size="md"
                leg={{
                  player_id: predictData.player_id,
                  player_name: name,
                  position: predictData.position,
                  team: null,
                  stat,
                  stat_label: STAT_LABELS[stat] ?? stat,
                  week,
                  season,
                  our_projection: projection,
                  floor,
                  ceiling,
                  book_line: bookLine,
                  boom_probability: predictData.boom_probability ?? null,
                  bust_probability: predictData.bust_probability ?? null,
                  fantasy_projection: predictData.projection.ppr_points ?? null,
                }}
              />
            )}
          </div>

          {/* Stat selector */}
          <div className="flex flex-wrap gap-1.5">
            {STATS.map((s) => (
              <button
                key={s}
                onClick={() => handleStatChange(s)}
                className="px-3 py-1.5 rounded-lg text-[11px] font-medium tracking-wide transition-all duration-150 border"
                style={stat === s
                  ? { background: `${posColor}20`, borderColor: `${posColor}50`, color: posColor }
                  : { background: 'rgba(13,27,48,0.6)', borderColor: 'rgba(255,255,255,0.06)', color: '#607B9B' }
                }
              >
                {STAT_LABELS[s] ?? s}
              </button>
            ))}
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
          <NoForecastBanner message={predictErr.structuredDetail.message as string | undefined} />
        ) : (
          <ErrorBanner error={predictErr} onRetry={() => void predictRefetch()} />
        )
      )}

      {/* Content */}
      {predictData && (
        <div className="flex flex-col gap-5 animate-fade-in">

          {/* Metric cards */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <ProjectionCard label="WK Proj"  value={projection} unit="yds" colorClass="text-oracle-blue" />
            {rosProjection !== null ? (
              <ProjectionCard label="ROS Proj" value={rosProjection} unit="yds" colorClass="text-oracle-purple" />
            ) : null}
            
            {/* Book Line Card (if available) */}
            {bookLine !== null ? (
              <div
                className="rounded-xl p-4 flex flex-col gap-2 relative overflow-hidden"
                style={{
                  background: edgePos ? 'rgba(0,255,163,0.06)' : 'rgba(255,59,92,0.06)',
                  border: `1px solid ${edgePos ? 'rgba(0,255,163,0.3)' : 'rgba(255,59,92,0.3)'}`,
                  boxShadow: `0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05)`,
                }}
              >
                <div
                  className="absolute top-0 right-0 w-16 h-16 rounded-bl-full opacity-30"
                  style={{ background: `radial-gradient(circle at top right, ${edgePos ? '#00FFA3' : '#FF3B5C'} 0%, transparent 70%)` }}
                />
                <span className="section-label">Sportsbook Line</span>
                <div className="flex items-baseline gap-1.5">
                  <span className="font-display text-4xl leading-none text-oracle-white">
                    {bookLine.toFixed(1)}
                  </span>
                  <span className="text-xs font-medium" style={{ color: 'rgba(255,255,255,0.5)' }}>yds</span>
                </div>
                <span className="text-xs font-bold" style={{ color: edgePos ? '#00FFA3' : '#FF3B5C' }}>
                  {edgePos ? '+' : ''}{edge?.toFixed(1)}% edge
                </span>
              </div>
            ) : (
              <ProjectionCard label="Book Line" value={0} subtext="No line avail" colorClass="text-oracle-muted" />
            )}

            <ProjectionCard label="Floor (p10)" value={floor}      unit="yds" colorClass="text-oracle-red" />
            <ProjectionCard label="Ceiling (p90)" value={ceiling}  unit="yds" colorClass="text-oracle-green" />
            <ProjectionCard
              label="Boom %"
              value={(predictData.boom_probability ?? 0) * 100}
              unit="%"
              colorClass="text-oracle-green"
            />
            <ProjectionCard
              label="Bust %"
              value={(predictData.bust_probability ?? 0) * 100}
              unit="%"
              colorClass="text-oracle-red"
            />
            <ProjectionCard
              label="Fantasy PPR"
              value={predictData.projection.ppr_points ?? 0}
              unit="pts"
              colorClass="text-oracle-purple"
            />
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <GlassPanel title="Confidence Interval">
              <PercentileFan p10={floor} p50={projection} p90={ceiling} label={STAT_LABELS[stat] ?? stat} />
            </GlassPanel>

            <GlassPanel title="Factor Attributions (SHAP)">
              {explainLoading && (
                <div className="flex justify-center py-8">
                  <LoadingSpinner size="md" />
                </div>
              )}
              {!explainLoading && explainData && explainData.top_factors.length > 0 && (
                <div className="flex flex-col gap-3">
                  <SHAPChart factors={explainData.top_factors} height={280} />
                  <div className="flex flex-wrap items-center gap-2 text-[11px] text-oracle-muted">
                    <span className="px-2.5 py-1 rounded-full"
                      style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
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
                <p className="text-oracle-muted text-sm py-8 text-center">
                  No SHAP factors available for this player / week.
                </p>
              )}
            </GlassPanel>
          </div>

          {/* Kalman estimates */}
          {(predictData.kalman_ability_estimate != null || predictData.confidence_score != null) && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {predictData.kalman_ability_estimate != null && (
                <ProjectionCard
                  label="Kalman Ability"
                  value={predictData.kalman_ability_estimate}
                  colorClass="text-oracle-cyan"
                />
              )}
              {predictData.kalman_uncertainty != null && (
                <ProjectionCard
                  label="Kalman σ"
                  value={predictData.kalman_uncertainty}
                  colorClass="text-oracle-muted"
                />
              )}
              {predictData.confidence_score != null && (
                <ProjectionCard
                  label="Confidence"
                  value={predictData.confidence_score * 100}
                  unit="%"
                  colorClass="text-oracle-amber"
                />
              )}
            </div>
          )}

          {/* What-If Studio */}
          {player_id && (
            <>
              {oddsData && (
                <div className="flex flex-wrap items-center gap-2 text-[11px] text-oracle-muted">
                  <span className="px-2.5 py-1 rounded-full"
                    style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
                    {oddsData.source === 'live' ? `${oddsData.provider} live line` : 'Mock demo line'}
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
