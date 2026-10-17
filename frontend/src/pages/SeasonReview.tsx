import { useState, useMemo } from 'react'
import clsx from 'clsx'
import { useBacktest } from '@/hooks/useBacktest'
import { LoadingSpinner } from '@/components/shared/LoadingSpinner'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { DataStalenessWarning } from '@/components/shared/DataStalenessWarning'
import { POSITIONS, BACKTEST_STATS, STAT_LABELS } from '@/lib/constants'
import { useCurrentSeason } from '@/hooks/useCurrentSeason'
import { positionTone, toneClass, type Tone } from '@/lib/tokens'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'

const SEASON_ITEMS = [
  { value: '2025', label: '2025 Season' },
  { value: '2024', label: '2024 Season' },
]
const STAT_ITEMS = BACKTEST_STATS.map((s) => ({ value: s, label: STAT_LABELS[s] || s }))

function MetricCard({
  title,
  value,
  subtitle,
  tone = 'info',
}: {
  title: string
  value: string | number
  subtitle?: string
  tone?: Tone
}) {
  return (
    <div
      className={clsx(
        toneClass(tone),
        'p-4 rounded-xl flex flex-col justify-between bg-panel-gradient border border-white/[0.06] shadow-[0_2px_12px_rgb(var(--shadow)/0.3)]',
      )}
    >
      <span className="text-xs font-semibold uppercase tracking-wider text-fg-muted">{title}</span>
      <div className="my-2">
        <span className="font-display text-3xl text-tone">
          {value}
        </span>
      </div>
      {subtitle && <span className="text-[11px] text-fg-muted">{subtitle}</span>}
    </div>
  )
}

export function SeasonReview() {
  const [selectedSeason, setSelectedSeason] = useState<number>(2025)
  // Must be a stat the shipped backtest artifact actually covers. fantasy_ppr
  // is not in it, so the page's default request returned 503 and every metric
  // rendered as an em-dash. See BACKTEST_STATS.
  const [selectedStat, setSelectedStat] = useState<string>('receiving_yards')
  const [selectedPositions, setSelectedPositions] = useState<string[]>(['WR', 'RB', 'TE', 'QB'])

  const { data: currentSeasonMeta } = useCurrentSeason()
  const currentSeason = currentSeasonMeta?.season

  const { data: backtest, isLoading, error } = useBacktest({
    stat: selectedStat,
    positions: selectedPositions,
  })

  const seasonMetrics = useMemo(() => {
    if (!backtest?.by_season) return []
    return backtest.by_season.filter((m) => m.season === selectedSeason)
  }, [backtest, selectedSeason])

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-line/60 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold font-display tracking-wider text-fg">
              Season Review
            </h1>
            <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-info/10 text-info border border-info/20">
              Accuracy & Historical Evaluation
            </span>
          </div>
          <p className="text-xs text-fg-muted mt-1">
            How our past projections compared with what actually happened.
          </p>
        </div>

        {/* Season selector limited to historical results 2024-2025 */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-fg-muted" id="season-label">Season:</span>
          <Select
            items={SEASON_ITEMS}
            value={String(selectedSeason)}
            onValueChange={(v) => { if (v) setSelectedSeason(Number(v)) }}
          >
            <SelectTrigger aria-labelledby="season-label" className="h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {SEASON_ITEMS.map((item) => (
                <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Results notice */}
      <div className="p-4 rounded-xl bg-surface-overlay/60 border border-line/80 flex items-start gap-3">
        <div className="text-xs text-fg-muted leading-relaxed">
          <span className="font-semibold text-fg">Past results only (2024–2025): </span>
          These scores compare our projections against what actually happened in those seasons.
          This season{currentSeason ? ` (${currentSeason})` : ''} is on the <strong className="text-accent">Current Season</strong> page.
        </div>
      </div>

      {backtest?.data_freshness && <DataStalenessWarning dataFreshness={backtest.data_freshness} />}

      {/* Control bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-surface-raised/40 p-4 rounded-xl border border-line/40">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-fg-muted" id="stat-label">Stat:</span>
          <Select
            items={STAT_ITEMS}
            value={selectedStat}
            onValueChange={(v) => { if (v) setSelectedStat(v) }}
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
      </div>

      {/* Main Content */}
      {isLoading ? (
        <LoadingSpinner />
      ) : error ? (
        <ErrorBanner error={error} />
      ) : (
        <div className="flex flex-col gap-6">
          {/* Top Aggregate Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Average miss"
              value={backtest?.overall_mae != null ? backtest.overall_mae.toFixed(2) : '—'}
              subtitle="Mean absolute error, in points"
              tone="info"
            />
            <MetricCard
              title="Big-miss penalty"
              value={backtest?.overall_rmse != null ? backtest.overall_rmse.toFixed(2) : '—'}
              subtitle="Root mean squared error; large misses count extra"
              tone="positive"
            />
            <MetricCard
              title="Range quality"
              value={backtest?.overall_crps != null ? backtest.overall_crps.toFixed(2) : '—'}
              subtitle="How well the predicted range matched reality"
              tone="highlight"
            />
            <MetricCard
              title="Evaluated Sample Count"
              value={seasonMetrics.reduce((sum, m) => sum + (m.n_games || 0), 0)}
              subtitle={`${selectedSeason} player-game observations`}
              tone="warning"
            />
          </div>

          {/* By-Position Breakdown Table */}
          <div className="rounded-xl overflow-hidden bg-panel-gradient border border-white/[0.06]">
            <div className="p-4 border-b border-line/50 flex items-center justify-between">
              <h2 className="text-sm font-bold text-fg">
                By position ({selectedSeason})
              </h2>
              <span className="text-xs text-fg-muted">Stat: {STAT_LABELS[selectedStat] || selectedStat}</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-line/60 bg-surface-overlay/40 text-fg-muted font-medium">
                    <th className="px-4 py-3">Position</th>
                    <th className="px-4 py-3">Games</th>
                    <th className="px-4 py-3">Our average miss</th>
                    <th className="px-4 py-3">Our big-miss penalty</th>
                    <th className="px-4 py-3">Our range quality</th>
                    <th className="px-4 py-3">Simple-method average miss</th>
                    <th className="px-4 py-3">In-range rate</th>
                    <th className="px-4 py-3">Better than simple method</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line/30">
                  {seasonMetrics.map((row) => (
                    <tr key={row.position} className="hover:bg-surface-raised/40 transition-colors">
                      <td className={clsx(toneClass(positionTone(row.position)), 'px-4 py-3 font-bold text-tone')}>
                        {row.position}
                      </td>
                      <td className="px-4 py-3 font-mono text-fg-muted">{row.n_games}</td>
                      <td className="px-4 py-3 font-mono font-semibold text-fg">{row.stack_mae.toFixed(2)}</td>
                      <td className="px-4 py-3 font-mono text-fg-muted">{row.stack_rmse.toFixed(2)}</td>
                      <td className="px-4 py-3 font-mono text-fg-muted">{row.stack_crps.toFixed(2)}</td>
                      <td className="px-4 py-3 font-mono text-fg-muted">{row.naive_mae.toFixed(2)}</td>
                      <td className="px-4 py-3 font-mono">
                        <span
                          className={`font-semibold ${
                            row.coverage_80 >= 0.75 && row.coverage_80 <= 0.85
                              ? 'text-positive'
                              : 'text-warning'
                          }`}
                        >
                          {(row.coverage_80 * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-4 py-3 font-mono">
                        <span
                          className={`font-semibold ${
                            row.baseline_improvement_pct >= 0 ? 'text-positive' : 'text-negative'
                          }`}
                        >
                          {row.baseline_improvement_pct >= 0 ? '+' : ''}
                          {/* Already a percentage in the backtest artifact, unlike
                              coverage_80 above, which is a fraction. Scaling it again
                              rendered a -20.05% regression as -2005.1%. */}
                          {row.baseline_improvement_pct.toFixed(1)}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
