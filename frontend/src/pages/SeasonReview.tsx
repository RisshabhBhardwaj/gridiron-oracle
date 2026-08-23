import { useState, useMemo } from 'react'
import { useBacktest } from '@/hooks/useBacktest'
import { LoadingSpinner } from '@/components/shared/LoadingSpinner'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { DataStalenessWarning } from '@/components/shared/DataStalenessWarning'
import { POSITIONS, STATS, STAT_LABELS } from '@/lib/constants'

const POS_COLORS: Record<string, string> = {
  WR: '#00C2FF',
  RB: '#00FFA3',
  TE: '#A855F7',
  QB: '#FFB800',
}

function MetricCard({
  title,
  value,
  subtitle,
  accentColor = '#00C2FF',
}: {
  title: string
  value: string | number
  subtitle?: string
  accentColor?: string
}) {
  return (
    <div
      className="p-4 rounded-xl flex flex-col justify-between"
      style={{
        background: 'linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.015) 100%)',
        border: '1px solid rgba(255,255,255,0.06)',
        boxShadow: '0 2px 12px rgba(0,0,0,0.3)',
      }}
    >
      <span className="text-xs font-semibold uppercase tracking-wider text-oracle-muted">{title}</span>
      <div className="my-2">
        <span className="font-display text-3xl text-oracle-white" style={{ color: accentColor }}>
          {value}
        </span>
      </div>
      {subtitle && <span className="text-[11px] text-oracle-muted">{subtitle}</span>}
    </div>
  )
}

export function SeasonReview() {
  const [selectedSeason, setSelectedSeason] = useState<number>(2025)
  const [selectedStat, setSelectedStat] = useState<string>('fantasy_ppr')
  const [selectedPositions, setSelectedPositions] = useState<string[]>(['WR', 'RB', 'TE', 'QB'])

  const { data: backtest, isLoading, error } = useBacktest({
    stat: selectedStat,
    positions: selectedPositions,
  })

  const seasonMetrics = useMemo(() => {
    if (!backtest?.by_season) return []
    return backtest.by_season.filter((m) => m.season === selectedSeason)
  }, [backtest, selectedSeason])

  const togglePosition = (pos: string) => {
    setSelectedPositions((prev) =>
      prev.includes(pos) ? (prev.length === 1 ? prev : prev.filter((p) => p !== pos)) : [...prev, pos],
    )
  }

  return (
    <div className="flex flex-col gap-6 p-4 sm:p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-oracle-border/60 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold font-display tracking-wider text-oracle-white">
              Season Review
            </h1>
            <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-oracle-blue/10 text-oracle-blue border border-oracle-blue/20">
              Accuracy & Historical Evaluation
            </span>
          </div>
          <p className="text-xs text-oracle-muted mt-1">
            Predicted vs. realized outcome audit and uncertainty calibration over historical game logs.
          </p>
        </div>

        {/* Season selector limited to historical ground truth 2024-2025 */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-oracle-muted">Evaluation Season:</label>
          <select
            value={selectedSeason}
            onChange={(e) => setSelectedSeason(Number(e.target.value))}
            className="oracle-select bg-oracle-surface border border-oracle-border text-xs rounded-lg px-3 py-1.5 text-oracle-white"
          >
            <option value={2025}>2025 Season</option>
            <option value={2024}>2024 Season</option>
          </select>
        </div>
      </div>

      {/* Data Floor / Ground-truth Notice */}
      <div className="p-4 rounded-xl bg-oracle-surface/60 border border-oracle-border/80 flex items-start gap-3">
        <span className="text-oracle-blue text-base font-bold">ℹ</span>
        <div className="text-xs text-oracle-muted leading-relaxed">
          <span className="font-semibold text-oracle-white">Evaluation Data Floor: </span>
          Accuracy evaluation benchmarks fitted projections against realized ground-truth in <code className="text-oracle-white">game_logs</code> (2024–2025 only). 
          Current-season forward projections (2026) live in the <strong className="text-oracle-green">Current Season</strong> dashboard.
        </div>
      </div>

      {backtest?.data_freshness && <DataStalenessWarning dataFreshness={backtest.data_freshness} />}

      {/* Control bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-oracle-card/40 p-4 rounded-xl border border-oracle-border/40">
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-oracle-muted">Target Stat:</label>
          <select
            value={selectedStat}
            onChange={(e) => setSelectedStat(e.target.value)}
            className="oracle-select bg-oracle-surface border border-oracle-border text-xs rounded-lg px-2.5 py-1.5 text-oracle-white"
          >
            {STATS.map((s) => (
              <option key={s} value={s}>
                {STAT_LABELS[s] || s}
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-1.5">
          {POSITIONS.map((pos) => {
            const active = selectedPositions.includes(pos)
            return (
              <button
                key={pos}
                onClick={() => togglePosition(pos)}
                className={`px-2.5 py-1 text-xs font-semibold rounded-md border transition-colors ${
                  active
                    ? 'bg-oracle-green text-oracle-dark border-transparent'
                    : 'border-oracle-border text-oracle-muted hover:border-oracle-border-bright'
                }`}
              >
                {pos}
              </button>
            )
          })}
        </div>
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
              title="Mean Absolute Error (MAE)"
              value={backtest?.overall_mae != null ? backtest.overall_mae.toFixed(2) : '—'}
              subtitle="Average prediction error magnitude"
              accentColor="#00C2FF"
            />
            <MetricCard
              title="Root Mean Squared Error (RMSE)"
              value={backtest?.overall_rmse != null ? backtest.overall_rmse.toFixed(2) : '—'}
              subtitle="Penalizes large deviations"
              accentColor="#00FFA3"
            />
            <MetricCard
              title="Continuous Ranked Prob Score (CRPS)"
              value={backtest?.overall_crps != null ? backtest.overall_crps.toFixed(2) : '—'}
              subtitle="Probabilistic distribution quality"
              accentColor="#A855F7"
            />
            <MetricCard
              title="Evaluated Sample Count"
              value={seasonMetrics.reduce((sum, m) => sum + (m.n_games || 0), 0)}
              subtitle={`${selectedSeason} player-game observations`}
              accentColor="#FFB800"
            />
          </div>

          {/* By-Position Breakdown Table */}
          <div
            className="rounded-xl overflow-hidden"
            style={{
              background: 'linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.015) 100%)',
              border: '1px solid rgba(255,255,255,0.06)',
            }}
          >
            <div className="p-4 border-b border-oracle-border/50 flex items-center justify-between">
              <h2 className="text-sm font-bold text-oracle-white">
                Positional Performance Audit ({selectedSeason})
              </h2>
              <span className="text-xs text-oracle-muted">Stat: {STAT_LABELS[selectedStat] || selectedStat}</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-oracle-border/60 bg-oracle-surface/40 text-oracle-muted font-medium">
                    <th className="px-4 py-3">Position</th>
                    <th className="px-4 py-3">Games</th>
                    <th className="px-4 py-3">Stack MAE</th>
                    <th className="px-4 py-3">Stack RMSE</th>
                    <th className="px-4 py-3">Stack CRPS</th>
                    <th className="px-4 py-3">Naive MAE</th>
                    <th className="px-4 py-3">80% Coverage</th>
                    <th className="px-4 py-3">Gain vs Baseline</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-oracle-border/30">
                  {seasonMetrics.map((row) => (
                    <tr key={row.position} className="hover:bg-oracle-card/40 transition-colors">
                      <td className="px-4 py-3 font-bold" style={{ color: POS_COLORS[row.position] }}>
                        {row.position}
                      </td>
                      <td className="px-4 py-3 font-mono text-oracle-muted">{row.n_games}</td>
                      <td className="px-4 py-3 font-mono font-semibold text-oracle-white">{row.stack_mae.toFixed(2)}</td>
                      <td className="px-4 py-3 font-mono text-oracle-muted">{row.stack_rmse.toFixed(2)}</td>
                      <td className="px-4 py-3 font-mono text-oracle-muted">{row.stack_crps.toFixed(2)}</td>
                      <td className="px-4 py-3 font-mono text-oracle-muted">{row.naive_mae.toFixed(2)}</td>
                      <td className="px-4 py-3 font-mono">
                        <span
                          className={`font-semibold ${
                            row.coverage_80 >= 0.75 && row.coverage_80 <= 0.85
                              ? 'text-oracle-green'
                              : 'text-oracle-yellow'
                          }`}
                        >
                          {(row.coverage_80 * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-4 py-3 font-mono">
                        <span
                          className={`font-semibold ${
                            row.baseline_improvement_pct >= 0 ? 'text-oracle-green' : 'text-oracle-red'
                          }`}
                        >
                          {row.baseline_improvement_pct >= 0 ? '+' : ''}
                          {(row.baseline_improvement_pct * 100).toFixed(1)}%
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
