import { useState, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Scatter, ZAxis, Line, ComposedChart,
} from 'recharts'
import { useBacktest } from '@/hooks/useBacktest'
import { BacktestPanel } from '@/components/BacktestPanel'
import { LoadingSpinner } from '@/components/shared/LoadingSpinner'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { DataStalenessWarning } from '@/components/shared/DataStalenessWarning'
import { POSITIONS, BACKTEST_STATS, STAT_LABELS } from '@/lib/constants'
import clsx from 'clsx'
import { Panel } from '@/components/shared/Panel'
import { positionTone, toneClass, token, type Tone } from '@/lib/tokens'

const CHART_THEME = {
  grid:      token('line'),
  axisLine:  token('line-strong'),
  axisText:  token('fg-muted'),
  tooltip: {
    contentStyle: {
      background: token('surface-raised', 0.95),
      border: `1px solid ${token('info', 0.2)}`,
      borderRadius: '12px',
      backdropFilter: 'blur(12px)',
      boxShadow: `0 8px 32px ${token('shadow', 0.6)}`,
    },
    labelStyle: { color: token('fg'), fontWeight: '600' },
  },
}

function coverageTone(coverage80: number): Tone {
  if (coverage80 >= 0.70 && coverage80 <= 0.90) return 'positive'
  if (coverage80 >= 0.60 && coverage80 <  0.70) return 'warning'
  return 'negative'
}

export function BacktestExplorer() {
  const [stat, setStat] = useState('receiving_yards')
  const [positions, setPositions] = useState<string[]>([...POSITIONS])

  const { data, isLoading, isError, error, refetch } = useBacktest({ stat, positions })

  function togglePosition(pos: string) {
    setPositions((prev) => prev.includes(pos) ? prev.filter((p) => p !== pos) : [...prev, pos])
  }

  const seasonBarData = useMemo(() => {
    if (!data) return []
    const map = new Map<number, { stackSum: number; naiveSum: number; count: number }>()
    for (const s of data.by_season) {
      const ex = map.get(s.season) ?? { stackSum: 0, naiveSum: 0, count: 0 }
      map.set(s.season, { stackSum: ex.stackSum + s.stack_mae, naiveSum: ex.naiveSum + s.naive_mae, count: ex.count + 1 })
    }
    return [...map.entries()].sort(([a], [b]) => a - b).map(([season, { stackSum, naiveSum, count }]) => ({
      season,
      stack_mae: parseFloat((stackSum / count).toFixed(2)),
      naive_mae: parseFloat((naiveSum / count).toFixed(2)),
    }))
  }, [data])

  const calibDiagonal = [{ x: 0, y: 0 }, { x: 1, y: 1 }]
  const calibPoints   = data?.calibration.map((c) => ({ x: c.predicted_prob, y: c.observed_freq, z: c.n_samples })) ?? []
  const metricNotes = data ? Object.entries(data.metric_notes).filter(([, note]) => note.length > 0) : []

  return (
    <div className="flex flex-col gap-6">

      {/* Header */}
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-3">
          <h1 className="font-display text-5xl tracking-widest bg-gradient-to-br from-highlight to-info bg-clip-text text-transparent">
            BACKTEST
          </h1>
        </div>
        <p className="text-sm text-fg-muted">Model accuracy and probabilistic calibration over historical seasons</p>
      </div>

      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-3 p-4 rounded-xl bg-surface-raised border border-white/[0.06]">
        <select value={stat} onChange={(e) => setStat(e.target.value)} className="oracle-select">
          {BACKTEST_STATS.map((s) => <option key={s} value={s}>{STAT_LABELS[s]}</option>)}
        </select>

        <div className="flex items-center gap-1.5">
          {POSITIONS.map((pos) => (
            <button
              key={pos}
              onClick={() => togglePosition(pos)}
              aria-pressed={positions.includes(pos)}
              className={clsx(
                toneClass(positionTone(pos)),
                'px-3 py-1.5 rounded-lg text-xs font-bold tracking-wider transition-all duration-150 border',
                positions.includes(pos)
                  ? 'bg-tone/20 border-tone/50 text-tone'
                  : 'bg-transparent border-white/[0.07] text-fg-muted hover:text-fg',
              )}
            >
              {pos}
            </button>
          ))}
        </div>
      </div>

      {isLoading && <div className="flex justify-center py-16"><LoadingSpinner size="lg" /></div>}
      {isError && error && <ErrorBanner error={error} onRetry={() => void refetch()} />}

      {data && !isLoading && (
        <>
          <DataStalenessWarning dataFreshness={data.data_freshness} />

          <div className="flex flex-col gap-2 rounded-xl p-4 bg-surface-raised border border-white/[0.06]">
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              <span className="px-2.5 py-1 rounded-full bg-white/[0.04] border border-white/[0.08] text-fg-muted">
                Source: {data.data_source}
              </span>
              <span className="text-fg-muted">
                {data.brier_score === null || data.simulated_pnl === null
                  ? 'Some metrics are intentionally blank because the current dataset cannot support a real calculation.'
                  : 'All headline metrics are populated from the current backtest dataset.'}
              </span>
            </div>
            {metricNotes.length > 0 && (
              <div className="flex flex-col gap-1 text-[11px] text-fg-muted">
                {metricNotes.map(([metric, note]) => (
                  <p key={metric}>
                    <span className="text-fg">{metric.replaceAll('_', ' ')}:</span> {note}
                  </p>
                ))}
              </div>
            )}
          </div>

          <BacktestPanel data={data} />

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Season MAE */}
            <Panel title="Season-by-Season MAE: Model vs Naive Baseline">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={seasonBarData} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
                  <XAxis dataKey="season" stroke={CHART_THEME.axisLine} tick={{ fontSize: 11, fill: CHART_THEME.axisText }} />
                  <YAxis stroke={CHART_THEME.axisLine} tick={{ fontSize: 11, fill: CHART_THEME.axisText }} />
                  <Tooltip {...CHART_THEME.tooltip} />
                  <Legend wrapperStyle={{ fontSize: '11px', color: CHART_THEME.axisText }} />
                  <Bar dataKey="stack_mae" name="Model MAE"     fill={token('info', 0.8)}  radius={[4,4,0,0]} />
                  <Bar dataKey="naive_mae" name="Naive Baseline" fill={token('fg-muted', 0.5)} radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </Panel>

            {/* Calibration */}
            <Panel title="Calibration Reliability Diagram">
              <ResponsiveContainer width="100%" height={280}>
                <ComposedChart margin={{ top: 8, right: 8, left: 0, bottom: 24 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
                  <XAxis
                    type="number" dataKey="x" domain={[0, 1]}
                    stroke={CHART_THEME.axisLine} tick={{ fontSize: 11, fill: CHART_THEME.axisText }}
                    label={{ value: 'Predicted Prob.', position: 'insideBottom', offset: -12, fill: CHART_THEME.axisText, fontSize: 11 }}
                  />
                  <YAxis
                    type="number" dataKey="y" domain={[0, 1]}
                    stroke={CHART_THEME.axisLine} tick={{ fontSize: 11, fill: CHART_THEME.axisText }}
                    label={{ value: 'Observed Freq.', angle: -90, position: 'insideLeft', offset: 10, fill: CHART_THEME.axisText, fontSize: 11 }}
                  />
                  <Tooltip {...CHART_THEME.tooltip} />
                  <Line
                    data={calibDiagonal} type="linear" dataKey="y"
                    stroke={token('fg-muted', 0.6)} strokeDasharray="5 5" dot={false}
                    name="Perfect Calibration" legendType="line"
                  />
                  <ZAxis range={[40, 200]} />
                  <Scatter data={calibPoints} fill={token('info', 0.8)} name="Model"
                    style={{ filter: `drop-shadow(0 0 4px ${token('info', 0.4)})` }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </Panel>
          </div>

          {/* Coverage table */}
          <Panel title="Coverage by Season & Position">
            <div className="overflow-x-auto -m-5">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-surface/60 border-b border-white/5">
                    {['Season', 'Position', 'Coverage 80%', 'Coverage 50%', 'N Games'].map((h, i) => (
                      <th
                        key={h}
                        className={clsx('px-5 py-3 text-[10px] font-semibold uppercase tracking-widest text-fg-muted', i > 1 ? 'text-right' : 'text-left')}
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.by_season.map((row) => (
                    <tr
                      key={`${row.season}-${row.position}`}
                      className="transition-colors border-b border-white/[0.04] even:bg-white/[0.015] hover:bg-info/[0.04]"
                    >
                      <td className="px-5 py-3 font-medium text-fg">{row.season}</td>
                      <td className="px-5 py-3">
                        <span className={clsx(toneClass(positionTone(row.position)), 'text-xs font-bold text-tone')}>
                          {row.position}
                        </span>
                      </td>
                      <td className={clsx(toneClass(coverageTone(row.coverage_80)), 'px-5 py-3 text-right font-bold font-mono text-tone')}>
                        {(row.coverage_80 * 100).toFixed(1)}%
                      </td>
                      <td className="px-5 py-3 text-right font-mono text-fg-muted">
                        {(row.coverage_50 * 100).toFixed(1)}%
                      </td>
                      <td className="px-5 py-3 text-right font-mono text-fg-muted">{row.n_games}</td>
                    </tr>
                  ))}
                  {data.by_season.length === 0 && (
                    <tr>
                      <td colSpan={5} className="px-5 py-8 text-center text-fg-muted text-sm">
                        No backtest data available for the selected filters.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}
    </div>
  )
}
