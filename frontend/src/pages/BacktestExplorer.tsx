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

const POS_COLORS: Record<string, string> = {
  WR: '#00C2FF', RB: '#00FFA3', TE: '#A855F7', QB: '#FFB800',
}

const CHART_THEME = {
  grid:    'rgba(26,47,78,0.6)',
  axis:    '#3D5A7A',
  tooltip: {
    contentStyle: {
      background: 'rgba(8,15,32,0.95)',
      border: '1px solid rgba(0,194,255,0.2)',
      borderRadius: '12px',
      backdropFilter: 'blur(12px)',
      boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
    },
    labelStyle: { color: '#F0F6FF', fontWeight: '600' },
  },
}

function coverageColor(coverage80: number): string {
  if (coverage80 >= 0.70 && coverage80 <= 0.90) return '#00FFA3'
  if (coverage80 >= 0.60 && coverage80 <  0.70) return '#FFB800'
  return '#FF3B5C'
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
      <div className="px-5 py-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <h3 className="text-sm font-semibold text-oracle-white">{title}</h3>
      </div>
      <div className="p-5">{children}</div>
    </div>
  )
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
          <h1
            className="font-display text-5xl tracking-widest"
            style={{
              background: 'linear-gradient(135deg, #A855F7 0%, #00C2FF 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            BACKTEST
          </h1>
        </div>
        <p className="text-sm text-oracle-muted">Model accuracy and probabilistic calibration over historical seasons</p>
      </div>

      {/* Filter bar */}
      <div
        className="flex flex-wrap items-center gap-3 p-4 rounded-xl"
        style={{ background: 'rgba(13,27,48,0.8)', border: '1px solid rgba(255,255,255,0.06)' }}
      >
        <select value={stat} onChange={(e) => setStat(e.target.value)} className="oracle-select">
          {BACKTEST_STATS.map((s) => <option key={s} value={s}>{STAT_LABELS[s]}</option>)}
        </select>

        <div className="flex items-center gap-1.5">
          {POSITIONS.map((pos) => (
            <button
              key={pos}
              onClick={() => togglePosition(pos)}
              className="px-3 py-1.5 rounded-lg text-xs font-bold tracking-wider transition-all duration-150 border"
              style={positions.includes(pos)
                ? { background: `${POS_COLORS[pos]}20`, borderColor: `${POS_COLORS[pos]}50`, color: POS_COLORS[pos] }
                : { background: 'transparent', borderColor: 'rgba(255,255,255,0.07)', color: '#607B9B' }
              }
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

          <div className="flex flex-col gap-2 rounded-xl p-4"
            style={{ background: 'rgba(13,27,48,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}>
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              <span className="px-2.5 py-1 rounded-full"
                style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: '#607B9B' }}>
                Source: {data.data_source}
              </span>
              <span className="text-oracle-muted">
                {data.brier_score === null || data.simulated_pnl === null
                  ? 'Some metrics are intentionally blank because the current dataset cannot support a real calculation.'
                  : 'All headline metrics are populated from the current backtest dataset.'}
              </span>
            </div>
            {metricNotes.length > 0 && (
              <div className="flex flex-col gap-1 text-[11px] text-oracle-muted">
                {metricNotes.map(([metric, note]) => (
                  <p key={metric}>
                    <span className="text-oracle-white">{metric.replaceAll('_', ' ')}:</span> {note}
                  </p>
                ))}
              </div>
            )}
          </div>

          <BacktestPanel data={data} />

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Season MAE */}
            <GlassPanel title="Season-by-Season MAE: Model vs Naive Baseline">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={seasonBarData} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
                  <XAxis dataKey="season" stroke={CHART_THEME.axis} tick={{ fontSize: 11, fill: CHART_THEME.axis }} />
                  <YAxis stroke={CHART_THEME.axis} tick={{ fontSize: 11, fill: CHART_THEME.axis }} />
                  <Tooltip {...CHART_THEME.tooltip} />
                  <Legend wrapperStyle={{ fontSize: '11px', color: '#607B9B' }} />
                  <Bar dataKey="stack_mae" name="Model MAE"     fill="rgba(0,194,255,0.8)"  radius={[4,4,0,0]} />
                  <Bar dataKey="naive_mae" name="Naive Baseline" fill="rgba(96,123,155,0.5)" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </GlassPanel>

            {/* Calibration */}
            <GlassPanel title="Calibration Reliability Diagram">
              <ResponsiveContainer width="100%" height={280}>
                <ComposedChart margin={{ top: 8, right: 8, left: 0, bottom: 24 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
                  <XAxis
                    type="number" dataKey="x" domain={[0, 1]}
                    stroke={CHART_THEME.axis} tick={{ fontSize: 11, fill: CHART_THEME.axis }}
                    label={{ value: 'Predicted Prob.', position: 'insideBottom', offset: -12, fill: CHART_THEME.axis, fontSize: 11 }}
                  />
                  <YAxis
                    type="number" dataKey="y" domain={[0, 1]}
                    stroke={CHART_THEME.axis} tick={{ fontSize: 11, fill: CHART_THEME.axis }}
                    label={{ value: 'Observed Freq.', angle: -90, position: 'insideLeft', offset: 10, fill: CHART_THEME.axis, fontSize: 11 }}
                  />
                  <Tooltip {...CHART_THEME.tooltip} />
                  <Line
                    data={calibDiagonal} type="linear" dataKey="y"
                    stroke="rgba(96,123,155,0.6)" strokeDasharray="5 5" dot={false}
                    name="Perfect Calibration" legendType="line"
                  />
                  <ZAxis range={[40, 200]} />
                  <Scatter data={calibPoints} fill="rgba(0,194,255,0.8)" name="Model"
                    style={{ filter: 'drop-shadow(0 0 4px rgba(0,194,255,0.4))' }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </GlassPanel>
          </div>

          {/* Coverage table */}
          <GlassPanel title="Coverage by Season & Position">
            <div className="overflow-x-auto -m-5">
              <table className="w-full text-sm">
                <thead>
                  <tr style={{ background: 'rgba(5,11,24,0.6)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    {['Season', 'Position', 'Coverage 80%', 'Coverage 50%', 'N Games'].map((h, i) => (
                      <th
                        key={h}
                        className={clsx('px-5 py-3 text-[10px] font-semibold uppercase tracking-widest text-oracle-muted', i > 1 ? 'text-right' : 'text-left')}
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.by_season.map((row, idx) => (
                    <tr
                      key={`${row.season}-${row.position}`}
                      className="transition-colors"
                      style={{
                        borderBottom: '1px solid rgba(255,255,255,0.04)',
                        background: idx % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)',
                      }}
                      onMouseEnter={(e) => { (e.currentTarget as HTMLTableRowElement).style.background = 'rgba(0,194,255,0.04)' }}
                      onMouseLeave={(e) => { (e.currentTarget as HTMLTableRowElement).style.background = idx % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)' }}
                    >
                      <td className="px-5 py-3 font-medium text-oracle-white">{row.season}</td>
                      <td className="px-5 py-3">
                        <span className="text-xs font-bold" style={{ color: POS_COLORS[row.position] ?? '#607B9B' }}>
                          {row.position}
                        </span>
                      </td>
                      <td className="px-5 py-3 text-right font-bold font-mono" style={{ color: coverageColor(row.coverage_80) }}>
                        {(row.coverage_80 * 100).toFixed(1)}%
                      </td>
                      <td className="px-5 py-3 text-right font-mono text-oracle-muted">
                        {(row.coverage_50 * 100).toFixed(1)}%
                      </td>
                      <td className="px-5 py-3 text-right font-mono text-oracle-muted">{row.n_games}</td>
                    </tr>
                  ))}
                  {data.by_season.length === 0 && (
                    <tr>
                      <td colSpan={5} className="px-5 py-8 text-center text-oracle-muted text-sm">
                        No backtest data available for the selected filters.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </GlassPanel>
        </>
      )}
    </div>
  )
}
