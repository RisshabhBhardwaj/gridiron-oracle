import { useMemo, useState } from 'react'
import { Bar, BarChart, Legend, ReferenceLine } from 'recharts'
import type { LegacyRow } from '@tanstack/react-table/legacy'
import type { BacktestRow } from '@/types/api'
import { DataTable, type TableColumn } from '@/components/ui/data-table'
import { ChartFrame, ChartGrid, ChartTooltip, ChartXAxis, ChartYAxis, chartTheme } from '@/components/charts/themed-chart'
import { Panel } from '@/components/shared/Panel'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { EvidenceVerdict } from '@/components/evidence/EvidenceVerdict'
import { EvidenceUnavailable } from '@/components/evidence/EvidenceUnavailable'
import { BaselineGuide } from '@/components/evidence/BaselineGuide'
import { ImprovementCell } from '@/components/evidence/ImprovementCell'
import { PositionFilter } from '@/components/evidence/PositionFilter'
import { useBacktest } from '@/hooks/useBacktest'
import { POSITIONS } from '@/lib/constants'
import { formatScore, isEvidenceUnavailable, positionsWithRows, statLabel, statsIn, verdictOf, verdictSentence } from '@/lib/evidence'
import { positionTone, toneClass, token } from '@/lib/tokens'
import { cn } from '@/lib/utils'
import { ChartFrameSkeleton, LoadingRegion, Skeleton, TableSkeleton } from '@/components/shared/Skeletons'
import { usePageTitle } from '@/hooks/usePageTitle'
import { PAGE_NAMES } from '@/lib/page-names'

function seriesColor(position: string): string {
  const tone = positionTone(position)
  return token(tone === 'muted' || tone === 'accent' ? 'fg-muted' : tone, 0.85)
}

const COLUMNS: TableColumn<BacktestRow>[] = [
  {
    header: 'Season',
    accessorKey: 'season',
    cell: ({ row }: { row: LegacyRow<BacktestRow> }) => <span className="font-medium text-fg">{row.original.season}</span>,
  },
  {
    header: 'Position',
    accessorKey: 'position',
    cell: ({ row }: { row: LegacyRow<BacktestRow> }) => (
      <span className={cn(toneClass(positionTone(row.original.position)), 'font-bold text-tone')}>{row.original.position}</span>
    ),
  },
  {
    header: 'Our score',
    accessorKey: 'model_score',
    align: 'right',
    mono: true,
    cell: ({ row }: { row: LegacyRow<BacktestRow> }) => <span className="font-semibold text-fg">{formatScore(row.original.model_score)}</span>,
  },
  {
    header: 'Simple method',
    accessorKey: 'naive_score',
    align: 'right',
    mono: true,
    cell: ({ row }: { row: LegacyRow<BacktestRow> }) => <span className="text-fg-muted">{formatScore(row.original.naive_score)}</span>,
  },
  {
    header: 'Trailing 3',
    hideOnPhone: true,
    accessorKey: 'trailing3_score',
    align: 'right',
    mono: true,
    cell: ({ row }: { row: LegacyRow<BacktestRow> }) => <span className="text-fg-muted">{formatScore(row.original.trailing3_score)}</span>,
  },
  {
    header: 'Marcel',
    hideOnPhone: true,
    accessorKey: 'marcel_score',
    align: 'right',
    mono: true,
    cell: ({ row }: { row: LegacyRow<BacktestRow> }) => <span className="text-fg-muted">{formatScore(row.original.marcel_score)}</span>,
  },
  {
    header: 'Better than simple method',
    accessorKey: 'improvement_vs_naive_pct',
    align: 'right',
    mono: true,
    cell: ({ row }: { row: LegacyRow<BacktestRow> }) => <ImprovementCell pct={row.original.improvement_vs_naive_pct} />,
  },
  {
    header: 'Player-games',
    hideOnPhone: true,
    accessorKey: 'n',
    align: 'right',
    mono: true,
    cell: ({ row }: { row: LegacyRow<BacktestRow> }) => <span className="text-fg-muted">{row.original.n.toLocaleString()}</span>,
  },
]

/** Same blocks, in the same order, as the loaded page: verdict, controls, chart, table. */
function BacktestSkeleton() {
  return (
    <LoadingRegion label="Loading results" className="flex flex-col gap-6">
      <Skeleton className="h-32 rounded-xl" />
      <Skeleton className="h-[4.5rem] rounded-xl" />
      <ChartFrameSkeleton height={280} />
      <TableSkeleton rows={6} columns={5} />
    </LoadingRegion>
  )
}

export function BacktestExplorer() {
  const [stat, setStat] = useState('receiving_yards')
  const [chosen, setChosen] = useState<string[]>([...POSITIONS])

  const { data, isPending, isError, error, refetch } = useBacktest({ stat })
  usePageTitle(PAGE_NAMES.backtest)

  const stats = useMemo(() => (data ? statsIn(data.available) : [stat]), [data, stat])
  const statItems = useMemo(() => stats.map((s) => ({ value: s, label: statLabel(s) })), [stats])
  const hasRows = useMemo(() => (data ? positionsWithRows(data.available, stat) : new Set<string>()), [data, stat])

  // Chips the stat has no rows for are disabled; if nothing chosen is left, show all that exist.
  const positions = useMemo(() => {
    const kept = chosen.filter((p) => hasRows.has(p))
    return kept.length > 0 ? kept : POSITIONS.filter((p) => hasRows.has(p))
  }, [chosen, hasRows])

  const rows = useMemo(
    () => (data ? data.rows.filter((r) => positions.includes(r.position)).sort((a, b) => b.season - a.season || a.position.localeCompare(b.position)) : []),
    [data, positions],
  )

  const chart = useMemo(() => {
    const seasons = [...new Set(rows.map((r) => r.season))].sort((a, b) => a - b)
    const points = seasons.map((season) => {
      const point: Record<string, number | null> = { season }
      for (const r of rows) if (r.season === season) point[r.position] = r.improvement_vs_naive_pct === null ? null : Number(r.improvement_vs_naive_pct.toFixed(1))
      return point
    })
    const series = POSITIONS.filter((p) => positions.includes(p))
    const summary = points
      .map((pt) => `${pt.season}: ${series.filter((p) => pt[p] != null).map((p) => `${p} ${(pt[p] as number) >= 0 ? '+' : ''}${pt[p]}%`).join(', ')}`)
      .join('; ')
    return { points, series, summary }
  }, [rows, positions])

  const metricLabel = rows[0]?.metric_label
  const selectionVerdict = useMemo(() => verdictOf(rows), [rows])

  const seasonsText = data && data.seasons.length > 0 ? `${data.seasons[0]} to ${data.seasons[data.seasons.length - 1]}` : ''

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1 border-b border-line/60 pb-5">
        <h1 className="font-display text-2xl font-bold tracking-wider text-fg">{PAGE_NAMES.backtest}</h1>
        <p className="text-sm text-fg-muted">
          How our projections did on past seasons, compared with simple methods. Each prediction uses only what was known before the game.
        </p>
      </header>

      {isPending && <BacktestSkeleton />}
      {/* Nothing published for this stat yet (a 404, or the evidence not mounted): not an error. */}
      {data === null && <EvidenceUnavailable onRetry={() => void refetch()} />}
      {isError && (isEvidenceUnavailable(error) ? <EvidenceUnavailable onRetry={() => void refetch()} /> : <ErrorBanner error={error} onRetry={() => void refetch()} />)}

      {data && (
        <>
          <EvidenceVerdict
            verdict={data.release_verdict}
            noun="season tests"
            tableUrl={data.table_url}
            lead={seasonsText ? `All stats and positions, ${seasonsText}` : undefined}
          />

          <div className="flex flex-wrap items-center gap-3 rounded-xl border border-white/[0.06] bg-surface-raised p-4">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-fg-muted" id="stat-label">Stat:</span>
              <Select items={statItems} value={stat} onValueChange={(v) => { if (v) setStat(v) }}>
                <SelectTrigger aria-labelledby="stat-label" className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {statItems.map((item) => (
                    <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <PositionFilter hasRows={hasRows} value={positions} onChange={setChosen} />
          </div>

          <p className="text-sm text-fg" aria-live="polite">
            <span className="font-semibold">{statLabel(stat)}: </span>
            {verdictSentence(selectionVerdict, 'season tests')}
          </p>

          <Panel title={`How much better than the simple method, by season: ${statLabel(stat)}`}>
            <ChartFrame
              height={280}
              label={`${statLabel(stat)}: ${verdictSentence(selectionVerdict, 'season tests')} Bar chart of the percent our projections beat the simple method by, per season. ${chart.summary}`}
              table={{
                caption: `Percent better than the simple method, by season: ${statLabel(stat)}`,
                head: ['Season', ...chart.series],
                rows: chart.points.map((pt) => [
                  String(pt.season),
                  ...chart.series.map((p) => (pt[p] == null ? '—' : `${(pt[p] as number) >= 0 ? '+' : ''}${pt[p]}%`)),
                ]),
              }}
            >
              <BarChart data={chart.points} margin={{ top: 8, right: 8, left: 8, bottom: 24 }}>
                <ChartGrid />
                <ChartXAxis
                  dataKey="season"
                  label={{ value: 'Season', position: 'insideBottom', offset: -12, fill: chartTheme.axisText, fontSize: 12 }}
                />
                <ChartYAxis
                  tickFormatter={(v: number) => `${v}%`}
                  label={{ value: '% better than simple method', angle: -90, position: 'insideLeft', offset: 0, fill: chartTheme.axisText, fontSize: 12, style: { textAnchor: 'middle' } }}
                />
                <ChartTooltip
                  formatter={(value, name) => {
                    const v = Number(value)
                    return [`${Math.abs(v).toFixed(1)}% ${v >= 0 ? 'better' : 'worse'} than the simple method`, name]
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px', color: chartTheme.axisText, paddingTop: 8 }} verticalAlign="top" />
                <ReferenceLine y={0} stroke={chartTheme.axisLine} />
                {chart.series.map((pos) => (
                  <Bar key={pos} dataKey={pos} name={pos} fill={seriesColor(pos)} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ChartFrame>
            <p className="mt-2 text-xs text-fg-muted">Above zero means our projections were closer than the simple method.</p>
          </Panel>

          <Panel title={`The numbers: ${statLabel(stat)}`}>
            {rows.length === 0 ? (
              <p className="py-6 text-center text-sm text-fg-muted">There are no results for this selection. Pick another position or stat.</p>
            ) : (
              <div className="flex flex-col gap-3">
                {metricLabel && <p className="text-xs text-fg-muted">Scores shown: {metricLabel}</p>}
                <div role="region" aria-label="Backtest results table; scrolls sideways on small screens" tabIndex={0} className="relative -mx-5 overflow-x-auto">
                  <DataTable columns={COLUMNS} data={rows} />
                </div>
              </div>
            )}
          </Panel>

          <Panel title="How to read this">
            <BaselineGuide cohortRules={data.cohort_rules} />
          </Panel>

          {(data.not_measured.coverage || data.not_measured.pnl) && (
            <p className="text-sm text-fg-muted">
              {[data.not_measured.coverage, data.not_measured.pnl].filter(Boolean).join(' ')}
            </p>
          )}
        </>
      )}
    </div>
  )
}
