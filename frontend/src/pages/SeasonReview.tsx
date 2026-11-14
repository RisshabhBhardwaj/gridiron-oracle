import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import type { LegacyRow } from '@tanstack/react-table/legacy'
import type { BacktestRow, BacktestVerdict } from '@/types/api'
import { useBacktest } from '@/hooks/useBacktest'
import { useCurrentSeason } from '@/hooks/useCurrentSeason'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { DataTable, type TableColumn } from '@/components/ui/data-table'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { EvidenceVerdict } from '@/components/evidence/EvidenceVerdict'
import { EvidenceUnavailable } from '@/components/evidence/EvidenceUnavailable'
import { BaselineGuide } from '@/components/evidence/BaselineGuide'
import { ImprovementCell } from '@/components/evidence/ImprovementCell'
import { PositionFilter } from '@/components/evidence/PositionFilter'
import { POSITIONS } from '@/lib/constants'
import { formatScore, isEvidenceUnavailable, positionsWithRows, statLabel, statsIn } from '@/lib/evidence'
import { positionTone, toneClass } from '@/lib/tokens'
import { cn } from '@/lib/utils'

function PositionCard({ row }: { row: BacktestRow }) {
  return (
    <div className={cn(toneClass(positionTone(row.position)), 'flex flex-col gap-3 rounded-xl border border-white/[0.06] bg-panel-gradient p-4')}>
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-bold text-tone">{row.position}</span>
        <ImprovementCell pct={row.improvement_vs_naive_pct} />
      </div>
      <dl className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <dt className="text-fg-muted">Our score</dt>
          <dd className="font-display text-2xl text-fg">{formatScore(row.model_score)}</dd>
        </div>
        <div>
          <dt className="text-fg-muted">Simple method</dt>
          <dd className="font-display text-2xl text-fg-muted">{formatScore(row.naive_score)}</dd>
        </div>
      </dl>
    </div>
  )
}

const COLUMNS: TableColumn<BacktestRow>[] = [
  {
    header: 'Position',
    accessorKey: 'position',
    cell: ({ row }: { row: LegacyRow<BacktestRow> }) => (
      <span className={cn(toneClass(positionTone(row.original.position)), 'font-bold text-tone')}>{row.original.position}</span>
    ),
  },
  {
    header: 'Player-games',
    accessorKey: 'n',
    align: 'right',
    mono: true,
    cell: ({ row }: { row: LegacyRow<BacktestRow> }) => <span className="text-fg-muted">{row.original.n.toLocaleString()}</span>,
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
    accessorKey: 'trailing3_score',
    align: 'right',
    mono: true,
    cell: ({ row }: { row: LegacyRow<BacktestRow> }) => <span className="text-fg-muted">{formatScore(row.original.trailing3_score)}</span>,
  },
  {
    header: 'Marcel',
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
]

function verdictOf(rows: BacktestRow[]): BacktestVerdict {
  const gains = rows.map((r) => r.improvement_vs_naive_pct).filter((g): g is number => g !== null)
  return {
    cell_seasons: rows.length,
    beat_naive: rows.filter((r) => r.beats_naive).length,
    beat_trailing3: rows.filter((r) => r.beats_trailing3).length,
    beat_marcel: rows.filter((r) => r.beats_marcel).length,
    beat_both: rows.filter((r) => r.beats_naive && r.beats_trailing3).length,
    causal_check_passed: rows.length,
    mean_improvement_vs_naive_pct: gains.length ? gains.reduce((a, b) => a + b, 0) / gains.length : null,
  }
}

export function SeasonReview() {
  const [selectedSeason, setSelectedSeason] = useState<number | null>(null)
  const [selectedStat, setSelectedStat] = useState('receiving_yards')
  const [chosen, setChosen] = useState<string[]>([...POSITIONS])

  const { data: currentSeasonMeta } = useCurrentSeason()
  const currentSeason = currentSeasonMeta?.season

  const { data, isLoading, isError, error, refetch } = useBacktest({ stat: selectedStat })

  const seasons = useMemo(() => (data ? [...data.seasons].sort((a, b) => b - a) : []), [data])
  const season = selectedSeason !== null && seasons.includes(selectedSeason) ? selectedSeason : (seasons[0] ?? null)
  const seasonItems = useMemo(() => seasons.map((s) => ({ value: String(s), label: `${s} Season` })), [seasons])

  const stats = useMemo(() => (data ? statsIn(data.available) : [selectedStat]), [data, selectedStat])
  const statItems = useMemo(() => stats.map((s) => ({ value: s, label: statLabel(s) })), [stats])
  const hasRows = useMemo(() => (data ? positionsWithRows(data.available, selectedStat) : new Set<string>()), [data, selectedStat])

  const positions = useMemo(() => {
    const kept = chosen.filter((p) => hasRows.has(p))
    return kept.length > 0 ? kept : POSITIONS.filter((p) => hasRows.has(p))
  }, [chosen, hasRows])

  const rows = useMemo(
    () =>
      data
        ? data.rows
            .filter((r) => r.season === season && positions.includes(r.position))
            .sort((a, b) => POSITIONS.indexOf(a.position as (typeof POSITIONS)[number]) - POSITIONS.indexOf(b.position as (typeof POSITIONS)[number]))
        : [],
    [data, season, positions],
  )
  const verdict = useMemo(() => verdictOf(rows), [rows])
  const metricLabel = rows[0]?.metric_label

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col justify-between gap-4 border-b border-line/60 pb-5 md:flex-row md:items-center">
        <div>
          <h1 className="font-display text-2xl font-bold tracking-wider text-fg">Season Review</h1>
          <p className="mt-1 text-sm text-fg-muted">How our past projections compared with what actually happened.</p>
        </div>
        {seasonItems.length > 0 && season !== null && (
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-fg-muted" id="season-label">Season:</span>
            <Select items={seasonItems} value={String(season)} onValueChange={(v) => { if (v) setSelectedSeason(Number(v)) }}>
              <SelectTrigger aria-labelledby="season-label" className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {seasonItems.map((item) => (
                  <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
      </header>

      <p className="text-sm text-fg-muted">
        <span className="font-semibold text-fg">Past results only{data && data.seasons.length > 0 ? ` (${data.seasons[0]}–${data.seasons[data.seasons.length - 1]})` : ''}. </span>
        {currentSeason ? `This season (${currentSeason}) is on the ` : 'This season is on the '}
        <Link to="/projections" className="text-accent underline-offset-4 hover:underline">Current Season</Link> page.
      </p>

      {isLoading && (
        <div className="flex flex-col gap-6" role="status" aria-busy="true" aria-label="Loading results">
          <div className="skeleton h-32 rounded-xl" />
          <div className="skeleton h-14 rounded-xl" />
          <div className="skeleton h-64 rounded-xl" />
        </div>
      )}
      {isError && (isEvidenceUnavailable(error) ? <EvidenceUnavailable onRetry={() => void refetch()} /> : <ErrorBanner error={error} onRetry={() => void refetch()} />)}

      {data && season !== null && (
        <>
          <EvidenceVerdict
            verdict={verdict}
            noun={verdict.cell_seasons === 1 ? 'position' : 'positions'}
            tableUrl={data.table_url}
            lead={`${statLabel(selectedStat)}, ${season}`}
          />

          <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-line/40 bg-surface-raised/40 p-4">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-fg-muted" id="stat-label">Stat:</span>
              <Select items={statItems} value={selectedStat} onValueChange={(v) => { if (v) setSelectedStat(v) }}>
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

          {rows.length === 0 ? (
            <p className="rounded-xl border border-line/40 py-8 text-center text-sm text-fg-muted">
              There are no results for this selection. Pick another season, position or stat.
            </p>
          ) : (
            <>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                {rows.map((r) => (
                  <PositionCard key={r.position} row={r} />
                ))}
              </div>

              <section className="overflow-hidden rounded-xl border border-white/[0.06] bg-panel-gradient">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line/50 p-4">
                  <h2 className="text-sm font-bold text-fg">By position ({season})</h2>
                  {metricLabel && <span className="text-xs text-fg-muted">Scores shown: {metricLabel}</span>}
                </div>
                <div className="relative overflow-x-auto">
                  <DataTable columns={COLUMNS} data={rows} className="text-xs" />
                </div>
              </section>
            </>
          )}

          <section className="rounded-xl border border-line/40 p-5">
            <h2 className="mb-3 text-sm font-bold text-fg">How to read this</h2>
            <BaselineGuide cohortRules={data.cohort_rules} />
          </section>
        </>
      )}
    </div>
  )
}
