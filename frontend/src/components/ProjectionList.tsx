import { useId, useMemo, useRef, useState } from 'react'
import clsx from 'clsx'
import { Link, useNavigate } from 'react-router-dom'
import { ChevronLeft, ChevronRight, LayoutGrid, Search, Table2 } from 'lucide-react'
import { DataTable, type TableColumn } from '@/components/ui/data-table'
import { PositionBadge } from '@/components/PositionBadge'
import { NOT_READY_PRIMARY } from '@/components/shared/NotReady'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { isForecastRow, provenanceLabel, provenanceShort, weekFigureLabel } from '@/lib/forecast-provenance'
import {
  filterRows,
  formatFigure,
  MEAN_OUTSIDE_NOTE,
  nextSort,
  pageSizeFor,
  paginate,
  playerHref,
  rangeExplanation,
  rangeView,
  rangeWords,
  rankRows,
  showingLine,
  sortRows,
  SORT_LABELS,
  DEFAULT_SORT,
  defaultSortFor,
  type ListScope,
  type ListView,
  type RankedRow,
  type SortId,
  type SortState,
} from '@/lib/projection-table'
import { STAT_LABELS } from '@/lib/constants'
import { positionTone, toneClass } from '@/lib/tokens'
import { cn, FOCUS_RING } from '@/lib/utils'
import type { SeasonPlayerProjection } from '@/types/api'
import type { LegacyRow } from '@tanstack/react-table/legacy'

const SORT_ITEMS = (Object.keys(SORT_LABELS) as SortId[]).map((id) => ({ value: id, label: SORT_LABELS[id] }))

/** Marks a headline number that sits outside its own range; the explanation is printed under the table. */
const OUTSIDE_MARK = '†'

function ProjectionFigure({ entry, scope, forecastLabelled }: { entry: RankedRow; scope: ListScope; forecastLabelled: boolean }) {
  const { row, stat } = entry
  const range = rangeView(stat)
  // A row that is not a real forecast says what it is, in the cell, whatever else is on the page.
  const simulated = scope === 'weekly' && !isForecastRow(row)
  return (
    <div className="flex flex-col items-end">
      <span className="font-display text-2xl leading-none tabular-nums text-fg">
        {formatFigure(stat.mean)}
        {range.meanOutside && (
          <sup className="ml-0.5 font-sans text-xs text-fg-muted">
            {OUTSIDE_MARK}
            <span className="sr-only"> (outside the range, see note under the table)</span>
          </sup>
        )}
      </span>
      {simulated && (
        <span className="mt-1 text-xs text-fg-muted" title={provenanceLabel(row.interval_method)}>
          {provenanceShort(row.interval_method)}
        </span>
      )}
      {!simulated && !forecastLabelled && scope === 'weekly' && (
        <span className="mt-1 text-xs text-fg-muted">Forecast</span>
      )}
    </div>
  )
}

interface ProjectionListProps {
  /** Rows to list, already limited to the positions and provenance the page wants to show. */
  rows: ReadonlyArray<SeasonPlayerProjection>
  statKey: string
  scope: ListScope
  season: number
  /** Weekly scope: the week shown. */
  week: number
  view: ListView
  onViewChange: (view: ListView) => void
  /** Leaves the narrowed position set (used by "Clear filters"). Omit when positions are not narrowed. */
  onResetPositions?: () => void
}

/**
 * The player list on Current Season: a sortable, filterable table by default and
 * the card grid as an option. Both share one filter, one sort and one page, so
 * switching view keeps your place. Eight hundred rows never render at once.
 */
export function ProjectionList({ rows, statKey, scope, season, week, view, onViewChange, onResetPositions }: ProjectionListProps) {
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<SortState>(DEFAULT_SORT)
  const [pageState, setPageState] = useState({ key: '', page: 0 })
  const topRef = useRef<HTMLDivElement>(null)
  const noteId = useId()
  const sortLabelId = useId()

  const ranked = useMemo(() => rankRows(rows, statKey), [rows, statKey])
  const shown = useMemo(() => sortRows(filterRows(ranked, query), sort), [ranked, query, sort])

  // The page number belongs to one combination of list, filter, sort and view. Change any of
  // them and it starts at page 1, with no effect needed to reset it.
  const pageKey = [scope, statKey, week, ranked.length, query, sort.id, sort.desc, view].join('|')
  const requested = pageState.key === pageKey ? pageState.page : 0
  const pageView = paginate(shown, requested, pageSizeFor(view))

  const goToPage = (page: number) => {
    setPageState({ key: pageKey, page })
    // Keep the reader at the top of the list, not at the bottom of the old page.
    topRef.current?.scrollIntoView({ block: 'start' })
  }

  const words = rangeWords(scope)
  const allForecast = scope === 'weekly' && ranked.every((r) => isForecastRow(r.row))
  const projectionHeader =
    scope === 'weekly'
      ? allForecast
        ? weekFigureLabel(week, statKey)
        : (STAT_LABELS[statKey] ?? statKey)
      : `Season total · ${STAT_LABELS[statKey] ?? statKey}`
  const anyOutside = pageView.items.some((e) => rangeView(e.stat).meanOutside)
  const hrefFor = (e: RankedRow) => playerHref(e.row, week, season, statKey)

  const columns: TableColumn<RankedRow>[] = useMemo(
    () => [
      {
        id: 'rank',
        header: () => (
          <>
            <span aria-hidden="true">#</span>
            <span className="sr-only">Rank</span>
          </>
        ),
        align: 'right',
        mono: true,
        className: 'w-10 pl-3 pr-1 sm:w-14 sm:px-4',
        cell: ({ row }: { row: LegacyRow<RankedRow> }) => <span className="text-fg-muted">{row.original.rank}</span>,
      },
      {
        id: 'name',
        header: 'Player',
        sortable: true,
        cell: ({ row }: { row: LegacyRow<RankedRow> }) => {
          const e = row.original
          const r = rangeView(e.stat)
          return (
            <div className="min-w-0">
              <Link
                to={hrefFor(e)}
                className={cn('rounded font-semibold text-fg hover:text-accent hover:underline', FOCUS_RING)}
              >
                {e.row.player_name}
              </Link>
              {/* Phones hide Pos, Team and the range columns; the answer sits under the name instead. */}
              <span className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-fg-muted sm:hidden">
                <PositionBadge position={e.row.position} />
                <span>{e.row.team ?? '—'}</span>
                <span className="font-mono tabular-nums">
                  {words.low} {formatFigure(r.floor)} · {words.high} {formatFigure(r.ceiling)}
                </span>
              </span>
            </div>
          )
        },
      },
      {
        id: 'position',
        header: 'Pos',
        sortable: true,
        hideOnPhone: true,
        cell: ({ row }: { row: LegacyRow<RankedRow> }) => <PositionBadge position={row.original.row.position} />,
      },
      {
        id: 'team',
        header: 'Team',
        hideOnPhone: true,
        cell: ({ row }: { row: LegacyRow<RankedRow> }) => <span className="text-fg-muted">{row.original.row.team ?? '—'}</span>,
      },
      {
        id: 'projection',
        header: projectionHeader,
        sortable: true,
        align: 'right',
        className: 'min-w-24 whitespace-normal pl-2 sm:min-w-36 sm:px-4',
        cell: ({ row }: { row: LegacyRow<RankedRow> }) => (
          <ProjectionFigure entry={row.original} scope={scope} forecastLabelled={allForecast} />
        ),
      },
      {
        id: 'floor',
        header: words.low,
        align: 'right',
        mono: true,
        hideOnPhone: true,
        cell: ({ row }: { row: LegacyRow<RankedRow> }) => (
          <span className="text-fg-muted">{formatFigure(rangeView(row.original.stat).floor)}</span>
        ),
      },
      {
        id: 'ceiling',
        header: words.high,
        align: 'right',
        mono: true,
        hideOnPhone: true,
        cell: ({ row }: { row: LegacyRow<RankedRow> }) => (
          <span className="text-fg-muted">{formatFigure(rangeView(row.original.stat).ceiling)}</span>
        ),
      },
    ],
    // hrefFor reads season, week and statKey, all of which are in the key below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [projectionHeader, scope, allForecast, words.low, words.high, season, week, statKey],
  )

  const clearFilters = () => {
    setQuery('')
    onResetPositions?.()
  }

  const caption =
    scope === 'weekly'
      ? `Week ${week} ${STAT_LABELS[statKey] ?? statKey}, players ranked by projection`
      : `${season} season total ${STAT_LABELS[statKey] ?? statKey}, players ranked by projection`

  return (
    <div className="flex flex-col gap-3">
      <div ref={topRef} className="flex scroll-mt-16 flex-wrap items-center gap-x-4 gap-y-3">
        <label className="relative block w-full sm:w-72">
          <span className="sr-only">Filter by name or team</span>
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-fg-muted" aria-hidden="true" />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter by name or team"
            autoComplete="off"
            className={cn(
              'h-9 w-full rounded-md border border-line bg-surface-raised pl-9 pr-3 text-sm text-fg placeholder:text-fg-subtle',
              'hover:border-line-strong',
              FOCUS_RING,
            )}
          />
        </label>
        <p className="text-xs text-fg-muted sm:ml-auto" role="status" aria-live="polite">
          {shown.length === 0 ? 'No players to show' : showingLine(pageView)}
        </p>
        {view === 'cards' && (
          <div className="flex items-center gap-2 text-xs text-fg-muted">
            <span id={sortLabelId}>Sort by</span>
            <Select
              items={SORT_ITEMS}
              value={sort.id}
              onValueChange={(v) => { if (v) setSort(defaultSortFor(v as SortId)) }}
            >
              <SelectTrigger aria-labelledby={sortLabelId} className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SORT_ITEMS.map((item) => (
                  <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
        <ToggleGroup
          aria-label="View"
          value={[view]}
          onValueChange={(next) => { if (next[0]) onViewChange(next[0] as ListView) }}
        >
          <ToggleGroupItem value="table" className="inline-flex items-center gap-1.5">
            <Table2 className="size-3.5" aria-hidden="true" />
            Table
          </ToggleGroupItem>
          <ToggleGroupItem value="cards" className="inline-flex items-center gap-1.5">
            <LayoutGrid className="size-3.5" aria-hidden="true" />
            Cards
          </ToggleGroupItem>
        </ToggleGroup>
      </div>

      <details className="group text-xs text-fg-muted">
        <summary className={cn('inline-flex cursor-pointer select-none items-center gap-1 rounded py-1 hover:text-fg', FOCUS_RING)}>
          How to read the range
        </summary>
        <p className="mt-1 max-w-3xl leading-relaxed">{rangeExplanation(scope)}</p>
      </details>

      {ranked.length === 0 ? (
        // Rows exist but none has a figure for this stat (for example passing yards for receivers).
        <div role="status" className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-line p-10 text-center">
          <p className="text-sm font-semibold text-fg">No {STAT_LABELS[statKey] ?? statKey} projections for these players</p>
          <p className="text-xs text-fg-muted">This stat is not projected for every position. Try another stat or position.</p>
          {onResetPositions && (
            <button type="button" onClick={onResetPositions} className={NOT_READY_PRIMARY}>
              Show all positions
            </button>
          )}
        </div>
      ) : shown.length === 0 ? (
        <div role="status" className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-line p-10 text-center">
          <p className="text-sm font-semibold text-fg">No players match “{query.trim()}”</p>
          <p className="text-xs text-fg-muted">Check the spelling, or try just a last name or a team.</p>
          <button type="button" onClick={clearFilters} className={NOT_READY_PRIMARY}>
            Clear filters
          </button>
        </div>
      ) : view === 'table' ? (
        <div className="overflow-clip rounded-xl border border-line">
          <DataTable
            columns={columns}
            data={pageView.items}
            stickyHeader
            caption={caption}
            sort={sort}
            onSortChange={(id) => setSort((cur) => nextSort(cur, id as SortId))}
          />
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {pageView.items.map((e) => (
            <PlayerCard
              key={e.row.player_id}
              entry={e}
              statKey={statKey}
              scope={scope}
              week={week}
              onClick={() => navigate(hrefFor(e))}
            />
          ))}
        </div>
      )}

      {anyOutside && (
        <p id={noteId} className="max-w-3xl text-xs leading-relaxed text-fg-muted">
          <span aria-hidden="true">{OUTSIDE_MARK} </span>
          {MEAN_OUTSIDE_NOTE}
        </p>
      )}

      {pageView.pageCount > 1 && (
        <nav aria-label="Pages" className="flex items-center justify-between gap-3 pt-1">
          <button
            type="button"
            onClick={() => goToPage(pageView.page - 1)}
            disabled={pageView.page === 0}
            className={cn(PAGER_BUTTON, FOCUS_RING)}
          >
            <ChevronLeft className="size-4" aria-hidden="true" />
            Previous page
          </button>
          <span className="text-xs text-fg-muted">
            Page {pageView.page + 1} of {pageView.pageCount}
          </span>
          <button
            type="button"
            onClick={() => goToPage(pageView.page + 1)}
            disabled={pageView.page >= pageView.pageCount - 1}
            className={cn(PAGER_BUTTON, FOCUS_RING)}
          >
            Next page
            <ChevronRight className="size-4" aria-hidden="true" />
          </button>
        </nav>
      )}
    </div>
  )
}

const PAGER_BUTTON =
  'inline-flex h-9 items-center gap-1 rounded-md border border-line bg-surface-raised px-3 text-sm font-medium text-fg transition-colors ' +
  'hover:border-line-strong disabled:pointer-events-none disabled:opacity-40'

interface PlayerCardProps {
  entry: RankedRow
  statKey: string
  scope: ListScope
  week: number
  onClick: () => void
}

function PlayerCard({ entry, statKey, scope, week, onClick }: PlayerCardProps) {
  const { row, stat } = entry
  // On the Weekly tab, a row that is not a real forecast says what it is on the card itself.
  const simulated = scope === 'weekly' && !isForecastRow(row)
  const range = rangeView(stat)
  const words = rangeWords(scope)

  return (
    <div
      className={clsx(
        toneClass(positionTone(row.position)),
        'relative rounded-xl overflow-hidden flex flex-col gap-3 p-4 group cursor-pointer transition-all duration-200 hover:-translate-y-0.5',
        'bg-panel-gradient border border-white/[0.06] shadow-[0_2px_12px_rgb(var(--shadow)/0.3)]',
      )}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter') onClick() }}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-fg truncate leading-tight">{row.player_name}</p>
          <p className="text-xs text-fg-muted mt-0.5">{row.team ?? 'UNK'}</p>
        </div>
        <div className="flex flex-shrink-0 items-center gap-1.5">
          {/* Slot: the availability chip (Questionable / Doubtful, decision 0012 D3) is not
              built. Only the "Unknown" note exists, once above the list (see CurrentSeason). */}
          <PositionBadge position={row.position} />
        </div>
      </div>

      <div className="flex items-baseline gap-1.5 mt-1">
        <span className="glow-text font-display text-4xl leading-none text-tone">
          {formatFigure(stat.mean)}
          {range.meanOutside && (
            <sup className="ml-0.5 font-sans text-sm text-fg-muted">
              {OUTSIDE_MARK}
              <span className="sr-only"> (outside the range, see note under the list)</span>
            </sup>
          )}
        </span>
        <span className="text-xs text-fg-muted">
          {scope === 'weekly' && !simulated ? weekFigureLabel(week, statKey) : STAT_LABELS[statKey] || statKey}
        </span>
      </div>
      {simulated && (
        <p className="-mt-1 text-xs leading-snug text-fg-muted">{provenanceLabel(row.interval_method)}</p>
      )}

      <div className="flex flex-col gap-1 mt-1">
        <div className="h-1.5 rounded-full overflow-hidden bg-surface-overlay/80">
          <div className="h-full w-full rounded-full bg-gradient-to-r from-negative/60 via-tone/50 to-positive/60" />
        </div>
        <div className="flex justify-between gap-2 font-mono text-xs text-fg-muted">
          <span>{words.low} {formatFigure(range.floor)}</span>
          <span>{words.high} {formatFigure(range.ceiling)}</span>
        </div>
      </div>
    </div>
  )
}
