/**
 * Rules for the Current Season player list (table and cards), kept out of the
 * component so they can be tested on their own: rank, filter, sort, range words,
 * paging and the remembered view. Only numbers the API sent are shown; nothing is
 * filled in or changed here.
 */

import type { SeasonPlayerProjection, SeasonStatProjection } from '@/types/api'

export type SortId = 'projection' | 'name' | 'position'
export interface SortState {
  id: SortId
  /** true = high to low / Z to A. */
  desc: boolean
}
export type ListView = 'table' | 'cards'
export type ListScope = 'weekly' | 'season'

/** Rows per page. A table row is short; a card is tall, so cards get a smaller, grid-friendly page. */
export const TABLE_PAGE_SIZE = 50
export const CARD_PAGE_SIZE = 48
export const pageSizeFor = (view: ListView): number => (view === 'table' ? TABLE_PAGE_SIZE : CARD_PAGE_SIZE)

/** Default direction when a sort is first chosen: best first for points, A to Z for words. */
export const DEFAULT_SORT: SortState = { id: 'projection', desc: true }
export function defaultSortFor(id: SortId): SortState {
  return { id, desc: id === 'projection' }
}

/** Clicking the active sort flips its direction; clicking another starts at that sort's default. */
export function nextSort(current: SortState, id: SortId): SortState {
  return current.id === id ? { id, desc: !current.desc } : defaultSortFor(id)
}

export const SORT_LABELS: Record<SortId, string> = {
  projection: 'Projection',
  name: 'Player name',
  position: 'Position',
}

export interface RankedRow {
  row: SeasonPlayerProjection
  /** Place in the projection order (1 = highest), fixed while filtering or re-sorting. */
  rank: number
  stat: SeasonStatProjection
}

type StatKey = 'passing_yards' | 'rushing_yards' | 'receiving_yards' | 'fantasy_ppr'

function byProjectionDesc(a: RankedRow, b: RankedRow): number {
  return b.stat.mean - a.stat.mean || a.row.player_name.localeCompare(b.row.player_name)
}

/**
 * Rank every row that has this stat by its projection, highest first. Rows with no
 * figure for the stat are left out (a card used to be hidden for them; a 0.0 would
 * be a made-up number).
 */
export function rankRows(rows: ReadonlyArray<SeasonPlayerProjection>, statKey: string): RankedRow[] {
  const withStat: RankedRow[] = []
  for (const row of rows) {
    const stat = row[statKey as StatKey]
    if (stat) withStat.push({ row, rank: 0, stat })
  }
  withStat.sort(byProjectionDesc)
  withStat.forEach((r, i) => { r.rank = i + 1 })
  return withStat
}

/** Lower-case letters and digits only, so "jamarr" finds "Ja'Marr" and "st brown" finds "St. Brown". */
function squash(text: string): string {
  return text.toLowerCase().normalize('NFKD').replace(/[^a-z0-9]/g, '')
}

/** Keep rows whose name or team contains every word typed. An empty box keeps everything. */
export function filterRows(rows: ReadonlyArray<RankedRow>, query: string): RankedRow[] {
  const words = query.trim().split(/\s+/).map(squash).filter(Boolean)
  if (words.length === 0) return [...rows]
  return rows.filter(({ row }) => {
    const hay = squash(`${row.player_name} ${row.team ?? ''}`)
    return words.every((w) => hay.includes(w))
  })
}

export function sortRows(rows: ReadonlyArray<RankedRow>, sort: SortState): RankedRow[] {
  const dir = sort.desc ? -1 : 1
  const out = [...rows]
  out.sort((a, b) => {
    if (sort.id === 'projection') return (a.stat.mean - b.stat.mean) * dir || a.rank - b.rank
    if (sort.id === 'name') {
      return a.row.player_name.localeCompare(b.row.player_name, 'en', { sensitivity: 'base' }) * dir || a.rank - b.rank
    }
    // Position: alphabetical groups, best projection first inside each group.
    return (a.row.position ?? '').localeCompare(b.row.position ?? '') * dir || a.rank - b.rank
  })
  return out
}

export interface PageView<T> {
  page: number
  pageCount: number
  /** 1-based, for "Showing 1–50 of 808". 0 when empty. */
  from: number
  to: number
  total: number
  items: T[]
}

/** The slice for one page. An out-of-range page number is pulled back to the nearest real page. */
export function paginate<T>(items: ReadonlyArray<T>, page: number, pageSize: number): PageView<T> {
  const total = items.length
  const pageCount = Math.max(1, Math.ceil(total / pageSize))
  const p = Math.min(Math.max(0, page), pageCount - 1)
  const start = p * pageSize
  const slice = items.slice(start, start + pageSize)
  return {
    page: p,
    pageCount,
    from: total === 0 ? 0 : start + 1,
    to: start + slice.length,
    total,
    items: slice,
  }
}

export function showingLine(view: PageView<unknown>): string {
  return `Showing ${view.from}–${view.to} of ${view.total}`
}

/** What to print for a range. The numbers are the API's own; only their names are ours. */
export interface RangeView {
  floor: number | null
  ceiling: number | null
  /** The headline number falls outside its own range (a zero-inflated stat). */
  meanOutside: boolean
}

export function rangeView(stat: SeasonStatProjection): RangeView {
  const floor = stat.p10 ?? null
  const ceiling = stat.p90 ?? null
  const meanOutside = (floor != null && stat.mean < floor) || (ceiling != null && stat.mean > ceiling)
  return { floor, ceiling, meanOutside }
}

export function formatFigure(value: number | null | undefined): string {
  return value == null ? '—' : value.toFixed(1)
}

/** The two ends of the range, named in plain words for the list's own time span. */
export function rangeWords(scope: ListScope): { low: string; high: string; span: string } {
  return scope === 'weekly'
    ? { low: 'Bad week', high: 'Great week', span: 'week' }
    : { low: 'Bad season', high: 'Great season', span: 'season' }
}

/** One sentence that says what the two range columns mean, 0.0 included. */
export function rangeExplanation(scope: ListScope): string {
  const { low, high, span } = rangeWords(scope)
  return (
    `${low} is the low end: about 1 in 10 ${span}s comes out below it. ` +
    `${high} is the high end: about 1 in 10 ${span}s comes out above it. ` +
    `A low end of 0.0 is real, not missing: in some simulated games the player does not play or does not record the stat.`
  )
}

/** The note under the table for a headline number that sits outside its own range. */
export const MEAN_OUTSIDE_NOTE =
  'The big number is an average over every simulated game, including games the player misses. ' +
  'Those zeros pull it outside the range printed next to it.'

/** The address of a player's page, with the week, season and stat the page needs. */
export function playerHref(row: SeasonPlayerProjection, week: number, season: number, statKey: string): string {
  return `/player/${row.player_id}?week=${week}&season=${season}&stat=${statKey}&name=${encodeURIComponent(row.player_name)}`
}

// ── Remembered view ─────────────────────────────────────────────────────────
// A convenience, not state: every read and write is wrapped, because storage can
// be blocked (private window, cleared site data) and the page must still work.

const VIEW_KEY = 'gridiron.projections.view'

export function readStoredView(): ListView {
  try {
    const v = window.localStorage.getItem(VIEW_KEY)
    return v === 'cards' ? 'cards' : 'table'
  } catch {
    return 'table'
  }
}

export function writeStoredView(view: ListView): void {
  try {
    window.localStorage.setItem(VIEW_KEY, view)
  } catch {
    // Not remembered; the choice still holds for this visit.
  }
}
