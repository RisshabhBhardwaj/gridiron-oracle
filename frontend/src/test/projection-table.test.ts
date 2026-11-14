import { afterEach, describe, expect, it } from 'vitest'
import {
  defaultSortFor,
  filterRows,
  nextSort,
  paginate,
  playerHref,
  rangeExplanation,
  rangeView,
  rangeWords,
  rankRows,
  readStoredView,
  showingLine,
  sortRows,
  writeStoredView,
} from '../lib/projection-table'
import { installMemoryStorage } from './memory-storage'
import type { SeasonPlayerProjection } from '../types/api'

function row(name: string, position: string, mean: number, team: string | null = 'AAA', p10 = mean - 5, p90 = mean + 5): SeasonPlayerProjection {
  return {
    player_id: `id-${name}`,
    player_name: name,
    position,
    team,
    passing_yards: null,
    rushing_yards: null,
    receiving_yards: null,
    fantasy_ppr: { mean, p10, p50: mean, p90 },
    degraded: false,
    interval_method: 'causal_oof_conformal_90',
    p_active: 1,
  }
}

const ROWS = [
  row("Ja'Marr Chase", 'WR', 18, 'CIN'),
  row('Josh Allen', 'QB', 22, 'BUF'),
  row('Amon-Ra St. Brown', 'WR', 17, 'DET'),
  row('Bijan Robinson', 'RB', 20, 'ATL'),
]

describe('rankRows', () => {
  it('ranks by projection, highest first, and leaves out rows with no figure for the stat', () => {
    const noStat = { ...row('No Stat', 'TE', 99), fantasy_ppr: null }
    const ranked = rankRows([...ROWS, noStat], 'fantasy_ppr')
    expect(ranked.map((r) => [r.rank, r.row.player_name])).toEqual([
      [1, 'Josh Allen'],
      [2, 'Bijan Robinson'],
      [3, "Ja'Marr Chase"],
      [4, 'Amon-Ra St. Brown'],
    ])
  })
})

describe('filterRows', () => {
  const ranked = rankRows(ROWS, 'fantasy_ppr')
  it('matches name or team, ignoring case, punctuation and spaces', () => {
    expect(filterRows(ranked, 'jamarr').map((r) => r.row.player_name)).toEqual(["Ja'Marr Chase"])
    expect(filterRows(ranked, 'st brown').map((r) => r.row.player_name)).toEqual(['Amon-Ra St. Brown'])
    expect(filterRows(ranked, 'buf').map((r) => r.row.player_name)).toEqual(['Josh Allen'])
  })
  it('needs every word, and an empty box keeps everyone', () => {
    expect(filterRows(ranked, 'josh cin')).toEqual([])
    expect(filterRows(ranked, '   ')).toHaveLength(4)
  })
  it('keeps the projection rank of a filtered row', () => {
    expect(filterRows(ranked, 'chase')[0].rank).toBe(3)
  })
})

describe('sortRows', () => {
  const ranked = rankRows(ROWS, 'fantasy_ppr')
  const names = (s: ReturnType<typeof sortRows>) => s.map((r) => r.row.player_name)
  it('sorts by projection, name and position in both directions', () => {
    expect(names(sortRows(ranked, { id: 'projection', desc: true }))[0]).toBe('Josh Allen')
    expect(names(sortRows(ranked, { id: 'projection', desc: false }))[0]).toBe('Amon-Ra St. Brown')
    expect(names(sortRows(ranked, { id: 'name', desc: false }))).toEqual([
      'Amon-Ra St. Brown', 'Bijan Robinson', "Ja'Marr Chase", 'Josh Allen',
    ])
    expect(names(sortRows(ranked, { id: 'name', desc: true }))[0]).toBe('Josh Allen')
    // Position A to Z: QB, RB, WR; the two receivers keep projection order.
    expect(names(sortRows(ranked, { id: 'position', desc: false }))).toEqual([
      'Josh Allen', 'Bijan Robinson', "Ja'Marr Chase", 'Amon-Ra St. Brown',
    ])
  })
  it('does not change the input', () => {
    const before = ranked.map((r) => r.rank)
    sortRows(ranked, { id: 'name', desc: false })
    expect(ranked.map((r) => r.rank)).toEqual(before)
  })
  it('nextSort starts at the default direction and flips on the active column', () => {
    expect(defaultSortFor('projection')).toEqual({ id: 'projection', desc: true })
    expect(defaultSortFor('name')).toEqual({ id: 'name', desc: false })
    expect(nextSort({ id: 'projection', desc: true }, 'name')).toEqual({ id: 'name', desc: false })
    expect(nextSort({ id: 'name', desc: false }, 'name')).toEqual({ id: 'name', desc: true })
  })
})

describe('paginate', () => {
  const items = Array.from({ length: 120 }, (_, i) => i)
  it('slices pages and says where you are', () => {
    const p1 = paginate(items, 0, 50)
    expect([p1.from, p1.to, p1.total, p1.pageCount, p1.items.length]).toEqual([1, 50, 120, 3, 50])
    expect(showingLine(p1)).toBe('Showing 1–50 of 120')
    const p3 = paginate(items, 2, 50)
    expect(showingLine(p3)).toBe('Showing 101–120 of 120')
  })
  it('pulls an out-of-range page back and handles an empty list', () => {
    expect(paginate(items, 99, 50).page).toBe(2)
    expect(paginate(items, -4, 50).page).toBe(0)
    const none = paginate([], 0, 50)
    expect([none.from, none.to, none.pageCount]).toEqual([0, 0, 1])
  })
})

describe('range words and zero floors', () => {
  it('names the ends in plain words for the list span, with no p10 or p90 jargon', () => {
    expect(rangeWords('weekly')).toMatchObject({ low: 'Bad week', high: 'Great week' })
    expect(rangeWords('season')).toMatchObject({ low: 'Bad season', high: 'Great season' })
    expect(rangeExplanation('weekly')).not.toMatch(/p10|p90|percentile/i)
    expect(rangeExplanation('weekly')).toMatch(/0\.0 is real, not missing/)
  })
  it('keeps a 0.0 floor as the API sent it', () => {
    const r = rangeView({ mean: 20.3, p10: 0, p50: 20, p90: 34.6 })
    expect(r.floor).toBe(0)
    expect(r.meanOutside).toBe(false)
  })
  it('flags a mean outside its range, on either side', () => {
    expect(rangeView({ mean: 7.8, p10: 0, p50: 0.08, p90: 12.6 }).meanOutside).toBe(false)
    expect(rangeView({ mean: 3, p10: 4, p50: 5, p90: 9 }).meanOutside).toBe(true)
    expect(rangeView({ mean: 10, p10: 0, p50: 0, p90: 8 }).meanOutside).toBe(true)
  })
})

describe('playerHref', () => {
  it('carries the week, season, stat and encoded name', () => {
    expect(playerHref(ROWS[0], 11, 2026, 'fantasy_ppr')).toBe(
      "/player/id-Ja'Marr Chase?week=11&season=2026&stat=fantasy_ppr&name=Ja'Marr%20Chase",
    )
  })
})

describe('remembered view', () => {
  afterEach(() => {
    delete (window as unknown as Record<string, unknown>).localStorage
  })
  it('defaults to the table and remembers the cards', () => {
    installMemoryStorage()
    expect(readStoredView()).toBe('table')
    writeStoredView('cards')
    expect(readStoredView()).toBe('cards')
    writeStoredView('table')
    expect(readStoredView()).toBe('table')
  })
  it('ignores an unknown stored value', () => {
    installMemoryStorage().setItem('gridiron.projections.view', 'carousel')
    expect(readStoredView()).toBe('table')
  })
  it('falls back to the table, and does not throw, when storage throws', () => {
    const storage = installMemoryStorage()
    storage.getItem = () => { throw new Error('blocked') }
    storage.setItem = () => { throw new Error('blocked') }
    expect(readStoredView()).toBe('table')
    expect(() => writeStoredView('cards')).not.toThrow()
  })
  it('falls back to the table when there is no storage at all', () => {
    Object.defineProperty(window, 'localStorage', { value: undefined, configurable: true, writable: true })
    expect(readStoredView()).toBe('table')
    expect(() => writeStoredView('cards')).not.toThrow()
  })
})
