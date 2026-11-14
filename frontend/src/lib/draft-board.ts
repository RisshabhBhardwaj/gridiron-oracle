/**
 * Rules for the draft board, kept out of the page so each one can be tested alone.
 *
 * The drafter needs three things: who to take, how our view differs from the
 * market, and how fresh the board is. Everything here turns API fields into
 * plain words. Model names and statistics never reach the page as body text.
 */

import type { CurrentSeasonResponse, DraftBoardResponse } from '@/types/api'

/** Column names, one word per thing, used by the table, the sort menu, the skeleton and the legend. */
export const BOARD_LABELS = {
  marketRank: 'Market rank',
  player: 'Player',
  pos: 'Pos',
  team: 'Team',
  avgPick: 'Avg. pick',
  ourRank: 'Our rank',
  ourPoints: 'Our points',
  value: 'Value vs market',
} as const

/**
 * Has the draft season passed for this board?
 *
 * `/season/current` reports week = (last played week) + 1, so week 1 means no
 * game has been played yet (backend/app/api/season.py). From week 2 on, a game
 * has been played and nobody is drafting a 2026 team any more. A board for an
 * earlier season is over by definition. Returns false while the current season
 * is unknown, so a placeholder never raises the notice.
 */
export function draftWindowOver(boardSeason: number, current: CurrentSeasonResponse | null | undefined): boolean {
  if (!current) return false
  if (current.season > boardSeason) return true
  return current.season === boardSeason && current.week >= 2
}

/** One calm sentence for a board whose draft window has passed. */
export function draftWindowNotice(boardSeason: number, current: CurrentSeasonResponse): string {
  return current.season > boardSeason
    ? `The ${boardSeason} season has been played. This is its preseason board.`
    : `The ${boardSeason} draft season is over. This is the preseason board.`
}

/** Both backend sources (`preseason_historical_per_game`, `preseason_projection_rows`) are pre-draft. */
export function isPreseasonBoard(projectionSource: string): boolean {
  return /preseason/i.test(projectionSource)
}

/** `2026-08-20` as "Aug 20, 2026", in UTC so the day never shifts with the viewer's time zone. */
export function formatBoardDate(iso: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso)
  if (!m) return iso
  const d = new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])))
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' })
}

/** "Based on preseason projections, updated Aug 20, 2026". */
export function freshnessLine(board: Pick<DraftBoardResponse, 'projection_source' | 'as_of'>): string {
  const basis = isPreseasonBoard(board.projection_source) ? 'preseason projections' : 'projections'
  return `Based on ${basis}, updated ${formatBoardDate(board.as_of)}`
}

const MARKET_NAMES: Record<string, string> = { sleeper: 'Sleeper', fantasypros: 'FantasyPros' }

/** Where the market ranks come from, in words ("Sleeper"). Unknown sources are capitalised. */
export function marketSourceName(source: string | null | undefined): string | null {
  if (!source) return null
  const key = source.toLowerCase()
  return MARKET_NAMES[key] ?? key.charAt(0).toUpperCase() + key.slice(1)
}

/**
 * How closely our order matches the market's, for the "About these numbers"
 * disclosure. The figure is the rank correlation the API computes for this board.
 */
export function agreementLine(rho: number | null | undefined, count: number): string | null {
  if (rho == null || !Number.isFinite(rho)) return null
  return `How closely our order matches the market's: ${rho.toFixed(2)}, where 1 means the same order and 0 means no link (across ${count} players).`
}

export interface ValueView {
  kind: 'up' | 'down' | 'even' | 'none'
  /** Printed text with a sign: "+12", "−3", "0", "—". */
  text: string
  /** Full meaning for screen readers and the title. */
  spoken: string
}

const MINUS = '−'

/**
 * Value vs market is (market rank − our rank), in draft spots. A positive
 * number means we rank the player higher than the market does. The sign is
 * always printed, so colour is never the only cue.
 */
export function valueView(value: number | null | undefined): ValueView {
  if (value == null || !Number.isFinite(value)) {
    return { kind: 'none', text: '—', spoken: 'We have no rank for this player' }
  }
  const n = Math.round(value)
  if (n === 0) return { kind: 'even', text: '0', spoken: 'We rank this player the same as the market' }
  const spots = Math.abs(n) === 1 ? '1 spot' : `${Math.abs(n)} spots`
  return n > 0
    ? { kind: 'up', text: `+${n}`, spoken: `We rank this player ${spots} higher than the market` }
    : { kind: 'down', text: `${MINUS}${Math.abs(n)}`, spoken: `We rank this player ${spots} lower than the market` }
}

/** One line under the table header explaining the Value column. */
export const VALUE_LEGEND =
  '+ means we rank a player higher than the market does, − lower, counted in draft spots.'
