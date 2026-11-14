import { ApiError } from '@/lib/api-client'
import { STATS, STAT_LABELS } from '@/lib/constants'
import type { BacktestCell, BacktestVerdict } from '@/types/api'

/** `detail.code` the api sends when the release's pinned evidence cannot be read. */
export const EVIDENCE_UNAVAILABLE = 'evidence_unavailable'

export function isEvidenceUnavailable(error: unknown): boolean {
  return (
    error instanceof ApiError &&
    error.status === 503 &&
    error.structuredDetail?.code === EVIDENCE_UNAVAILABLE
  )
}

export function statLabel(stat: string): string {
  return STAT_LABELS[stat] ?? stat.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

/** Stats the release measures, common ones first, then the rest A to Z. */
export function statsIn(available: BacktestCell[]): string[] {
  const present = new Set(available.map((c) => c.stat))
  const known = (STATS as readonly string[]).filter((s) => present.has(s))
  const rest = [...present].filter((s) => !known.includes(s)).sort()
  return [...known, ...rest]
}

export function positionsWithRows(available: BacktestCell[], stat: string): Set<string> {
  return new Set(available.filter((c) => c.stat === stat).map((c) => c.position))
}

/** Scores below 1 (count targets) need a third decimal to be readable. */
export function formatScore(n: number): string {
  return n.toFixed(Math.abs(n) < 1 ? 3 : 2)
}

export function formatPct(n: number): string {
  return `${Math.abs(n).toFixed(1)}%`
}

/**
 * One plain sentence: better or not than the simple method, and by how much.
 * `noun` names what is counted, e.g. "player-stat seasons" or "positions".
 */
export function verdictSentence(v: BacktestVerdict, noun: string): string {
  const total = v.cell_seasons
  if (total === 0) return 'There are no results for this selection.'
  const n = v.beat_naive
  const head =
    n === total
      ? `Better than the simple method in all ${total} ${noun}`
      : n === 0
        ? `Not better than the simple method in any of ${total} ${noun}`
        : `Better than the simple method in ${n} of ${total} ${noun}`
  const gain = v.mean_improvement_vs_naive_pct
  if (gain === null) return `${head}.`
  return gain >= 0
    ? `${head}, by ${formatPct(gain)} on average.`
    : `${head}, and ${formatPct(gain)} worse on average.`
}

/** The other two baselines, in one line. */
export function otherBaselinesSentence(v: BacktestVerdict): string {
  if (v.cell_seasons === 0) return ''
  return `It also beat a trailing 3-game average in ${v.beat_trailing3} of ${v.cell_seasons} and the Marcel method in ${v.beat_marcel} of ${v.cell_seasons}.`
}

/** Plain words for the comparison group. Unknown rules are shown as the server wrote them. */
export function cohortPlain(rule: string): string {
  if (rule.includes('at least this many completed games earlier in the same season')) {
    return 'Players with at least one earlier game that season.'
  }
  return rule.charAt(0).toUpperCase() + rule.slice(1) + (rule.endsWith('.') ? '' : '.')
}
