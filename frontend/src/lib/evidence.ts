import { ApiError } from '@/lib/api-client'
import { STATS, STAT_LABELS } from '@/lib/constants'
import type { BacktestCell, BacktestRow, BacktestVerdict } from '@/types/api'

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
 * The verdict for a set of rows. Improvement is averaged per scoring metric:
 * average-miss scores and count-error (deviance) scores are on different scales,
 * so they are never averaged together.
 */
export function verdictOf(rows: BacktestRow[]): BacktestVerdict {
  const byMetric: Record<string, number[]> = {}
  for (const r of rows) {
    if (r.improvement_vs_naive_pct !== null) (byMetric[r.metric] ??= []).push(r.improvement_vs_naive_pct)
  }
  return {
    cell_seasons: rows.length,
    beat_naive: rows.filter((r) => r.beats_naive).length,
    beat_trailing3: rows.filter((r) => r.beats_trailing3).length,
    beat_marcel: rows.filter((r) => r.beats_marcel).length,
    beat_both: rows.filter((r) => r.beats_naive && r.beats_trailing3).length,
    causal_check_passed: rows.length,
    improvement_by_metric: Object.fromEntries(
      Object.entries(byMetric).map(([m, g]) => [m, { cell_seasons: g.length, mean_pct: g.reduce((a, b) => a + b, 0) / g.length }]),
    ),
  }
}

/** The percent a sentence may state on its own: only when every cell shares one metric. */
function singleMetricGain(v: BacktestVerdict): number | null {
  const all = Object.values(v.improvement_by_metric)
  return all.length === 1 ? all[0].mean_pct : null
}

/** The percent that sets the tone of a verdict: the single metric's, else the average-miss one. */
export function toneGain(v: BacktestVerdict): number | null {
  return singleMetricGain(v) ?? v.improvement_by_metric.mae?.mean_pct ?? null
}

/**
 * One plain sentence: better or not than the simple method, and by how much.
 * `noun` names what is counted, e.g. "player-stat seasons" or "positions".
 * The percent appears only when all the cells share one scoring metric.
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
  const gain = singleMetricGain(v)
  if (gain === null) return `${head}.`
  return gain >= 0
    ? `${head}, by ${formatPct(gain)} on average.`
    : `${head}, and ${formatPct(gain)} worse on average.`
}

/**
 * When a verdict mixes scoring metrics, the one percent we can state honestly:
 * the average miss on yards and fantasy points. Count stats get no blended percent.
 * Null when the verdict is a single metric (the main sentence has it) or has no average-miss cells.
 */
export function mixedMetricSentence(v: BacktestVerdict): string | null {
  if (Object.keys(v.improvement_by_metric).length < 2) return null
  const mae = v.improvement_by_metric.mae
  if (!mae) return null
  const size = mae.mean_pct >= 0 ? 'smaller' : 'larger'
  return `On yards and fantasy points, our average miss is ${formatPct(mae.mean_pct)} ${size}.`
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
