import { describe, expect, it } from 'vitest'
import { mixedMetricSentence, verdictOf, verdictSentence } from '@/lib/evidence'
import type { BacktestRow, BacktestVerdict } from '@/types/api'

const base = { cell_seasons: 10, beat_naive: 9, beat_trailing3: 10, beat_marcel: 8, beat_both: 9, causal_check_passed: 10 }
const v = (improvement_by_metric: BacktestVerdict['improvement_by_metric'], over: Partial<BacktestVerdict> = {}): BacktestVerdict => ({
  ...base, ...over, improvement_by_metric,
})

describe('verdictSentence', () => {
  it('adds the percent when every cell shares one metric', () => {
    const s = verdictSentence(v({ mae: { cell_seasons: 10, mean_pct: 10.4 } }), 'season tests')
    expect(s).toBe('Better than the simple method in 9 of 10 season tests, by 10.4% on average.')
  })

  it('keeps the percent for a count stat on its own (one metric)', () => {
    const s = verdictSentence(v({ poisson_deviance: { cell_seasons: 10, mean_pct: 50.2 } }), 'season tests')
    expect(s).toContain('by 50.2% on average')
  })

  it('never blends metrics: a mixed verdict is the count alone', () => {
    const s = verdictSentence(
      v({ mae: { cell_seasons: 5, mean_pct: 10.5 }, poisson_deviance: { cell_seasons: 5, mean_pct: 50.2 } }),
      'season tests',
    )
    expect(s).toBe('Better than the simple method in 9 of 10 season tests.')
  })
})

describe('mixedMetricSentence', () => {
  it('states the average-miss gain over yards and fantasy points only', () => {
    const s = mixedMetricSentence(v({ mae: { cell_seasons: 5, mean_pct: 10.52 }, poisson_deviance: { cell_seasons: 5, mean_pct: 50 } }))
    expect(s).toBe('On yards and fantasy points, our average miss is 10.5% smaller.')
  })

  it('says larger when the model is worse', () => {
    const s = mixedMetricSentence(v({ mae: { cell_seasons: 5, mean_pct: -3 }, poisson_deviance: { cell_seasons: 5, mean_pct: 5 } }))
    expect(s).toBe('On yards and fantasy points, our average miss is 3.0% larger.')
  })

  it('is empty for a single-metric verdict, where the main sentence already has the percent', () => {
    expect(mixedMetricSentence(v({ mae: { cell_seasons: 10, mean_pct: 10 } }))).toBeNull()
  })

  it('is empty when no cell is scored by average miss', () => {
    expect(mixedMetricSentence(v({ poisson_deviance: { cell_seasons: 10, mean_pct: 50 } }))).toBeNull()
  })
})

describe('verdictOf', () => {
  const row = (metric: string, pct: number | null, beats = true): BacktestRow =>
    ({ metric, improvement_vs_naive_pct: pct, beats_naive: beats, beats_trailing3: true, beats_marcel: true }) as BacktestRow

  it('averages each metric separately', () => {
    const out = verdictOf([row('mae', 10), row('mae', 20), row('poisson_deviance', 50), row('poisson_deviance', -10, false)])
    expect(out.cell_seasons).toBe(4)
    expect(out.beat_naive).toBe(3)
    expect(out.improvement_by_metric).toEqual({
      mae: { cell_seasons: 2, mean_pct: 15 },
      poisson_deviance: { cell_seasons: 2, mean_pct: 20 },
    })
  })

  it('has no percent for an empty selection', () => {
    expect(verdictOf([]).improvement_by_metric).toEqual({})
  })
})
