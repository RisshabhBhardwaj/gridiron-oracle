import { describe, expect, it } from 'vitest'
import { ApiError } from '../lib/api-client'
import {
  explainFailure,
  isCurrentOrLater,
  isLiveLineWeek,
  isNoForecastError,
  parseWeek,
  repeatsHeroFigure,
  weekFigure,
} from '../lib/player-page'
import { factories } from './factories'

function predictWith(intervalMethod: string, p10: number | null = 42.1, p90: number | null = 142.8) {
  const base = factories.predictResponse()
  return { ...base, interval_method: intervalMethod, percentiles: { ...base.percentiles, p10, p90 } }
}

describe('weekFigure: what the page may show as the Week N number', () => {
  it.each(['causal_oof_conformal_90', 'mapie_enbpi'])('shows the projection of a %s row, with its range', (method) => {
    expect(weekFigure(predictWith(method))).toEqual({ kind: 'forecast', value: 87.4, low: 42.1, high: 142.8 })
  })

  it.each(['season_simulator_mc', 'playing_time_gaussian_mc', 'scenario_monte_carlo', 'some_future_method', ''])(
    'never shows a %j row as a week forecast',
    (method) => {
      expect(weekFigure(predictWith(method))).toEqual({ kind: 'none' })
    },
  )

  it('gives no figure for a missing answer', () => {
    expect(weekFigure(undefined)).toEqual({ kind: 'none' })
    expect(weekFigure(null)).toEqual({ kind: 'none' })
  })

  it('keeps the figure but drops a range the model did not give', () => {
    expect(weekFigure(predictWith('causal_oof_conformal_90', null, null))).toEqual({
      kind: 'forecast', value: 87.4, low: null, high: null,
    })
  })

  it('uses the stated projection for the stat, falling back to p50 (the point estimate)', () => {
    const base = predictWith('causal_oof_conformal_90')
    const missing = { ...base, projection: { ...base.projection, receiving_yards: null } }
    expect(weekFigure(missing)).toMatchObject({ kind: 'forecast', value: base.percentiles.p50 })
  })

  it('gives no figure when no number is finite', () => {
    const base = predictWith('causal_oof_conformal_90')
    const broken = {
      ...base,
      projection: { ...base.projection, receiving_yards: null },
      percentiles: { ...base.percentiles, p50: Number.NaN },
    }
    expect(weekFigure(broken)).toEqual({ kind: 'none' })
  })
})

describe('parseWeek', () => {
  it('accepts a week from 1 to 22', () => {
    expect(parseWeek('1', 9)).toBe(1)
    expect(parseWeek('18', 9)).toBe(18)
    expect(parseWeek('22', 9)).toBe(22)
  })

  it.each([null, '', ' ', 'abc', '0', '23', '-3', '5.5', 'NaN'])('falls back for %j', (raw) => {
    expect(parseWeek(raw, 9)).toBe(9)
  })
})

describe('repeatsHeroFigure', () => {
  it('is true for a fantasy stat and for an equal figure', () => {
    expect(repeatsHeroFigure('fantasy_ppr', 15.2, 15.2)).toBe(true)
    expect(repeatsHeroFigure('ppr_points', 20, 15)).toBe(true)
    expect(repeatsHeroFigure('receptions', 15.21, 15.2)).toBe(true)
  })

  it('is false for a different figure, and when there is no fantasy figure', () => {
    expect(repeatsHeroFigure('receiving_yards', 87.4, 15.2)).toBe(false)
    expect(repeatsHeroFigure('receiving_yards', 87.4, null)).toBe(false)
    expect(repeatsHeroFigure('fantasy_ppr', 15.2, undefined)).toBe(false)
  })
})

describe('which week the sportsbook is pricing', () => {
  const now = { season: 2026, week: 11 }

  it('only the current week of the current season has a live line', () => {
    expect(isLiveLineWeek({ season: 2026, week: 11 }, now)).toBe(true)
    expect(isLiveLineWeek({ season: 2026, week: 5 }, now)).toBe(false)
    expect(isLiveLineWeek({ season: 2025, week: 11 }, now)).toBe(false)
  })

  it('tells a played week from a current or future one', () => {
    expect(isCurrentOrLater({ season: 2026, week: 11 }, now)).toBe(true)
    expect(isCurrentOrLater({ season: 2026, week: 12 }, now)).toBe(true)
    expect(isCurrentOrLater({ season: 2027, week: 1 }, now)).toBe(true)
    expect(isCurrentOrLater({ season: 2026, week: 10 }, now)).toBe(false)
    expect(isCurrentOrLater({ season: 2025, week: 18 }, now)).toBe(false)
  })
})

describe('errors', () => {
  const noForecast = new ApiError(404, 'x', { forecast_available: false, message: 'm' })

  it('only a 404 that says forecast_available=false is the "no forecast" state', () => {
    expect(isNoForecastError(noForecast)).toBe(true)
    expect(isNoForecastError(new ApiError(404, 'route missing'))).toBe(false)
    expect(isNoForecastError(new ApiError(500, 'boom'))).toBe(false)
    expect(isNoForecastError(new Error('Failed to fetch'))).toBe(false)
    expect(isNoForecastError(null)).toBe(false)
  })

  it('a missing breakdown (404) is "not published"; every other failure is a failure', () => {
    expect(explainFailure(new ApiError(404, 'none'))).toBe('not-published')
    expect(explainFailure(new ApiError(503, 'down'))).toBe('failed')
    expect(explainFailure(new ApiError(500, 'boom'))).toBe('failed')
    expect(explainFailure(new TypeError('Failed to fetch'))).toBe('failed')
  })
})
