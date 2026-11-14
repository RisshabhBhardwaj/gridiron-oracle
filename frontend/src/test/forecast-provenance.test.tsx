import { describe, expect, it } from 'vitest'
import {
  classifyProvenance,
  isForecastRow,
  needsSimulationNotice,
  toSearchablePlayers,
  weekFigureLabel,
} from '../lib/forecast-provenance'
import { factories } from './factories'

describe('classifyProvenance', () => {
  it('calls the season simulator a simulation', () => {
    expect(classifyProvenance('season_simulator_mc')).toBe('simulation')
  })

  it('calls any other named method a forecast, so a real producer needs no UI change', () => {
    expect(classifyProvenance('causal_oof_conformal_90')).toBe('forecast')
    expect(classifyProvenance('some_future_method')).toBe('forecast')
  })

  it('fails closed when the method is missing or blank', () => {
    expect(classifyProvenance(undefined)).toBe('unknown')
    expect(classifyProvenance(null)).toBe('unknown')
    expect(classifyProvenance('  ')).toBe('unknown')
    expect(isForecastRow({ interval_method: '' })).toBe(false)
  })
})

describe('needsSimulationNotice', () => {
  const sim = { interval_method: 'season_simulator_mc' }
  const real = { interval_method: 'causal_oof_conformal_90' }

  it('is true when any row is a simulation, false when all are forecasts or there are none', () => {
    expect(needsSimulationNotice([sim, sim])).toBe(true)
    expect(needsSimulationNotice([real, sim])).toBe(true)
    expect(needsSimulationNotice([real, real])).toBe(false)
    expect(needsSimulationNotice([])).toBe(false)
  })
})

describe('toSearchablePlayers', () => {
  it('gives no number for simulation rows and the mean for forecast rows', () => {
    const rows = factories.seasonWeekProjectionsResponse().projections
    const sim = toSearchablePlayers(rows, 'fantasy_ppr')
    expect(sim.map((p) => p.projection)).toEqual([null, null])

    const real = toSearchablePlayers(
      rows.map((r) => ({ ...r, interval_method: 'causal_oof_conformal_90' })),
      'fantasy_ppr',
    )
    expect(real[0].projection).toBe(19.8)
  })

  it('gives no number when the forecast row has no value for the stat', () => {
    const rows = factories
      .seasonWeekProjectionsResponse()
      .projections.map((r) => ({ ...r, interval_method: 'causal_oof_conformal_90' }))
    // Jefferson has no rushing yards.
    expect(toSearchablePlayers(rows, 'rushing_yards')[0].projection).toBeNull()
  })
})

describe('weekFigureLabel', () => {
  it('names the week and the stat', () => {
    expect(weekFigureLabel(11, 'fantasy_ppr')).toBe('Week 11 · Fantasy Points (PPR)')
  })
})
