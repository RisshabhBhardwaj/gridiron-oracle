import { describe, expect, it } from 'vitest'
import {
  classifyProvenance,
  GENERIC_SIMULATION_LABEL,
  isForecastRow,
  needsSimulationNotice,
  noticeLabel,
  NOT_A_FORECAST_LABEL,
  provenanceLabel,
  SIMULATION_LABEL,
  toSearchablePlayers,
  weekFigureLabel,
} from '../lib/forecast-provenance'
import { factories } from './factories'

describe('classifyProvenance', () => {
  it.each(['causal_oof_conformal_90', 'mapie_enbpi'])('calls %s a forecast (fitted model, causal intervals)', (m) => {
    expect(classifyProvenance(m)).toBe('forecast')
    expect(isForecastRow({ interval_method: m })).toBe(true)
  })

  it.each(['season_simulator_mc', 'playing_time_gaussian_mc', 'scenario_monte_carlo'])(
    'calls %s a simulation, not a forecast',
    (m) => {
      expect(classifyProvenance(m)).toBe('simulation')
      expect(isForecastRow({ interval_method: m })).toBe(false)
    },
  )

  it.each(['unavailable', 'posterior_samples', 'some_future_method', '', '  '])(
    'fails closed: %j is not a forecast',
    (m) => {
      expect(classifyProvenance(m)).toBe('unknown')
      expect(isForecastRow({ interval_method: m })).toBe(false)
    },
  )

  it('fails closed when the method is missing', () => {
    expect(classifyProvenance(undefined)).toBe('unknown')
    expect(classifyProvenance(null)).toBe('unknown')
    expect(isForecastRow({})).toBe(false)
  })
})

describe('labels', () => {
  it('says "preseason" only for the preseason simulator', () => {
    expect(provenanceLabel('season_simulator_mc')).toBe(SIMULATION_LABEL)
    expect(provenanceLabel('playing_time_gaussian_mc')).toBe(GENERIC_SIMULATION_LABEL)
    expect(provenanceLabel('scenario_monte_carlo')).toBe(GENERIC_SIMULATION_LABEL)
  })

  it('uses a neutral label for a method it cannot place', () => {
    expect(provenanceLabel('unavailable')).toBe(NOT_A_FORECAST_LABEL)
    expect(provenanceLabel(undefined)).toBe(NOT_A_FORECAST_LABEL)
  })

  it('uses one shared label for a list only when every non-forecast row agrees', () => {
    const real = { interval_method: 'mapie_enbpi' }
    expect(noticeLabel([real, { interval_method: 'season_simulator_mc' }])).toBe(SIMULATION_LABEL)
    expect(noticeLabel([{ interval_method: 'season_simulator_mc' }, { interval_method: 'playing_time_gaussian_mc' }])).toBe(NOT_A_FORECAST_LABEL)
  })
})

describe('needsSimulationNotice', () => {
  const sim = { interval_method: 'season_simulator_mc' }
  const real = { interval_method: 'causal_oof_conformal_90' }

  it('is true for rows with an unknown method, and when any row is a simulation, false when all are forecasts or there are none', () => {
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
