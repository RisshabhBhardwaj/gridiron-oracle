import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import { factories } from './factories'

export const handlers = [
  http.get('/api/projections/week/:week', () =>
    HttpResponse.json(factories.weekProjectionsResponse()),
  ),

  http.get('/api/projections/season/:season', () =>
    HttpResponse.json(factories.seasonProjectionsResponse()),
  ),

  http.get('/api/predict', () =>
    HttpResponse.json(factories.predictResponse()),
  ),

  http.get('/api/season/current', () =>
    HttpResponse.json({ season: 2025, week: 5 }),
  ),

  http.get('/api/explain/:player_id', () =>
    HttpResponse.json(factories.explainResponse()),
  ),

  http.post('/api/scenario', () =>
    HttpResponse.json(factories.scenarioResponse()),
  ),

  http.get('/api/backtest', () =>
    HttpResponse.json(factories.backtestResponse()),
  ),

  http.get('/api/settings', () =>
    HttpResponse.json(factories.settingsResponse()),
  ),

  http.put('/api/settings', () =>
    HttpResponse.json(factories.settingsResponse()),
  ),
]

export const server = setupServer(...handlers)
