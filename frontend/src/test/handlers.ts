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

  http.get('/api/draft/board', () =>
    HttpResponse.json({
      season: 2026, source: 'fantasypros', projection_source: 'preseason_historical_per_game',
      as_of: '2026-08-01', scoring: 'ppr', count: 2, spearman_rho: 0.42,
      note: 'Causal preseason data only.',
      players: [
        { player_name: 'Ja\'Marr Chase', position: 'WR', team: 'CIN', adp: 1.2, player_id: '00-003', source: 'fantasypros', model_rank: 1, model_fantasy_ppr: 289.2, adp_rank: 1, value_vs_adp: 0 },
        { player_name: 'Bijan Robinson', position: 'RB', team: 'ATL', adp: 2.4, player_id: '00-004', source: 'fantasypros', model_rank: 4, model_fantasy_ppr: 246.1, adp_rank: 2, value_vs_adp: -2 },
      ],
    }),
  ),

  http.get('/api/settings', () =>
    HttpResponse.json(factories.settingsResponse()),
  ),

  http.put('/api/settings', () =>
    HttpResponse.json(factories.settingsResponse()),
  ),
]

export const server = setupServer(...handlers)
