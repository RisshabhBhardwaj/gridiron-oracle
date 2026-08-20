/**
 * Strongly-typed factory functions returning valid mock data for each API response type.
 */
import type {
  WeekProjectionsResponse,
  PredictResponse,
  ExplainResponse,
  ScenarioResponse,
  BacktestResponse,
  SettingsResponse,
  SeasonProjectionsResponse,
  WeekPlayerProjection,
  FactorItem,
} from '../types/api'

function freshTimestamp(): string {
  return new Date().toISOString()
}

function staleTimestamp(): string {
  // 49 hours ago — clearly > 24h
  return new Date(Date.now() - 49 * 60 * 60 * 1000).toISOString()
}

function makePlayer(overrides?: Partial<WeekPlayerProjection>): WeekPlayerProjection {
  return {
    player_id: 'test-001',
    player_name: 'Justin Jefferson',
    position: 'WR',
    team: 'MIN',
    stat: 'receiving_yards',
    projection: 87.4,
    floor: 42.1,
    ceiling: 142.8,
    interval_method: 'causal_oof_conformal_90',
    served_learner: 'lgbm_identity',
    degraded: false,
    pipeline_run_id: 'stack_materialize_test',
    p5: null,
    p25: null,
    p75: null,
    p95: null,
    boom_probability: 0.34,
    bust_probability: 0.18,
    fantasy_projection: 15.2,
    fantasy_floor: 8.1,
    fantasy_ceiling: 24.7,
    model_version: 'v1.0.0-test',
    data_freshness: freshTimestamp(),
    ...overrides,
  }
}

export const factories = {
  weekProjectionsResponse(overrides?: { stale?: boolean }): WeekProjectionsResponse {
    const ts = overrides?.stale ? staleTimestamp() : freshTimestamp()
    return {
      week: 5,
      season: 2025,
      stat: 'receiving_yards',
      count: 3,
      projections: [
        makePlayer({ data_freshness: ts }),
        makePlayer({
          player_id: 'test-002',
          player_name: 'CeeDee Lamb',
          position: 'WR',
          team: 'DAL',
          projection: 95.1,
          floor: 48.0,
          ceiling: 155.0,
          data_freshness: ts,
        }),
        makePlayer({
          player_id: 'test-003',
          player_name: 'Dalvin Cook',
          position: 'RB',
          team: 'NYJ',
          stat: 'rushing_yards',
          projection: 72.3,
          floor: 30.0,
          ceiling: 120.0,
          data_freshness: ts,
        }),
      ],
      data_freshness: ts,
    }
  },

  seasonProjectionsResponse(): SeasonProjectionsResponse {
    return {
      season: 2025,
      start_week: 6,
      count: 2,
      projections: [
        {
          player_id: 'test-001',
          player_name: 'Justin Jefferson',
          position: 'WR',
          team: 'MIN',
          passing_yards: null,
          rushing_yards: null,
          receiving_yards: { mean: 1024.5, p10: 850.0, p50: 1010.0, p90: 1215.0 },
          fantasy_ppr: { mean: 214.3, p10: 175.0, p50: 210.0, p90: 252.0 },
          degraded: false,
          interval_method: 'playing_time_enbpi',
          p_active: 0.92,
        },
        {
          player_id: 'test-002',
          player_name: 'CeeDee Lamb',
          position: 'WR',
          team: 'DAL',
          passing_yards: null,
          rushing_yards: null,
          receiving_yards: { mean: 998.0, p10: 790.0, p50: 980.0, p90: 1190.0 },
          fantasy_ppr: { mean: 205.7, p10: 168.0, p50: 201.0, p90: 244.0 },
          degraded: false,
          interval_method: 'playing_time_enbpi',
          p_active: 0.90,
        },
      ],
      data_freshness: new Date().toISOString(),
    }
  },

  predictResponse(overrides?: { stale?: boolean }): PredictResponse {
    const ts = overrides?.stale ? staleTimestamp() : freshTimestamp()
    return {
      player: 'Justin Jefferson',
      player_id: 'test-001',
      week: 5,
      season: 2025,
      position: 'WR',
      stat: 'receiving_yards',
      projection: {
        pass_attempts: null,
        completions: null,
        passing_yards: null,
        passing_tds: null,
        interceptions: null,
        carries: null,
        rushing_yards: null,
        rushing_tds: null,
        receptions: null,
        receiving_yards: 87.4,
        receiving_tds: null,
        targets: null,
        fantasy_ppr: 15.2,
        fumbles: null,
        sacks_taken: null,
        ppr_points: 15.2,
      },
      percentiles: {
        p10: 42.1,
        p50: 87.4,
        p90: 142.8,
      },
      interval_method: 'causal_oof_conformal_90',
      served_learner: 'lgbm_identity',
      degraded: false,
      pipeline_run_id: 'stack_materialize_test',
      confidence_score: 0.78,
      kalman_ability_estimate: 89.0,
      kalman_uncertainty: 12.5,
      boom_probability: 0.34,
      bust_probability: 0.18,
      top_factors: [
        { feature: 'kalman_est_receiving_yards', impact: 12.3, label: 'Player Form (Kalman)' },
        { feature: 'opp_def_rank', impact: -5.1, label: 'Opponent Defense Rank' },
        { feature: 'snap_share', impact: 8.7, label: 'Snap Share' },
      ],
      attribution_source: 'model_artifact',
      model_version: 'v1.0.0-test',
      data_freshness: ts,
    }
  },

  explainResponse(): ExplainResponse {
    const factors: FactorItem[] = [
      { feature: 'kalman_est_receiving_yards', label: 'Player Form (Kalman)', impact: 12.3, value: 89.0 },
      { feature: 'opp_def_rank', label: 'Opponent Defense Rank', impact: -5.1, value: 28.0 },
      { feature: 'snap_share', label: 'Snap Share', impact: 8.7, value: 0.82 },
      { feature: 'target_share', label: 'Target Share', impact: 6.2, value: 0.28 },
      { feature: 'temp_f', label: 'Temperature (°F)', impact: -1.4, value: 34.0 },
    ]
    return {
      player_id: 'test-001',
      week: 5,
      season: 2025,
      stat: 'receiving_yards',
      position: 'WR',
      base_value: 72.1,
      attribution_source: 'model_artifact',
      top_factors: factors,
      data_freshness: new Date().toISOString(),
    }
  },

  scenarioResponse(): ScenarioResponse {
    return {
      player_id: 'test-001',
      player_name: 'Justin Jefferson',   // backend returns this
      week: 5,
      season: 2025,
      stat: 'receiving_yards',
      base_projection: 87.4,
      scenario_projection: 91.2,
      delta: 3.8,
      delta_pct: 4.35,
      percentiles: { p10: 44.0, p50: 91.2, p90: 148.0 },
      boom_probability: 0.36,
      bust_probability: 0.16,
      fantasy_projection: 15.9,
      data_freshness: new Date().toISOString(),
    }
  },

  backtestResponse(): BacktestResponse {
    return {
      model_version: 'v1.0.0-test',
      seasons: [2022, 2023, 2025],
      positions: ['WR', 'RB', 'TE', 'QB'],
      stat: 'receiving_yards',
      overall_mae: 18.42,
      overall_rmse: 24.11,
      overall_crps: 0.312,
      brier_score: 0.198,
      calibration: [
        { predicted_prob: 0.1, observed_freq: 0.09, n_samples: 120 },
        { predicted_prob: 0.3, observed_freq: 0.28, n_samples: 210 },
        { predicted_prob: 0.5, observed_freq: 0.51, n_samples: 340 },
        { predicted_prob: 0.7, observed_freq: 0.69, n_samples: 280 },
        { predicted_prob: 0.9, observed_freq: 0.88, n_samples: 150 },
      ],
      simulated_pnl: 14.72,
      sharpe_ratio: 1.34,
      max_drawdown: -8.20,
      data_source: 'csv',
      metric_notes: {},
      by_season: [
        {
          season: 2022,
          position: 'WR',
          stat: 'receiving_yards',
          n_games: 532,
          stack_mae: 17.8,
          stack_rmse: 23.4,
          stack_crps: 0.301,
          naive_mae: 24.2,
          coverage_80: 0.79,
          coverage_50: 0.51,
          baseline_improvement_pct: 26.4,
        },
        {
          season: 2023,
          position: 'WR',
          stat: 'receiving_yards',
          n_games: 548,
          stack_mae: 18.1,
          stack_rmse: 24.0,
          stack_crps: 0.315,
          naive_mae: 23.9,
          coverage_80: 0.81,
          coverage_50: 0.49,
          baseline_improvement_pct: 24.3,
        },
        {
          season: 2025,
          position: 'WR',
          stat: 'receiving_yards',
          n_games: 301,
          stack_mae: 19.2,
          stack_rmse: 25.0,
          stack_crps: 0.322,
          naive_mae: 25.1,
          coverage_80: 0.77,
          coverage_50: 0.52,
          baseline_improvement_pct: 23.5,
        },
      ],
      data_freshness: new Date().toISOString(),
    }
  },

  settingsResponse(): SettingsResponse {
    return {
      weights: {
        kalman_form: 35,
        seasonal_baseline: 20,
        matchup: 20,
        weather_venue: 10,
        team_context: 5,
        roster_injury: 5,
        rule_meta: 5,
      },
      fantasy_scoring: 'ppr',
      engine_exposure_cap: 0.25,
      weight_preset_name: 'Default',
    }
  },
}
