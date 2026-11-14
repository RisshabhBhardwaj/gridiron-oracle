/**
 * Strongly-typed factory functions returning valid mock data for each API response type.
 */
import type {
  WeekProjectionsResponse,
  PredictResponse,
  ExplainResponse,
  ScenarioResponse,
  BacktestCell,
  BacktestResponse,
  BacktestRow,
  BacktestVerdict,
  SettingsResponse,
  SeasonProjectionsResponse,
  SeasonWeekProjectionsResponse,
  SeasonTeamWinsResponse,
  TeamGameWeekResponse,
  GameDriveSimResponse,
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
    weighting: 'default',
    weights_reason: 'defaults',
    fantasy_scoring: 'ppr',
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
          interval_method: 'playing_time_gaussian_mc',
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
          interval_method: 'playing_time_gaussian_mc',
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
      weighting: 'default',
      weights_reason: 'defaults',
      fantasy_scoring: 'ppr',
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

  backtestResponse(stat: string | null = null): BacktestResponse {
    const COHORT = 'player has at least this many completed games earlier in the same season (min_prior_games=1)'
    const mk = (
      s: string, position: string, season: number, model: number, naive: number,
      t3: number, marcel: number, beatsMarcel = true, metric = 'mae',
    ): BacktestRow => ({
      stat: s, position, season, metric,
      metric_label: metric === 'mae' ? 'Average miss, in yards (lower is better)' : 'Count error score, Poisson deviance (lower is better)',
      n: 500, cohort_rule: COHORT,
      model_score: model, naive_score: naive, trailing3_score: t3, marcel_score: marcel,
      beats_naive: model < naive, beats_trailing3: model < t3, beats_marcel: beatsMarcel,
      improvement_vs_naive_pct: naive ? ((naive - model) / naive) * 100 : null,
    })
    const all: BacktestRow[] = [
      mk('receiving_yards', 'WR', 2024, 22.9, 25.0, 24.0, 23.5),
      mk('receiving_yards', 'WR', 2025, 23.24, 25.17, 24.1, 23.9),
      mk('receiving_yards', 'TE', 2025, 15.1, 16.4, 15.9, 15.6),
      mk('receiving_yards', 'RB', 2025, 12.2, 13.0, 12.7, 12.5, false),
      mk('rushing_yards', 'RB', 2024, 20.1, 22.4, 21.5, 20.9),
      mk('rushing_yards', 'RB', 2025, 19.6, 21.8, 21.0, 20.4),
      mk('rushing_yards', 'QB', 2025, 9.4, 9.1, 9.8, 9.6),
      mk('receptions', 'TE', 2025, 0.95, 1.3, 1.2, 1.1, true, 'poisson_deviance'),
    ]
    const verdictOf = (rows: BacktestRow[]): BacktestVerdict => {
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
    const rows = stat ? all.filter((r) => r.stat === stat) : all
    const seen = new Set<string>()
    const available: BacktestCell[] = []
    for (const r of all) {
      const k = `${r.stat}:${r.position}`
      if (!seen.has(k)) {
        seen.add(k)
        available.push({ stat: r.stat, position: r.position, metric: r.metric, metric_label: r.metric_label })
      }
    }
    return {
      release: 'causal_20261114',
      generated_at: '2026-11-14T12:00:00Z',
      seasons: [2024, 2025],
      stat,
      positions: ['QB', 'RB', 'TE', 'WR'],
      verdict: verdictOf(rows),
      release_verdict: verdictOf(all),
      cohort_rules: [COHORT],
      rows,
      available,
      not_measured: {
        coverage: "We don't yet publish how often outcomes land inside our range.",
        pnl: "We don't yet publish simulated betting results.",
      },
      table_url: '/backtest/table.csv',
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
      settings_source: 'file',
    }
  },

  seasonWeekProjectionsResponse(): SeasonWeekProjectionsResponse {
    return {
      season: 2026,
      start_week: 1,
      week: 5,
      count: 2,
      projections: [
        {
          player_id: 'test-001',
          player_name: 'Justin Jefferson',
          position: 'WR',
          team: 'MIN',
          passing_yards: null,
          rushing_yards: null,
          receiving_yards: { mean: 92.4, p10: 48.0, p50: 90.0, p90: 145.0 },
          fantasy_ppr: { mean: 19.8, p10: 9.5, p50: 19.0, p90: 31.2 },
          degraded: false,
          interval_method: 'season_simulator_mc',
          p_active: 1.0,
        },
        {
          player_id: 'test-002',
          player_name: 'Bijan Robinson',
          position: 'RB',
          team: 'ATL',
          passing_yards: null,
          rushing_yards: { mean: 78.5, p10: 35.0, p50: 76.0, p90: 125.0 },
          receiving_yards: { mean: 32.0, p10: 10.0, p50: 30.0, p90: 60.0 },
          fantasy_ppr: { mean: 17.5, p10: 8.0, p50: 17.0, p90: 27.5 },
          degraded: false,
          interval_method: 'season_simulator_mc',
          p_active: 1.0,
        },
      ],
      data_freshness: freshTimestamp(),
    }
  },

  seasonTeamWinsResponse(): SeasonTeamWinsResponse {
    return {
      season: 2026,
      start_week: 1,
      count: 4,
      teams: [
        { team: 'KC', wins_mean: 11.8, wins_p10: 9, wins_p90: 14 },
        { team: 'SF', wins_mean: 11.2, wins_p10: 9, wins_p90: 14 },
        { team: 'BAL', wins_mean: 10.9, wins_p10: 8, wins_p90: 13 },
        { team: 'MIN', wins_mean: 8.7, wins_p10: 6, wins_p90: 11 },
      ],
      data_freshness: freshTimestamp(),
    }
  },

  teamGameWeekResponse(): TeamGameWeekResponse {
    return {
      season: 2026,
      week: 5,
      count: 2,
      games: [
        {
          game_id: '2026_05_MIN_GB',
          team: 'MIN',
          opponent: 'GB',
          season: 2026,
          week: 5,
          is_home: true,
          points: 24.2,
          yards: 345.0,
          pass_rate: 0.62,
          win_probability: 0.54,
          model_run_id: 'weekly_auto_2026',
        },
        {
          game_id: '2026_05_MIN_GB',
          team: 'GB',
          opponent: 'MIN',
          season: 2026,
          week: 5,
          is_home: false,
          points: 22.8,
          yards: 330.0,
          pass_rate: 0.58,
          win_probability: 0.46,
          model_run_id: 'weekly_auto_2026',
        },
      ],
    }
  },

  mockDraftProfilesResponse() {
    return {
      season: 2026,
      count: 8,
      profiles: [
        {
          owner_id: 'owner_1',
          display_name: 'Manager Alpha',
          draft_slot: 1,
          drafts: 4,
          picks: 68,
          first_qb_pick_shrunk: 42.5,
          first_te_pick_shrunk: 65.0,
          early_rb_share_shrunk: 0.45,
          early_wr_share_shrunk: 0.55,
          adp_delta_mean_shrunk: -2.4,
          adp_delta_sd_shrunk: 8.2,
        },
        {
          owner_id: 'owner_2',
          display_name: 'Manager Beta',
          draft_slot: 2,
          drafts: 4,
          picks: 68,
          first_qb_pick_shrunk: 15.0,
          first_te_pick_shrunk: 30.0,
          early_rb_share_shrunk: 0.35,
          early_wr_share_shrunk: 0.65,
          adp_delta_mean_shrunk: 1.1,
          adp_delta_sd_shrunk: 6.5,
        },
        {
          owner_id: 'owner_3',
          display_name: 'Manager Gamma',
          draft_slot: 3,
          drafts: 4,
          picks: 68,
          first_qb_pick_shrunk: 48.0,
          first_te_pick_shrunk: 60.0,
          early_rb_share_shrunk: 0.5,
          early_wr_share_shrunk: 0.4,
          adp_delta_mean_shrunk: 0.0,
          adp_delta_sd_shrunk: 5.0,
        },
        {
          owner_id: 'owner_4',
          display_name: 'Manager Delta',
          draft_slot: 4,
          drafts: 4,
          picks: 68,
          first_qb_pick_shrunk: 50.0,
          first_te_pick_shrunk: 62.0,
          early_rb_share_shrunk: 0.4,
          early_wr_share_shrunk: 0.5,
          adp_delta_mean_shrunk: -1.0,
          adp_delta_sd_shrunk: 5.5,
        },
        {
          owner_id: 'owner_5',
          display_name: 'Manager Epsilon',
          draft_slot: 5,
          drafts: 4,
          picks: 68,
          first_qb_pick_shrunk: 52.0,
          first_te_pick_shrunk: 64.0,
          early_rb_share_shrunk: 0.45,
          early_wr_share_shrunk: 0.45,
          adp_delta_mean_shrunk: 0.5,
          adp_delta_sd_shrunk: 4.8,
        },
        {
          owner_id: 'owner_6',
          display_name: 'Manager Zeta',
          draft_slot: 6,
          drafts: 4,
          picks: 68,
          first_qb_pick_shrunk: 45.0,
          first_te_pick_shrunk: 58.0,
          early_rb_share_shrunk: 0.55,
          early_wr_share_shrunk: 0.35,
          adp_delta_mean_shrunk: -0.5,
          adp_delta_sd_shrunk: 6.0,
        },
        {
          owner_id: 'owner_7',
          display_name: 'Manager Eta',
          draft_slot: 7,
          drafts: 4,
          picks: 68,
          first_qb_pick_shrunk: 40.0,
          first_te_pick_shrunk: 55.0,
          early_rb_share_shrunk: 0.6,
          early_wr_share_shrunk: 0.3,
          adp_delta_mean_shrunk: 1.5,
          adp_delta_sd_shrunk: 7.0,
        },
        {
          owner_id: 'owner_8',
          display_name: 'Manager Theta',
          draft_slot: 8,
          drafts: 4,
          picks: 68,
          first_qb_pick_shrunk: 35.0,
          first_te_pick_shrunk: 50.0,
          early_rb_share_shrunk: 0.5,
          early_wr_share_shrunk: 0.4,
          adp_delta_mean_shrunk: -3.0,
          adp_delta_sd_shrunk: 9.0,
        },
      ],
      note: 'Per-manager behavior is estimated with empirical Bayes shrinkage.',
    }
  },

  mockDraftPickResponse() {
    return {
      season: 2026,
      pick_number: 3,
      owner_id: 'owner_3',
      player: {
        player_name: 'CeeDee Lamb',
        position: 'WR',
        team: 'DAL',
        adp: 3.1,
        player_id: '00-0036322',
        source: 'fantasypros',
        model_rank: 2,
        model_fantasy_ppr: 280.5,
        adp_rank: 3,
        value_vs_adp: 1,
      },
      seed: 42,
    }
  },

  driveSimResponse(): GameDriveSimResponse {
    return {
      game_id: '2026_05_MIN_GB',
      season: 2026,
      week: 5,
      home_team: 'MIN',
      away_team: 'GB',
      seed: 42,
      anchor: {
        home_points: 24.5,
        away_points: 22.8,
        home_yards: 350.0,
        away_yards: 330.0,
        home_pass_rate: 0.58,
        away_pass_rate: 0.58,
        home_win_probability: 0.54,
      },
      summary: {
        simulated_home_points: 24,
        simulated_away_points: 20,
        simulated_home_yards: 345.0,
        simulated_away_yards: 310.0,
        total_drives: 22,
        simulated_winner: 'MIN',
      },
      drives: [
        {
          drive_number: 1,
          possession_team: 'GB',
          quarter: 1,
          start_field_pos: 25,
          end_field_pos: 65,
          plays_count: 5,
          yards_gained: 40,
          outcome: 'PUNT',
          points_scored: 0,
          home_score_after: 0,
          away_score_after: 0,
          plays: [
            {
              play_number: 1,
              down: 1,
              ytg: 10,
              field_pos: 25,
              play_type: 'pass',
              yards_gained: 12,
              is_turnover: false,
              is_first_down: true,
              is_touchdown: false,
              is_safety: false,
              end_field_pos: 37,
            },
          ],
        },
        {
          drive_number: 2,
          possession_team: 'MIN',
          quarter: 1,
          start_field_pos: 25,
          end_field_pos: 100,
          plays_count: 6,
          yards_gained: 75,
          outcome: 'TOUCHDOWN',
          points_scored: 7,
          home_score_after: 7,
          away_score_after: 0,
          plays: [
            {
              play_number: 1,
              down: 1,
              ytg: 10,
              field_pos: 25,
              play_type: 'run',
              yards_gained: 75,
              is_turnover: false,
              is_first_down: true,
              is_touchdown: true,
              is_safety: false,
              end_field_pos: 100,
            },
          ],
        },
      ],
      note: 'Simulated via drive-by-drive Markov path walk reconciled to fitted team game predictions.',
    }
  },
}
