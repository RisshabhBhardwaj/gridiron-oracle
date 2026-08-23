/**
 * TypeScript interfaces matching the Pydantic models in the FastAPI backend.
 * FastAPI serializes datetime as ISO 8601 strings — typed as string here.
 */

// ── predict.py ────────────────────────────────────────────────────────────────

export interface StatProjection {
  pass_attempts: number | null
  completions: number | null
  passing_yards: number | null
  passing_tds: number | null
  interceptions: number | null
  carries: number | null
  rushing_yards: number | null
  rushing_tds: number | null
  receptions: number | null
  receiving_yards: number | null
  receiving_tds: number | null
  targets: number | null
  fantasy_ppr: number | null
  fumbles: number | null
  sacks_taken: number | null
  ppr_points: number | null
}

export interface Percentiles {
  p10: number | null
  p50: number
  p90: number | null
}

export interface SHAPFactor {
  feature: string
  impact: number
  label: string
}

export interface PredictResponse {
  player: string
  player_id: string
  week: number
  season: number
  position: string
  stat: string
  projection: StatProjection
  percentiles: Percentiles
  interval_method: string
  served_learner: string
  degraded: boolean
  pipeline_run_id: string | null
  confidence_score: number | null
  kalman_ability_estimate: number | null
  kalman_uncertainty: number | null
  boom_probability: number | null
  bust_probability: number | null
  top_factors: SHAPFactor[]
  attribution_source: string
  model_version: string
  data_freshness: string
}

export interface WeekPlayerProjection {
  player_id: string
  player_name: string
  position: string
  team: string | null
  stat: string
  projection: number
  floor: number | null
  ceiling: number | null
  interval_method: string
  served_learner: string
  degraded: boolean
  pipeline_run_id: string | null
  p5: number | null
  p25: number | null
  p75: number | null
  p95: number | null
  boom_probability: number | null
  bust_probability: number | null
  fantasy_projection: number | null
  fantasy_floor: number | null
  fantasy_ceiling: number | null
  model_version: string
  data_freshness: string
}

export interface WeekProjectionsResponse {
  week: number
  season: number
  stat: string
  count: number
  projections: WeekPlayerProjection[]
  data_freshness: string
}

export interface SeasonStatProjection {
  mean: number
  p10: number
  p50: number
  p90: number
}

export interface SeasonPlayerProjection {
  player_id: string
  player_name: string
  position: string
  team: string | null
  passing_yards: SeasonStatProjection | null
  rushing_yards: SeasonStatProjection | null
  receiving_yards: SeasonStatProjection | null
  fantasy_ppr: SeasonStatProjection | null
  degraded: boolean
  interval_method: string
  p_active: number | null
}

export interface SeasonProjectionsResponse {
  season: number
  start_week: number
  count: number
  projections: SeasonPlayerProjection[]
  data_freshness: string
}

export interface SeasonWeekProjectionsResponse {
  season: number
  start_week: number
  week: number
  count: number
  projections: SeasonPlayerProjection[]
  data_freshness: string
}

export interface TeamWinProjection {
  team: string
  wins_mean: number
  wins_p10?: number | null
  wins_p90?: number | null
}

export interface SeasonTeamWinsResponse {
  season: number
  start_week: number
  count: number
  teams: TeamWinProjection[]
  data_freshness: string
}

export interface TeamGamePrediction {
  game_id: string
  team: string
  opponent: string
  season: number
  week: number
  is_home: boolean
  points: number | null
  yards: number | null
  pass_rate: number | null
  win_probability: number | null
  model_run_id: string
  note?: string
}

export interface TeamGameWeekResponse {
  season: number
  week: number
  count: number
  games: TeamGamePrediction[]
}

export interface PairedGame {
  game_id: string
  season: number
  week: number
  home_team: string
  away_team: string
  home_points: number | null
  away_points: number | null
  home_yards: number | null
  away_yards: number | null
  home_pass_rate: number | null
  away_pass_rate: number | null
  home_win_probability: number | null
  away_win_probability: number | null
  model_run_id: string
  note?: string
}

export interface DraftBoardPlayer {
  player_name: string
  position: string | null
  team: string | null
  adp: number | null
  player_id: string | null
  source: string
  model_rank: number | null
  model_fantasy_ppr: number | null
  adp_rank: number | null
  value_vs_adp: number | null
  board_tail?: boolean
}

export interface DraftBoardResponse {
  season: number
  source: string
  scoring: string
  count: number
  players: DraftBoardPlayer[]
  spearman_rho: number | null
  projection_source: string
  as_of: string
  note: string
}

export interface ManagerProfile {
  owner_id: string
  display_name: string
  draft_slot?: number | null
  drafts: number
  picks: number
  first_qb_pick_shrunk: number
  first_te_pick_shrunk: number
  adp_delta_mean_shrunk: number
  adp_delta_sd_shrunk: number
  early_rb_share_shrunk: number
  early_wr_share_shrunk: number
}

export interface MockDraftProfilesResponse {
  season: number
  count: number
  profiles: ManagerProfile[]
  note: string
}

export interface PickedItem {
  pick_no: number
  round: number
  slot: number
  owner_id: string
  player: DraftBoardPlayer
}

export interface PickRequest {
  season: number
  draft_order: string[]
  picks_so_far: PickedItem[]
  user_slot?: number
  auto_pick_user?: boolean
  seed?: number
}

export interface PickResponse {
  season: number
  new_picks: PickedItem[]
  is_user_turn: boolean
  is_draft_complete: boolean
  next_turn_slot: number | null
  seed: number
}

// ── explain.py ────────────────────────────────────────────────────────────────

export interface FactorItem {
  feature: string
  label: string
  impact: number
  value: number
}

export interface ExplainResponse {
  player_id: string
  week: number
  season: number
  stat: string
  position: string
  base_value: number
  attribution_source: string
  top_factors: FactorItem[]
  data_freshness: string
}

// ── scenario.py ───────────────────────────────────────────────────────────────

export interface ScenarioOverrides {
  wind_speed_mph?: number
  temperature_f?: number
  snap_share?: number
  primary_defender_grade?: number
  dome_override?: boolean
}

export interface ScenarioRequest {
  player_id: string
  week: number
  season: number
  stat: string
  overrides: ScenarioOverrides
}

export interface ScenarioResponse {
  player_id: string
  player_name: string          // ← Added: backend returns this
  stat: string
  week: number
  season: number
  base_projection: number
  scenario_projection: number
  delta: number
  delta_pct: number
  percentiles: Percentiles
  boom_probability: number | null
  bust_probability: number | null
  fantasy_projection: number | null
  data_freshness: string
}

// ── backtest.py ───────────────────────────────────────────────────────────────

export interface CalibrationPoint {
  predicted_prob: number
  observed_freq: number
  n_samples: number
}

export interface SeasonMetrics {
  season: number
  position: string
  stat: string
  n_games: number
  stack_mae: number
  stack_rmse: number
  stack_crps: number
  naive_mae: number
  coverage_80: number
  coverage_50: number
  baseline_improvement_pct: number
}

export interface BacktestResponse {
  model_version: string
  seasons: number[]
  positions: string[]
  stat: string
  overall_mae: number
  overall_rmse: number
  overall_crps: number
  brier_score: number | null
  calibration: CalibrationPoint[]
  simulated_pnl: number | null
  sharpe_ratio: number | null
  max_drawdown: number | null
  by_season: SeasonMetrics[]
  data_source: string
  metric_notes: Record<string, string>
  data_freshness: string
}

// ── settings.py ───────────────────────────────────────────────────────────────

export interface FeatureWeights {
  kalman_form: number
  seasonal_baseline: number
  matchup: number
  weather_venue: number
  team_context: number
  roster_injury: number
  rule_meta: number
}

export type FantasyScoring = 'ppr' | 'half_ppr' | 'standard'

export interface SettingsResponse {
  weights: FeatureWeights
  fantasy_scoring: FantasyScoring
  engine_exposure_cap: number
  weight_preset_name: string
}

export interface SettingsUpdateRequest {
  weights?: FeatureWeights
  fantasy_scoring?: FantasyScoring
  engine_exposure_cap?: number
  weight_preset_name?: string
}

// ── season ────────────────────────────────────────────────────────────────────

export interface CurrentSeasonResponse {
  season: number
  week: number
}

// ── alerts ────────────────────────────────────────────────────────────────────

// Matches backend AlertSeverity enum values exactly
export type AlertSeverity = 'info' | 'warning' | 'edge' | 'injury'

export interface AlertItem {
  id: string
  severity: AlertSeverity
  title: string            // ← Corrected: was 'message'
  body: string             // ← Added: backend returns body text
  player_id: string | null
  player_name: string | null
  stat: string | null
  value: number | null
  timestamp: string        // ← Corrected: was 'created_at'
}

export interface AlertsResponse {
  alerts: AlertItem[]
  count: number
}

// ── drive-sim ─────────────────────────────────────────────────────────────────

export interface GameAnchor {
  home_points: number
  away_points: number
  home_yards: number
  away_yards: number
  home_pass_rate: number
  away_pass_rate: number
  home_win_probability: number
}

export interface GameSimSummary {
  simulated_home_points: number
  simulated_away_points: number
  simulated_home_yards: number
  simulated_away_yards: number
  total_drives: number
  simulated_winner: string
}

export interface SimulatedPlayDTO {
  play_number: number
  down: number
  ytg: number
  field_pos: number
  play_type: string
  yards_gained: number
  is_turnover: boolean
  is_first_down: boolean
  is_touchdown: boolean
  is_safety: boolean
  end_field_pos: number
}

export interface SimulatedDriveDTO {
  drive_number: number
  possession_team: string
  quarter: number
  start_field_pos: number
  end_field_pos: number
  plays_count: number
  yards_gained: number
  outcome: string
  points_scored: number
  home_score_after: number
  away_score_after: number
  plays: SimulatedPlayDTO[]
}

export interface GameDriveSimResponse {
  game_id: string
  season: number
  week: number
  home_team: string
  away_team: string
  seed: number
  anchor: GameAnchor
  summary: GameSimSummary
  drives: SimulatedDriveDTO[]
  note: string
}
