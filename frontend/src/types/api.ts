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
  p10: number
  p50: number
  p90: number
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
  floor: number
  ceiling: number
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
}

export interface SeasonProjectionsResponse {
  season: number
  start_week: number
  count: number
  projections: SeasonPlayerProjection[]
  data_freshness: string
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
