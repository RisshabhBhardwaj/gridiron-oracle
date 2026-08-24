// Offline fallbacks — live values are fetched from /season/current via the
// useCurrentSeason hook. Callers should gate their own requests on
// `isPlaceholderData` rather than fetching against these: 2025 has no
// team_game_predictions at all, so a request set fired against the fallback
// 404s rather than merely being stale.
export const CURRENT_SEASON = 2026
export const CURRENT_WEEK = 1

export const POSITIONS = ['WR', 'RB', 'TE', 'QB'] as const
export type Position = (typeof POSITIONS)[number]

// All 15 stats supported by the backend (VALID_STATS in projection.py)
export const STATS = [
  // Most common — shown first in dropdowns
  'receiving_yards',
  'rushing_yards',
  'passing_yards',
  'receptions',
  'targets',
  'carries',
  // Touchdowns
  'receiving_tds',
  'rushing_tds',
  'passing_tds',
  // Passing (QB-specific)
  'pass_attempts',
  'completions',
  'interceptions',
  // Cross-position / composite
  'fantasy_ppr',
  'fumbles',
  'sacks_taken',
] as const
export type Stat = (typeof STATS)[number]

export const STAT_LABELS: Record<string, string> = {
  receiving_yards: 'Receiving Yards',
  rushing_yards: 'Rushing Yards',
  passing_yards: 'Passing Yards',
  receptions: 'Receptions',
  targets: 'Targets',
  carries: 'Carries',
  receiving_tds: 'Receiving TDs',
  rushing_tds: 'Rushing TDs',
  passing_tds: 'Passing TDs',
  pass_attempts: 'Pass Attempts',
  completions: 'Completions',
  interceptions: 'Interceptions',
  fantasy_ppr: 'Fantasy Points (PPR)',
  fumbles: 'Fumbles',
  sacks_taken: 'Sacks Taken',
  ppr_points: 'Fantasy Points (PPR)',
}

// Exactly the 4 stats populated in season_simulation_weeks
export const SEASON_SIM_STATS = [
  'fantasy_ppr',
  'receiving_yards',
  'rushing_yards',
  'passing_yards',
] as const
export type SeasonSimStat = (typeof SEASON_SIM_STATS)[number]

// Exactly the stats present in ml/backtest_results/backtest_2019_2025.csv,
// which is the only backtest source that ships to production — there is no
// backtest_results table in Neon, and PRODUCT_MODE=artifact_backed forbids the
// synthetic fallback. Any stat outside this list makes /backtest return 503,
// which is how Season Review shipped defaulting to a dead surface.
export const BACKTEST_STATS = [
  'receiving_yards',
  'rushing_yards',
  'passing_yards',
  'receptions',
  'targets',
  'carries',
  'receiving_tds',
  'rushing_tds',
  'passing_tds',
] as const
export type BacktestStat = (typeof BACKTEST_STATS)[number]
