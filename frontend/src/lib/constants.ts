// Offline fallbacks — live values are fetched from /season/current via useCurrentSeason hook.
export const CURRENT_SEASON = 2025
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
