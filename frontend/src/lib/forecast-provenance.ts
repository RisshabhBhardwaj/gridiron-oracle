/**
 * Where a weekly number came from, decided in one place.
 *
 * The weekly endpoint serves whatever `get_season_projections` returns. Today that
 * is the preseason season simulation (`season_simulator_mc`), or a flat-rate Monte
 * Carlo (`playing_time_gaussian_mc`) once the visitor changes the start week. Neither
 * knows about games already played, so neither is a forecast for that week
 * (decision 0011).
 *
 * ALLOW-LIST, fails closed: a row is a forecast only when its `interval_method` is
 * listed in FORECAST_INTERVAL_METHODS, meaning the number comes from the fitted
 * weekly model with causal (earlier-seasons-only) intervals. Anything else,
 * including a missing method, `unavailable` and any method this file has never
 * heard of, is NOT a forecast. A new forward producer must add its method to the
 * allow-list deliberately; until then its rows show the no-forecast notice.
 *
 * Checked against the backend on 2026-11-14:
 * - `causal_oof_conformal_90`: pinned stack, intervals from earlier seasons only
 *   (scripts/materialize_stack_projections.py).
 * - `mapie_enbpi`: EnbPI-lite on earlier seasons' residuals (ml/conformal.py).
 * - `posterior_samples` is NOT listed: it only says the row has Bayesian draws
 *   (services/projection.py), from the unpinned pipeline path, not causal intervals.
 */

import { STAT_LABELS } from '@/lib/constants'
import type { SearchablePlayer } from '@/context/SearchContext'
import type { SeasonPlayerProjection } from '@/types/api'

/** The only methods that mark a weekly forecast. */
export const FORECAST_INTERVAL_METHODS: ReadonlySet<string> = new Set([
  'causal_oof_conformal_90',
  'mapie_enbpi',
])

/** Methods known to be simulations: the season simulator and plain Monte Carlo. */
export const SIMULATION_INTERVAL_METHODS: ReadonlySet<string> = new Set([
  'season_simulator_mc',
  'playing_time_gaussian_mc',
  'scenario_monte_carlo',
])

/** The one simulation that is specifically the preseason run, never updated in season. */
const PRESEASON_SIMULATOR = 'season_simulator_mc'

export type Provenance = 'simulation' | 'forecast' | 'unknown'

export function classifyProvenance(intervalMethod: string | null | undefined): Provenance {
  const method = intervalMethod?.trim()
  if (!method) return 'unknown'
  if (FORECAST_INTERVAL_METHODS.has(method)) return 'forecast'
  if (SIMULATION_INTERVAL_METHODS.has(method)) return 'simulation'
  return 'unknown'
}

export function isForecastRow(row: { interval_method?: string | null }): boolean {
  return classifyProvenance(row.interval_method) === 'forecast'
}

/**
 * True when at least one row is not a verified forecast, so the page must say
 * so. False for an empty list (nothing to label) and when every row is a forecast.
 */
export function needsSimulationNotice(rows: ReadonlyArray<{ interval_method?: string | null }>): boolean {
  return rows.some((row) => !isForecastRow(row))
}

/** Label for the preseason simulator's numbers. */
export const SIMULATION_LABEL = 'Preseason simulation, not updated for games played'
/** Label for other Monte Carlo output. */
export const GENERIC_SIMULATION_LABEL = 'Simulation, not a weekly forecast'
/** Label when we cannot say how the number was made. */
export const NOT_A_FORECAST_LABEL = 'Not a verified weekly forecast'

/** The label to print on a number that is not a forecast, by how it was made. */
export function provenanceLabel(intervalMethod: string | null | undefined): string {
  const kind = classifyProvenance(intervalMethod)
  if (kind === 'unknown') return NOT_A_FORECAST_LABEL
  return intervalMethod?.trim() === PRESEASON_SIMULATOR ? SIMULATION_LABEL : GENERIC_SIMULATION_LABEL
}

/**
 * One label for a list of rows: the shared label when every non-forecast row has
 * the same one, otherwise the neutral not-a-forecast label.
 */
export function noticeLabel(rows: ReadonlyArray<{ interval_method?: string | null }>): string {
  const labels = new Set(rows.filter((r) => !isForecastRow(r)).map((r) => provenanceLabel(r.interval_method)))
  return labels.size === 1 ? [...labels][0] : NOT_A_FORECAST_LABEL
}

/** Label for a real weekly figure. Shared by the Weekly cards and the search dialog. */
export function weekFigureLabel(week: number, stat: string): string {
  return `Week ${week} · ${STAT_LABELS[stat] ?? stat}`
}

/**
 * Rows for the search dialog. Search opens the player page, which shows a week
 * forecast or says there is none, so search shows the same: the weekly mean for a
 * real forecast row, and `null` (no number) for a simulation row.
 */
export function toSearchablePlayers(rows: ReadonlyArray<SeasonPlayerProjection>, stat: keyof SeasonPlayerProjection & string): SearchablePlayer[] {
  return rows.map((p) => {
    const figure = p[stat as 'fantasy_ppr']
    return {
      player_id: p.player_id,
      player_name: p.player_name,
      position: p.position,
      team: p.team,
      stat,
      projection: isForecastRow(p) && figure ? figure.mean : null,
    }
  })
}
