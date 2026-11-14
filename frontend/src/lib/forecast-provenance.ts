/**
 * Where a weekly number came from, decided in one place.
 *
 * The weekly endpoint (`/projections/season/{season}/weeks/{week}`) today serves
 * the preseason season simulation read out at one week. It ignores games already
 * played, so it must never be read as a forecast for that week (decision 0011).
 * When a real forward forecast is served, its rows carry a different
 * `interval_method` and every banner and label built on this helper disappears
 * with no UI change.
 *
 * Rule: a row is a forecast unless its method is a known simulation or is
 * missing. A missing method fails closed (not a forecast).
 */

import { STAT_LABELS } from '@/lib/constants'
import type { SearchablePlayer } from '@/context/SearchContext'
import type { SeasonPlayerProjection } from '@/types/api'

/** `interval_method` values that mean "preseason simulation, not updated for games played". */
export const SIMULATION_INTERVAL_METHODS: ReadonlySet<string> = new Set(['season_simulator_mc'])

export type Provenance = 'simulation' | 'forecast' | 'unknown'

export function classifyProvenance(intervalMethod: string | null | undefined): Provenance {
  const method = intervalMethod?.trim()
  if (!method) return 'unknown'
  return SIMULATION_INTERVAL_METHODS.has(method) ? 'simulation' : 'forecast'
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

/** The label shown on every simulated number, wherever it appears. */
export const SIMULATION_LABEL = 'Preseason simulation, not updated for games played'

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
