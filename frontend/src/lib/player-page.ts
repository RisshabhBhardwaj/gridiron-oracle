/**
 * Rules for the player page, kept out of the component so each one can be tested alone.
 *
 * The page shows one trusted number for the week being viewed, or says there is none.
 * "Trusted" is the same rule the Weekly tab and search use (lib/forecast-provenance.ts):
 * only an allow-listed forecast method counts. A simulation row never becomes a Week N
 * figure on this page.
 */

import { ApiError, isNotReady } from '@/lib/api-client'
import { isForecastRow } from '@/lib/forecast-provenance'
import type { PredictResponse, StatProjection } from '@/types/api'

/** The weeks /predict accepts. */
const FIRST_WEEK = 1
const LAST_WEEK = 22

/** Week from the address, or the fallback when it is missing or not a week. */
export function parseWeek(raw: string | null, fallback: number): number {
  if (raw === null || raw.trim() === '') return fallback
  const n = Number(raw)
  return Number.isInteger(n) && n >= FIRST_WEEK && n <= LAST_WEEK ? n : fallback
}

export type WeekFigure =
  | { kind: 'forecast'; value: number; low: number | null; high: number | null }
  | { kind: 'none' }

function finite(n: number | null | undefined): n is number {
  return typeof n === 'number' && Number.isFinite(n)
}

/**
 * What the page may show as the Week N figure for a /predict answer.
 *
 * The number is the model's point estimate (the expected value). `percentiles.p50`
 * is that same number: the API fills it from `projection` (backend/app/api/predict.py),
 * it is not a separate median of a distribution. Search and the Weekly cards print the
 * mean, so all three show the same quantity. The page labels it "projection", never
 * "median" or "typical".
 *
 * Anything that is not an allow-listed forecast row, including a missing answer, gives
 * `none`.
 */
export function weekFigure(predict: PredictResponse | null | undefined): WeekFigure {
  if (!predict || !isForecastRow({ interval_method: predict.interval_method })) return { kind: 'none' }
  const stated = predict.projection?.[predict.stat as keyof StatProjection]
  const value = finite(stated) ? stated : predict.percentiles?.p50
  if (!finite(value)) return { kind: 'none' }
  const low = predict.percentiles?.p10
  const high = predict.percentiles?.p90
  return { kind: 'forecast', value, low: finite(low) ? low : null, high: finite(high) ? high : null }
}

/** Fantasy points repeat the hero when the stat is already fantasy points, or the two match. */
export function repeatsHeroFigure(stat: string, heroValue: number, fantasyPoints: number | null | undefined): boolean {
  if (!finite(fantasyPoints)) return false
  if (stat === 'fantasy_ppr' || stat === 'ppr_points') return true
  return fantasyPoints.toFixed(1) === heroValue.toFixed(1)
}

export interface SeasonPoint {
  season: number
  week: number
}

/** True for the week the sportsbooks are pricing now: the current week of the current season. */
export function isLiveLineWeek(viewed: SeasonPoint, current: SeasonPoint): boolean {
  return viewed.season === current.season && viewed.week === current.week
}

/** True for the current week or a later one; false for a week that has been played. */
export function isCurrentOrLater(viewed: SeasonPoint, current: SeasonPoint): boolean {
  return viewed.season > current.season || (viewed.season === current.season && viewed.week >= current.week)
}

/**
 * Why the "what drives this" panel has nothing to show:
 * - `not-published`: the server says there is no breakdown for this stat (404). Retrying will not help.
 * - `failed`: anything else (5xx, network, proxy). A real failure, so the panel offers Retry.
 */
export function explainFailure(error: unknown): 'not-published' | 'failed' {
  return isNotReady(error) ? 'not-published' : 'failed'
}

/**
 * True for /predict's honest "no forecast for this week yet" answer: a 404 that says
 * `forecast_available: false`. Any other failure (including a 404 from a missing route)
 * is a real error and gets the error state with Retry.
 */
export function isNoForecastError(error: unknown): boolean {
  return error instanceof ApiError && isNotReady(error) && error.structuredDetail?.forecast_available === false
}
