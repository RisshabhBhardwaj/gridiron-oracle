import { useQuery } from '@tanstack/react-query'
import { apiClient } from '../lib/api-client'
import type { SeasonWeekProjectionsResponse } from '../types/api'

interface Params {
  season: number
  week: number
  startWeek?: number
  positions?: string[]
  /** Hold the request until the caller has a real season (see useCurrentSeason). */
  enabled?: boolean
}

export function useSeasonWeekProjections(params: Params) {
  const { season, week, startWeek = 1, positions, enabled = true } = params

  return useQuery({
    queryKey: ['seasonWeekProjections', season, startWeek, week, positions],
    queryFn: async () => {
      const qs = new URLSearchParams()
      qs.set('start_week', String(startWeek))
      if (positions && positions.length > 0) {
        positions.forEach((pos) => qs.append('positions', pos))
      }
      return apiClient.get<SeasonWeekProjectionsResponse>(
        `/projections/season/${season}/weeks/${week}?${qs.toString()}`,
      )
    },
    staleTime: 1000 * 60 * 5,
    enabled: enabled && Boolean(season && week),
  })
}
