import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type { WeekProjectionsResponse } from '@/types/api'

interface UseWeekProjectionsParams {
  week: number
  season: number
  stat: string
  positions: string[]
}

export function useWeekProjections({
  week,
  season,
  stat,
  positions,
}: UseWeekProjectionsParams) {
  return useQuery<WeekProjectionsResponse>({
    queryKey: ['week-projections', week, season, stat, positions],
    queryFn: () => {
      const posParams = positions.map((p) => `positions=${encodeURIComponent(p)}`).join('&')
      const path = `/projections/week/${week}?season=${season}&stat=${encodeURIComponent(stat)}&${posParams}`
      return apiClient.get<WeekProjectionsResponse>(path)
    },
    staleTime: 5 * 60 * 1000,
  })
}
