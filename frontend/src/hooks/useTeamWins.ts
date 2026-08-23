import { useQuery } from '@tanstack/react-query'
import { apiClient } from '../lib/api-client'
import type { SeasonTeamWinsResponse } from '../types/api'

interface Params {
  season: number
  startWeek?: number
}

export function useTeamWins(params: Params) {
  const { season, startWeek = 1 } = params

  return useQuery({
    queryKey: ['teamWins', season, startWeek],
    queryFn: async () => {
      const qs = new URLSearchParams()
      qs.set('start_week', String(startWeek))
      return apiClient.get<SeasonTeamWinsResponse>(
        `/projections/season/${season}/team-wins?${qs.toString()}`,
      )
    },
    staleTime: 1000 * 60 * 5,
    enabled: Boolean(season),
  })
}
