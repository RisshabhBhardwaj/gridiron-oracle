import { useQuery } from '@tanstack/react-query'
import { apiClient } from '../lib/api-client'
import type { SeasonTeamWinsResponse } from '../types/api'

interface Params {
  season: number
  startWeek?: number
  /** Hold the request until the caller has a real season (see useCurrentSeason). */
  enabled?: boolean
}

export function useTeamWins(params: Params) {
  const { season, startWeek = 1, enabled = true } = params

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
    enabled: enabled && Boolean(season),
  })
}
