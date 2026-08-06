import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type { PredictResponse } from '@/types/api'

interface UsePredictParams {
  player: string
  week: number
  season: number
  stat: string
  enabled?: boolean
}

export function usePredict({ player, week, season, stat, enabled = true }: UsePredictParams) {
  return useQuery<PredictResponse>({
    queryKey: ['predict', player, week, season, stat],
    queryFn: () => {
      const path = `/predict?player=${encodeURIComponent(player)}&week=${week}&season=${season}&stat=${encodeURIComponent(stat)}`
      return apiClient.get<PredictResponse>(path)
    },
    enabled: enabled && player.length > 0,
    staleTime: 5 * 60 * 1000,
  })
}
