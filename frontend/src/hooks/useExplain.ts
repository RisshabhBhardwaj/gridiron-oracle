import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type { ExplainResponse } from '@/types/api'

interface UseExplainParams {
  player_id: string
  week: number
  season: number
  stat: string
  top_n?: number
}

export function useExplain({ player_id, week, season, stat, top_n = 10 }: UseExplainParams) {
  return useQuery<ExplainResponse>({
    queryKey: ['explain', player_id, week, season, stat],
    queryFn: () => {
      const path = `/explain/${encodeURIComponent(player_id)}?week=${week}&season=${season}&stat=${encodeURIComponent(stat)}&top_n=${top_n}`
      return apiClient.get<ExplainResponse>(path)
    },
    enabled: player_id.length > 0,
    staleTime: 5 * 60 * 1000,
  })
}
