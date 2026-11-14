import { useQuery } from '@tanstack/react-query'
import { apiClient, isNotReady } from '@/lib/api-client'
import type { ExplainResponse } from '@/types/api'

interface UseExplainParams {
  player_id: string
  week: number
  season: number
  stat: string
  top_n?: number
  /** Hold the request until the caller knows it is wanted. Defaults to true. */
  enabled?: boolean
}

export function useExplain({ player_id, week, season, stat, top_n = 10, enabled = true }: UseExplainParams) {
  return useQuery<ExplainResponse>({
    queryKey: ['explain', player_id, week, season, stat],
    queryFn: () => {
      const path = `/explain/${encodeURIComponent(player_id)}?week=${week}&season=${season}&stat=${encodeURIComponent(stat)}&top_n=${top_n}`
      return apiClient.get<ExplainResponse>(path)
    },
    enabled: enabled && player_id.length > 0,
    // A 404 means "no breakdown published": retrying cannot change it. Anything else gets one retry.
    retry: (failureCount, error) => !isNotReady(error) && failureCount < 1,
    staleTime: 5 * 60 * 1000,
  })
}
