import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type { BacktestResponse } from '@/types/api'

/**
 * Evidence from the served release. One request per stat returns the whole
 * release verdict, the rows for that stat (all positions) and the list of
 * every stat the release measures, so both pages filter positions locally.
 */
export function useBacktest({ stat }: { stat: string }) {
  return useQuery<BacktestResponse>({
    queryKey: ['backtest', stat],
    queryFn: () => apiClient.get<BacktestResponse>(`/backtest?stat=${encodeURIComponent(stat)}`),
    staleTime: 60 * 60 * 1000,
  })
}
