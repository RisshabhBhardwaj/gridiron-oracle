import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type { BacktestResponse } from '@/types/api'

interface UseBacktestParams {
  stat: string
  positions: string[]
}

export function useBacktest({ stat, positions }: UseBacktestParams) {
  return useQuery<BacktestResponse>({
    queryKey: ['backtest', stat, positions],
    queryFn: () => {
      const posParams = positions.map((p) => `positions=${encodeURIComponent(p)}`).join('&')
      const path = `/backtest?stat=${encodeURIComponent(stat)}&${posParams}`
      return apiClient.get<BacktestResponse>(path)
    },
    staleTime: 60 * 60 * 1000,
  })
}
