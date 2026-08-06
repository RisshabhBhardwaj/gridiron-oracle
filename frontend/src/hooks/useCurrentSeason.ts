import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type { CurrentSeasonResponse } from '@/types/api'
import { CURRENT_SEASON, CURRENT_WEEK } from '@/lib/constants'

/**
 * Fetches the most recent (season, week) from /season/current.
 *
 * staleTime: 1 hour — the current week changes at most once per day.
 * placeholderData: offline fallback so the UI is never blank.
 */
export function useCurrentSeason() {
  return useQuery<CurrentSeasonResponse>({
    queryKey: ['season', 'current'],
    queryFn: () => apiClient.get<CurrentSeasonResponse>('/season/current'),
    staleTime: 60 * 60 * 1000,
    placeholderData: { season: CURRENT_SEASON, week: CURRENT_WEEK },
  })
}
