import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type { HealthResponse } from '@/types/api'

export type ApiStatus = 'checking' | 'live' | 'unavailable'

/** How often the nav status re-checks while the tab is open and visible. */
export const HEALTH_POLL_MS = 5 * 60 * 1000

/**
 * Whether the data service is reachable and serving published results.
 *
 * "Live" means: /health answered AND the served model artifacts are ready
 * (`artifact_mode_ready` is not false). The report's own `status` is ignored on
 * purpose: "degraded" is the normal live state (optional tracking is not deployed)
 * and says nothing about whether visitors can get data.
 *
 * Polls every few minutes and stops while the tab is hidden. No retries: one
 * failed check already means "unavailable"; the next poll corrects it.
 */
export function useApiStatus(): ApiStatus {
  const { data, isError, isPending } = useQuery<HealthResponse>({
    queryKey: ['health'],
    queryFn: () => apiClient.get<HealthResponse>('/health'),
    refetchInterval: HEALTH_POLL_MS,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: true,
    staleTime: HEALTH_POLL_MS / 2,
    retry: false,
  })
  if (isError) return 'unavailable'
  if (isPending || !data) return 'checking'
  return data.artifact_mode_ready === false ? 'unavailable' : 'live'
}
