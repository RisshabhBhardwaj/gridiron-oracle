import { useQuery, useMutation } from '@tanstack/react-query'
import { apiClient } from '../lib/api-client'
import type { MockDraftProfilesResponse, PickRequest, PickResponse } from '../types/api'

export function useMockDraftProfiles(season: number = 2026) {
  return useQuery({
    queryKey: ['mockDraftProfiles', season],
    queryFn: async () => {
      return apiClient.get<MockDraftProfilesResponse>(`/mock-draft/profiles?season=${season}`)
    },
    staleTime: 1000 * 60 * 10,
    enabled: Boolean(season),
  })
}

export function useMockDraftPick() {
  return useMutation({
    mutationFn: async (req: PickRequest) => {
      return apiClient.post<PickResponse>('/mock-draft/pick', req)
    },
  })
}
