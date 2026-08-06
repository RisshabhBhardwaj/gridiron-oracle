import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type { SettingsResponse, SettingsUpdateRequest } from '@/types/api'

const SETTINGS_KEY = ['settings']

export function useSettings() {
  return useQuery<SettingsResponse>({
    queryKey: SETTINGS_KEY,
    queryFn: () => apiClient.get<SettingsResponse>('/settings'),
    staleTime: Infinity,
  })
}

export function useUpdateSettings() {
  const queryClient = useQueryClient()

  return useMutation<SettingsResponse, Error, SettingsUpdateRequest>({
    mutationFn: (req: SettingsUpdateRequest) =>
      apiClient.put<SettingsResponse>('/settings', req),
    onSuccess: (data) => {
      queryClient.setQueryData(SETTINGS_KEY, data)
    },
  })
}
