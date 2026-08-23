import { useQuery } from '@tanstack/react-query'
import { apiClient } from '../lib/api-client'
import type { GameDriveSimResponse } from '../types/api'

export function useGameDriveSim(gameId: string | undefined, seed?: number) {
  return useQuery({
    queryKey: ['gameDriveSim', gameId, seed],
    queryFn: async () => {
      if (!gameId) throw new Error('Game ID is required')
      const url = seed != null
        ? `/games/${encodeURIComponent(gameId)}/drive-sim?seed=${seed}`
        : `/games/${encodeURIComponent(gameId)}/drive-sim`
      return apiClient.get<GameDriveSimResponse>(url)
    },
    enabled: Boolean(gameId),
    staleTime: 1000 * 60 * 5,
  })
}
