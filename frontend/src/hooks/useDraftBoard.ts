import { useQuery } from '@tanstack/react-query'
import { apiClient } from '../lib/api-client'
import type { DraftBoardPlayer, DraftBoardResponse } from '../types/api'

export type { DraftBoardPlayer, DraftBoardResponse }

interface Params {
  season: number
  source?: string | null
  scoring?: string
  position?: string | null
}

async function fetchDraftBoard(params: Params): Promise<DraftBoardResponse> {
  const qs = new URLSearchParams()
  qs.set('season', String(params.season))
  if (params.source) qs.set('source', params.source)
  qs.set('scoring', params.scoring ?? 'ppr')
  if (params.position) qs.set('position', params.position)
  return apiClient.get<DraftBoardResponse>(`/draft/board?${qs.toString()}`)
}

export function useDraftBoard(params: Params) {
  return useQuery({
    queryKey: ['draftBoard', params.season, params.source, params.scoring, params.position],
    queryFn: () => fetchDraftBoard(params),
    staleTime: 1000 * 60 * 5,
    retry: 1,
  })
}
