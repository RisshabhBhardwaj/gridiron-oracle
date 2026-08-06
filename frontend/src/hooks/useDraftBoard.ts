import { useQuery } from '@tanstack/react-query'

export interface DraftBoardPlayer {
  player_name: string
  position: string | null
  team: string | null
  adp: number
  player_id: string | null
  source: string
  model_rank: number | null
  model_fantasy_ppr: number | null
  adp_rank: number | null
  value_vs_adp: number | null
}

export interface DraftBoardResponse {
  season: number
  source: string
  scoring: string
  count: number
  players: DraftBoardPlayer[]
  spearman_rho: number | null
  note: string
}

interface Params {
  season: number
  source?: string
  scoring?: string
  position?: string | null
}

async function fetchDraftBoard(params: Params): Promise<DraftBoardResponse> {
  const qs = new URLSearchParams()
  qs.set('season', String(params.season))
  qs.set('source', params.source ?? 'historical')
  qs.set('scoring', params.scoring ?? 'ppr')
  if (params.position) qs.set('position', params.position)
  const res = await fetch(`/api/draft/board?${qs.toString()}`)
  if (!res.ok) {
    const body = await res.text()
    throw new Error(body || res.statusText)
  }
  return res.json()
}

export function useDraftBoard(params: Params) {
  return useQuery({
    queryKey: ['draftBoard', params.season, params.source, params.scoring, params.position],
    queryFn: () => fetchDraftBoard(params),
    staleTime: 1000 * 60 * 5,
    retry: 1,
  })
}
