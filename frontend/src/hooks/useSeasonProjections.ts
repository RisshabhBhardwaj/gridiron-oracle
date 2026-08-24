import { useQuery } from '@tanstack/react-query'
import type { SeasonProjectionsResponse } from '@/types/api'

interface UseSeasonProjectionsParams {
  season: number
  startWeek: number
  positions?: string[]
  /** Hold the request until the caller has a real season (see useCurrentSeason). */
  enabled?: boolean
}

async function fetchSeasonProjections({
  season,
  startWeek,
  positions = ['WR', 'RB', 'TE', 'QB'],
}: UseSeasonProjectionsParams): Promise<SeasonProjectionsResponse> {
  const params = new URLSearchParams()
  params.append('start_week', startWeek.toString())
  positions.forEach((p) => params.append('positions', p))

  const url = `/api/projections/season/${season}?${params.toString()}`
  
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`Failed to fetch season projections: ${res.statusText}`)
  }
  return res.json()
}

export function useSeasonProjections(params: UseSeasonProjectionsParams) {
  return useQuery({
    queryKey: ['seasonProjections', params.season, params.startWeek, params.positions],
    queryFn: () => fetchSeasonProjections(params),
    staleTime: 1000 * 60 * 5, // 5 minutes
    retry: 1,
    enabled: params.enabled ?? true,
  })
}
