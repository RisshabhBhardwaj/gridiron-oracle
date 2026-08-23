import { useQuery } from '@tanstack/react-query'
import { apiClient } from '../lib/api-client'
import type { PairedGame, TeamGamePrediction, TeamGameWeekResponse } from '../types/api'

interface Params {
  season: number
  week: number
}

export function useTeamGames(params: Params) {
  const { season, week } = params

  return useQuery({
    queryKey: ['teamGames', season, week],
    queryFn: async () => {
      return apiClient.get<TeamGameWeekResponse>(`/team-games/${season}/${week}`)
    },
    select: (data): PairedGame[] => {
      const byGame: Record<string, { home?: TeamGamePrediction; away?: TeamGamePrediction }> = {}
      for (const row of data.games) {
        if (!byGame[row.game_id]) {
          byGame[row.game_id] = {}
        }
        if (row.is_home) {
          byGame[row.game_id].home = row
        } else {
          byGame[row.game_id].away = row
        }
      }

      const pairs: PairedGame[] = []
      for (const [gameId, { home, away }] of Object.entries(byGame)) {
        if (home && away) {
          pairs.push({
            game_id: gameId,
            season: home.season,
            week: home.week,
            home_team: home.team,
            away_team: away.team,
            home_points: home.points,
            away_points: away.points,
            home_yards: home.yards,
            away_yards: away.yards,
            home_pass_rate: home.pass_rate,
            away_pass_rate: away.pass_rate,
            home_win_probability: home.win_probability,
            away_win_probability: away.win_probability,
            model_run_id: home.model_run_id,
            note: home.note,
          })
        }
      }
      return pairs
    },
    staleTime: 1000 * 60 * 5,
    enabled: Boolean(season && week),
  })
}
