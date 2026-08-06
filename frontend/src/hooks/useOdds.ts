/**
 * useOdds — fetches NFL player prop odds from The Odds API.
 * Uses a client-side fetch to api.the-odds-api.com when an API key is configured.
 * Falls back to our mock data when key is absent (dev mode).
 *
 * The Odds API free tier: 500 req/month — we cache aggressively.
 */
import { useQuery } from '@tanstack/react-query'

export interface PlayerPropOdd {
  player_name: string
  stat_type: string    // e.g. "player_receptions", "player_rushing_yards"
  line: number         // the over/under line
  over_price: number   // American odds for over (e.g. -110)
  under_price: number  // American odds for under
  bookmaker: string
}

export interface OddsResponse {
  player_props: PlayerPropOdd[]
  last_updated: string
  from_cache: boolean
  source: 'mock' | 'live'
  provider: string
}

// Maps our internal stat keys to The Odds API stat_type values
export const STAT_TO_ODDS_TYPE: Record<string, string> = {
  receiving_yards:  'player_reception_yards',
  rushing_yards:    'player_rushing_yards',
  passing_yards:    'player_passing_yards',
  receptions:       'player_receptions',
  passing_tds:      'player_passing_tds',
  rushing_tds:      'player_rushing_tds',
  receiving_tds:    'player_receiving_tds',
}

/** Mock prop lines for dev (no API key needed) */
function generateMockOdds(playerNames: string[], stat: string): PlayerPropOdd[] {
  const lines: Record<string, number> = {
    receiving_yards: 65,
    rushing_yards:   75,
    passing_yards:   245,
    receptions:      5.5,
    passing_tds:     1.5,
    rushing_tds:     0.5,
    receiving_tds:   0.5,
  }
  const baseLine = lines[stat] ?? 50

  return playerNames.slice(0, 20).map((name) => {
    // seeded random variation per player
    const seed = name.split('').reduce((a, c) => a + c.charCodeAt(0), 0)
    const variation = ((seed % 40) - 20) / 100 // ±20%
    const line = Math.round((baseLine * (1 + variation)) * 2) / 2 // round to 0.5

    return {
      player_name:  name,
      stat_type:    STAT_TO_ODDS_TYPE[stat] ?? 'player_receptions',
      line,
      over_price:   -110,
      under_price:  -110,
      bookmaker:    'DraftKings',
    }
  })
}

interface UseOddsOptions {
  playerNames: string[]
  stat: string
  enabled?: boolean
}

export function useOdds({ playerNames, stat, enabled = true }: UseOddsOptions) {
  return useQuery<OddsResponse>({
    queryKey: ['odds', stat, playerNames.slice(0, 5).join(',')],
    queryFn: async (): Promise<OddsResponse> => {
      // Check if user has configured an API key
      const apiKey = import.meta.env.VITE_ODDS_API_KEY as string | undefined

      if (!apiKey || playerNames.length === 0) {
        // Return mock data — useful for dev and demos
        await new Promise((r) => setTimeout(r, 200)) // simulate latency
        return {
          player_props: generateMockOdds(playerNames, stat),
          last_updated: new Date().toISOString(),
          from_cache: false,
          source: 'mock',
          provider: 'Local mock lines',
        }
      }

      // Real API call — key present
      const sport = 'americanfootball_nfl'
      const markets = STAT_TO_ODDS_TYPE[stat] ?? 'player_receptions'
      const url = `https://api.the-odds-api.com/v4/sports/${sport}/odds/?apiKey=${apiKey}&markets=${markets}&regions=us&oddsFormat=american`

      const res = await fetch(url)
      if (!res.ok) throw new Error(`Odds API error: ${res.status}`)

      // Parse and flatten
      const raw = (await res.json()) as Array<{
        bookmakers: Array<{
          key: string
          markets: Array<{
            key: string
            outcomes: Array<{ name: string; price: number; point: number; description: string }>
          }>
        }>
      }>

      const props: PlayerPropOdd[] = []
      for (const game of raw) {
        for (const book of game.bookmakers) {
          for (const market of book.markets) {
            if (market.key !== markets) continue
            const overOutcome  = market.outcomes.find((o) => o.name === 'Over')
            const underOutcome = market.outcomes.find((o) => o.name === 'Under')
            if (!overOutcome || !underOutcome) continue
            props.push({
              player_name:  overOutcome.description,
              stat_type:    market.key,
              line:         overOutcome.point,
              over_price:   overOutcome.price,
              under_price:  underOutcome.price,
              bookmaker:    book.key,
            })
          }
        }
      }

      return {
        player_props: props,
        last_updated: new Date().toISOString(),
        from_cache: false,
        source: 'live',
        provider: 'The Odds API',
      }
    },
    enabled: enabled && playerNames.length > 0,
    staleTime: 5 * 60 * 1000, // 5 minutes — respect free-tier limits
    retry: 1,
  })
}

/** Get the book line for a specific player from odds data. */
export function findPlayerLine(
  props: PlayerPropOdd[],
  playerName: string,
): number | null {
  // Try exact match first, then fuzzy (last name)
  const exact = props.find((p) => p.player_name.toLowerCase() === playerName.toLowerCase())
  if (exact) return exact.line

  const lastName = playerName.split(' ').pop()?.toLowerCase() ?? ''
  const fuzzy = props.find((p) => p.player_name.toLowerCase().includes(lastName))
  return fuzzy?.line ?? null
}
