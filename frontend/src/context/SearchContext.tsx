/**
 * SearchContext — global state for the command-palette player search overlay.
 */
import { createContext, useCallback, useContext, useState, useEffect, type ReactNode } from 'react'
import type { WeekPlayerProjection } from '@/types/api'

interface SearchCtx {
  isOpen: boolean
  open: () => void
  close: () => void
  toggle: () => void
  query: string
  setQuery: (q: string) => void
  /** All players loaded from latest projections — set by Dashboard on load */
  players: WeekPlayerProjection[]
  setPlayers: (p: WeekPlayerProjection[]) => void
  /** Filtered subset matching query */
  results: WeekPlayerProjection[]
  defaultWeek: number
  defaultSeason: number
  defaultStat: string
  setDefaults: (defaults: { week: number; season: number; stat: string }) => void
}

const SearchContext = createContext<SearchCtx | null>(null)

export function SearchProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen]   = useState(false)
  const [query, setQuery]     = useState('')
  const [players, setPlayers] = useState<WeekPlayerProjection[]>([])
  const [defaults, setDefaultsState] = useState({ week: 1, season: 2025, stat: 'receiving_yards' })

  const setDefaults = useCallback((next: { week: number; season: number; stat: string }) => {
    setDefaultsState((prev) => (
      prev.week === next.week && prev.season === next.season && prev.stat === next.stat
        ? prev
        : next
    ))
  }, [])

  // Cmd+K / Ctrl+K shortcut
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setIsOpen((prev) => !prev)
      }
      if (e.key === 'Escape') setIsOpen(false)
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const q = query.trim().toLowerCase()
  const results = q.length < 1
    ? players.slice(0, 20)
    : [...players]
        .map((player) => {
          const fullName = player.player_name.toLowerCase()
          const nameParts = fullName.split(/\s+/).filter(Boolean)
          const first = nameParts[0] ?? ''
          const last = nameParts[nameParts.length - 1] ?? ''
          const team = (player.team ?? '').toLowerCase()
          const position = player.position.toLowerCase()

          let score = -1
          if (first.startsWith(q) || last.startsWith(q)) score = 5
          else if (nameParts.some((part) => part.startsWith(q))) score = 4
          else if (fullName.startsWith(q)) score = 3
          else if (fullName.includes(q)) score = 2
          else if (team.startsWith(q) || position.startsWith(q)) score = 1
          else if (team.includes(q) || position.includes(q)) score = 0

          return { player, score }
        })
        .filter((entry) => entry.score >= 0)
        .sort((a, b) => {
          if (a.score !== b.score) return b.score - a.score
          return b.player.projection - a.player.projection
        })
        .map((entry) => entry.player)
        .slice(0, 20)

  const ctx: SearchCtx = {
    isOpen,
    open:       () => setIsOpen(true),
    close:      () => { setIsOpen(false); setQuery('') },
    toggle:     () => setIsOpen((p) => !p),
    query,
    setQuery,
    players,
    setPlayers,
    results,
    defaultWeek: defaults.week,
    defaultSeason: defaults.season,
    defaultStat: defaults.stat,
    setDefaults,
  }

  return <SearchContext.Provider value={ctx}>{children}</SearchContext.Provider>
}

export function useSearch(): SearchCtx {
  const ctx = useContext(SearchContext)
  if (!ctx) throw new Error('useSearch must be used inside SearchProvider')
  return ctx
}
