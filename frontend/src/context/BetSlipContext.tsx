/**
 * BetSlipContext — global state for the parlay / bet slip.
 * Persisted to sessionStorage so refreshing doesn't lose the slip.
 */
import { createContext, useContext, useReducer, useEffect, type ReactNode } from 'react'

export interface BetSlipLeg {
  id: string               // unique: `${player_id}-${stat}-${week}`
  player_id: string
  player_name: string
  position: string
  team: string | null
  stat: string
  stat_label: string
  week: number
  season: number
  our_projection: number   // our model's median (p50)
  floor: number            // p10
  ceiling: number          // p90
  book_line: number | null // sportsbook line (null if no odds loaded)
  boom_probability: number | null
  bust_probability: number | null
  fantasy_projection: number | null
}

interface BetSlipState {
  legs: BetSlipLeg[]
  isOpen: boolean
}

type BetSlipAction =
  | { type: 'ADD'; leg: BetSlipLeg }
  | { type: 'REMOVE'; id: string }
  | { type: 'CLEAR' }
  | { type: 'TOGGLE_OPEN' }
  | { type: 'SET_OPEN'; open: boolean }
  | { type: 'SET_BOOK_LINE'; id: string; line: number }

function reducer(state: BetSlipState, action: BetSlipAction): BetSlipState {
  switch (action.type) {
    case 'ADD':
      if (state.legs.find((l) => l.id === action.leg.id)) return state
      return { ...state, legs: [...state.legs, action.leg], isOpen: true }
    case 'REMOVE':
      return { ...state, legs: state.legs.filter((l) => l.id !== action.id) }
    case 'CLEAR':
      return { ...state, legs: [] }
    case 'TOGGLE_OPEN':
      return { ...state, isOpen: !state.isOpen }
    case 'SET_OPEN':
      return { ...state, isOpen: action.open }
    case 'SET_BOOK_LINE':
      return {
        ...state,
        legs: state.legs.map((l) =>
          l.id === action.id ? { ...l, book_line: action.line } : l,
        ),
      }
    default:
      return state
  }
}

const STORAGE_KEY = 'gridiron_bet_slip'

function loadFromStorage(): BetSlipState {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw) as { legs: BetSlipLeg[] }
      return { legs: parsed.legs ?? [], isOpen: false }
    }
  } catch {}
  return { legs: [], isOpen: false }
}

interface BetSlipCtx {
  state: BetSlipState
  addLeg: (leg: BetSlipLeg) => void
  removeLeg: (id: string) => void
  clearSlip: () => void
  toggleOpen: () => void
  setOpen: (open: boolean) => void
  hasLeg: (id: string) => boolean
  setBookLine: (id: string, line: number) => void
}

const BetSlipContext = createContext<BetSlipCtx | null>(null)

export function BetSlipProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, undefined, loadFromStorage)

  // Persist legs to sessionStorage
  useEffect(() => {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ legs: state.legs }))
  }, [state.legs])

  const ctx: BetSlipCtx = {
    state,
    addLeg:     (leg)  => dispatch({ type: 'ADD', leg }),
    removeLeg:  (id)   => dispatch({ type: 'REMOVE', id }),
    clearSlip:  ()     => dispatch({ type: 'CLEAR' }),
    toggleOpen: ()     => dispatch({ type: 'TOGGLE_OPEN' }),
    setOpen:    (open) => dispatch({ type: 'SET_OPEN', open }),
    hasLeg:     (id)   => state.legs.some((l) => l.id === id),
    setBookLine:(id, line) => dispatch({ type: 'SET_BOOK_LINE', id, line }),
  }

  return <BetSlipContext.Provider value={ctx}>{children}</BetSlipContext.Provider>
}

export function useBetSlip(): BetSlipCtx {
  const ctx = useContext(BetSlipContext)
  if (!ctx) throw new Error('useBetSlip must be used inside BetSlipProvider')
  return ctx
}

// ── Parlay math ─────────────────────────────────────────────────────────────

/**
 * Convert American odds to decimal multiplier.
 * e.g. -110 → 1.909,  +150 → 2.5
 */
export function americanToDecimal(american: number): number {
  if (american > 0) return 1 + american / 100
  return 1 + 100 / Math.abs(american)
}

/** Convert decimal multiplier to American odds display string. */
export function decimalToAmerican(decimal: number): string {
  if (decimal >= 2) return `+${Math.round((decimal - 1) * 100)}`
  const american = -Math.round(100 / (decimal - 1))
  return `${american}`
}

/** Calculate combined parlay decimal odds from individual legs at standard -110. */
export function parlayOdds(nLegs: number): number {
  // Standard -110 = 1.909 per leg
  return Math.pow(1.909, nLegs)
}

/** Edge = (our projection - book line) / book line * 100 */
export function calcEdge(projection: number, bookLine: number): number {
  if (bookLine === 0) return 0
  return ((projection - bookLine) / Math.abs(bookLine)) * 100
}
