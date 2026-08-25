/**
 * Percentage edge of our projection over a sportsbook's posted line.
 *
 * Previously lived in context/BetSlipContext.tsx. The bet-slip / parlay
 * builder was removed, but this is a pure helper with nothing to do with the
 * slip — the player and projection surfaces still show edge vs. the book line.
 */
export function calcEdge(projection: number, bookLine: number): number {
  if (bookLine === 0) return 0
  return ((projection - bookLine) / Math.abs(bookLine)) * 100
}
