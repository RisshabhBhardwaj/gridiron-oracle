/**
 * Display formatting utilities for the Gridiron Oracle frontend.
 */

export function formatYards(n: number): string {
  return `${n.toFixed(1)} yds`
}

export function formatPct(n: number): string {
  return `${(n * 100).toFixed(1)}%`
}

export function formatPctWhole(n: number): string {
  return `${Math.round(n)}%`
}

export function formatDelta(n: number): string {
  const sign = n >= 0 ? '+' : ''
  return `${sign}${n.toFixed(1)}`
}

export function formatDate(iso: string): string {
  const date = new Date(iso)
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

export function hoursAgo(iso: string): number {
  const now = Date.now()
  const then = new Date(iso).getTime()
  return (now - then) / (1000 * 60 * 60)
}

export function isStale(iso: string): boolean {
  return hoursAgo(iso) > 24
}
