/**
 * One name per page. The nav, the page heading, the browser tab title, Home's
 * cards and every link to the page read from here, so a page cannot be called
 * "Backtesting" in one place and "Backtest" in another.
 */
export const SITE_NAME = 'Gridiron Oracle'

export const PAGE_NAMES = {
  home: 'Home',
  projections: 'Current Season',
  season: 'Season Review',
  draft: 'Draft Board',
  backtest: 'Backtest',
  settings: 'Settings',
} as const

/** "Draft Board · Gridiron Oracle". With no page name, just the site name. */
export function pageTitle(name?: string | null): string {
  const trimmed = name?.trim()
  return trimmed ? `${trimmed} · ${SITE_NAME}` : SITE_NAME
}
