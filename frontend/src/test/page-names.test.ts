import { describe, expect, it } from 'vitest'
import { PAGE_NAMES, SITE_NAME, pageTitle } from '../lib/page-names'

describe('pageTitle', () => {
  it('puts the page name first, then the site name', () => {
    expect(pageTitle(PAGE_NAMES.draft)).toBe('Draft Board · Gridiron Oracle')
    expect(pageTitle(PAGE_NAMES.backtest)).toBe('Backtest · Gridiron Oracle')
  })
  it('is the bare site name with no page name', () => {
    expect(pageTitle()).toBe(SITE_NAME)
    expect(pageTitle('')).toBe(SITE_NAME)
  })
  it('uses one word for the backtest page', () => {
    expect(PAGE_NAMES.backtest).toBe('Backtest')
  })
})
