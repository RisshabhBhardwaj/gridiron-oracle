import { describe, expect, it } from 'vitest'
import {
  agreementLine,
  draftWindowNotice,
  draftWindowOver,
  formatBoardDate,
  freshnessLine,
  marketSourceName,
  valueView,
} from '../lib/draft-board'

describe('draftWindowOver', () => {
  it('is open in the preseason: week 1 means no game has been played', () => {
    expect(draftWindowOver(2026, { season: 2026, week: 1 })).toBe(false)
  })

  it('is over once a game has been played (week 2 or later)', () => {
    expect(draftWindowOver(2026, { season: 2026, week: 2 })).toBe(true)
    expect(draftWindowOver(2026, { season: 2026, week: 11 })).toBe(true)
  })

  it('is over for a board from an earlier season, whatever the week', () => {
    expect(draftWindowOver(2025, { season: 2026, week: 1 })).toBe(true)
  })

  it('is open for a board of a season that has not started', () => {
    expect(draftWindowOver(2027, { season: 2026, week: 11 })).toBe(false)
  })

  it('stays quiet while the current season is unknown', () => {
    expect(draftWindowOver(2026, undefined)).toBe(false)
    expect(draftWindowOver(2026, null)).toBe(false)
  })

  it('says it plainly, with no model names', () => {
    expect(draftWindowNotice(2026, { season: 2026, week: 11 })).toBe(
      'The 2026 draft season is over. This is the preseason board.',
    )
    expect(draftWindowNotice(2025, { season: 2026, week: 1 })).toMatch(/2025 season has been played/)
  })
})

describe('valueView', () => {
  it('prints a plus sign and an up cue when we rank the player higher than the market', () => {
    expect(valueView(12)).toEqual({
      kind: 'up',
      text: '+12',
      spoken: 'We rank this player 12 spots higher than the market',
    })
  })

  it('prints a real minus sign and a down cue when we rank the player lower', () => {
    const v = valueView(-3)
    expect(v.kind).toBe('down')
    expect(v.text).toBe('−3')
    expect(v.spoken).toBe('We rank this player 3 spots lower than the market')
  })

  it('says "1 spot", not "1 spots"', () => {
    expect(valueView(1).spoken).toMatch(/1 spot higher/)
    expect(valueView(-1).spoken).toMatch(/1 spot lower/)
  })

  it('treats zero and rounding-to-zero as even, with no sign', () => {
    expect(valueView(0).text).toBe('0')
    expect(valueView(0.2).kind).toBe('even')
  })

  it('shows a dash when we have no rank', () => {
    expect(valueView(null).text).toBe('—')
    expect(valueView(undefined).kind).toBe('none')
    expect(valueView(Number.NaN).kind).toBe('none')
  })
})

describe('freshnessLine', () => {
  it('turns the source and date into plain words', () => {
    expect(freshnessLine({ projection_source: 'preseason_historical_per_game', as_of: '2026-08-20' })).toBe(
      'Based on preseason projections, updated Aug 20, 2026',
    )
  })

  it('never prints the internal source name or a statistic', () => {
    const line = freshnessLine({ projection_source: 'preseason_projection_rows', as_of: '2026-08-20' })
    expect(line).not.toMatch(/_|ρ|n=/)
  })

  it('does not call a non-preseason source preseason', () => {
    expect(freshnessLine({ projection_source: 'weekly_rows', as_of: '2026-10-01' })).toBe(
      'Based on projections, updated Oct 1, 2026',
    )
  })

  it('keeps the calendar day whatever the time zone', () => {
    expect(formatBoardDate('2026-01-01')).toBe('Jan 1, 2026')
    expect(formatBoardDate('not a date')).toBe('not a date')
  })
})

describe('agreementLine and market name', () => {
  it('explains the rank agreement in plain words, with its scale and sample', () => {
    expect(agreementLine(0.5756, 143)).toBe(
      "How closely our order matches the market's: 0.58, where 1 means the same order and 0 means no link (across 143 players).",
    )
  })

  it('is omitted when the API has no figure', () => {
    expect(agreementLine(null, 143)).toBeNull()
  })

  it('names the market source', () => {
    expect(marketSourceName('sleeper')).toBe('Sleeper')
    expect(marketSourceName('fantasypros')).toBe('FantasyPros')
    expect(marketSourceName('espn')).toBe('Espn')
    expect(marketSourceName(null)).toBeNull()
  })
})
