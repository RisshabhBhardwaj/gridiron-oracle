import { afterEach, describe, expect, it } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { http, HttpResponse } from 'msw'
import { SearchProvider } from '../context/SearchContext'
import { CurrentSeason } from '../pages/CurrentSeason'
import { server } from './handlers'
import { factories } from './factories'
import { installMemoryStorage } from './memory-storage'
import type { SeasonPlayerProjection } from '../types/api'

const FIRST = ['Aaron', 'Bo', 'Cam', 'Dak', 'Eli', 'Fred', 'Gus', 'Hank', 'Ian', 'Jon']
const POS = ['QB', 'RB', 'WR', 'TE']

/** n distinct players; projection falls with the index, so rank 1 is "Aaron Alpha0". */
function makeRows(n: number, method = 'causal_oof_conformal_90'): SeasonPlayerProjection[] {
  return Array.from({ length: n }, (_, i) => {
    const mean = 300 - i
    return {
      player_id: `p-${i}`,
      player_name: `${FIRST[i % FIRST.length]} ${String.fromCharCode(65 + (i % 26))}player${i}`,
      position: POS[i % POS.length],
      team: ['MIN', 'BUF', 'DAL', 'SF'][i % 4],
      passing_yards: null,
      rushing_yards: null,
      receiving_yards: null,
      fantasy_ppr: { mean, p10: mean - 40, p50: mean, p90: mean + 40 },
      degraded: false,
      interval_method: method,
      p_active: 1,
    }
  })
}

function serveWeekly(rows: SeasonPlayerProjection[]) {
  server.use(
    http.get('/api/projections/season/:season/weeks/:week', () =>
      HttpResponse.json({ ...factories.seasonWeekProjectionsResponse(), count: rows.length, projections: rows }),
    ),
  )
}
function serveSeason(rows: SeasonPlayerProjection[]) {
  server.use(
    http.get('/api/projections/season/:season', () =>
      HttpResponse.json({ ...factories.seasonProjectionsResponse(), count: rows.length, projections: rows }),
    ),
  )
}

function Here() {
  const loc = useLocation()
  return <p data-testid="here">{loc.pathname}{loc.search}</p>
}

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <SearchProvider>
        <MemoryRouter initialEntries={['/projections']}>
          <Routes>
            <Route path="/projections" element={<CurrentSeason />} />
            <Route path="/player/:id" element={<Here />} />
          </Routes>
        </MemoryRouter>
      </SearchProvider>
    </QueryClientProvider>,
  )
}

/** Names in the table's body, top to bottom. */
function bodyNames(): string[] {
  const table = screen.getByRole('table')
  return within(table)
    .getAllByRole('row')
    .slice(1)
    .map((r) => within(r).getAllByRole('cell')[1].querySelector('a')?.textContent ?? '')
}

afterEach(() => {
  delete (window as unknown as Record<string, unknown>).localStorage
})

describe('Current Season player table', () => {
  it('is the default view, with rank, player link, position, team, projection and the range in plain words', async () => {
    serveWeekly(makeRows(3))
    renderPage()
    const table = await screen.findByRole('table')
    const headers = within(table).getAllByRole('columnheader').map((h) => h.textContent?.replace(/\s+/g, ' ').trim())
    expect(headers).toEqual(['#Rank', 'Player', 'Pos', 'Team', 'Week 5 · Fantasy Points (PPR)', 'Bad week', 'Great week'])
    const link = screen.getByRole('link', { name: 'Aaron Aplayer0' })
    expect(link).toHaveAttribute('href', expect.stringMatching(/^\/player\/p-0\?week=5&season=2025&stat=fantasy_ppr&name=Aaron%20Aplayer0$/))
    // 300 - 40 and 300 + 40: the API's own p10 and p90, printed under their plain names.
    const first = within(table).getAllByRole('row')[1]
    expect(within(first).getByText('260.0')).toBeInTheDocument()
    expect(within(first).getByText('340.0')).toBeInTheDocument()
    expect(document.body.textContent).not.toMatch(/p10|p90/)
  })

  it('opens the player page when the name is clicked', async () => {
    serveWeekly(makeRows(2))
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByRole('link', { name: 'Aaron Aplayer0' }))
    expect(await screen.findByTestId('here')).toHaveTextContent('/player/p-0?week=5&season=2025&stat=fantasy_ppr')
  })

  it('right-aligns the numbers in tabular figures', async () => {
    serveWeekly(makeRows(2))
    renderPage()
    const table = await screen.findByRole('table')
    const row = within(table).getAllByRole('row')[1]
    const cells = within(row).getAllByRole('cell')
    expect(cells[0]).toHaveClass('text-right')
    expect(cells[5]).toHaveClass('text-right', 'tabular-nums')
    expect(within(cells[4]).getByText('300.0')).toHaveClass('tabular-nums')
  })

  describe('sorting', () => {
    it('starts on projection, high to low, and says so with aria-sort', async () => {
      serveWeekly(makeRows(5))
      renderPage()
      await screen.findByRole('table')
      const projection = screen.getByRole('columnheader', { name: /Week 5 · Fantasy Points/ })
      expect(projection).toHaveAttribute('aria-sort', 'descending')
      expect(screen.getByRole('columnheader', { name: 'Player' })).toHaveAttribute('aria-sort', 'none')
      expect(bodyNames()[0]).toBe('Aaron Aplayer0')
    })

    it('sorts by name A to Z, then Z to A, and announces each', async () => {
      serveWeekly(makeRows(6))
      const user = userEvent.setup()
      renderPage()
      await screen.findByRole('table')
      const header = screen.getByRole('columnheader', { name: 'Player' })
      await user.click(within(header).getByRole('button'))
      expect(header).toHaveAttribute('aria-sort', 'ascending')
      const az = bodyNames()
      expect(az).toEqual([...az].sort((a, b) => a.localeCompare(b)))
      expect(screen.getByRole('columnheader', { name: /Week 5/ })).toHaveAttribute('aria-sort', 'none')
      await user.click(within(header).getByRole('button'))
      expect(header).toHaveAttribute('aria-sort', 'descending')
      expect(bodyNames()).toEqual([...az].reverse())
    })

    it('sorts by position, and keeps each player\'s projection rank', async () => {
      serveWeekly(makeRows(8))
      const user = userEvent.setup()
      renderPage()
      await screen.findByRole('table')
      await user.click(within(screen.getByRole('columnheader', { name: 'Pos' })).getByRole('button'))
      const rows = within(screen.getByRole('table')).getAllByRole('row').slice(1)
      const positions = rows.map((r) => within(r).getAllByRole('cell')[2].textContent)
      expect(positions).toEqual([...positions].sort())
      // Position sort puts QBs first; the first QB is still rank 1, the second is rank 5.
      expect(within(rows[0]).getAllByRole('cell')[0]).toHaveTextContent('1')
      expect(within(rows[1]).getAllByRole('cell')[0]).toHaveTextContent('5')
    })

    it('clicking projection after another sort goes back to best first', async () => {
      serveWeekly(makeRows(4))
      const user = userEvent.setup()
      renderPage()
      await screen.findByRole('table')
      await user.click(within(screen.getByRole('columnheader', { name: 'Player' })).getByRole('button'))
      await user.click(within(screen.getByRole('columnheader', { name: /Week 5/ })).getByRole('button'))
      expect(screen.getByRole('columnheader', { name: /Week 5/ })).toHaveAttribute('aria-sort', 'descending')
      expect(bodyNames()[0]).toBe('Aaron Aplayer0')
    })
  })

  describe('filter box', () => {
    it('filters by name or team and keeps the real rank', async () => {
      serveWeekly(makeRows(40))
      const user = userEvent.setup()
      renderPage()
      await screen.findByRole('table')
      await user.type(screen.getByRole('searchbox', { name: 'Filter by name or team' }), 'player7')
      expect(bodyNames()).toEqual(['Hank Hplayer7'])
      const row = within(screen.getByRole('table')).getAllByRole('row')[1]
      expect(within(row).getAllByRole('cell')[0]).toHaveTextContent('8') // rank 8, not 1
      expect(screen.getByText('Showing 1–1 of 1')).toBeInTheDocument()
    })

    it('a team code filters too', async () => {
      serveWeekly(makeRows(12))
      const user = userEvent.setup()
      renderPage()
      await screen.findByRole('table')
      await user.type(screen.getByRole('searchbox'), 'buf')
      expect(bodyNames()).toHaveLength(3) // indexes 1, 5, 9
    })

    it('says nothing matched and "Clear filters" brings the list back', async () => {
      serveWeekly(makeRows(5))
      const user = userEvent.setup()
      renderPage()
      await screen.findByRole('table')
      await user.type(screen.getByRole('searchbox'), 'zzzz')
      expect(screen.getByText('No players match “zzzz”')).toBeInTheDocument()
      expect(screen.queryByRole('table')).not.toBeInTheDocument()
      await user.click(screen.getByRole('button', { name: 'Clear filters' }))
      expect(await screen.findByRole('table')).toBeInTheDocument()
      expect(bodyNames()).toHaveLength(5)
      expect(screen.getByRole('searchbox')).toHaveValue('')
    })
  })

  describe('paging', () => {
    it('shows 50 of 120, then the next page, then the last, and starts back at page 1 after a filter', async () => {
      serveWeekly(makeRows(120))
      const user = userEvent.setup()
      renderPage()
      await screen.findByRole('table')
      expect(bodyNames()).toHaveLength(50)
      expect(screen.getByText('Showing 1–50 of 120')).toBeInTheDocument()
      expect(screen.getByText('Page 1 of 3')).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /Previous page/ })).toBeDisabled()

      await user.click(screen.getByRole('button', { name: /Next page/ }))
      expect(screen.getByText('Showing 51–100 of 120')).toBeInTheDocument()
      expect(bodyNames()[0]).toBe('Aaron Yplayer50')
      expect(within(within(screen.getByRole('table')).getAllByRole('row')[1]).getAllByRole('cell')[0]).toHaveTextContent('51')

      await user.click(screen.getByRole('button', { name: /Next page/ }))
      expect(screen.getByText('Showing 101–120 of 120')).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /Next page/ })).toBeDisabled()

      await user.click(screen.getByRole('button', { name: /Previous page/ }))
      expect(screen.getByText('Showing 51–100 of 120')).toBeInTheDocument()

      await user.type(screen.getByRole('searchbox'), 'player1')
      expect(screen.getByText(/^Showing 1–/)).toBeInTheDocument()
    })

    it('scrolls the list back into view on a page change', async () => {
      serveWeekly(makeRows(60))
      const calls: Element[] = []
      const original = Element.prototype.scrollIntoView
      Element.prototype.scrollIntoView = function (this: Element) { calls.push(this) }
      try {
        const user = userEvent.setup()
        renderPage()
        await screen.findByRole('table')
        await user.click(screen.getByRole('button', { name: /Next page/ }))
        expect(calls.length).toBeGreaterThan(0)
      } finally {
        Element.prototype.scrollIntoView = original
      }
    })

    it('has no pager when everything fits on one page', async () => {
      serveWeekly(makeRows(10))
      renderPage()
      await screen.findByRole('table')
      expect(screen.queryByRole('navigation', { name: 'Pages' })).not.toBeInTheDocument()
      expect(screen.getByText('Showing 1–10 of 10')).toBeInTheDocument()
    })

    it('renders one page of rows, not all of them', async () => {
      serveWeekly(makeRows(808))
      renderPage()
      await screen.findByRole('table')
      expect(within(screen.getByRole('table')).getAllByRole('row')).toHaveLength(51)
      expect(screen.getByText('Showing 1–50 of 808')).toBeInTheDocument()
    })
  })

  describe('provenance rule in table form', () => {
    it('shows no row for the simulation until the visitor asks, and then labels every row', async () => {
      serveWeekly(makeRows(4, 'season_simulator_mc'))
      const user = userEvent.setup()
      renderPage()
      expect(await screen.findByText('No forecast yet for Week 5, 2025')).toBeInTheDocument()
      expect(screen.queryByRole('table')).not.toBeInTheDocument()

      await user.click(screen.getByRole('button', { name: /show these numbers anyway/i }))
      const table = await screen.findByRole('table')
      const rows = within(table).getAllByRole('row').slice(1)
      expect(rows).toHaveLength(4)
      for (const r of rows) expect(within(r).getByText('Preseason simulation')).toBeInTheDocument()
      // The section says it in full, and the column header does not claim a Week 5 forecast.
      expect(screen.getByText('Preseason simulation, not updated for games played')).toBeInTheDocument()
      expect(screen.queryByText(/Week 5 · Fantasy Points/)).not.toBeInTheDocument()
      expect(within(table).getByRole('columnheader', { name: /Fantasy Points \(PPR\)/ })).toBeInTheDocument()

      await user.click(screen.getByRole('button', { name: /hide these numbers/i }))
      expect(screen.queryByRole('table')).not.toBeInTheDocument()
    })

    it('labels a simulation row even when it sits among real forecasts, and never labels a forecast row as one', async () => {
      const rows = makeRows(4)
      rows[1] = { ...rows[1], interval_method: 'playing_time_gaussian_mc' }
      rows[2] = { ...rows[2], interval_method: 'some_new_method' }
      serveWeekly(rows)
      const user = userEvent.setup()
      renderPage()
      await user.click(await screen.findByRole('button', { name: /show these numbers anyway/i }))
      const table = await screen.findByRole('table')
      const body = within(table).getAllByRole('row').slice(1)
      expect(within(body[0]).queryByText(/simulation|Not verified/i)).not.toBeInTheDocument()
      expect(within(body[1]).getByText('Simulation')).toBeInTheDocument()
      expect(within(body[2]).getByText('Not verified')).toBeInTheDocument()
      expect(within(body[3]).queryByText(/simulation|Not verified/i)).not.toBeInTheDocument()
      // The real forecasts say so, because the header no longer does.
      expect(within(body[0]).getByText('Forecast')).toBeInTheDocument()
    })

    it('a real forecast shows no simulation wording anywhere', async () => {
      serveWeekly(makeRows(3))
      renderPage()
      await screen.findByRole('table')
      expect(screen.queryByText(/simulation/i)).not.toBeInTheDocument()
      expect(screen.queryByText('Forecast')).not.toBeInTheDocument()
    })

    it('Season-Long rows are season totals, named so, with season wording for the range', async () => {
      serveSeason(makeRows(3, 'season_simulator_mc'))
      const user = userEvent.setup()
      renderPage()
      await user.click(screen.getByRole('tab', { name: 'Season-Long' }))
      const table = await screen.findByRole('table')
      const headers = within(table).getAllByRole('columnheader').map((h) => h.textContent?.replace(/\s+/g, ' ').trim())
      expect(headers).toContain('Season total · Fantasy Points (PPR)')
      expect(headers).toContain('Bad season')
      expect(headers).toContain('Great season')
      expect(headers.join(' ')).not.toMatch(/week/i)
      // The season total is not a weekly forecast, so it carries no weekly provenance tag.
      expect(screen.queryByText('Preseason simulation')).not.toBeInTheDocument()
    })
  })

  describe('phone column set', () => {
    it('hides Pos, Team and the range on phones; keeps rank, player and projection, and puts the range under the name', async () => {
      serveWeekly(makeRows(2))
      renderPage()
      const table = await screen.findByRole('table')
      const byName = (name: RegExp | string) => within(table).getByRole('columnheader', { name })
      for (const hidden of ['Pos', 'Team', 'Bad week', 'Great week']) {
        expect(byName(hidden)).toHaveClass('hidden', 'sm:table-cell')
      }
      for (const shown of [/Rank/, 'Player', /Week 5/]) {
        expect(byName(shown)).not.toHaveClass('hidden')
      }
      const sub = within(screen.getAllByRole('row')[1]).getByText(/Bad week 260\.0 · Great week 340\.0/)
      expect(sub.parentElement).toHaveClass('sm:hidden')
    })
  })

  describe('range and zero floors', () => {
    it('keeps a 0.0 low end as sent, under the "Bad week" name, with the explanation one tap away', async () => {
      const rows = makeRows(2)
      rows[0] = { ...rows[0], fantasy_ppr: { mean: 500, p10: 0, p50: 480, p90: 600 } }
      serveWeekly(rows)
      const user = userEvent.setup()
      renderPage()
      const table = await screen.findByRole('table')
      const first = within(table).getAllByRole('row')[1]
      expect(within(first).getAllByRole('cell')[5]).toHaveTextContent('0.0')
      const summary = screen.getByText('How to read the range')
      expect(summary.tagName).toBe('SUMMARY')
      await user.click(summary)
      expect(screen.getByText(/A low end of 0\.0 is real, not missing/)).toBeInTheDocument()
    })

    it('marks a headline number outside its range and prints the reason as visible text, not a tooltip', async () => {
      const rows = makeRows(2)
      rows[0] = { ...rows[0], fantasy_ppr: { mean: 500, p10: 510, p50: 505, p90: 560 } }
      serveWeekly(rows)
      renderPage()
      const table = await screen.findByRole('table')
      const first = within(table).getAllByRole('row')[1]
      expect(within(first).getByText(/†/)).toBeInTheDocument()
      const note = screen.getByText(/The big number is an average over every simulated game/)
      expect(note).toBeVisible()
      expect(note).not.toHaveAttribute('title')
      // The second row is inside its range: no mark on it.
      expect(within(within(table).getAllByRole('row')[2]).queryByText(/†/)).not.toBeInTheDocument()
    })

    it('prints no note when no number is outside its range', async () => {
      serveWeekly(makeRows(3))
      renderPage()
      await screen.findByRole('table')
      expect(screen.queryByText(/The big number is an average/)).not.toBeInTheDocument()
    })
  })

  describe('table and card views', () => {
    it('can switch to cards, and the choice is remembered for next time', async () => {
      installMemoryStorage()
      serveWeekly(makeRows(3))
      const user = userEvent.setup()
      const { unmount } = renderPage()
      await screen.findByRole('table')
      expect(screen.getByRole('button', { name: 'Table' })).toHaveAttribute('aria-pressed', 'true')
      await user.click(screen.getByRole('button', { name: 'Cards' }))
      expect(screen.queryByRole('table')).not.toBeInTheDocument()
      expect(screen.getByText('Aaron Aplayer0')).toBeInTheDocument()
      expect(screen.getAllByText('Week 5 · Fantasy Points (PPR)').length).toBe(3)
      expect(window.localStorage.getItem('gridiron.projections.view')).toBe('cards')
      unmount()

      renderPage()
      expect(await screen.findByText('Aaron Aplayer0')).toBeInTheDocument()
      expect(screen.queryByRole('table')).not.toBeInTheDocument()
      expect(screen.getByRole('button', { name: 'Cards' })).toHaveAttribute('aria-pressed', 'true')
      // And back to the table.
      await user.click(screen.getByRole('button', { name: 'Table' }))
      expect(await screen.findByRole('table')).toBeInTheDocument()
      expect(window.localStorage.getItem('gridiron.projections.view')).toBe('table')
    })

    it('a card is a real link to the player page, not a clickable div', async () => {
      installMemoryStorage().setItem('gridiron.projections.view', 'cards')
      serveWeekly(makeRows(2))
      renderPage()
      const link = await screen.findByRole('link', { name: /Aaron Aplayer0/ })
      expect(link).toHaveAttribute('href', expect.stringContaining('/player/p-0?week=5&season=2025'))
      expect(screen.queryByRole('button', { name: /Aaron Aplayer0/ })).not.toBeInTheDocument()
    })

    it('works when storage is blocked: starts on the table and still switches', async () => {
      const storage = installMemoryStorage()
      storage.getItem = () => { throw new Error('blocked') }
      storage.setItem = () => { throw new Error('blocked') }
      serveWeekly(makeRows(3))
      const user = userEvent.setup()
      renderPage()
      expect(await screen.findByRole('table')).toBeInTheDocument()
      await user.click(screen.getByRole('button', { name: 'Cards' }))
      expect(screen.queryByRole('table')).not.toBeInTheDocument()
      expect(screen.getByText('Aaron Aplayer0')).toBeInTheDocument()
    })

    it('works when there is no storage at all', async () => {
      Object.defineProperty(window, 'localStorage', { value: undefined, configurable: true, writable: true })
      serveWeekly(makeRows(3))
      const user = userEvent.setup()
      renderPage()
      expect(await screen.findByRole('table')).toBeInTheDocument()
      await user.click(screen.getByRole('button', { name: 'Cards' }))
      expect(screen.getByText('Aaron Aplayer0')).toBeInTheDocument()
    })

    it('cards share the filter and the page, use plain range words, and label a simulation card', async () => {
      installMemoryStorage().setItem('gridiron.projections.view', 'cards')
      serveWeekly(makeRows(60, 'season_simulator_mc'))
      const user = userEvent.setup()
      renderPage()
      await user.click(await screen.findByRole('button', { name: /show these numbers anyway/i }))
      expect(await screen.findByText('Showing 1–48 of 60')).toBeInTheDocument()
      expect(screen.getAllByText('Preseason simulation, not updated for games played')).toHaveLength(49) // heading + 48 cards
      expect(screen.getAllByText(/Bad week 260\.0/).length).toBeGreaterThan(0)
      expect(document.body.textContent).not.toMatch(/p10|p90/)
      await user.type(screen.getByRole('searchbox'), 'player59')
      expect(screen.getByText('Showing 1–1 of 1')).toBeInTheDocument()
    })
  })

  describe('states', () => {
    it('shows a table-shaped skeleton while the list loads, not the card grid', async () => {
      let release: () => void = () => {}
      const gate = new Promise<void>((resolve) => { release = resolve })
      server.use(
        http.get('/api/projections/season/:season/weeks/:week', async () => {
          await gate
          return HttpResponse.json({ ...factories.seasonWeekProjectionsResponse(), projections: makeRows(3) })
        }),
      )
      renderPage()
      const region = await screen.findByText('Loading player projections')
      expect(region.parentElement).toHaveAttribute('aria-busy', 'true')
      expect(region.parentElement?.querySelector('.grid')).toBeNull()
      release()
      expect(await screen.findByRole('table')).toBeInTheDocument()
      await waitFor(() => expect(screen.queryByText('Loading player projections')).not.toBeInTheDocument())
    })

    it('says no one has a figure for the stat, with a way back, instead of an empty table', async () => {
      const rows = makeRows(3).map((r) => ({ ...r, fantasy_ppr: null }))
      serveWeekly(rows)
      renderPage()
      expect(await screen.findByText(/No Fantasy Points \(PPR\) projections for these players/)).toBeInTheDocument()
      expect(screen.queryByRole('table')).not.toBeInTheDocument()
    })
  })
})
