import { describe, expect, it } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { http, HttpResponse } from 'msw'
import { DraftBoard } from '../pages/DraftBoard'
import { server } from './handlers'

function renderBoard() {
  return render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <DraftBoard />
    </QueryClientProvider>,
  )
}

const BOARD = {
  season: 2026, source: 'sleeper', projection_source: 'preseason_historical_per_game',
  as_of: '2026-08-20', scoring: 'ppr', count: 3, spearman_rho: 0.5756, note: 'Backend note that must never show.',
  players: [
    { player_name: 'Alpha Back', position: 'RB', team: 'DET', adp: 1, player_id: 'a', source: 'sleeper', model_rank: 1, model_fantasy_ppr: 311.9, adp_rank: 1, value_vs_adp: 0 },
    { player_name: 'Bravo Wide', position: 'WR', team: 'SEA', adp: 5, player_id: 'b', source: 'sleeper', model_rank: 17, model_fantasy_ppr: 248.9, adp_rank: 5, value_vs_adp: -12 },
    { player_name: 'Charlie Tight', position: 'TE', team: 'LV', adp: 20, player_id: 'c', source: 'sleeper', model_rank: 8, model_fantasy_ppr: 200.2, adp_rank: 20, value_vs_adp: 12 },
  ],
}

describe('DraftBoard', () => {
  it('defaults to the upcoming draft season', async () => {
    renderBoard()
    expect(screen.getByRole('combobox', { name: 'Draft season' })).toHaveTextContent('2026')
    expect(await screen.findByText("Ja'Marr Chase")).toBeInTheDocument()
  })

  it('says how fresh the board is in plain words, with no model names or statistics', async () => {
    server.use(http.get('/api/draft/board', () => HttpResponse.json(BOARD)))
    renderBoard()
    expect(await screen.findByText('Based on preseason projections, updated Aug 20, 2026')).toBeInTheDocument()
    const text = document.body.textContent ?? ''
    expect(text).not.toMatch(/preseason_historical_per_game/)
    expect(text).not.toMatch(/ρ=|n=3/)
    expect(text).not.toMatch(/Backend note/)
  })

  it('keeps the rank agreement in a plain-words disclosure for the analyst', async () => {
    server.use(http.get('/api/draft/board', () => HttpResponse.json(BOARD)))
    renderBoard()
    await screen.findByText('Alpha Back')
    expect(screen.getByText('About these numbers')).toBeInTheDocument()
    expect(screen.getByText(/How closely our order matches the market's: 0\.58/)).toBeInTheDocument()
    expect(screen.getByText(/Sleeper drafters/)).toBeInTheDocument()
  })

  it('names columns in plain words and right-aligns the numbers', async () => {
    server.use(http.get('/api/draft/board', () => HttpResponse.json(BOARD)))
    renderBoard()
    await screen.findByText('Alpha Back')
    const headers = within(screen.getByRole('table')).getAllByRole('columnheader')
    expect(headers.map((h) => h.textContent)).toEqual([
      'Market rank', 'Player', 'Pos', 'Team', 'Avg. pick', 'Our rank', 'Our points', 'Value vs market',
    ])
    for (const name of ['Market rank', 'Avg. pick', 'Our rank', 'Our points', 'Value vs market']) {
      expect(screen.getByRole('columnheader', { name })).toHaveClass('text-right')
    }
  })

  it('keeps the answer on a phone: market rank, player, our rank and value stay; the rest is hidden', async () => {
    server.use(http.get('/api/draft/board', () => HttpResponse.json(BOARD)))
    renderBoard()
    await screen.findByText('Alpha Back')
    const headers = within(screen.getByRole('table')).getAllByRole('columnheader')
    const hiddenOnPhone = headers.filter((h) => h.className.includes('hidden')).map((h) => h.textContent)
    const shownOnPhone = headers.filter((h) => !h.className.includes('hidden')).map((h) => h.textContent)
    expect(shownOnPhone).toEqual(['Market rank', 'Player', 'Our rank', 'Value vs market'])
    expect(hiddenOnPhone).toEqual(['Pos', 'Team', 'Avg. pick', 'Our points'])
  })

  it('puts position, team and points under the name for the hidden phone columns', async () => {
    server.use(http.get('/api/draft/board', () => HttpResponse.json(BOARD)))
    renderBoard()
    const name = await screen.findByText('Bravo Wide')
    const cell = name.closest('td')!
    expect(within(cell).getByText('WR')).toBeInTheDocument()
    expect(within(cell).getByText('SEA')).toBeInTheDocument()
    expect(within(cell).getByText(/249 pts/)).toBeInTheDocument()
  })

  it('shows value with a sign and a spoken meaning, not colour alone', async () => {
    server.use(http.get('/api/draft/board', () => HttpResponse.json(BOARD)))
    renderBoard()
    await screen.findByText('Alpha Back')
    expect(screen.getByText('+12')).toBeInTheDocument()
    expect(screen.getByText('−12')).toBeInTheDocument()
    expect(screen.getByText('We rank this player 12 spots higher than the market')).toBeInTheDocument()
    expect(screen.getByText('We rank this player 12 spots lower than the market')).toBeInTheDocument()
    expect(screen.getByText(/means we rank a player higher than the market does, − lower/)).toBeInTheDocument()
  })

  it('has no Refresh button in board mode', async () => {
    server.use(http.get('/api/draft/board', () => HttpResponse.json(BOARD)))
    renderBoard()
    await screen.findByText('Alpha Back')
    expect(screen.queryByRole('button', { name: /refresh/i })).not.toBeInTheDocument()
  })

  it('says the draft season is over once games are played', async () => {
    server.use(
      http.get('/api/draft/board', () => HttpResponse.json(BOARD)),
      http.get('/api/season/current', () => HttpResponse.json({ season: 2026, week: 11 })),
    )
    renderBoard()
    expect(await screen.findByText('The 2026 draft season is over. This is the preseason board.')).toBeInTheDocument()
  })

  it('says nothing about the draft window before the season starts', async () => {
    server.use(
      http.get('/api/draft/board', () => HttpResponse.json(BOARD)),
      http.get('/api/season/current', () => HttpResponse.json({ season: 2026, week: 1 })),
    )
    renderBoard()
    await screen.findByText('Alpha Back')
    expect(screen.queryByText(/draft season is over/)).not.toBeInTheDocument()
  })

  it('changes the requested season and filters by position', async () => {
    const user = userEvent.setup()
    renderBoard()
    await screen.findByText("Ja'Marr Chase")
    await user.click(screen.getByRole('combobox', { name: 'Draft season' }))
    await user.click(await screen.findByRole('option', { name: '2025' }))
    expect(screen.getByRole('combobox', { name: 'Draft season' })).toHaveTextContent('2025')
    await user.click(screen.getByRole('button', { name: 'WR' }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'WR' })).toHaveAttribute('aria-pressed', 'true'))
  })

  it('collapses the filters behind one button on phones, and the button tells the state', async () => {
    const user = userEvent.setup()
    renderBoard()
    const toggle = screen.getByRole('button', { name: /^Filters/ })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(document.getElementById('draft-filters')).toHaveClass('hidden')
    await user.click(toggle)
    expect(toggle).toHaveAttribute('aria-expanded', 'true')
    expect(document.getElementById('draft-filters')).not.toHaveClass('hidden')
  })

  it('shows a table-shaped skeleton while loading', () => {
    server.use(http.get('/api/draft/board', () => new Promise(() => {})))
    renderBoard()
    const loading = screen.getByRole('status', { name: 'Loading the draft board' })
    expect(within(loading).getAllByRole('row', { hidden: true }).length).toBeGreaterThan(5)
  })

  it('shows an alert with Retry when the board fails, without raw error text', async () => {
    server.use(http.get('/api/draft/board', () => HttpResponse.json({ detail: 'psycopg2 stack trace' }, { status: 500 })))
    renderBoard()
    // The hook retries once before it gives up.
    const alert = await screen.findByRole('alert', {}, { timeout: 4000 })
    expect(within(alert).getByRole('button', { name: 'Retry' })).toBeInTheDocument()
    expect(alert).not.toHaveTextContent(/psycopg2/)
  })

  it('shows a designed not-ready state with a next step on 404', async () => {
    const user = userEvent.setup()
    server.use(http.get('/api/draft/board', ({ request }) =>
      new URL(request.url).searchParams.get('season') === '2026'
        ? HttpResponse.json({ detail: 'none' }, { status: 404 })
        : HttpResponse.json({ ...BOARD, season: 2025 }),
    ))
    renderBoard()
    expect(await screen.findByText('The 2026 draft board is not ready yet')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Show the 2025 board' }))
    expect(await screen.findByText('Alpha Back')).toBeInTheDocument()
  })

  it('says so when a position filter matches no one, and clears it', async () => {
    const user = userEvent.setup()
    server.use(http.get('/api/draft/board', ({ request }) =>
      new URL(request.url).searchParams.get('position') === 'QB'
        ? HttpResponse.json({ ...BOARD, count: 0, players: [] })
        : HttpResponse.json(BOARD),
    ))
    renderBoard()
    await screen.findByText('Alpha Back')
    await user.click(screen.getByRole('button', { name: 'QB' }))
    expect(await screen.findByText('No QB players on this board')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Clear filters' }))
    expect(await screen.findByText('Alpha Back')).toBeInTheDocument()
  })
})
