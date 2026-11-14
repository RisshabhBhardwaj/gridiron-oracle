import { describe, expect, it } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { http, HttpResponse } from 'msw'
import { Bar, BarChart } from 'recharts'
import { SearchProvider } from '../context/SearchContext'
import { NavBar } from '../components/layout/NavBar'
import { PageShell } from '../components/layout/PageShell'
import { SkipLink } from '../components/layout/SkipLink'
import { ChartFrame } from '../components/charts/themed-chart'
import { Home } from '../pages/Home'
import { Settings } from '../pages/Settings'
import { GameDetail } from '../pages/GameDetail'
import { PlayerDetail } from '../pages/PlayerDetail'
import { server } from './handlers'

function shell(ui: React.ReactElement, path = '/') {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <SearchProvider>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route path="*" element={ui} />
          </Routes>
        </MemoryRouter>
      </SearchProvider>
    </QueryClientProvider>,
  )
}

describe('search buttons: the name starts with what the button shows', () => {
  it('nav Search button', () => {
    shell(<NavBar />)
    // The name is the visible words, with no separate aria-label to drift from them; the shortcut hint is hidden from the name.
    const button = document.getElementById('nav-search-btn') as HTMLElement
    expect(button).not.toHaveAttribute('aria-label')
    expect(button).toHaveAccessibleName('Search players')
    expect(button).toHaveTextContent(/^Search players/)
    expect(button).toHaveAttribute('aria-keyshortcuts', 'Meta+K Control+K')
    expect(button.querySelector('kbd')).toHaveAttribute('aria-hidden', 'true')
  })

  it('Home search button', () => {
    shell(<Home />)
    expect(screen.getByRole('button', { name: 'Search players, teams, or positions' })).toBeInTheDocument()
  })
})

describe('skip link', () => {
  it('is the first thing a keyboard reaches and moves focus to the main content', async () => {
    const user = userEvent.setup()
    shell(
      <>
        <SkipLink />
        <NavBar />
        <PageShell><h1>Page</h1></PageShell>
      </>,
    )
    await user.tab()
    const link = screen.getByRole('link', { name: 'Skip to main content' })
    expect(link).toHaveFocus()
    expect(link).toHaveAttribute('href', '#main-content')
    await user.keyboard('{Enter}')
    expect(screen.getByRole('main')).toHaveFocus()
  })
})

describe('exactly one h1 in every state', () => {
  const h1s = () => screen.queryAllByRole('heading', { level: 1 })

  it('Settings while loading, and after', async () => {
    server.use(http.get('/api/settings', async () => { await new Promise((r) => setTimeout(r, 50)); return HttpResponse.json({}) }))
    shell(<Settings />, '/settings')
    expect(h1s()).toHaveLength(1)
    expect(await screen.findByRole('heading', { level: 1, name: 'Settings' })).toBeInTheDocument()
    expect(h1s()).toHaveLength(1)
  })

  it('Settings when the request fails', async () => {
    server.use(http.get('/api/settings', () => HttpResponse.json({ detail: 'x' }, { status: 500 })))
    shell(<Settings />, '/settings')
    await screen.findByRole('alert')
    expect(h1s()).toHaveLength(1)
  })

  it('Game page while loading and when the simulation is not ready', async () => {
    server.use(http.get('/api/games/:game_id/drive-sim', () => new HttpResponse(null, { status: 404 })))
    shell(<Routes><Route path="/projections/game/:gameId" element={<GameDetail />} /></Routes>, '/projections/game/2026_05_MIN_GB')
    expect(h1s()).toHaveLength(1)
    await screen.findByText("This game's simulation isn't ready yet")
    expect(h1s()).toHaveLength(1)
  })

  it('Player page with no player in the address', () => {
    shell(<Routes><Route path="/player/:player_id" element={<PlayerDetail />} /></Routes>, '/player/abc')
    expect(h1s()).toHaveLength(1)
    expect(screen.getByRole('heading', { level: 1, name: 'No player selected' })).toBeInTheDocument()
  })
})

describe('chart with a table fallback', () => {
  const chart = (
    <ChartFrame
      height={120}
      label="Bar chart of wins. 2024: 12; 2025: 8"
      table={{ caption: 'Wins by season', head: ['Season', 'Wins'], rows: [['2024', 12], ['2025', 8]] }}
    >
      <BarChart data={[{ a: 1 }]}><Bar dataKey="a" /></BarChart>
    </ChartFrame>
  )

  it('is an image with a plain name until the visitor asks for the table', async () => {
    const user = userEvent.setup()
    render(chart)
    expect(screen.getByRole('img', { name: /Bar chart of wins/ })).toBeInTheDocument()
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
    const toggle = screen.getByRole('button', { name: 'Show as table' })
    expect(toggle).toHaveAttribute('aria-pressed', 'false')

    await user.click(toggle)
    const table = screen.getByRole('table', { name: 'Wins by season' })
    expect(within(table).getByRole('columnheader', { name: 'Season' })).toBeInTheDocument()
    expect(within(table).getByRole('rowheader', { name: '2025' })).toBeInTheDocument()
    expect(within(table).getByRole('cell', { name: '8' })).toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Show as chart' })).toHaveAttribute('aria-pressed', 'true')

    await user.click(screen.getByRole('button', { name: 'Show as chart' }))
    expect(screen.getByRole('img', { name: /Bar chart of wins/ })).toBeInTheDocument()
  })

  it('shows no toggle when there is no table to offer', () => {
    render(<ChartFrame height={120} label="x"><BarChart data={[{ a: 1 }]}><Bar dataKey="a" /></BarChart></ChartFrame>)
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })
})
