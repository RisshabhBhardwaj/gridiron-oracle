import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { SearchProvider } from '../context/SearchContext'
import { CurrentSeason } from '../pages/CurrentSeason'
import { server } from './handlers'

function Here() {
  const loc = useLocation()
  return <p data-testid="here">{loc.pathname}{loc.search}</p>
}

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <SearchProvider>
        <MemoryRouter initialEntries={['/projections']}>
          <Routes>
            <Route path="/projections" element={ui} />
            <Route path="/player/:id" element={<Here />} />
          </Routes>
        </MemoryRouter>
      </SearchProvider>
    </QueryClientProvider>,
  )
}

describe('CurrentSeason', () => {
  it('renders page header and default Weekly tab', async () => {
    renderWithProviders(<CurrentSeason />)
    expect(screen.getByText('Current Season')).toBeInTheDocument()
    expect(screen.getByText('Weekly')).toBeInTheDocument()
    expect(screen.getByText('Season-Long')).toBeInTheDocument()
    expect(screen.getByText('Games')).toBeInTheDocument()

    // Should load weekly player projections
    await waitFor(() => {
      expect(screen.getByText('Justin Jefferson')).toBeInTheDocument()
    })
  })

  it('switches to Games tab and displays win totals and matchups', async () => {
    renderWithProviders(<CurrentSeason />)

    const gamesTabBtn = screen.getByText('Games')
    fireEvent.click(gamesTabBtn)

    await waitFor(() => {
      expect(screen.getByText(/Projected Season Win Totals/i)).toBeInTheDocument()
      expect(screen.getByText(/Matchup Projections/i)).toBeInTheDocument()
    })
  })

  it('switches to Season-Long tab', async () => {
    renderWithProviders(<CurrentSeason />)

    const seasonTabBtn = screen.getByText('Season-Long')
    fireEvent.click(seasonTabBtn)

    await waitFor(() => {
      expect(screen.getByText('Start Week:')).toBeInTheDocument()
    })
  })

  it('requests only the current week once /season/current resolves', async () => {
    // Regression: the page copied the current week into state from an effect, so
    // for one render it asked for the offline fallback week (1) before the real
    // one (5). That wasted request also made the cards vanish and reappear, which
    // raced the click in the next test on slow machines (CI).
    const weeklyPaths: string[] = []
    const onStart = ({ request }: { request: Request }) => {
      const path = new URL(request.url).pathname
      if (/\/projections\/season\/\d+\/weeks\/\d+$/.test(path)) weeklyPaths.push(path)
    }
    server.events.on('request:start', onStart)
    try {
      renderWithProviders(<CurrentSeason />)
      await screen.findByText('Justin Jefferson')
      expect(weeklyPaths).toEqual(['/api/projections/season/2025/weeks/5'])
    } finally {
      server.events.removeListener('request:start', onStart)
    }
  })

  it('opens the player page with the name, week, season and stat in the address', async () => {
    // Regression: the card used to open /player/<id> alone, and the player page
    // answered "No player selected".
    renderWithProviders(<CurrentSeason />)
    fireEvent.click(await screen.findByText('Justin Jefferson'))
    const here = await screen.findByTestId('here')
    expect(here).toHaveTextContent(/^\/player\/[^?]+\?week=\d+&season=\d+&stat=\w+&name=Justin%20Jefferson$/)
  })
})
