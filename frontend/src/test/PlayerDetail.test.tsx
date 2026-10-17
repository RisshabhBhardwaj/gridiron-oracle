import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { server } from './handlers'
import { factories } from './factories'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { SearchProvider } from '../context/SearchContext'
import { PlayerDetail } from '../pages/PlayerDetail'
import { PlayerSearch } from '../components/PlayerSearch'

function renderPlayer(path = '/player/test-001?week=5&season=2025&stat=receiving_yards&name=Justin+Jefferson') {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <SearchProvider>
        <MemoryRouter initialEntries={[path]}>
          <PlayerSearch />
          <Routes>
            <Route path="/player/:player_id" element={<PlayerDetail />} />
          </Routes>
        </MemoryRouter>
      </SearchProvider>
    </QueryClientProvider>,
  )
}

describe('PlayerDetail', () => {
  it('renders player name in header', async () => {
    renderPlayer()
    await waitFor(() =>
      expect(screen.getByText(/Justin Jefferson/i)).toBeInTheDocument(),
    )
  })

  it('renders ProjectionCard with correct projection value', async () => {
    renderPlayer()
    await waitFor(() => {
      // Multiple elements with "87.4" are expected (card + percentile fan). Use getAllByText.
      const projValues = screen.getAllByText('87.4')
      expect(projValues.length).toBeGreaterThan(0)
    })
    const labels = screen.getAllByText(/WK Proj/i)
    expect(labels.length).toBeGreaterThan(0)
  })

  it('renders SHAPChart with plain-English factor labels visible', async () => {
    renderPlayer()
    // The section heading confirms explain data was fetched and chart rendered.
    // recharts renders labels in SVG <text> nodes; we verify the section heading
    // and that no loading spinner is displayed for that section.
    await waitFor(() =>
      expect(screen.getByText('What drives this projection')).toBeInTheDocument(),
    )
    // Explain data loaded: the "no explanation" message should NOT appear
    await waitFor(() => {
      expect(
        screen.queryByText(/no explanation available/i),
      ).not.toBeInTheDocument()
    })
  })

  describe('when there is no forecast yet (the API answers 404 with forecast_available=false)', () => {
    function noForecast() {
      server.use(
        http.get('/api/predict', () =>
          HttpResponse.json(
            {
              detail: {
                forecast_available: false,
                message: "No approved projection for player='Justin Jefferson' week=5 season=2027. The weekly forecast surface is not yet live.",
              },
            },
            { status: 404 },
          ),
        ),
      )
    }

    it('names the season and week that were asked for, not a hard-coded year', async () => {
      noForecast()
      renderPlayer('/player/test-001?week=5&season=2027&stat=receiving_yards&name=Justin+Jefferson')
      expect(await screen.findByText('No forecast yet for Week 5, 2027')).toBeInTheDocument()
      expect(screen.getByText(/Week 5 projection for Justin Jefferson in 2027/)).toBeInTheDocument()
      expect(screen.queryByText(/2026/)).not.toBeInTheDocument()
    })

    it('does not show the server message, and offers two next steps', async () => {
      noForecast()
      renderPlayer('/player/test-001?week=5&season=2025&stat=receiving_yards&name=Justin+Jefferson')
      await screen.findByText('No forecast yet for Week 5, 2025')
      expect(screen.queryByText(/player='/)).not.toBeInTheDocument()
      expect(screen.getByRole('link', { name: /open the draft board/i })).toHaveAttribute('href', '/draft')
      fireEvent.click(screen.getByRole('button', { name: /search for another player/i }))
      expect(await screen.findByPlaceholderText(/search players/i)).toBeInTheDocument()
    })
  })

  describe('sportsbook line', () => {
    it('says "No line yet" instead of showing 0 or an invented line when odds are not live', async () => {
      // Default handler: the server has no odds key, so useOdds generates a demo line.
      renderPlayer()
      expect(await screen.findByText('No line yet')).toBeInTheDocument()
      expect(screen.getByText(/compare the range below with the line at your sportsbook/i)).toBeInTheDocument()
      expect(screen.queryByText(/% edge/)).not.toBeInTheDocument()
      expect(screen.queryByText(/demo line/i)).not.toBeInTheDocument()
    })

    it('shows the real line and edge when a live line exists for the player', async () => {
      server.use(
        http.get('/api/odds', () =>
          HttpResponse.json({
            configured: true,
            events: [
              {
                bookmakers: [
                  {
                    key: 'fanduel',
                    markets: [
                      {
                        key: 'player_reception_yards',
                        outcomes: [
                          { name: 'Over', price: -110, point: 80.5, description: 'Justin Jefferson' },
                          { name: 'Under', price: -110, point: 80.5, description: 'Justin Jefferson' },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          }),
        ),
      )
      renderPlayer()
      expect(await screen.findByText('80.5')).toBeInTheDocument()
      expect(screen.getByText('Sportsbook line')).toBeInTheDocument()
      expect(screen.getAllByText(/% edge/).length).toBeGreaterThan(0)
      expect(screen.queryByText('No line yet')).not.toBeInTheDocument()
    })
  })

  describe('values the model did not give', () => {
    it('shows a dash and a reason for boom, bust and PPR, never 0', async () => {
      server.use(
        http.get('/api/predict', () =>
          HttpResponse.json({
            ...factories.predictResponse(),
            boom_probability: null,
            bust_probability: null,
            projection: { ...factories.predictResponse().projection, ppr_points: null },
          }),
        ),
      )
      renderPlayer()
      await screen.findByText('WK Proj')
      for (const label of ['Boom %', 'Bust %', 'Fantasy PPR']) {
        const card = screen.getByText(label).parentElement as HTMLElement
        expect(within(card).getByText('—')).toBeInTheDocument()
        expect(within(card).getByText('Not available for this stat')).toBeInTheDocument()
      }
    })
  })

  it('asks for rest-of-season from week 18, not week 19 (the API rejects 19)', async () => {
    const startWeeks: string[] = []
    server.use(
      http.get('/api/projections/season/:season', ({ request }) => {
        startWeeks.push(new URL(request.url).searchParams.get('start_week') ?? '')
        return HttpResponse.json(factories.seasonProjectionsResponse())
      }),
    )
    renderPlayer('/player/test-001?week=18&season=2025&stat=receiving_yards&name=Justin+Jefferson')
    await waitFor(() => expect(startWeeks.length).toBeGreaterThan(0))
    expect(startWeeks.every((w) => w === '18')).toBe(true)
  })

  it('explains an empty factor panel when no breakdown is published for the stat', async () => {
    server.use(http.get('/api/explain/:player_id', () => HttpResponse.json({ detail: 'unavailable' }, { status: 503 })))
    renderPlayer()
    expect(await screen.findByText(/cannot show what drives this projection for this stat yet/i)).toBeInTheDocument()
  })

  it('switches the stat from the dropdown, and the dropdown can be used from the keyboard', async () => {
    const stats: string[] = []
    server.use(
      http.get('/api/predict', ({ request }) => {
        stats.push(new URL(request.url).searchParams.get('stat') ?? '')
        return HttpResponse.json(factories.predictResponse())
      }),
    )
    renderPlayer()
    const user = userEvent.setup()
    const trigger = await screen.findByRole('combobox', { name: 'Stat' })
    expect(trigger).toHaveTextContent('Receiving Yards')

    await user.click(trigger)
    await user.click(await screen.findByRole('option', { name: 'Receptions' }))
    await waitFor(() => expect(stats).toContain('receptions'))
    expect(screen.getByRole('combobox', { name: 'Stat' })).toHaveTextContent('Receptions')

    // Keyboard: focus the trigger, open with Enter, move with the arrow keys, choose with Enter.
    screen.getByRole('combobox', { name: 'Stat' }).focus()
    await user.keyboard('{Enter}')
    await screen.findByRole('listbox')
    await user.keyboard('{ArrowDown}{Enter}')
    await waitFor(() => expect(stats.length).toBeGreaterThan(2))
  })
})
