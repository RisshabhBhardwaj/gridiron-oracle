import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { http, HttpResponse } from 'msw'
import { SearchProvider } from '../context/SearchContext'
import { CurrentSeason } from '../pages/CurrentSeason'
import { PlayerDetail } from '../pages/PlayerDetail'
import { PlayerSearch } from '../components/PlayerSearch'
import { server } from './handlers'
import { factories } from './factories'

/** Serve weekly rows as a real forecast (any non-simulator interval_method). */
function serveForecastRows() {
  server.use(
    http.get('/api/projections/season/:season/weeks/:week', () => {
      const body = factories.seasonWeekProjectionsResponse()
      body.projections = body.projections.map((p) => ({ ...p, interval_method: 'causal_oof_conformal_90' }))
      return HttpResponse.json(body)
    }),
  )
}

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
          <PlayerSearch />
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
    serveForecastRows()
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
    serveForecastRows()
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
    serveForecastRows()
    renderWithProviders(<CurrentSeason />)
    fireEvent.click(await screen.findByText('Justin Jefferson'))
    const here = await screen.findByTestId('here')
    expect(here).toHaveTextContent(/^\/player\/[^?]+\?week=\d+&season=\d+&stat=\w+&name=Justin%20Jefferson$/)
  })

  describe('Weekly tab provenance', () => {
    it('says there is no forecast and shows no number when the rows are the preseason simulation', async () => {
      renderWithProviders(<CurrentSeason />)
      expect(await screen.findByText('No forecast yet for Week 5, 2025')).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /see season-long projections/i })).toBeInTheDocument()
      expect(screen.getByRole('link', { name: /open the draft board/i })).toHaveAttribute('href', '/draft')
      expect(screen.queryByText('Justin Jefferson')).not.toBeInTheDocument()
      expect(screen.queryByText('19.8')).not.toBeInTheDocument()
    })

    it('labels the section and every card when the visitor asks for the simulation', async () => {
      renderWithProviders(<CurrentSeason />)
      fireEvent.click(await screen.findByRole('button', { name: /show these numbers anyway/i }))
      expect(await screen.findByText('Justin Jefferson')).toBeInTheDocument()
      const label = 'Preseason simulation, not updated for games played'
      // One heading plus one line on each of the two cards.
      expect(screen.getAllByText(label)).toHaveLength(3)
      expect(screen.queryByText(/Week 5 · Fantasy Points/)).not.toBeInTheDocument()
      fireEvent.click(screen.getByRole('button', { name: /hide these numbers/i }))
      expect(screen.queryByText('Justin Jefferson')).not.toBeInTheDocument()
    })

    it.each(['playing_time_gaussian_mc', 'scenario_monte_carlo', 'unavailable', 'posterior_samples'])(
      'treats %s rows as not a forecast too',
      async (method) => {
        server.use(
          http.get('/api/projections/season/:season/weeks/:week', () => {
            const body = factories.seasonWeekProjectionsResponse()
            body.projections = body.projections.map((p) => ({ ...p, interval_method: method }))
            return HttpResponse.json(body)
          }),
        )
        renderWithProviders(<CurrentSeason />)
        expect(await screen.findByText('No forecast yet for Week 5, 2025')).toBeInTheDocument()
        expect(screen.queryByText('Justin Jefferson')).not.toBeInTheDocument()
        fireEvent.click(screen.getByRole('button', { name: /show these numbers anyway/i }))
        await screen.findByText('Justin Jefferson')
        const own = /unavailable|posterior_samples/.test(method)
          ? 'Not a verified weekly forecast'
          : 'Simulation, not a weekly forecast'
        expect(screen.getAllByText(own)).toHaveLength(3)
        expect(screen.queryByText(/Preseason simulation/)).not.toBeInTheDocument()
      },
    )

    it('shows the notice when the start week is changed to Week 3 and the endpoint answers with Monte Carlo rows', async () => {
      const startWeeks: string[] = []
      server.use(
        http.get('/api/projections/season/:season/weeks/:week', ({ request }) => {
          const sw = new URL(request.url).searchParams.get('start_week') ?? ''
          startWeeks.push(sw)
          const body = factories.seasonWeekProjectionsResponse()
          const method = sw === '1' ? 'causal_oof_conformal_90' : 'playing_time_gaussian_mc'
          body.projections = body.projections.map((p) => ({ ...p, interval_method: method }))
          return HttpResponse.json(body)
        }),
      )
      const user = userEvent.setup()
      renderWithProviders(<CurrentSeason />)
      // start_week=1 serves real-forecast rows here: no notice.
      expect(await screen.findByText('Justin Jefferson')).toBeInTheDocument()
      expect(screen.queryByText(/No forecast yet/)).not.toBeInTheDocument()
      // The start week lives on the Season-Long tab but also drives the Weekly request.
      await user.click(screen.getByText('Season-Long'))
      await user.click(await screen.findByRole('combobox', { name: 'Start Week:' }))
      await user.click(await screen.findByRole('option', { name: 'From Week 3' }))
      await user.click(screen.getByText('Weekly'))
      expect(await screen.findByText('No forecast yet for Week 5, 2025')).toBeInTheDocument()
      expect(startWeeks).toContain('3')
      expect(screen.queryByText('Justin Jefferson')).not.toBeInTheDocument()
    })

    it('shows no banner and no simulation label when the rows are a real forecast', async () => {
      serveForecastRows()
      renderWithProviders(<CurrentSeason />)
      expect(await screen.findByText('Justin Jefferson')).toBeInTheDocument()
      expect(screen.queryByText(/No forecast yet/)).not.toBeInTheDocument()
      expect(screen.queryByText(/Preseason simulation/)).not.toBeInTheDocument()
      expect(screen.queryByRole('button', { name: /these numbers/i })).not.toBeInTheDocument()
      expect(screen.getAllByText('Week 5 · Fantasy Points (PPR)').length).toBeGreaterThan(0)
    })

    it('offers the season-long tab as the next step', async () => {
      renderWithProviders(<CurrentSeason />)
      fireEvent.click(await screen.findByRole('button', { name: /see season-long projections/i }))
      expect(await screen.findByText('Start Week:')).toBeInTheDocument()
    })
  })

  describe('search shows what the page it opens shows', () => {
    it('shows no number for a simulation row, and the player page then also says no forecast', async () => {
      server.use(
        http.get('/api/predict', () =>
          HttpResponse.json({ detail: { forecast_available: false, message: 'none' } }, { status: 404 }),
        ),
      )
      render(
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <SearchProvider>
            <MemoryRouter initialEntries={['/projections']}>
              <PlayerSearch />
              <Routes>
                <Route path="/projections" element={<CurrentSeason />} />
                <Route path="/player/:player_id" element={<PlayerDetail />} />
              </Routes>
            </MemoryRouter>
          </SearchProvider>
        </QueryClientProvider>,
      )
      await screen.findByText('No forecast yet for Week 5, 2025')
      fireEvent.keyDown(window, { key: 'k', metaKey: true })
      const option = await screen.findByRole('option', { name: /Justin Jefferson/ })
      // "No forecast yet" is said once, under the search box; the row itself shows no figure.
      expect(screen.getByText('No Week 5 forecast yet')).toBeInTheDocument()
      expect(option).not.toHaveTextContent('19.8')
      fireEvent.click(option)
      expect(await screen.findByText('No forecast yet for Week 5, 2025')).toBeInTheDocument()
    })

    it('shows the same number and label as the Weekly card for a real forecast row', async () => {
      serveForecastRows()
      renderWithProviders(<CurrentSeason />)
      await screen.findByText('Justin Jefferson')
      fireEvent.keyDown(window, { key: 'k', metaKey: true })
      const option = await screen.findByRole('option', { name: /Justin Jefferson/ })
      expect(option).toHaveTextContent('19.8')
      expect(option).toHaveTextContent('Week 5 · Fantasy Points (PPR)')
    })
  })

  describe('Games tab', () => {
    function openGames() {
      renderWithProviders(<CurrentSeason />)
      fireEvent.click(screen.getByText('Games'))
    }

    it('turns the 404 "not ready" answer into a calm state with a next step, not an error', async () => {
      server.use(
        http.get('/api/team-games/:season/:week', () =>
          HttpResponse.json({ detail: 'No team-game predictions for season=2025 week=5.' }, { status: 404 }),
        ),
      )
      openGames()
      expect(await screen.findByText(/Matchup forecasts for Week 5 aren.t ready yet/)).toBeInTheDocument()
      expect(screen.getByText(/Season win totals are above/)).toBeInTheDocument()
      expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument()
      // The win totals still render.
      expect(await screen.findByText('11.8')).toBeInTheDocument()
      fireEvent.click(screen.getByRole('button', { name: /see season-long projections/i }))
      expect(await screen.findByText('Start Week:')).toBeInTheDocument()
    })

    it('still shows the error state, with Retry, for a server failure', async () => {
      server.use(
        http.get('/api/team-games/:season/:week', () =>
          HttpResponse.json({ detail: 'boom' }, { status: 500 }),
        ),
      )
      openGames()
      expect(await screen.findByText('Something went wrong')).toBeInTheDocument()
      expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
      expect(screen.queryByText(/aren.t ready yet/)).not.toBeInTheDocument()
    })

    it('does not retry a 404', async () => {
      let calls = 0
      server.use(
        http.get('/api/team-games/:season/:week', () => {
          calls += 1
          return HttpResponse.json({ detail: 'none' }, { status: 404 })
        }),
      )
      openGames()
      await screen.findByText(/aren.t ready yet/)
      expect(calls).toBe(1)
    })
  })
})

describe('CurrentSeason: loading, not ready and failed are different states', () => {
  const emptyWeek = () =>
    http.get('/api/projections/season/:season/weeks/:week', () =>
      HttpResponse.json({ ...factories.seasonWeekProjectionsResponse(), count: 0, projections: [] }),
    )

  it('shows no empty state and no guessed week while the current week is still loading', async () => {
    let release: () => void = () => {}
    const gate = new Promise<void>((resolve) => { release = resolve })
    server.use(
      http.get('/api/season/current', async () => {
        await gate
        return HttpResponse.json({ season: 2025, week: 5 })
      }),
      emptyWeek(),
    )
    renderWithProviders(<CurrentSeason />)
    // Loading: a skeleton, never the empty or not-ready notices, and no "Week 1" fallback.
    expect(screen.getByText('Loading player projections')).toBeInTheDocument()
    expect(screen.queryByText(/No player projections/)).not.toBeInTheDocument()
    expect(screen.queryByText(/aren.t ready yet/)).not.toBeInTheDocument()
    expect(screen.queryByText('Week 1')).not.toBeInTheDocument()
    expect(screen.queryByText(/2026 Season/)).not.toBeInTheDocument()
    release()
    // Once settled, empty data is a not-ready state with a next step.
    expect(await screen.findByText(/Week 5 projections aren.t ready yet/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Check again' })).toBeInTheDocument()
  })

  it('treats a 404 and an empty list as "not ready", never as an error', async () => {
    server.use(
      http.get('/api/projections/season/:season/weeks/:week', () =>
        HttpResponse.json({ detail: 'none' }, { status: 404 }),
      ),
    )
    renderWithProviders(<CurrentSeason />)
    expect(await screen.findByText(/Week 5 projections aren.t ready yet/)).toBeInTheDocument()
    expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /see season-long projections/i })).toBeInTheDocument()
  })

  it('shows one alert with Retry for a server failure, and Retry asks again', async () => {
    let calls = 0
    server.use(
      http.get('/api/projections/season/:season/weeks/:week', () => {
        calls += 1
        return HttpResponse.json({ detail: 'Traceback /srv/x.py' }, { status: 500 })
      }),
    )
    renderWithProviders(<CurrentSeason />)
    expect(await screen.findByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getAllByRole('alert')).toHaveLength(1)
    expect(screen.queryByText(/Traceback/)).not.toBeInTheDocument()
    expect(screen.queryByText(/aren.t ready yet/)).not.toBeInTheDocument()
    const before = calls
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(calls).toBeGreaterThan(before))
  })

  it('shows an error with Retry, and asks for nothing else, when the current week cannot be loaded', async () => {
    let weeklyCalls = 0
    server.use(
      http.get('/api/season/current', () => HttpResponse.json({ detail: 'boom' }, { status: 500 })),
      http.get('/api/projections/season/:season/weeks/:week', () => {
        weeklyCalls += 1
        return HttpResponse.json(factories.seasonWeekProjectionsResponse())
      }),
    )
    renderWithProviders(<CurrentSeason />)
    expect(await screen.findByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
    expect(weeklyCalls).toBe(0)
    expect(screen.queryByText(/No player projections/)).not.toBeInTheDocument()
  })

  it('on the Games tab, a 404 for win totals is not ready and a 500 is an error', async () => {
    server.use(
      http.get('/api/projections/season/:season/team-wins', () =>
        HttpResponse.json({ detail: 'none' }, { status: 404 }),
      ),
    )
    const { unmount } = renderWithProviders(<CurrentSeason />)
    fireEvent.click(screen.getByText('Games'))
    expect(await screen.findByText(/Season win totals aren.t ready yet/)).toBeInTheDocument()
    expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument()
    unmount()

    server.use(
      http.get('/api/projections/season/:season/team-wins', () =>
        HttpResponse.json({ detail: 'boom' }, { status: 500 }),
      ),
    )
    renderWithProviders(<CurrentSeason />)
    fireEvent.click(screen.getByText('Games'))
    expect(await screen.findByText('Something went wrong')).toBeInTheDocument()
    expect(screen.queryByText(/win totals aren.t ready yet/)).not.toBeInTheDocument()
  })

  describe('injury status "Unknown"', () => {
    function serveStatus(status: (i: number) => string | undefined) {
      server.use(
        http.get('/api/projections/season/:season/weeks/:week', () => {
          const body = factories.seasonWeekProjectionsResponse()
          body.projections = body.projections.map((p, i) => ({
            ...p,
            interval_method: 'causal_oof_conformal_90',
            availability_status: status(i),
          }))
          return HttpResponse.json(body)
        }),
      )
    }

    it('says so once for the whole list, not on every card', async () => {
      serveStatus(() => 'Unknown')
      renderWithProviders(<CurrentSeason />)
      await screen.findByText('Justin Jefferson')
      expect(screen.getAllByText('Injury status not checked for this week')).toHaveLength(1)
    })

    it('says nothing for any other status or when the field is absent', async () => {
      serveStatus((i) => (i === 0 ? 'Questionable' : undefined))
      renderWithProviders(<CurrentSeason />)
      await screen.findByText('Justin Jefferson')
      expect(screen.queryByText(/Injury status not checked/)).not.toBeInTheDocument()
      expect(screen.queryByText(/Questionable/)).not.toBeInTheDocument()
    })
  })
})
