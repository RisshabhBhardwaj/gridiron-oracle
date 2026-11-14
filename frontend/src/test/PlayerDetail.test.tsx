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

// The default /api/season/current handler says it is Week 5 of 2025, so the default
// URL below is the current week, and week=3 is a week that has been played.
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
            <Route path="/" element={<p>Home page</p>} />
            <Route path="/player/:player_id" element={<PlayerDetail />} />
          </Routes>
        </MemoryRouter>
      </SearchProvider>
    </QueryClientProvider>,
  )
}

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

function liveLine() {
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
}

describe('PlayerDetail', () => {
  describe('header', () => {
    it('shows name, position, team and week once the forecast is in', async () => {
      renderPlayer()
      expect(await screen.findByRole('heading', { level: 1, name: /justin jefferson/i })).toBeInTheDocument()
      expect(await screen.findByText('MIN')).toBeInTheDocument()
      expect(screen.getByText('WR')).toBeInTheDocument()
      expect(screen.getByText(/Week 5 · 2025/)).toBeInTheDocument()
    })

    it('marks a week that has been played', async () => {
      renderPlayer('/player/test-001?week=3&season=2025&stat=receiving_yards&name=Justin+Jefferson')
      expect(await screen.findByText(/Week 3 · 2025 · already played/)).toBeInTheDocument()
    })

    it('does not grey the name out (no transparent gradient text)', async () => {
      renderPlayer()
      const heading = await screen.findByRole('heading', { level: 1, name: /justin jefferson/i })
      expect(heading.className).not.toMatch(/text-transparent|bg-clip-text/)
    })

    it('has a slot for availability but shows none (decision 0012 D3)', async () => {
      renderPlayer()
      await screen.findByText('MIN')
      expect(screen.queryByText(/questionable|out for|injur|doubtful/i)).not.toBeInTheDocument()
    })

    it('keeps name, position and team on screen while there is no forecast', async () => {
      noForecast()
      renderPlayer('/player/test-001?week=5&season=2027&stat=receiving_yards&name=Justin+Jefferson')
      expect(await screen.findByText('No forecast yet for Week 5, 2027')).toBeInTheDocument()
      expect(screen.getByRole('heading', { level: 1, name: /justin jefferson/i })).toBeInTheDocument()
      expect(await screen.findByText('MIN')).toBeInTheDocument()
      expect(screen.getByText('WR')).toBeInTheDocument()
    })

    it('goes Home from Back when the page was opened straight from a link', async () => {
      renderPlayer()
      const user = userEvent.setup()
      await user.click(await screen.findByRole('button', { name: /back/i }))
      expect(await screen.findByText('Home page')).toBeInTheDocument()
    })
  })

  describe('the one number', () => {
    it('labels the hero like search does ("Week N · stat") and shows the projection once', async () => {
      renderPlayer()
      const hero = await screen.findByRole('region', { name: 'Week 5 · Receiving Yards' })
      expect(within(hero).getByText('87.4')).toBeInTheDocument()
      // The old tile label is gone, and the figure does not repeat anywhere on the page.
      expect(screen.queryByText(/WK Proj/i)).not.toBeInTheDocument()
      expect(screen.getAllByText('87.4')).toHaveLength(1)
    })

    it('shows the range as "Bad week" and "Great week"', async () => {
      renderPlayer()
      const hero = await screen.findByRole('region', { name: 'Week 5 · Receiving Yards' })
      expect(within(hero).getByText('Bad week')).toBeInTheDocument()
      expect(within(hero).getByText('42.1')).toBeInTheDocument()
      expect(within(hero).getByText('Great week')).toBeInTheDocument()
      expect(within(hero).getByText('142.8')).toBeInTheDocument()
    })

    it('has at most three supporting tiles, none empty', async () => {
      renderPlayer()
      await screen.findByRole('region', { name: 'Week 5 · Receiving Yards' })
      expect(screen.getByText('Fantasy points, PPR')).toBeInTheDocument()
      expect(screen.getByText('Boom chance')).toBeInTheDocument()
      expect(screen.getByText('Bust chance')).toBeInTheDocument()
      expect(screen.queryByText('—')).not.toBeInTheDocument()
    })

    it('leaves out tiles the model did not fill, instead of showing dashes', async () => {
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
      await screen.findByRole('region', { name: 'Week 5 · Receiving Yards' })
      expect(screen.queryByText('—')).not.toBeInTheDocument()
      expect(screen.queryByText('Boom chance')).not.toBeInTheDocument()
      expect(screen.queryByText('Bust chance')).not.toBeInTheDocument()
      expect(screen.queryByText(/Fantasy points/)).not.toBeInTheDocument()
    })

    it('does not repeat the figure as a fantasy tile when the stat is fantasy points', async () => {
      server.use(
        http.get('/api/predict', () =>
          HttpResponse.json({
            ...factories.predictResponse(),
            stat: 'fantasy_ppr',
            projection: { ...factories.predictResponse().projection, fantasy_ppr: 15.2, ppr_points: 15.2 },
          }),
        ),
      )
      renderPlayer('/player/test-001?week=5&season=2025&stat=fantasy_ppr&name=Justin+Jefferson')
      await screen.findByRole('region', { name: /Week 5 · /i })
      expect(screen.getAllByText('15.2')).toHaveLength(1)
      expect(screen.queryByText(/Fantasy points/)).not.toBeInTheDocument()
    })

    it('does not use the word "median" or "typical" for the number', async () => {
      renderPlayer()
      await screen.findByRole('region', { name: 'Week 5 · Receiving Yards' })
      expect(screen.queryByText(/median|typical/i)).not.toBeInTheDocument()
    })
  })

  describe('provenance: a simulation is never a week forecast', () => {
    it.each(['season_simulator_mc', 'playing_time_gaussian_mc', 'scenario_monte_carlo', 'unavailable', ''])(
      'a %j row shows the no-forecast banner, not a Week 5 figure',
      async (method) => {
        server.use(
          http.get('/api/predict', () =>
            HttpResponse.json({ ...factories.predictResponse(), interval_method: method }),
          ),
        )
        renderPlayer()
        expect(await screen.findByText('No forecast yet for Week 5, 2025')).toBeInTheDocument()
        expect(screen.queryByRole('region', { name: /Week 5 · /i })).not.toBeInTheDocument()
        expect(screen.queryByText('87.4')).not.toBeInTheDocument()
        expect(screen.queryByText('Bad week')).not.toBeInTheDocument()
        // No What-If or explain panel on a number we do not stand behind.
        expect(screen.queryByText('What-If Studio')).not.toBeInTheDocument()
        expect(screen.queryByText('What drives this projection')).not.toBeInTheDocument()
      },
    )

    it('shows the simulated figure only after asking, labelled as the preseason simulation', async () => {
      server.use(
        http.get('/api/predict', () =>
          HttpResponse.json({ ...factories.predictResponse(), interval_method: 'season_simulator_mc' }),
        ),
      )
      renderPlayer()
      await screen.findByText('No forecast yet for Week 5, 2025')
      expect(screen.queryByText('87.4')).not.toBeInTheDocument()

      await userEvent.setup().click(screen.getByRole('button', { name: /show the preseason simulation/i }))
      expect(await screen.findByText('87.4')).toBeInTheDocument()
      expect(screen.getByText('Preseason simulation, not updated for games played')).toBeInTheDocument()
      // Still no Week 5 forecast label on it.
      expect(screen.queryByRole('region', { name: /Week 5 · /i })).not.toBeInTheDocument()
    })

    it('offers the rest-of-season simulation for the current week, labelled as a simulation', async () => {
      noForecast()
      renderPlayer()
      await screen.findByText('No forecast yet for Week 5, 2025')
      await userEvent.setup().click(await screen.findByRole('button', { name: /show the preseason simulation/i }))
      expect(await screen.findByText('Rest of season · Receiving Yards')).toBeInTheDocument()
      expect(screen.getByText('Simulation, not a weekly forecast')).toBeInTheDocument()
    })

    it('does not offer a preseason simulation for a week that has been played', async () => {
      noForecast()
      renderPlayer('/player/test-001?week=3&season=2025&stat=receiving_yards&name=Justin+Jefferson')
      await screen.findByText('No forecast yet for Week 3, 2025')
      expect(screen.queryByRole('button', { name: /show the preseason simulation/i })).not.toBeInTheDocument()
    })
  })

  describe('when there is no forecast yet (the API answers 404 with forecast_available=false)', () => {
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

    it('treats a 404 without forecast_available=false as a real failure with Retry', async () => {
      server.use(http.get('/api/predict', () => HttpResponse.json({ detail: 'Not Found' }, { status: 404 })))
      renderPlayer()
      expect(await screen.findByRole('button', { name: 'Retry' }, { timeout: 5000 })).toBeInTheDocument()
      expect(screen.queryByText(/No forecast yet/)).not.toBeInTheDocument()
    })
  })

  describe('failures', () => {
    it('a 500 from /predict is an error with Retry, and Retry asks again', async () => {
      let calls = 0
      server.use(
        http.get('/api/predict', () => {
          calls += 1
          // The first request and its two automatic retries fail; the visitor's Retry succeeds.
          return calls <= 3
            ? HttpResponse.json({ detail: 'boom' }, { status: 500 })
            : HttpResponse.json(factories.predictResponse())
        }),
      )
      renderPlayer()
      // /predict retries a failure twice (1s, 2s back-off) before it gives up.
      const retry = await screen.findByRole('button', { name: 'Retry' }, { timeout: 5000 })
      expect(screen.queryByText(/No forecast yet/)).not.toBeInTheDocument()
      fireEvent.click(retry)
      expect(await screen.findByRole('region', { name: 'Week 5 · Receiving Yards' })).toBeInTheDocument()
      expect(calls).toBe(4)
    })
  })

  describe('loading', () => {
    it('shows a page-shaped skeleton and the name, never the no-forecast state, until the answer arrives', async () => {
      let release: () => void = () => {}
      const gate = new Promise<void>((resolve) => { release = resolve })
      server.use(
        http.get('/api/predict', async () => {
          await gate
          return HttpResponse.json(factories.predictResponse())
        }),
      )
      renderPlayer()
      expect(await screen.findByRole('status', { name: 'Loading projection' })).toBeInTheDocument()
      expect(screen.getByRole('heading', { level: 1, name: /justin jefferson/i })).toBeInTheDocument()
      expect(screen.queryByText(/No forecast yet/)).not.toBeInTheDocument()
      expect(screen.queryByText('Bad week')).not.toBeInTheDocument()

      release()
      expect(await screen.findByRole('region', { name: 'Week 5 · Receiving Yards' })).toBeInTheDocument()
      expect(screen.queryByRole('status', { name: 'Loading projection' })).not.toBeInTheDocument()
    })

    it('waits for the current week when the link has none, and asks for that week, not a placeholder', async () => {
      const asked: string[] = []
      server.use(
        http.get('/api/predict', ({ request }) => {
          const u = new URL(request.url)
          asked.push(`${u.searchParams.get('season')}/${u.searchParams.get('week')}`)
          return HttpResponse.json(factories.predictResponse())
        }),
      )
      renderPlayer('/player/test-001?stat=receiving_yards&name=Justin+Jefferson')
      await screen.findByRole('region', { name: 'Week 5 · Receiving Yards' })
      expect(asked).toEqual(['2025/5'])
    })

    it('falls back to the current week for a week in the link that is not a week', async () => {
      const asked: string[] = []
      server.use(
        http.get('/api/predict', ({ request }) => {
          asked.push(new URL(request.url).searchParams.get('week') ?? '')
          return HttpResponse.json(factories.predictResponse())
        }),
      )
      renderPlayer('/player/test-001?week=banana&season=2025&stat=receiving_yards&name=Justin+Jefferson')
      await screen.findByRole('region', { name: 'Week 5 · Receiving Yards' })
      expect(asked).toEqual(['5'])
    })
  })

  describe('staleness banner', () => {
    it('does not warn from the feature-build timestamp: a played week is not "stale"', async () => {
      server.use(
        http.get('/api/predict', () => HttpResponse.json(factories.predictResponse({ stale: true }))),
      )
      renderPlayer()
      await screen.findByRole('region', { name: 'Week 5 · Receiving Yards' })
      expect(screen.queryByText(/data last updated/i)).not.toBeInTheDocument()
      expect(screen.queryByText(/may be stale/i)).not.toBeInTheDocument()
    })
  })

  describe('explain panel', () => {
    it('shows the factors when the breakdown is published', async () => {
      renderPlayer()
      expect(await screen.findByText('What drives this projection')).toBeInTheDocument()
      expect(await screen.findByText('Simplified explanation, not the full breakdown')).toBeInTheDocument()
      expect(screen.queryByText(/no explanation available/i)).not.toBeInTheDocument()
      expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    })

    it.each([
      ['a 503', () => HttpResponse.json({ detail: 'unavailable' }, { status: 503 })],
      ['a 500', () => HttpResponse.json({ detail: 'boom' }, { status: 500 })],
      ['a network failure', () => HttpResponse.error()],
    ])('%s is an error with Retry, never "not available"', async (_name, respond) => {
      server.use(http.get('/api/explain/:player_id', respond))
      renderPlayer()
      const alert = await screen.findByRole('alert', {}, { timeout: 4000 })
      expect(within(alert).getByRole('button', { name: 'Retry' })).toBeInTheDocument()
      expect(screen.queryByText(/do not have a breakdown/i)).not.toBeInTheDocument()
    })

    it('Retry asks again and shows the breakdown when it succeeds', async () => {
      let calls = 0
      server.use(
        http.get('/api/explain/:player_id', () => {
          calls += 1
          // The first request and its one automatic retry fail; the visitor's Retry succeeds.
          return calls <= 2
            ? HttpResponse.json({ detail: 'unavailable' }, { status: 503 })
            : HttpResponse.json(factories.explainResponse())
        }),
      )
      renderPlayer()
      const alert = await screen.findByRole('alert', {}, { timeout: 4000 })
      fireEvent.click(within(alert).getByRole('button', { name: 'Retry' }))
      await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument())
      expect(calls).toBe(3)
    })

    it('a 404 ("not published") is a plain message with no Retry', async () => {
      server.use(http.get('/api/explain/:player_id', () => HttpResponse.json({ detail: 'none' }, { status: 404 })))
      renderPlayer()
      expect(await screen.findByText(/do not have a breakdown for this stat yet/i)).toBeInTheDocument()
      expect(screen.queryByRole('alert')).not.toBeInTheDocument()
      expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument()
    })
  })

  describe('sportsbook line', () => {
    it('shows no line, no empty tile and no vendor chip when odds are not live', async () => {
      renderPlayer()
      await screen.findByRole('region', { name: 'Week 5 · Receiving Yards' })
      await waitFor(() => expect(screen.queryByText('No line yet')).not.toBeInTheDocument())
      expect(screen.queryByText(/sportsbook line/i)).not.toBeInTheDocument()
      expect(screen.queryByText(/% edge/)).not.toBeInTheDocument()
      expect(screen.queryByText(/live line/i)).not.toBeInTheDocument()
      expect(screen.queryByText(/odds api/i)).not.toBeInTheDocument()
    })

    it('shows the real line, and how far the projection is from it, for the current week', async () => {
      liveLine()
      renderPlayer()
      expect(await screen.findByText('80.5')).toBeInTheDocument()
      expect(screen.getByText(/Sportsbook line/)).toBeInTheDocument()
      expect(screen.getByText(/Our projection is .*% above the line/)).toBeInTheDocument()
    })

    it('shows no line beside a week that has been played, even if the book has one', async () => {
      let oddsRequests = 0
      liveLine()
      server.use(http.get('/api/odds', () => { oddsRequests += 1; return HttpResponse.json({ configured: true, events: [] }) }))
      renderPlayer('/player/test-001?week=3&season=2025&stat=receiving_yards&name=Justin+Jefferson')
      await screen.findByRole('region', { name: 'Week 3 · Receiving Yards' })
      await screen.findByText('What drives this projection')
      expect(oddsRequests).toBe(0)
      expect(screen.queryByText('80.5')).not.toBeInTheDocument()
      expect(screen.queryByText(/Sportsbook line/)).not.toBeInTheDocument()
    })
  })

  describe('What-If Studio', () => {
    it('shows the change beside the sliders once one moves, and Reset clears it', async () => {
      renderPlayer()
      const slider = await screen.findByLabelText('Wind Speed')
      expect(screen.getByText(/move a slider to see how the projection changes/i)).toBeInTheDocument()

      slider.focus()
      await userEvent.setup().keyboard('{ArrowRight}')
      expect(await screen.findByText('+3.8', {}, { timeout: 3000 })).toBeInTheDocument()
      expect(screen.getByText(/from 87.4 to/)).toBeInTheDocument()

      fireEvent.click(screen.getByRole('button', { name: 'Reset' }))
      await waitFor(() => expect(screen.queryByText('+3.8')).not.toBeInTheDocument())
      expect(screen.getByText(/move a slider to see how the projection changes/i)).toBeInTheDocument()
    })

    it('says so, with Retry, when the scenario cannot be computed', async () => {
      server.use(http.post('/api/scenario', () => HttpResponse.json({ detail: 'boom' }, { status: 500 })))
      renderPlayer()
      const slider = await screen.findByLabelText('Wind Speed')
      slider.focus()
      await userEvent.setup().keyboard('{ArrowRight}')
      const retry = await screen.findAllByRole('button', { name: 'Retry' }, { timeout: 3000 })
      expect(retry.length).toBeGreaterThan(0)
    })

    it('is not on the page when there is no forecast to move', async () => {
      noForecast()
      renderPlayer()
      await screen.findByText('No forecast yet for Week 5, 2025')
      expect(screen.queryByText('What-If Studio')).not.toBeInTheDocument()
    })

    it('says nothing about weights anywhere on the page (decision 0012 D7)', async () => {
      server.use(
        http.get('/api/predict', () =>
          HttpResponse.json({ ...factories.predictResponse(), weighting: 'user', weights_reason: null }),
        ),
      )
      renderPlayer()
      await screen.findByText('What-If Studio')
      expect(screen.queryByText(/weight/i)).not.toBeInTheDocument()
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

  describe('scoring and settings labels', () => {
    it('says nothing extra for a default response and labels the fantasy tile PPR', async () => {
      renderPlayer()
      await screen.findByRole('region', { name: 'Week 5 · Receiving Yards' })
      expect(screen.queryByRole('note')).not.toBeInTheDocument()
      expect(screen.getByText('Fantasy points, PPR')).toBeInTheDocument()
    })

    it('marks user-adjusted numbers as a what-if, without the word "weights"', async () => {
      server.use(
        http.get('/api/predict', () =>
          HttpResponse.json({ ...factories.predictResponse(), weighting: 'user', weights_reason: null }),
        ),
      )
      renderPlayer()
      const note = await screen.findByRole('note')
      expect(note).toHaveTextContent(/reflect your saved settings/i)
      expect(note).not.toHaveTextContent(/weight/i)
    })

    it('names the scoring on the fantasy tile when it is not PPR', async () => {
      server.use(
        http.get('/api/predict', () =>
          HttpResponse.json({ ...factories.predictResponse(), fantasy_scoring: 'half_ppr' }),
        ),
      )
      renderPlayer()
      expect(await screen.findByText('Fantasy points, half-PPR')).toBeInTheDocument()
    })
  })
})
