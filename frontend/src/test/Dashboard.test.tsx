import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { http, HttpResponse } from 'msw'
import { server } from './handlers'
import { factories } from './factories'
import { SearchProvider } from '../context/SearchContext'
import { Dashboard } from '../pages/Dashboard'

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <SearchProvider>
        <MemoryRouter initialEntries={['/']}>
          <Routes>
            <Route path="/" element={ui} />
          </Routes>
        </MemoryRouter>
      </SearchProvider>
    </QueryClientProvider>,
  )
}

describe('Dashboard', () => {
  it('shows loading spinner while fetching', () => {
    server.use(
      http.get('/api/projections/week/:week', async () => {
        await new Promise((resolve) => setTimeout(resolve, 50))
        return HttpResponse.json(factories.weekProjectionsResponse())
      }),
    )
    const { container } = renderWithProviders(<Dashboard />)
    expect(container.querySelectorAll('.skeleton').length).toBeGreaterThan(0)
  })

  it('renders projection content after data loads', async () => {
    renderWithProviders(<Dashboard />)
    await waitFor(() =>
      expect(screen.getByText('Justin Jefferson')).toBeInTheDocument(),
    )
  })

  it('shows DataStalenessWarning when data_freshness > 24h old', async () => {
    server.use(
      http.get('/api/projections/week/:week', () =>
        HttpResponse.json(factories.weekProjectionsResponse({ stale: true })),
      ),
    )
    renderWithProviders(<Dashboard />)
    await waitFor(() =>
      expect(screen.getByText(/projections may be stale/i)).toBeInTheDocument(),
    )
  })

  it('does NOT show DataStalenessWarning when data is fresh', async () => {
    renderWithProviders(<Dashboard />)
    await waitFor(() =>
      expect(screen.getByText('Justin Jefferson')).toBeInTheDocument(),
    )
    expect(screen.queryByText(/projections may be stale/i)).not.toBeInTheDocument()
  })

  it('clicking Projection column header changes sort direction', async () => {
    renderWithProviders(<Dashboard />)
    await waitFor(() =>
      expect(screen.getByText('Justin Jefferson')).toBeInTheDocument(),
    )
    // Find and click "Projection" header
    const projHeader = screen.getByRole('button', { name: /projection/i })
    fireEvent.click(projHeader)
    // After first click desc→asc, up arrow appears
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /projection.*↑/i })).toBeInTheDocument()
    })
    // Second click toggles direction
    fireEvent.click(screen.getByRole('button', { name: /projection.*↑/i }))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /projection.*↓/i })).toBeInTheDocument()
    })
  })

  it('position toggle WR click removes WR rows when only WR data present', async () => {
    // Override to return only WR players
    server.use(
      http.get('/api/projections/week/:week', () =>
        HttpResponse.json({
          ...factories.weekProjectionsResponse(),
          projections: [
            {
              player_id: 'wr-only',
              player_name: 'WR Only Player',
              position: 'WR',
              team: 'TEST',
              stat: 'receiving_yards',
              projection: 70.0,
              floor: 30.0,
              ceiling: 110.0,
              boom_probability: 0.3,
              bust_probability: 0.2,
              fantasy_projection: 12.0,
              fantasy_floor: 6.0,
              fantasy_ceiling: 20.0,
              model_version: 'v1',
              data_freshness: new Date().toISOString(),
            },
          ],
          count: 1,
        }),
      ),
    )
    renderWithProviders(<Dashboard />)
    await waitFor(() =>
      expect(screen.getByText('WR Only Player')).toBeInTheDocument(),
    )

    // Now toggle WR off — this sets positions to RB/TE/QB which won't match WR player
    // The query will re-run with new positions and we mock empty response
    server.use(
      http.get('/api/projections/week/:week', () =>
        HttpResponse.json({
          ...factories.weekProjectionsResponse(),
          projections: [],
          count: 0,
        }),
      ),
    )
    const wrButton = screen.getByRole('button', { name: 'WR' })
    fireEvent.click(wrButton)
    await waitFor(() =>
      expect(screen.queryByText('WR Only Player')).not.toBeInTheDocument(),
    )
  })
})
