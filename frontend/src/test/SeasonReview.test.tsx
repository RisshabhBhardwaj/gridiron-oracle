import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { SeasonReview } from '../pages/SeasonReview'
import { server } from './handlers'

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/season']}>
        <Routes>
          <Route path="/season" element={ui} />
          <Route path="/projections" element={<p>projections page</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('SeasonReview', () => {
  it('leads with a verdict for the latest season and links to the data', async () => {
    renderWithProviders(<SeasonReview />)
    expect(screen.getByText('Season Review')).toBeInTheDocument()
    expect(screen.getByText(/Past results only/i)).toBeInTheDocument()
    const verdict = await screen.findByRole('region', { name: 'Verdict' })
    // Receiving yards 2025: WR, TE, RB all beat the simple method
    expect(verdict).toHaveTextContent('Better than the simple method in all 3 positions')
    expect(within(verdict).getByRole('link', { name: /Download all the numbers/ })).toHaveAttribute('href', '/api/backtest/table.csv')
    expect(screen.getByRole('combobox', { name: 'Season:' })).toHaveTextContent('2025 Season')
  })

  it('shows WR receiving yards 2025 as 23.24 vs 25.17 with an up arrow word, not colour alone', async () => {
    renderWithProviders(<SeasonReview />)
    await screen.findByRole('region', { name: 'Verdict' })
    const table = screen.getByRole('table')
    const wr = within(table).getAllByRole('row').find((r) => /WR/.test(r.textContent ?? ''))
    expect(wr).toHaveTextContent('23.24')
    expect(wr).toHaveTextContent('25.17')
    expect(wr).toHaveTextContent(/better/i)
  })

  it('disables position chips with no rows and shows no stale-data or unmeasured metrics', async () => {
    renderWithProviders(<SeasonReview />)
    await screen.findByRole('region', { name: 'Verdict' })
    expect(screen.getByRole('button', { name: 'QB, no results for this stat' })).toBeDisabled()
    const text = document.body.textContent ?? ''
    for (const banned of [/crps/i, /brier/i, /rmse/i, /calibration/i, /stale/i]) expect(text).not.toMatch(banned)
  })

  it('allows changing season and stat', async () => {
    const user = userEvent.setup()
    renderWithProviders(<SeasonReview />)
    await screen.findByRole('region', { name: 'Verdict' })

    await user.click(screen.getByRole('combobox', { name: 'Season:' }))
    await user.click(await screen.findByRole('option', { name: '2024 Season' }))
    expect(screen.getByRole('combobox', { name: 'Season:' })).toHaveTextContent('2024 Season')

    await user.click(screen.getByRole('combobox', { name: 'Stat:' }))
    await user.click(await screen.findByRole('option', { name: 'Rushing Yards' }))
    await waitFor(() => expect(screen.getByRole('combobox', { name: 'Stat:' })).toHaveTextContent('Rushing Yards'))
  })

  it('shows a designed unavailable state on a 503', async () => {
    server.use(
      http.get('/api/backtest', () =>
        HttpResponse.json({ detail: { message: 'x', code: 'evidence_unavailable' } }, { status: 503 }),
      ),
    )
    renderWithProviders(<SeasonReview />)
    expect(await screen.findByText("Past results aren't available right now")).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument()
  })
})
