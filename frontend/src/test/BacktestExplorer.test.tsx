import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BacktestExplorer } from '../pages/BacktestExplorer'
import { server } from './handlers'

function renderBacktest() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/backtest']}>
        <Routes>
          <Route path="/backtest" element={<BacktestExplorer />} />
          <Route path="/projections" element={<p>projections page</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('BacktestExplorer', () => {
  it('leads with a plain verdict and a link to the data', async () => {
    renderBacktest()
    const verdict = await screen.findByRole('region', { name: 'Verdict' })
    expect(verdict).toHaveTextContent(/Better than the simple method in 7 of 8 player-stat seasons/)
    const link = within(verdict).getByRole('link', { name: /Download all the numbers/ })
    expect(link).toHaveAttribute('href', '/api/backtest/table.csv')
  })

  it('gives a percent only over yards and fantasy points, never blended with count stats', async () => {
    renderBacktest()
    const verdict = await screen.findByRole('region', { name: 'Verdict' })
    // The first sentence is a count, with no percent.
    expect(verdict).toHaveTextContent(/Better than the simple method in 7 of 8 player-stat seasons\.On yards/)
    expect(verdict).toHaveTextContent(/On yards and fantasy points, our average miss is \d+\.\d% smaller\./)
    expect(verdict).not.toHaveTextContent(/on average\./)
  })

  it('shows release-measured WR 2025 numbers: 23.24 vs 25.17', async () => {
    renderBacktest()
    const table = await screen.findByRole('table')
    const wr2025 = within(table).getAllByRole('row').find((r) => /2025/.test(r.textContent ?? '') && /WR/.test(r.textContent ?? ''))
    expect(wr2025).toBeDefined()
    expect(wr2025).toHaveTextContent('23.24')
    expect(wr2025).toHaveTextContent('25.17')
  })

  it('has no developer-facing or unmeasured metrics', async () => {
    renderBacktest()
    await screen.findByRole('table')
    const text = document.body.textContent ?? ''
    for (const banned of [/csv\b(?!\))/i, /pnl/i, /brier/i, /crps/i, /sharpe/i, /drawdown/i, /calibration/i, /rmse/i]) {
      expect(text).not.toMatch(banned)
    }
    expect(screen.getByText(/We don't yet publish how often outcomes land inside our range/)).toBeInTheDocument()
  })

  it('labels the chart and its axes for assistive tech', async () => {
    renderBacktest()
    await screen.findByRole('table')
    const chart = screen.getByRole('img', { name: /percent our projections beat the simple method/i })
    expect(chart).toHaveAccessibleName(/Receiving Yards/)
    expect(screen.getByText('Above zero means our projections were closer than the simple method.')).toBeInTheDocument()
  })

  it('explains the cohort rule and the baselines in plain words', async () => {
    renderBacktest()
    await screen.findByRole('table')
    expect(screen.getByText(/Players with at least one earlier game that season/)).toBeInTheDocument()
    expect(screen.getByText(/Marcel/, { selector: 'dt' })).toBeInTheDocument()
  })

  it('disables position chips that have no rows for the chosen stat', async () => {
    const user = userEvent.setup()
    renderBacktest()
    await screen.findByRole('table')
    // Receiving yards has no QB rows
    expect(screen.getByRole('button', { name: 'QB, no results for this stat' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'WR' })).toBeEnabled()
    await user.click(screen.getByRole('combobox', { name: 'Stat:' }))
    await user.click(await screen.findByRole('option', { name: 'Rushing Yards' }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'WR, no results for this stat' })).toBeDisabled())
    expect(screen.getByRole('button', { name: 'QB' })).toBeEnabled()
  })

  it('shows a designed unavailable state, with the next step, on a 503', async () => {
    server.use(
      http.get('/api/backtest', () =>
        HttpResponse.json({ detail: { message: 'x', code: 'evidence_unavailable' } }, { status: 503 }),
      ),
    )
    renderBacktest()
    expect(await screen.findByText("Past results aren't available right now")).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /this season's projections/ })).toHaveAttribute('href', '/projections')
    expect(document.body.textContent).not.toMatch(/evidence_unavailable/)
  })
})
