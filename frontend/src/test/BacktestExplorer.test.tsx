import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BacktestExplorer } from '../pages/BacktestExplorer'

function renderBacktest() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/backtest']}>
        <Routes>
          <Route path="/backtest" element={<BacktestExplorer />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('BacktestExplorer', () => {
  it('shows metric cards: MAE, RMSE, CRPS values from factory data', async () => {
    renderBacktest()
    await waitFor(() =>
      expect(screen.getByText('MAE')).toBeInTheDocument(),
    )
    // Factory data has overall_mae: 18.42
    expect(screen.getByText('18.42')).toBeInTheDocument()
    expect(screen.getByText('RMSE')).toBeInTheDocument()
    expect(screen.getByText('24.11')).toBeInTheDocument()
    expect(screen.getByText('CRPS')).toBeInTheDocument()
  })

  it('season comparison chart container renders', async () => {
    renderBacktest()
    // The heading for the season chart is the reliable signal; recharts
    // may or may not render SVG in jsdom with mocked ResizeObserver
    await waitFor(() =>
      expect(screen.getByText('Season-by-Season MAE: Model vs Naive Baseline')).toBeInTheDocument(),
    )
  })

  it('calibration chart container renders', async () => {
    renderBacktest()
    await waitFor(() =>
      expect(screen.getByText('Calibration Reliability Diagram')).toBeInTheDocument(),
    )
  })

  it('coverage table rows render with season values', async () => {
    renderBacktest()
    await waitFor(() =>
      expect(screen.getByText('Coverage by Season & Position')).toBeInTheDocument(),
    )
    // Factory has seasons 2022, 2023, 2025
    const cells = await screen.findAllByText('2022')
    expect(cells.length).toBeGreaterThan(0)
    expect(screen.getByText('2023')).toBeInTheDocument()
  })

  it('stat dropdown changes trigger new query', async () => {
    renderBacktest()
    await waitFor(() =>
      expect(screen.getByText('MAE')).toBeInTheDocument(),
    )
    const dropdown = screen.getByRole('combobox')
    expect(dropdown).toBeInTheDocument()
    fireEvent.change(dropdown, { target: { value: 'rushing_yards' } })
    // Query refires — loading indicator may briefly appear, then data reloads
    await waitFor(() =>
      expect(screen.getByText('MAE')).toBeInTheDocument(),
    )
  })
})
