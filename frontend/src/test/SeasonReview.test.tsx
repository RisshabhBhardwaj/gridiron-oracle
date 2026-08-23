import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { SeasonReview } from '../pages/SeasonReview'

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/season']}>
        <Routes>
          <Route path="/season" element={ui} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('SeasonReview', () => {
  it('renders season review header and ground truth data floor notice', async () => {
    renderWithProviders(<SeasonReview />)
    expect(screen.getByText('Season Review')).toBeInTheDocument()
    expect(screen.getByText(/Evaluation Data Floor/i)).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByText('Mean Absolute Error (MAE)')).toBeInTheDocument()
      expect(screen.getByText('Root Mean Squared Error (RMSE)')).toBeInTheDocument()
      expect(screen.getByText(/Positional Performance Audit/i)).toBeInTheDocument()
    })
  })

  it('allows changing target stat and season', async () => {
    renderWithProviders(<SeasonReview />)

    await waitFor(() => {
      expect(screen.getByText('Mean Absolute Error (MAE)')).toBeInTheDocument()
    })

    const seasonSelect = screen.getByDisplayValue('2025 Season')
    fireEvent.change(seasonSelect, { target: { value: '2024' } })

    expect(screen.getByDisplayValue('2024 Season')).toBeInTheDocument()
  })
})
