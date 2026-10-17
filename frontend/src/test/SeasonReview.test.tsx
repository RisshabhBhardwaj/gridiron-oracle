import { describe, it, expect } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
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
  it('renders season review header and past-results notice in plain words', async () => {
    renderWithProviders(<SeasonReview />)
    expect(screen.getByText('Season Review')).toBeInTheDocument()
    expect(screen.getByText(/Past results only/i)).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByText('Average miss')).toBeInTheDocument()
      expect(screen.getByText('Big-miss penalty')).toBeInTheDocument()
      expect(screen.getByText(/By position/i)).toBeInTheDocument()
    })
  })

  it('allows changing target stat and season', async () => {
    const user = userEvent.setup()
    renderWithProviders(<SeasonReview />)

    await waitFor(() => {
      expect(screen.getByText('Average miss')).toBeInTheDocument()
    })

    const seasonSelect = screen.getByRole('combobox', { name: 'Season:' })
    expect(seasonSelect).toHaveTextContent('2025 Season')
    await user.click(seasonSelect)
    await user.click(await screen.findByRole('option', { name: '2024 Season' }))
    expect(screen.getByRole('combobox', { name: 'Season:' })).toHaveTextContent('2024 Season')

    const statSelect = screen.getByRole('combobox', { name: 'Stat:' })
    await user.click(statSelect)
    await user.click(await screen.findByRole('option', { name: 'Rushing Yards' }))
    expect(screen.getByRole('combobox', { name: 'Stat:' })).toHaveTextContent('Rushing Yards')
  })
})
