import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { SearchProvider } from '../context/SearchContext'
import { CurrentSeason } from '../pages/CurrentSeason'

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
})
