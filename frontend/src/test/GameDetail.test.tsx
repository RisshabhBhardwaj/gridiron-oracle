import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { GameDetail } from '../pages/GameDetail'

function renderWithProviders(ui: React.ReactElement, initialEntry: string = '/projections/game/2026_05_MIN_GB') {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialEntry]}>
        <Routes>
          <Route path="/projections/game/:gameId" element={ui} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('GameDetail', () => {
  it('renders game matchup header, anchor comparison, and drive timeline', async () => {
    renderWithProviders(<GameDetail />)

    await waitFor(() => {
      expect(screen.getByText('GB @ MIN')).toBeInTheDocument()
      expect(screen.getByText(/Final Simulated Result/i)).toBeInTheDocument()
      expect(screen.getByText('MIN Wins')).toBeInTheDocument()
      expect(screen.getByText('Drive Sequence Timeline')).toBeInTheDocument()
      expect(screen.getByText('Drive 1')).toBeInTheDocument()
      expect(screen.getByText('Drive 2')).toBeInTheDocument()
    })
  })

  it('allows expanding drive play-by-play table', async () => {
    renderWithProviders(<GameDetail />)

    await waitFor(() => {
      expect(screen.getByText('Drive 2')).toBeInTheDocument()
    })

    const drive2Btn = screen.getByText('Drive 2')
    fireEvent.click(drive2Btn)

    await waitFor(() => {
      expect(screen.getByText('TOUCHDOWN 🏈')).toBeInTheDocument()
    })
  })
})
