import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { http, HttpResponse } from 'msw'
import { GameDetail } from '../pages/GameDetail'
import { server } from './handlers'

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
      expect(screen.getByText(/Simulated result/i)).toBeInTheDocument()
      expect(screen.getByText('MIN win')).toBeInTheDocument()
      expect(screen.getByText('Drives')).toBeInTheDocument()
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
      expect(screen.getByText('Touchdown')).toBeInTheDocument()
    })
  })

  it('is "not ready" for a 404, with a way back, and never an error', async () => {
    server.use(http.get('/api/games/:game_id/drive-sim', () => HttpResponse.json({ detail: 'none' }, { status: 404 })))
    renderWithProviders(<GameDetail />)
    expect(await screen.findByText(/simulation isn.t ready yet/)).toBeInTheDocument()
    expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument()
    expect(screen.getAllByRole('link', { name: /Back to Current Season/ }).length).toBeGreaterThan(0)
  })

  it('shows one alert with Retry for a server failure', async () => {
    server.use(http.get('/api/games/:game_id/drive-sim', () => HttpResponse.json({ detail: 'boom' }, { status: 500 })))
    renderWithProviders(<GameDetail />)
    expect(await screen.findByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getAllByRole('alert')).toHaveLength(1)
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
    expect(screen.queryByText(/simulation isn.t ready yet/)).not.toBeInTheDocument()
  })

  it('shows a skeleton, not the page, while loading, and sets the tab title when loaded', async () => {
    renderWithProviders(<GameDetail />)
    expect(screen.getByText('Loading the game simulation')).toBeInTheDocument()
    await screen.findByText('GB @ MIN')
    expect(document.title).toBe('GB @ MIN · Gridiron Oracle')
  })
})
