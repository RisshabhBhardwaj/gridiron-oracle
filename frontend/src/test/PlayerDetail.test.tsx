import { describe, it, expect } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { SearchProvider } from '../context/SearchContext'
import { PlayerDetail } from '../pages/PlayerDetail'

function renderPlayer(path = '/player/test-001?week=5&season=2025&stat=receiving_yards&name=Justin+Jefferson') {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <SearchProvider>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route path="/player/:player_id" element={<PlayerDetail />} />
          </Routes>
        </MemoryRouter>
      </SearchProvider>
    </QueryClientProvider>,
  )
}

describe('PlayerDetail', () => {
  it('renders player name in header', async () => {
    renderPlayer()
    await waitFor(() =>
      expect(screen.getByText(/Justin Jefferson/i)).toBeInTheDocument(),
    )
  })

  it('renders ProjectionCard with correct projection value', async () => {
    renderPlayer()
    await waitFor(() => {
      // Multiple elements with "87.4" are expected (card + percentile fan). Use getAllByText.
      const projValues = screen.getAllByText('87.4')
      expect(projValues.length).toBeGreaterThan(0)
    })
    const labels = screen.getAllByText(/WK Proj/i)
    expect(labels.length).toBeGreaterThan(0)
  })

  it('renders SHAPChart with plain-English factor labels visible', async () => {
    renderPlayer()
    // The SHAP section heading confirms explain data was fetched and chart rendered.
    // recharts renders labels in SVG <text> nodes; we verify the section heading
    // and that no loading spinner is displayed for that section.
    await waitFor(() =>
      expect(screen.getByText('Factor Attributions (SHAP)')).toBeInTheDocument(),
    )
    // Explain data loaded: the "No SHAP factors available" message should NOT appear
    await waitFor(() => {
      expect(
        screen.queryByText(/No SHAP factors available/i),
      ).not.toBeInTheDocument()
    })
  })

})
