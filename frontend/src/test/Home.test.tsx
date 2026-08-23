import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { SearchProvider } from '../context/SearchContext'
import { Home } from '../pages/Home'

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

describe('Home', () => {
  it('renders the search input', () => {
    renderWithProviders(<Home />)
    expect(screen.getByPlaceholderText(/search players, teams, or positions/i)).toBeInTheDocument()
  })

  it('shows a matching result row when typing a partial player name', async () => {
    renderWithProviders(<Home />)
    const input = screen.getByPlaceholderText(/search players, teams, or positions/i)

    fireEvent.focus(input)
    fireEvent.change(input, { target: { value: 'jeff' } })

    await waitFor(() => {
      expect(screen.getByText('Justin Jefferson')).toBeInTheDocument()
    })
  })

  it('shows an empty state when the search matches no players', async () => {
    renderWithProviders(<Home />)
    const input = screen.getByPlaceholderText(/search players, teams, or positions/i)

    fireEvent.focus(input)
    fireEvent.change(input, { target: { value: 'zzzznomatch' } })

    await waitFor(() => {
      expect(screen.getByText(/no matching players found/i)).toBeInTheDocument()
    })
  })
})
