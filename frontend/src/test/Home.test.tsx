import { describe, it, expect } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { SearchProvider } from '../context/SearchContext'
import { PlayerSearch } from '../components/PlayerSearch'
import { Home } from '../pages/Home'

function Here() {
  const loc = useLocation()
  return <p data-testid="here">{loc.pathname}{loc.search}</p>
}

function renderWithProviders() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <SearchProvider>
        <MemoryRouter initialEntries={['/']}>
          <Routes>
            <Route path="/" element={<><Home /><PlayerSearch /></>} />
            <Route path="/player/:id" element={<Here />} />
          </Routes>
        </MemoryRouter>
      </SearchProvider>
    </QueryClientProvider>,
  )
}

describe('Home', () => {
  it('offers one search entry point that opens the shared dialog', async () => {
    const user = userEvent.setup()
    renderWithProviders()
    await user.click(screen.getByRole('button', { name: /search players, teams, or positions/i }))
    expect(screen.getByRole('combobox', { name: /search players, positions, teams/i })).toBeInTheDocument()
  })

  it('finds a player through the dialog and opens their page', async () => {
    const user = userEvent.setup()
    renderWithProviders()
    await user.click(screen.getByRole('button', { name: /search players, teams, or positions/i }))
    const box = screen.getByRole('combobox', { name: /search players, positions, teams/i })
    await user.type(box, 'jeff')
    await user.click(await screen.findByRole('option', { name: /justin jefferson/i }))
    const here = await screen.findByTestId('here')
    expect(here.textContent).toMatch(/^\/player\/[^?]+\?.*name=Justin%20Jefferson/)
  })

  it('says so in the dialog when nothing matches', async () => {
    const user = userEvent.setup()
    renderWithProviders()
    await user.click(screen.getByRole('button', { name: /search players, teams, or positions/i }))
    const box = screen.getByRole('combobox', { name: /search players, positions, teams/i })
    await user.type(box, 'zzzznomatch')
    await waitFor(() => {
      expect(screen.getByText(/no players found for/i)).toBeInTheDocument()
    })
  })
})
