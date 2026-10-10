import { useEffect } from 'react'
import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { SearchProvider, useSearch } from '../context/SearchContext'
import { PlayerSearch } from '../components/PlayerSearch'
import { NavBar } from '../components/layout/NavBar'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App } from '../App'

function renderShell() {
  return render(
    <SearchProvider>
      <MemoryRouter>
        <NavBar />
        <PlayerSearch />
      </MemoryRouter>
    </SearchProvider>,
  )
}

describe('PlayerSearch', () => {
  it('opens from the Search button without crashing, then closes with Escape', () => {
    // Regression: the component called a hook after an early return, so the first
    // render that opened it threw "Rendered more hooks than during the previous
    // render" and, with no error boundary, blanked the whole app.
    renderShell()
    expect(screen.queryByPlaceholderText(/search players/i)).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /search players/i }))
    const input = screen.getByPlaceholderText(/search players/i)
    expect(input).toBeInTheDocument()
    fireEvent.keyDown(input, { key: 'Escape' })
    expect(screen.queryByPlaceholderText(/search players/i)).not.toBeInTheDocument()
  })

  it('opens and closes with Cmd+K', () => {
    renderShell()
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    expect(screen.getByPlaceholderText(/search players/i)).toBeInTheDocument()
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    expect(screen.queryByPlaceholderText(/search players/i)).not.toBeInTheDocument()
  })
})

function Seed() {
  const { setPlayers } = useSearch()
  useEffect(() => {
    setPlayers([
      { player_id: '00-001', player_name: 'Justin Jefferson', position: 'WR', team: 'MIN', stat: 'receiving_yards', projection: 87.4 },
      { player_id: '00-002', player_name: 'Bijan Robinson', position: 'RB', team: 'ATL', stat: 'rushing_yards', projection: 71.2 },
    ])
  }, [setPlayers])
  return null
}

function Where() {
  const loc = useLocation()
  return <p data-testid="where">{loc.pathname}{loc.search}</p>
}

describe('PlayerSearch as a dialog', () => {
  function renderSeeded() {
    return render(
      <SearchProvider>
        <MemoryRouter>
          <NavBar />
          <Seed />
          <PlayerSearch />
          <Where />
        </MemoryRouter>
      </SearchProvider>,
    )
  }

  async function openAndFocus() {
    fireEvent.click(screen.getByRole('button', { name: /search players/i }))
    await screen.findByRole('dialog')
    await waitFor(() => expect(screen.getByRole('combobox')).toHaveFocus())
  }

  it('is a labelled modal dialog and puts focus in the search box', async () => {
    renderSeeded()
    fireEvent.click(screen.getByRole('button', { name: /search players/i }))
    const dialog = await screen.findByRole('dialog', { name: 'Search players' })
    expect(dialog).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('combobox', { name: /search players/i })).toHaveFocus())
  })

  it('filters as you type and opens the highlighted player with the keyboard', async () => {
    const user = userEvent.setup()
    renderSeeded()
    await openAndFocus()

    await user.keyboard('bijan')
    expect(await screen.findByRole('option', { name: /Bijan Robinson/ })).toHaveAttribute('aria-selected', 'true')
    expect(screen.queryByRole('option', { name: /Justin Jefferson/ })).not.toBeInTheDocument()

    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.getByTestId('where')).toHaveTextContent(/^\/player\/00-002\?.*name=Bijan%20Robinson/))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('moves the highlight with the arrow keys and opens a player with a click', async () => {
    const user = userEvent.setup()
    renderSeeded()
    await openAndFocus()
    const input = screen.getByRole('combobox')
    await user.keyboard('{ArrowDown}')
    expect(input).toHaveAttribute('aria-activedescendant', screen.getAllByRole('option')[1].id)

    await user.click(screen.getByRole('option', { name: /Justin Jefferson/ }))
    await waitFor(() => expect(screen.getByTestId('where')).toHaveTextContent('/player/00-001'))
  })

  it('has a visible way to close it, and Escape works too', async () => {
    const user = userEvent.setup()
    renderSeeded()
    fireEvent.click(screen.getByRole('button', { name: /search players/i }))
    await screen.findByRole('dialog')
    await user.click(screen.getByRole('button', { name: /close search/i }))
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: /search players/i }))
    await screen.findByRole('dialog')
    await user.keyboard('{Escape}')
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  })
})

describe('search in the real app shell', () => {
  it('loads the search dialog on first open (it is not in the landing bundle)', async () => {
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MemoryRouter initialEntries={['/settings']}>
          <App />
        </MemoryRouter>
      </QueryClientProvider>,
    )
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    expect(await screen.findByRole('dialog', { name: 'Search players' })).toBeInTheDocument()
  })
})
