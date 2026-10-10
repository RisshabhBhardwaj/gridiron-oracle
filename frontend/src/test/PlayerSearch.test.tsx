import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { SearchProvider } from '../context/SearchContext'
import { PlayerSearch } from '../components/PlayerSearch'
import { NavBar } from '../components/layout/NavBar'

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
