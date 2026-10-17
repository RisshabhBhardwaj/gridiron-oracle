import { describe, expect, it } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { SearchProvider } from '../context/SearchContext'
import { NavBar } from '../components/layout/NavBar'

function renderNav() {
  return render(
    <SearchProvider>
      <MemoryRouter initialEntries={['/']}>
        <Routes>
          <Route path="/" element={<NavBar />} />
          <Route path="/draft" element={<p>draft page</p>} />
        </Routes>
      </MemoryRouter>
    </SearchProvider>,
  )
}

describe('NavBar mobile menu', () => {
  it('opens a dialog with the nav links and follows one', async () => {
    const user = userEvent.setup()
    renderNav()
    await user.click(screen.getByRole('button', { name: /open navigation menu/i }))
    const dialog = await screen.findByRole('dialog', { name: 'Menu' })
    expect(dialog).toBeInTheDocument()
    await user.click(screen.getByRole('link', { name: 'Draft' }))
    await waitFor(() => expect(screen.getByText('draft page')).toBeInTheDocument())
  })

  it('closes with the Escape key', async () => {
    const user = userEvent.setup()
    renderNav()
    await user.click(screen.getByRole('button', { name: /open navigation menu/i }))
    await screen.findByRole('dialog', { name: 'Menu' })
    await user.keyboard('{Escape}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Menu' })).not.toBeInTheDocument())
  })
})
