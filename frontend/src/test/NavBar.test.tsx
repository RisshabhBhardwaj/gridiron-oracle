import { describe, expect, it } from 'vitest'
import { http, HttpResponse } from 'msw'
import { server } from './handlers'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { SearchProvider } from '../context/SearchContext'
import { NavBar } from '../components/layout/NavBar'

function renderNav() {
  return render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <SearchProvider>
        <MemoryRouter initialEntries={['/']}>
          <Routes>
            <Route path="/" element={<NavBar />} />
            <Route path="/draft" element={<p>draft page</p>} />
          </Routes>
        </MemoryRouter>
      </SearchProvider>
    </QueryClientProvider>,
  )
}

describe('NavBar mobile menu', () => {
  it('opens a dialog with the nav links and follows one', async () => {
    const user = userEvent.setup()
    renderNav()
    await user.click(screen.getByRole('button', { name: /open navigation menu/i }))
    const dialog = await screen.findByRole('dialog', { name: 'Menu' })
    expect(dialog).toBeInTheDocument()
    await user.click(screen.getByRole('link', { name: 'Draft Board' }))
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

describe('NavBar status', () => {
  const health = (body: object, status = 200) =>
    server.use(http.get('/api/health', () => HttpResponse.json(body, { status })))

  it('says "Live" when the service answers and its artifacts are ready, even if status is "degraded"', async () => {
    health({ status: 'degraded', artifact_mode_ready: true })
    renderNav()
    expect(await screen.findByText('Live')).toBeInTheDocument()
    expect(screen.queryByText('Data unavailable')).not.toBeInTheDocument()
  })

  it('says "Live" when artifact readiness is not reported', async () => {
    health({ status: 'ok' })
    renderNav()
    expect(await screen.findByText('Live')).toBeInTheDocument()
  })

  it('says "Data unavailable" in words when the service cannot be reached', async () => {
    health({ detail: 'down' }, 503)
    renderNav()
    expect(await screen.findByText('Data unavailable')).toBeInTheDocument()
    expect(screen.queryByText('Live')).not.toBeInTheDocument()
  })

  it('says "Data unavailable" when the served artifacts are not ready', async () => {
    health({ status: 'degraded', artifact_mode_ready: false })
    renderNav()
    expect(await screen.findByText('Data unavailable')).toBeInTheDocument()
  })
})

describe('NavBar naming', () => {
  it('uses the same words as the page headings', async () => {
    const user = userEvent.setup()
    renderNav()
    await user.click(screen.getByRole('button', { name: /open navigation menu/i }))
    const menu = await screen.findByRole('navigation', { name: 'Mobile' })
    for (const name of ['Current Season', 'Season Review', 'Draft Board', 'Backtest', 'Settings']) {
      expect(within(menu).getByRole('link', { name })).toBeInTheDocument()
    }
    expect(within(menu).queryByRole('link', { name: 'Backtesting' })).not.toBeInTheDocument()
  })
})
