import { describe, expect, it } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { DraftBoard } from '../pages/DraftBoard'

function renderBoard() {
  return render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <DraftBoard />
    </QueryClientProvider>,
  )
}

describe('DraftBoard', () => {
  it('defaults to the upcoming draft season and renders causal metadata', async () => {
    renderBoard()
    expect(screen.getByRole('combobox', { name: 'Draft season' })).toHaveTextContent('2026')
    expect(await screen.findByText("Ja'Marr Chase")).toBeInTheDocument()
    expect(screen.getByLabelText('Draft projection metadata')).toHaveTextContent('as of 2026-08-01')
  })

  it('changes the requested season and filters by position', async () => {
    const user = userEvent.setup()
    renderBoard()
    await screen.findByText("Ja'Marr Chase")
    const season = screen.getByRole('combobox', { name: 'Draft season' })
    await user.click(season)
    await user.click(await screen.findByRole('option', { name: '2025' }))
    expect(screen.getByRole('combobox', { name: 'Draft season' })).toHaveTextContent('2025')
    await user.click(screen.getByRole('button', { name: 'WR' }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'WR' })).toHaveAttribute('aria-pressed', 'true'))
  })
})
