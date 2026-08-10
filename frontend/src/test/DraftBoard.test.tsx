import { describe, expect, it } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
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
    expect(screen.getByRole('combobox', { name: 'Draft season' })).toHaveValue('2026')
    expect(await screen.findByText("Ja'Marr Chase")).toBeInTheDocument()
    expect(screen.getByLabelText('Draft projection metadata')).toHaveTextContent('as of 2026-08-01')
  })

  it('changes the requested season and filters by position', async () => {
    renderBoard()
    await screen.findByText("Ja'Marr Chase")
    fireEvent.change(screen.getByRole('combobox', { name: 'Draft season' }), { target: { value: '2025' } })
    fireEvent.click(screen.getByRole('button', { name: 'WR' }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'WR' })).toHaveClass('text-oracle-green'))
  })
})
