import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { DraftBoard } from '../pages/DraftBoard'

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/draft']}>
        <Routes>
          <Route path="/draft" element={ui} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('DraftBoard & MockDraft', () => {
  it('renders default draft board table', async () => {
    renderWithProviders(<DraftBoard />)
    expect(screen.getByRole('heading', { level: 1, name: 'Draft Board' })).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByText("Ja'Marr Chase")).toBeInTheDocument()
      expect(screen.getByText('Bijan Robinson')).toBeInTheDocument()
    })
  })

  it('switches to Mock Draft Mode and displays manager profiles and start button', async () => {
    renderWithProviders(<DraftBoard />)

    const mockModeBtn = screen.getByText('Mock Draft Mode')
    fireEvent.click(mockModeBtn)

    expect(screen.getByText('Mock Draft Simulator')).toBeInTheDocument()
    expect(screen.getByText('Start Mock Draft')).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByText(/Manager Alpha/i)).toBeInTheDocument()
    })
  })
})
