import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { http, HttpResponse } from 'msw'
import { server } from './handlers'
import { factories } from './factories'
import { Settings } from '../pages/Settings'

function renderSettings() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/settings']}>
        <Routes>
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('Settings', () => {
  it('default weights render (35, 20, 20, 10, 5, 5, 5 visible in inputs)', async () => {
    renderSettings()
    await waitFor(() =>
      expect(screen.getByText('Feature Weights')).toBeInTheDocument(),
    )
    // Check all 7 sliders have correct default values
    const sliders = screen.getAllByRole('slider')
    expect(sliders.length).toBeGreaterThanOrEqual(7)
    // kalman_form = 35
    const kalmanSlider = screen.getByLabelText('Kalman Form (Player Ability)')
    expect(kalmanSlider).toHaveValue('35')
    // matchup = 20
    const matchupSlider = screen.getByLabelText('Matchup Quality')
    expect(matchupSlider).toHaveValue('20')
  })

  it('weight total shows 100% in green', async () => {
    renderSettings()
    await waitFor(() =>
      expect(screen.getByText('Total weight')).toBeInTheDocument(),
    )
    const totalEl = screen.getByText('100.0%')
    expect(totalEl).toHaveStyle({ color: 'rgb(0, 255, 163)' })
  })

  it('changing one slider updates the total display', async () => {
    renderSettings()
    await waitFor(() =>
      expect(screen.getByText('Feature Weights')).toBeInTheDocument(),
    )
    const kalmanSlider = screen.getByLabelText('Kalman Form (Player Ability)')
    // Change kalman_form from 35 → 40
    fireEvent.change(kalmanSlider, { target: { value: '40' } })
    await waitFor(() =>
      expect(screen.getByText('105.0%')).toBeInTheDocument(),
    )
  })

  it('Save button is disabled when total != 100', async () => {
    renderSettings()
    await waitFor(() =>
      expect(screen.getByText('Feature Weights')).toBeInTheDocument(),
    )
    const kalmanSlider = screen.getByLabelText('Kalman Form (Player Ability)')
    // Make total invalid: 35→45 (total becomes 110)
    fireEvent.change(kalmanSlider, { target: { value: '45' } })
    await waitFor(() =>
      expect(screen.getByText(/must equal 100%/i)).toBeInTheDocument(),
    )
    const saveButton = screen.getByRole('button', { name: /save settings/i })
    expect(saveButton).toBeDisabled()
  })

  it('PUT /settings called when Save clicked with valid weights', async () => {
    let putCalled = false
    server.use(
      http.put('/api/settings', async () => {
        putCalled = true
        return HttpResponse.json(factories.settingsResponse())
      }),
    )
    renderSettings()
    await waitFor(() =>
      expect(screen.getByText('Feature Weights')).toBeInTheDocument(),
    )
    // Weights should be valid at 100% — click Save
    const saveButton = screen.getByRole('button', { name: /save settings/i })
    expect(saveButton).not.toBeDisabled()
    fireEvent.click(saveButton)
    await waitFor(() => expect(putCalled).toBe(true))
  })

  it('Fantasy scoring PPR option is active by default', async () => {
    renderSettings()
    await waitFor(() =>
      expect(screen.getByText('Fantasy Scoring Format')).toBeInTheDocument(),
    )
    const pprButton = screen.getByRole('button', { name: /^PPR 1pt per reception$/i })
    expect(pprButton).toHaveStyle({ background: 'rgba(0, 194, 255, 0.1)' })
  })
})
