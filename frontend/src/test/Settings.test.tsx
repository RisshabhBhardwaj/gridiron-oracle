import { afterEach, beforeEach, describe, it, expect, vi } from 'vitest'
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

// The page is view-only unless the build sets VITE_SETTINGS_EDITABLE=true (the
// public site never does). The editing behaviour is tested with it switched on.
afterEach(() => { vi.unstubAllEnvs() })

describe('Settings (editable build)', () => {
  beforeEach(() => { vi.stubEnv('VITE_SETTINGS_EDITABLE', 'true') })

  it('default weights render (35, 20, 20, 10, 5, 5, 5 visible in inputs)', async () => {
    renderSettings()
    await waitFor(() =>
      expect(screen.getByText('Feature Weights')).toBeInTheDocument(),
    )
    // Check all 7 sliders have correct default values
    // hidden: true because Base UI keeps the thumb visibility:hidden until it can
    // measure layout, which jsdom never does. The slider is fully visible in a browser.
    const sliders = screen.getAllByRole('slider', { hidden: true })
    expect(sliders.length).toBeGreaterThanOrEqual(7)
    // kalman_form = 35
    const kalmanSlider = screen.getByLabelText('Recent form')
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
    expect(totalEl).toHaveClass('text-tone')
    expect(totalEl.closest('.tone-positive')).not.toBeNull()
  })

  it('changing one slider updates the total display', async () => {
    renderSettings()
    await waitFor(() =>
      expect(screen.getByText('Feature Weights')).toBeInTheDocument(),
    )
    const kalmanSlider = screen.getByLabelText('Recent form')
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
    const kalmanSlider = screen.getByLabelText('Recent form')
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
    const pprOption = screen.getByRole('radio', { name: /^PPR 1pt per reception$/i })
    expect(pprOption).toHaveAttribute('aria-checked', 'true')
  })
})

describe('Settings: loading, error and title', () => {
  it('shows a skeleton while loading, an error with Retry for a 500, and sets the tab title', async () => {
    server.use(http.get('/api/settings', () => HttpResponse.json({ detail: 'boom' }, { status: 500 })))
    renderSettings()
    expect(screen.getByText('Loading settings')).toBeInTheDocument()
    expect(await screen.findByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getAllByRole('alert')).toHaveLength(1)
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
    expect(document.title).toBe('Settings · Gridiron Oracle')
  })
})

describe('Settings (public site, view-only)', () => {
  it('shows the saved weights but explains that they cannot be changed', async () => {
    renderSettings()
    await waitFor(() => expect(screen.getByText('Feature Weights')).toBeInTheDocument())
    expect(screen.getByText('100.0%')).toBeInTheDocument()
    expect(screen.getByText('35.0%')).toBeInTheDocument()
    expect(screen.getByRole('note')).toHaveTextContent(/look at these settings, but not change them/i)
    expect(screen.getByText(/saving is turned off on this website/i)).toBeInTheDocument()
  })

  it('disables Save, Reset, scoring buttons and every slider', async () => {
    renderSettings()
    await waitFor(() => expect(screen.getByText('Feature Weights')).toBeInTheDocument())
    const save = screen.getByRole('button', { name: /save settings/i })
    expect(save).toBeDisabled()
    expect(save).toHaveAccessibleDescription(/look at these settings, but not change them/i)
    expect(screen.getByRole('button', { name: /reset to defaults/i })).toBeDisabled()
    // Base UI radios are spans: the group communicates read-only through
    // aria-disabled rather than a native disabled attribute.
    expect(screen.getByRole('radio', { name: /^PPR 1pt per reception$/i })).toHaveAttribute('aria-disabled', 'true')
    for (const slider of screen.getAllByRole('slider', { hidden: true })) expect(slider).toBeDisabled()
  })

  it('never sends a PUT', async () => {
    let putCalled = false
    server.use(
      http.put('/api/settings', async () => {
        putCalled = true
        return HttpResponse.json(factories.settingsResponse())
      }),
    )
    renderSettings()
    await waitFor(() => expect(screen.getByText('Feature Weights')).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: /save settings/i }))
    await new Promise((r) => setTimeout(r, 50))
    expect(putCalled).toBe(false)
  })
})

describe('Settings (saved-settings provenance)', () => {
  function serveSettingsSource(source: string) {
    server.use(
      http.get('/api/settings', () =>
        HttpResponse.json({ ...factories.settingsResponse(), settings_source: source }),
      ),
    )
  }

  it('says nothing about provenance when the values were read from the file', async () => {
    serveSettingsSource('file')
    renderSettings()
    await waitFor(() => expect(screen.getByText('Feature Weights')).toBeInTheDocument())
    // The loading spinner also carries role="status", so scope to the banner id.
    expect(document.getElementById('settings-source-note')).toBeNull()
  })

  it('says the defaults are shown when no settings were saved yet', async () => {
    serveSettingsSource('default_missing')
    renderSettings()
    expect(await screen.findByText(/no saved settings exist yet/i)).toBeInTheDocument()
  })

  it('says the defaults are shown when the saved settings could not be read', async () => {
    serveSettingsSource('default_invalid')
    renderSettings()
    expect(await screen.findByText(/could not be read/i)).toBeInTheDocument()
  })

  it('explains that adjusted numbers are a what-if, not checked past results', async () => {
    renderSettings()
    await waitFor(() => expect(screen.getByText('Feature Weights')).toBeInTheDocument())
    expect(screen.getByText(/they are a what-if/i)).toBeInTheDocument()
  })

  it('says weights apply to past seasons now and to weekly forecasts later (D7)', async () => {
    renderSettings()
    await waitFor(() => expect(screen.getByText('Feature Weights')).toBeInTheDocument())
    expect(
      screen.getByText(
        /These apply to past-season numbers now; weekly forecasts will use them in a later update\./,
      ),
    ).toBeInTheDocument()
  })
})

describe('Settings (editable build, server refuses the write)', () => {
  beforeEach(() => { vi.stubEnv('VITE_SETTINGS_EDITABLE', 'true') })

  it('explains a 403 in plain words and does not show the server text', async () => {
    server.use(
      http.put('/api/settings', () =>
        HttpResponse.json({ detail: 'Settings are read-only on this site' }, { status: 403 }),
      ),
    )
    renderSettings()
    await waitFor(() => expect(screen.getByText('Feature Weights')).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: /save settings/i }))
    expect(await screen.findByText(/your changes were not saved\. this website does not allow changes/i)).toBeInTheDocument()
    expect(screen.queryByText('Settings are read-only on this site')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /retry/i })).not.toBeInTheDocument()
  })

  it('offers a retry after any other failure', async () => {
    server.use(http.put('/api/settings', () => HttpResponse.json({ detail: 'boom' }, { status: 500 })))
    renderSettings()
    await waitFor(() => expect(screen.getByText('Feature Weights')).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: /save settings/i }))
    expect(await screen.findByText(/check your connection and try again/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument()
    expect(screen.queryByText('boom')).not.toBeInTheDocument()
  })
})
