import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ErrorBanner, BACKEND_UNAVAILABLE, BACKEND_UNAVAILABLE_MESSAGE } from '../components/shared/ErrorBanner'
import { ApiError } from '../lib/api-client'

describe('ErrorBanner', () => {
  it('explains a dead server in plain words without commands or raw text', () => {
    render(<ErrorBanner error={new Error('ECONNREFUSED 127.0.0.1:8000')} />)
    expect(screen.getByText('No connection to the server')).toBeInTheDocument()
    expect(screen.getByText(/check your internet connection/i)).toBeInTheDocument()
    expect(screen.queryByText(/ECONNREFUSED/)).not.toBeInTheDocument()
    expect(screen.queryByText(/make up/)).not.toBeInTheDocument()
  })

  it('never shows the raw error text for unexpected failures', () => {
    render(<ErrorBanner error={new Error('/srv/app/model.onnx: sha mismatch 9f3a…')} />)
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByText(/please try again/i)).toBeInTheDocument()
    expect(screen.queryByText(/sha mismatch/)).not.toBeInTheDocument()
  })

  it('shows one calm line (no error framing, no retry) when the backend is moving hosts', () => {
    const err = new ApiError(502, BACKEND_UNAVAILABLE_MESSAGE, undefined, BACKEND_UNAVAILABLE)
    render(<ErrorBanner error={err} onRetry={() => {}} />)
    expect(screen.getByRole('status')).toHaveTextContent(BACKEND_UNAVAILABLE_MESSAGE)
    expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument()
  })

  it('keeps the error framing for other API errors', () => {
    render(<ErrorBanner error={new ApiError(500, 'boom')} onRetry={() => {}} />)
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
  })

  it('shows a caller-supplied message and fires retry', async () => {
    const user = userEvent.setup()
    const onRetry = vi.fn()
    render(<ErrorBanner message="Your changes were not saved." onRetry={onRetry} />)
    expect(screen.getByText('Your changes were not saved.')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Retry' }))
    expect(onRetry).toHaveBeenCalledOnce()
  })
})
