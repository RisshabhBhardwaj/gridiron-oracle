import { afterEach, describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ProjectionCard } from '../components/ProjectionCard'

function stubReducedMotion(reduce: boolean) {
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: reduce && query.includes('prefers-reduced-motion'),
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  }))
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('ProjectionCard', () => {
  it('shows the real number at once when the system asks for reduced motion', () => {
    // The count-up only runs on animation frames. Without this, the card could sit
    // at "0.0", a confident wrong number, until a frame ran.
    stubReducedMotion(true)
    render(<ProjectionCard label="WK Proj" value={87.4} unit="yds" />)
    expect(screen.getByText('87.4')).toBeInTheDocument()
  })

  it('shows a dash, not 0.0, when there is no value', () => {
    stubReducedMotion(true)
    render(<ProjectionCard label="Boom %" value={null} unit="%" />)
    expect(screen.getByText('—')).toBeInTheDocument()
    expect(screen.queryByText('0.0')).not.toBeInTheDocument()
  })

  it('shows emptyText instead of a dash when given', () => {
    stubReducedMotion(true)
    render(<ProjectionCard label="Sportsbook line" value={null} emptyText="No line yet" />)
    expect(screen.getByText('No line yet')).toBeInTheDocument()
    expect(screen.queryByText('—')).not.toBeInTheDocument()
  })
})
