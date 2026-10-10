/**
 * Behaviour of the shared primitives in src/components/ui. Dialog and Select are
 * also covered through PlayerSearch and PlayerDetail; Slider through Settings.
 */
import { useState } from 'react'
import { describe, it, expect } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Slider } from '../components/ui/slider'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import { cn } from '../lib/utils'

describe('Slider', () => {
  function Harness({ disabled = false }: { disabled?: boolean }) {
    const [value, setValue] = useState(20)
    return (
      <div>
        <Slider aria-label="Matchup" min={0} max={100} step={5} value={value} onValueChange={setValue} disabled={disabled} />
        <output>{value}</output>
      </div>
    )
  }

  it('has an accessible name and moves with the arrow keys, Home and End', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const slider = screen.getByLabelText('Matchup')
    expect(slider).toHaveValue('20')
    slider.focus()
    await user.keyboard('{ArrowRight}')
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('25'))
    await user.keyboard('{End}')
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('100'))
    await user.keyboard('{Home}')
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('0'))
  })

  it('does nothing when disabled', async () => {
    const user = userEvent.setup()
    render(<Harness disabled />)
    const slider = screen.getByLabelText('Matchup')
    expect(slider).toBeDisabled()
    await user.keyboard('{ArrowRight}')
    expect(screen.getByRole('status')).toHaveTextContent('20')
  })
})

describe('Tabs', () => {
  it('uses tablist/tab/tabpanel roles and switches with the arrow keys', async () => {
    const user = userEvent.setup()
    render(
      <Tabs defaultValue="weekly">
        <TabsList aria-label="View">
          <TabsTrigger value="weekly">Weekly</TabsTrigger>
          <TabsTrigger value="season">Season-Long</TabsTrigger>
        </TabsList>
        <TabsContent value="weekly">Weekly view</TabsContent>
        <TabsContent value="season">Season view</TabsContent>
      </Tabs>,
    )
    expect(screen.getByRole('tablist', { name: 'View' })).toBeInTheDocument()
    const weekly = screen.getByRole('tab', { name: 'Weekly' })
    expect(weekly).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tabpanel')).toHaveTextContent('Weekly view')

    weekly.focus()
    await user.keyboard('{ArrowRight}')
    await waitFor(() => expect(screen.getByRole('tab', { name: 'Season-Long' })).toHaveAttribute('aria-selected', 'true'))
    expect(screen.getByRole('tabpanel')).toHaveTextContent('Season view')
  })
})

describe('cn', () => {
  it('lets the last class win when two set the same property', () => {
    expect(cn('px-2 text-fg', 'px-4')).toBe('text-fg px-4')
    expect(cn('top-1/2', false && 'top-20', 'top-20')).toBe('top-20')
  })
})
