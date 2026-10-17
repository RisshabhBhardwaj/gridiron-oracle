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
import { Switch } from '../components/ui/switch'
import { RadioGroup, RadioGroupItem } from '../components/ui/radio-group'
import { ToggleGroup, ToggleGroupItem } from '../components/ui/toggle-group'
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

describe('Switch', () => {
  function Harness() {
    const [checked, setChecked] = useState(false)
    return (
      <div>
        <Switch aria-label="Dome game" checked={checked} onCheckedChange={setChecked} />
        <output>{checked ? 'on' : 'off'}</output>
      </div>
    )
  }

  it('toggles with click and the Space key', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const sw = screen.getByRole('switch', { name: 'Dome game' })
    expect(sw).toHaveAttribute('aria-checked', 'false')
    await user.click(sw)
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('on'))
    expect(sw).toHaveAttribute('aria-checked', 'true')
    sw.focus()
    await user.keyboard(' ')
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('off'))
  })
})

describe('RadioGroup', () => {
  function Harness() {
    const [value, setValue] = useState('ppr')
    return (
      <div>
        <RadioGroup value={value} onValueChange={setValue} aria-label="Scoring">
          <RadioGroupItem value="ppr" label="Full PPR" />
          <RadioGroupItem value="half" label="Half PPR" />
          <RadioGroupItem value="std" label="Standard" />
        </RadioGroup>
        <output>{value}</output>
      </div>
    )
  }

  it('moves between options with the arrow keys', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    expect(screen.getByRole('radiogroup', { name: 'Scoring' })).toBeInTheDocument()
    const ppr = screen.getByRole('radio', { name: 'Full PPR' })
    expect(ppr).toHaveAttribute('aria-checked', 'true')
    ppr.focus()
    await user.keyboard('{ArrowDown}')
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('half'))
    expect(screen.getByRole('radio', { name: 'Half PPR' })).toHaveAttribute('aria-checked', 'true')
  })
})

describe('ToggleGroup', () => {
  function Harness() {
    const [value, setValue] = useState<string[]>(['WR'])
    return (
      <div>
        <ToggleGroup multiple value={value} onValueChange={setValue} aria-label="Positions">
          <ToggleGroupItem value="WR">WR</ToggleGroupItem>
          <ToggleGroupItem value="RB">RB</ToggleGroupItem>
        </ToggleGroup>
        <output>{value.join(',')}</output>
      </div>
    )
  }

  it('presses and unpresses with click and the Space key', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const wr = screen.getByRole('button', { name: 'WR' })
    const rb = screen.getByRole('button', { name: 'RB' })
    expect(wr).toHaveAttribute('aria-pressed', 'true')
    await user.click(rb)
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('WR,RB'))
    rb.focus()
    await user.keyboard(' ')
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('WR'))
  })
})

describe('cn', () => {
  it('lets the last class win when two set the same property', () => {
    expect(cn('px-2 text-fg', 'px-4')).toBe('text-fg px-4')
    expect(cn('top-1/2', false && 'top-20', 'top-20')).toBe('top-20')
  })
})
