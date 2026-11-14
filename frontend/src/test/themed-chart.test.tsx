import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { Bar, BarChart } from 'recharts'
import { ChartGrid, ChartTooltip, ChartXAxis, ChartYAxis } from '@/components/charts/themed-chart'

const data = [
  { season: 2024, WR: 12 },
  { season: 2025, WR: 8 },
]

// Recharts 2 finds axes, grid and tooltip by type name and reads their config from
// the element's own props, so a wrapper that does not extend the real component
// draws nothing. These tests fail if the themed wrappers ever go silent again.
describe('themed chart wrappers', () => {
  it('draws axes with tick labels and axis titles', () => {
    const { container } = render(
      <BarChart width={400} height={240} data={data}>
        <ChartGrid />
        <ChartXAxis dataKey="season" label={{ value: 'Season', position: 'insideBottom' }} />
        <ChartYAxis tickFormatter={(v: number) => `${v}%`} label={{ value: 'Better by', angle: -90 }} />
        <ChartTooltip />
        <Bar dataKey="WR" />
      </BarChart>,
    )
    expect(container.querySelector('.recharts-xAxis')).not.toBeNull()
    expect(container.querySelector('.recharts-yAxis')).not.toBeNull()
    expect(container.querySelector('.recharts-cartesian-grid')).not.toBeNull()
    const text = container.textContent ?? ''
    expect(text).toContain('2024')
    expect(text).toContain('2025')
    expect(text).toContain('Season')
    expect(text).toContain('Better by')
    expect(text).toMatch(/\d+%/)
  })

  it('applies the theme tokens to the axis line and tick text', () => {
    const { container } = render(
      <BarChart width={400} height={240} data={data}>
        <ChartXAxis dataKey="season" />
        <Bar dataKey="WR" />
      </BarChart>,
    )
    const line = container.querySelector('.recharts-xAxis .recharts-cartesian-axis-line')
    expect(line?.getAttribute('stroke')).toBeTruthy()
    const tick = container.querySelector('.recharts-xAxis .recharts-cartesian-axis-tick-value')
    expect(tick?.getAttribute('fill')).toBeTruthy()
  })
})
