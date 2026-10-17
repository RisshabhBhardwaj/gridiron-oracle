import type { ComponentProps, ReactElement } from 'react'
import {
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { token } from '@/lib/tokens'

/**
 * One themed wrapper for every Recharts chart. Axes, grid and tooltip read
 * the project tokens, so a chart never hard-codes a colour.
 *
 * Usage:
 *   <ChartFrame height={280}>
 *     <BarChart data={rows} …>
 *       <ChartGrid />
 *       <ChartXAxis dataKey="season" />
 *       <ChartYAxis />
 *       <ChartTooltip />
 *       …
 *     </BarChart>
 *   </ChartFrame>
 */
export const chartTheme = {
  grid: token('line'),
  axisLine: token('line-strong'),
  axisText: token('fg-muted'),
  tooltipContentStyle: {
    background: token('surface-raised', 0.95),
    border: `1px solid ${token('info', 0.2)}`,
    borderRadius: '12px',
    backdropFilter: 'blur(12px)',
    boxShadow: `0 8px 32px ${token('shadow', 0.6)}`,
  },
  tooltipLabelStyle: { color: token('fg'), fontWeight: '600' },
}

export function ChartFrame({ height, children }: { height: number; children: ReactElement }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      {children}
    </ResponsiveContainer>
  )
}

export function ChartGrid(props: ComponentProps<typeof CartesianGrid>) {
  return <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} {...props} />
}

export function ChartXAxis(props: ComponentProps<typeof XAxis>) {
  return (
    <XAxis
      stroke={chartTheme.axisLine}
      tick={{ fontSize: 11, fill: chartTheme.axisText }}
      {...props}
    />
  )
}

export function ChartYAxis(props: ComponentProps<typeof YAxis>) {
  return (
    <YAxis
      stroke={chartTheme.axisLine}
      tick={{ fontSize: 11, fill: chartTheme.axisText }}
      {...props}
    />
  )
}

export function ChartTooltip() {
  return (
    <Tooltip
      contentStyle={chartTheme.tooltipContentStyle}
      labelStyle={chartTheme.tooltipLabelStyle}
    />
  )
}
