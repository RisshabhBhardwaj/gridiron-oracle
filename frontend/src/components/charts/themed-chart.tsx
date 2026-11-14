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

/**
 * `label` is the chart's accessible name (role="img"): say what it shows and
 * the main takeaway, since the SVG itself is not readable by a screen reader.
 */
export function ChartFrame({
  height,
  children,
  label,
}: {
  height: number
  children: ReactElement
  label?: string
}) {
  return (
    <div role={label ? 'img' : undefined} aria-label={label}>
      <ResponsiveContainer width="100%" height={height}>
        {children}
      </ResponsiveContainer>
    </div>
  )
}

/*
 * Recharts 2 finds axes, grid and tooltip by their type name and reads an axis's
 * settings from the element's own props. A plain wrapper component is therefore
 * ignored and draws nothing. The axes extend the real Recharts classes (so the
 * name and static config are inherited) and carry the theme as default props;
 * the grid and tooltip wrappers take Recharts' display names.
 */
export class ChartXAxis extends XAxis {
  static defaultProps = {
    ...XAxis.defaultProps,
    stroke: chartTheme.axisLine,
    tick: { fontSize: 11, fill: chartTheme.axisText },
  }
}

export class ChartYAxis extends YAxis {
  static defaultProps = {
    ...YAxis.defaultProps,
    stroke: chartTheme.axisLine,
    tick: { fontSize: 11, fill: chartTheme.axisText },
  }
}

export function ChartGrid(props: ComponentProps<typeof CartesianGrid>) {
  return <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} {...props} />
}
ChartGrid.displayName = 'CartesianGrid'

export function ChartTooltip(props: ComponentProps<typeof Tooltip>) {
  return (
    <Tooltip
      contentStyle={chartTheme.tooltipContentStyle}
      labelStyle={chartTheme.tooltipLabelStyle}
      {...props}
    />
  )
}
ChartTooltip.displayName = 'Tooltip'
