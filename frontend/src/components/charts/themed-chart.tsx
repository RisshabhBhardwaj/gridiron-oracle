import { useId, useState, type ComponentProps, type ReactElement } from 'react'
import {
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { token } from '@/lib/tokens'
import { cn, FOCUS_RING } from '@/lib/utils'

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

/** The same numbers as the chart, as a plain table: header cells, then one array of cells per row. */
export interface ChartTable {
  caption: string
  head: string[]
  rows: Array<Array<string | number>>
}

/**
 * `label` is the chart's accessible name (role="img"): say the takeaway first, then the
 * numbers, since the SVG itself is not readable by a screen reader. `table` is the
 * same data as a real table; a visible "Show as table" button swaps the chart for it,
 * for screen reader users, for anyone who wants exact numbers, and for copying.
 */
export function ChartFrame({
  height,
  children,
  label,
  table,
}: {
  height: number
  children: ReactElement
  label?: string
  table?: ChartTable
}) {
  const [asTable, setAsTable] = useState(false)
  const tableId = useId()

  return (
    <div>
      {table && (
        <div className="mb-2 flex justify-end">
          <button
            type="button"
            aria-pressed={asTable}
            aria-controls={tableId}
            onClick={() => setAsTable((v) => !v)}
            className={cn(
              'inline-flex h-8 items-center rounded-md border border-line bg-surface-raised px-3 text-xs font-medium text-fg transition-colors hover:border-line-strong',
              FOCUS_RING,
            )}
          >
            {asTable ? 'Show as chart' : 'Show as table'}
          </button>
        </div>
      )}
      <div id={tableId}>
        {table && asTable ? (
          <div role="region" aria-label={table.caption} tabIndex={0} className="overflow-x-auto rounded-lg border border-line">
            <table className="w-full text-left text-xs tabular-nums">
              <caption className="sr-only">{table.caption}</caption>
              <thead className="bg-surface-raised text-fg-muted">
                <tr>
                  {table.head.map((h, i) => (
                    <th key={h} scope="col" className={cn('px-3 py-2 font-semibold uppercase tracking-wider', i > 0 && 'text-right')}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-line/40">
                {table.rows.map((r) => (
                  <tr key={String(r[0])}>
                    {r.map((cell, i) =>
                      i === 0 ? (
                        <th key={i} scope="row" className="px-3 py-2 font-semibold text-fg">{cell}</th>
                      ) : (
                        <td key={i} className="px-3 py-2 text-right font-mono text-fg-muted">{cell}</td>
                      ),
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div role={label ? 'img' : undefined} aria-label={label}>
            <ResponsiveContainer width="100%" height={height}>
              {children}
            </ResponsiveContainer>
          </div>
        )}
      </div>
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
    tick: { fontSize: 12, fill: chartTheme.axisText },
  }
}

export class ChartYAxis extends YAxis {
  static defaultProps = {
    ...YAxis.defaultProps,
    stroke: chartTheme.axisLine,
    tick: { fontSize: 12, fill: chartTheme.axisText },
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
