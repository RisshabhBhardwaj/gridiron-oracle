import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell,
} from 'recharts'
import clsx from 'clsx'
import type { FactorItem } from '@/types/api'
import { toneClass, token } from '@/lib/tokens'

interface SHAPChartProps  { factors: FactorItem[]; height?: number }
interface ChartDatum      { label: string; feature: string; impact: number; value: number }
interface TTPayload       { payload: ChartDatum }
interface TTProps         { active?: boolean; payload?: TTPayload[] }

function CustomTooltip({ active, payload }: TTProps) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  const pos = d.impact >= 0
  return (
    <div
      className={clsx(
        toneClass(pos ? 'positive' : 'negative'),
        'rounded-xl px-4 py-3 text-xs backdrop-blur-xl',
        'bg-surface-raised/95 border border-tone/30 shadow-[0_8px_32px_rgb(var(--shadow)/0.6),0_0_12px_rgb(var(--tone)/0.1)]',
      )}
    >
      <p className="font-semibold text-fg mb-1">{d.label}</p>
      <p className="text-fg-muted">Feature value: <span className="text-fg font-mono">{d.value.toFixed(3)}</span></p>
      <p className="text-tone">
        Impact: <span className="font-mono font-bold">{pos ? '+' : ''}{d.impact.toFixed(3)}</span>
      </p>
    </div>
  )
}

export function SHAPChart({ factors, height = 280 }: SHAPChartProps) {
  const sorted: ChartDatum[] = [...factors]
    .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
    .map((f) => ({ label: f.label, feature: f.feature, impact: f.impact, value: f.value }))

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={sorted} layout="vertical" margin={{ top: 4, right: 20, left: 8, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={token('line')} horizontal={false} />
        <XAxis
          type="number"
          stroke={token('line-strong')}
          tick={{ fontSize: 10, fill: token('fg-muted') }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          type="category"
          dataKey="label"
          width={150}
          stroke="transparent"
          tick={{ fontSize: 10, fill: token('fg-muted') }}
          tickLine={false}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: token('fg', 0.03) }} />
        <ReferenceLine x={0} stroke={token('line-strong')} strokeWidth={1} />
        <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
          {sorted.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={token(entry.impact >= 0 ? 'positive' : 'negative', 0.8)}
              style={{
                filter: `drop-shadow(0 0 4px ${token(entry.impact >= 0 ? 'positive' : 'negative', 0.4)})`,
              }}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
