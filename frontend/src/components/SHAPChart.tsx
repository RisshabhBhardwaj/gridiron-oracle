import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell,
} from 'recharts'
import type { FactorItem } from '@/types/api'

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
      className="rounded-xl px-4 py-3 text-xs"
      style={{
        background: 'rgba(8,15,32,0.95)',
        border: `1px solid ${pos ? 'rgba(0,255,163,0.3)' : 'rgba(255,59,92,0.3)'}`,
        boxShadow: `0 8px 32px rgba(0,0,0,0.6), 0 0 12px ${pos ? 'rgba(0,255,163,0.1)' : 'rgba(255,59,92,0.1)'}`,
        backdropFilter: 'blur(12px)',
      }}
    >
      <p className="font-semibold text-oracle-white mb-1">{d.label}</p>
      <p className="text-oracle-muted">Feature value: <span className="text-oracle-white font-mono">{d.value.toFixed(3)}</span></p>
      <p style={{ color: pos ? '#00FFA3' : '#FF3B5C' }}>
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
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(26,47,78,0.8)" horizontal={false} />
        <XAxis
          type="number"
          stroke="rgba(96,123,155,0.5)"
          tick={{ fontSize: 10, fill: '#607B9B' }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          type="category"
          dataKey="label"
          width={150}
          stroke="transparent"
          tick={{ fontSize: 10, fill: '#607B9B' }}
          tickLine={false}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
        <ReferenceLine x={0} stroke="rgba(96,123,155,0.4)" strokeWidth={1} />
        <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
          {sorted.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={entry.impact >= 0 ? 'rgba(0,255,163,0.8)' : 'rgba(255,59,92,0.8)'}
              style={{
                filter: `drop-shadow(0 0 4px ${entry.impact >= 0 ? 'rgba(0,255,163,0.4)' : 'rgba(255,59,92,0.4)'})`,
              }}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
