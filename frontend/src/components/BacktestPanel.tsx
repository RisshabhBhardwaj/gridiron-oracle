import type { BacktestResponse } from '@/types/api'

interface BacktestPanelProps { data: BacktestResponse }

function safeFixed(n: number | null | undefined, digits: number): string {
  if (n === null || n === undefined || !isFinite(n) || isNaN(n)) return '—'
  return n.toFixed(digits)
}

interface MetricCardProps {
  label: string
  value: string
  subtext?: string
  color?: string
  glow?: string
}

function MetricCard({ label, value, subtext, color = '#F0F6FF', glow }: MetricCardProps) {
  return (
    <div
      className="relative rounded-xl p-4 flex flex-col gap-2 overflow-hidden"
      style={{
        background: 'linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%)',
        border: glow ? `1px solid ${glow}30` : '1px solid rgba(255,255,255,0.06)',
        boxShadow: glow ? `0 4px 20px rgba(0,0,0,0.3), 0 0 12px ${glow}15` : '0 4px 20px rgba(0,0,0,0.3)',
      }}
    >
      {glow && (
        <div
          className="absolute top-0 right-0 w-12 h-12 rounded-bl-full opacity-30"
          style={{ background: `radial-gradient(circle at top right, ${glow}50, transparent 70%)` }}
        />
      )}
      <span className="section-label">{label}</span>
      <span
        className="font-display text-3xl leading-none"
        style={{ color, textShadow: glow ? `0 0 16px ${glow}60` : 'none' }}
      >
        {value}
      </span>
      {subtext && <span className="text-[10px] text-oracle-muted leading-tight">{subtext}</span>}
    </div>
  )
}

export function BacktestPanel({ data }: BacktestPanelProps) {
  const pnlPositive = data.simulated_pnl !== null && data.simulated_pnl >= 0
  const pnlValue = data.simulated_pnl === null ? '—'
    : `${data.simulated_pnl >= 0 ? '+' : ''}${safeFixed(data.simulated_pnl, 2)}`

  const sharpeColor = data.sharpe_ratio === null ? '#607B9B'
    : data.sharpe_ratio >= 1 ? '#00FFA3'
    : data.sharpe_ratio >= 0.5 ? '#FFB800' : '#FF3B5C'

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
      <MetricCard
        label="MAE"
        value={safeFixed(data.overall_mae, 2)}
        subtext="Mean Absolute Error"
        color="#00C2FF"
        glow="#00C2FF"
      />
      <MetricCard
        label="RMSE"
        value={safeFixed(data.overall_rmse, 2)}
        subtext="Root Mean Squared Error"
        color="#00C2FF"
        glow="#00C2FF"
      />
      <MetricCard
        label="CRPS"
        value={safeFixed(data.overall_crps, 3)}
        subtext="Probabilistic accuracy"
        color="#A855F7"
        glow="#A855F7"
      />
      <MetricCard
        label="Brier Score"
        value={safeFixed(data.brier_score, 3)}
        subtext={data.metric_notes?.brier_score ?? 'Binary-event calibration when available'}
        color="#A855F7"
        glow="#A855F7"
      />
      <MetricCard
        label="Simulated P&L"
        value={pnlValue}
        subtext={data.metric_notes?.simulated_pnl ?? 'Backtest wagering output when available'}
        color={data.simulated_pnl === null ? '#607B9B' : pnlPositive ? '#00FFA3' : '#FF3B5C'}
        glow={data.simulated_pnl === null ? undefined : pnlPositive ? '#00FFA3' : '#FF3B5C'}
      />
      <MetricCard
        label="Sharpe Ratio"
        value={safeFixed(data.sharpe_ratio, 2)}
        subtext={data.metric_notes?.sharpe_ratio ?? 'Risk-adjusted return'}
        color={sharpeColor}
        glow={sharpeColor}
      />
      <MetricCard
        label="Max Drawdown"
        value={data.max_drawdown === null ? '—' : `${safeFixed(data.max_drawdown, 2)}u`}
        subtext={data.metric_notes?.max_drawdown ?? 'Largest peak-to-trough loss'}
        color={data.max_drawdown === null ? '#607B9B' : '#FF3B5C'}
        glow={data.max_drawdown === null ? undefined : '#FF3B5C'}
      />
      <MetricCard
        label="Seasons"
        value={data.seasons.length > 0 ? data.seasons.join(', ') : '—'}
        subtext={`${data.stat} · ${data.positions.join('/')} · ${data.data_source}`}
      />
    </div>
  )
}
