import clsx from 'clsx'
import type { BacktestResponse } from '@/types/api'
import { STAT_LABELS } from '@/lib/constants'
import { toneClass, type Tone } from '@/lib/tokens'

interface BacktestPanelProps { data: BacktestResponse }

function safeFixed(n: number | null | undefined, digits: number): string {
  if (n === null || n === undefined || !isFinite(n) || isNaN(n)) return '—'
  return n.toFixed(digits)
}

interface MetricCardProps {
  label: string
  value: string
  subtext?: string
  /** Colours the value. Leave out for a plain, uncoloured card. "muted" means no data. */
  tone?: Tone
}

function MetricCard({ label, value, subtext, tone }: MetricCardProps) {
  const glowing = tone !== undefined && tone !== 'muted'
  return (
    <div
      className={clsx(
        tone && toneClass(tone),
        'relative rounded-xl p-4 flex flex-col gap-2 overflow-hidden bg-panel-gradient',
        glowing
          ? 'border border-tone/20 shadow-[0_4px_20px_rgb(var(--shadow)/0.3),0_0_12px_rgb(var(--tone)/0.08)]'
          : 'border border-white/[0.06] shadow-[0_4px_20px_rgb(var(--shadow)/0.3)]',
      )}
    >
      {glowing && (
        <div className="corner-glow absolute top-0 right-0 w-12 h-12 rounded-bl-full opacity-30" />
      )}
      <span className="section-label">{label}</span>
      <span className={clsx('font-display text-3xl leading-none', tone ? 'text-tone' : 'text-fg', glowing && 'glow-text')}>
        {value}
      </span>
      {subtext && <span className="text-[10px] text-fg-muted leading-tight">{subtext}</span>}
    </div>
  )
}

export function BacktestPanel({ data }: BacktestPanelProps) {
  const pnlPositive = data.simulated_pnl !== null && data.simulated_pnl >= 0
  const pnlValue = data.simulated_pnl === null ? '—'
    : `${data.simulated_pnl >= 0 ? '+' : ''}${safeFixed(data.simulated_pnl, 2)}`

  const sharpeTone: Tone = data.sharpe_ratio === null ? 'muted'
    : data.sharpe_ratio >= 1 ? 'positive'
    : data.sharpe_ratio >= 0.5 ? 'warning' : 'negative'

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
      <MetricCard
        label="MAE"
        value={safeFixed(data.overall_mae, 2)}
        subtext="Average miss, in points"
        tone="info"
      />
      <MetricCard
        label="RMSE"
        value={safeFixed(data.overall_rmse, 2)}
        subtext="Large misses count extra"
        tone="info"
      />
      <MetricCard
        label="CRPS"
        value={safeFixed(data.overall_crps, 3)}
        subtext="How well the predicted range matched reality"
        tone="highlight"
      />
      <MetricCard
        label="Brier Score"
        value={safeFixed(data.brier_score, 3)}
        subtext={data.metric_notes?.brier_score ?? 'Probability accuracy (lower is better)'}
        tone="highlight"
      />
      <MetricCard
        label="Simulated P&L"
        value={pnlValue}
        subtext={data.metric_notes?.simulated_pnl ?? 'Backtest wagering output when available'}
        tone={data.simulated_pnl === null ? 'muted' : pnlPositive ? 'positive' : 'negative'}
      />
      <MetricCard
        label="Sharpe Ratio"
        value={safeFixed(data.sharpe_ratio, 2)}
        subtext={data.metric_notes?.sharpe_ratio ?? 'Return per unit of risk'}
        tone={sharpeTone}
      />
      <MetricCard
        label="Max Drawdown"
        value={data.max_drawdown === null ? '—' : `${safeFixed(data.max_drawdown, 2)}u`}
        subtext={data.metric_notes?.max_drawdown ?? 'Largest peak-to-trough loss'}
        tone={data.max_drawdown === null ? 'muted' : 'negative'}
      />
      <MetricCard
        label="Seasons"
        value={data.seasons.length > 0 ? data.seasons.join(', ') : '—'}
        subtext={`${STAT_LABELS[data.stat] ?? data.stat} · ${data.positions.join(', ')}`}
      />
    </div>
  )
}
