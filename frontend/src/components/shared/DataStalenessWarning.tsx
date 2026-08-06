import { hoursAgo } from '@/lib/formatters'

interface DataStalenessWarningProps {
  dataFreshness: string
}

export function DataStalenessWarning({ dataFreshness }: DataStalenessWarningProps) {
  const hours = hoursAgo(dataFreshness)
  if (hours < 24) return null

  const label = hours < 48 ? `${Math.round(hours)}h ago` : `${Math.round(hours / 24)}d ago`

  return (
    <div
      className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium"
      style={{
        background: 'rgba(255,184,0,0.1)',
        border: '1px solid rgba(255,184,0,0.3)',
        color: '#FFB800',
        boxShadow: '0 0 12px rgba(255,184,0,0.1)',
      }}
    >
      <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
      Data last updated {label} — projections may be stale
    </div>
  )
}
