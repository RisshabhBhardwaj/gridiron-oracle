import { TriangleAlert } from 'lucide-react'
import { hoursAgo } from '@/lib/formatters'

interface DataStalenessWarningProps {
  dataFreshness: string
}

export function DataStalenessWarning({ dataFreshness }: DataStalenessWarningProps) {
  const hours = hoursAgo(dataFreshness)
  if (hours < 24) return null

  const label = hours < 48 ? `${Math.round(hours)}h ago` : `${Math.round(hours / 24)}d ago`

  return (
    <div className="tone-warning inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium bg-tone/10 border border-tone/30 text-tone shadow-[0_0_12px_rgb(var(--tone)/0.1)]">
      <TriangleAlert className="size-3.5 flex-shrink-0" aria-hidden="true" />
      Data last updated {label} — projections may be stale
    </div>
  )
}
