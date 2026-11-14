import clsx from 'clsx'
import { useApiStatus, type ApiStatus } from '@/hooks/useApiStatus'

const LABEL: Record<ApiStatus, string> = {
  checking: 'Checking',
  live: 'Live',
  unavailable: 'Data unavailable',
}

/**
 * Nav status: a dot plus a word, so the state never rests on colour alone.
 * "Live" and "Checking" stay out of the way on narrow screens (so the badge appearing or
 * going away never moves the header); a problem is always shown.
 */
export function ApiStatusBadge() {
  const status = useApiStatus()
  return (
    <div
      role="status"
      data-status={status}
      className={clsx(
        'items-center gap-2 px-2 py-1',
        status === 'unavailable' ? 'flex' : 'hidden lg:flex',
      )}
    >
      <span className="relative flex h-1.5 w-1.5" aria-hidden="true">
        {status === 'live' && (
          <span className="absolute inline-flex h-full w-full animate-ping-slow rounded-full bg-accent opacity-75" />
        )}
        <span
          className={clsx(
            'relative inline-flex h-1.5 w-1.5 rounded-full',
            status === 'live' && 'bg-accent',
            status === 'unavailable' && 'bg-negative',
            status === 'checking' && 'bg-fg-muted/60',
          )}
        />
      </span>
      <span
        className={clsx(
          'font-mono text-xs font-medium uppercase tracking-wider',
          // Mono font: 8ch holds "Checking" and "Live" alike, so the answer arriving moves nothing.
          status === 'unavailable' ? 'text-negative' : 'min-w-[8ch] text-fg-muted',
        )}
      >
        {LABEL[status]}
      </span>
    </div>
  )
}
