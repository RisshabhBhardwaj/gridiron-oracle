import { TriangleAlert, WifiOff } from 'lucide-react'
import { ApiError } from '@/lib/api-client'

interface ErrorBannerProps {
  error?: Error | null
  onRetry?: () => void
  message?: string
}

/** The /api proxy's code for "the backend cannot be reached" (see functions/api/[[path]].js). */
export const BACKEND_UNAVAILABLE = 'backend_unavailable'
export const BACKEND_UNAVAILABLE_MESSAGE = 'Live data is moving to a new server and will be back soon'

/**
 * Plain-words failure notice. Never shows raw error text or developer
 * commands: server messages can leak paths, keys and stack traces, and a
 * fantasy player cannot act on them. Callers with a specific, user-safe
 * explanation pass it as `message`.
 */
export function ErrorBanner({ error, onRetry, message }: ErrorBannerProps) {
  // Expected while the backend moves hosts: one calm line, no error framing, no retry.
  if (error instanceof ApiError && error.code === BACKEND_UNAVAILABLE) {
    return <p role="status">{BACKEND_UNAVAILABLE_MESSAGE}</p>
  }

  const isOffline = error?.message?.includes('ECONNREFUSED') || error?.message?.includes('Failed to fetch')

  return (
    <div className="tone-negative relative rounded-xl p-5 flex items-start gap-4 overflow-hidden border border-tone/25 bg-gradient-to-br from-tone/10 to-tone/[0.03] shadow-[0_4px_24px_rgb(var(--tone)/0.08)]">
      {/* Glow edge */}
      <div className="absolute left-0 top-0 bottom-0 w-0.5 rounded-l-xl bg-gradient-to-b from-tone to-tone/20" />

      {/* Icon */}
      <div className="flex-shrink-0 flex items-center justify-center w-9 h-9 rounded-lg mt-0.5 bg-tone/15 border border-tone/30">
        {isOffline ? (
          <WifiOff className="size-5 text-tone" aria-hidden="true" />
        ) : (
          <TriangleAlert className="size-5 text-tone" aria-hidden="true" />
        )}
      </div>

      {/* Text */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-tone">
          {isOffline ? 'No connection to the server' : 'Something went wrong'}
        </p>
        <p className="text-xs mt-1 text-tone/70">
          {message ?? (isOffline
            ? 'Check your internet connection and try again in a moment.'
            : 'Please try again. If it keeps failing, come back later.')}
        </p>
        {onRetry && (
          <button
            onClick={onRetry}
            className="mt-3 text-xs font-semibold px-3 py-1.5 rounded-lg transition-all duration-150 bg-tone/15 border border-tone/30 text-tone hover:bg-tone/25"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  )
}
