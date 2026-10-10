interface ErrorBannerProps {
  error?: Error | null
  onRetry?: () => void
  message?: string
}

export function ErrorBanner({ error, onRetry, message }: ErrorBannerProps) {
  const isOffline = error?.message?.includes('ECONNREFUSED') || error?.message?.includes('Failed to fetch')

  return (
    <div className="tone-negative relative rounded-xl p-5 flex items-start gap-4 overflow-hidden border border-tone/25 bg-gradient-to-br from-tone/10 to-tone/[0.03] shadow-[0_4px_24px_rgb(var(--tone)/0.08)]">
      {/* Glow edge */}
      <div className="absolute left-0 top-0 bottom-0 w-0.5 rounded-l-xl bg-gradient-to-b from-tone to-tone/20" />

      {/* Icon */}
      <div className="flex-shrink-0 flex items-center justify-center w-9 h-9 rounded-lg mt-0.5 bg-tone/15 border border-tone/30">
        {isOffline ? (
          <svg className="w-5 h-5 text-tone" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M18.364 5.636a9 9 0 010 12.728M15.536 8.464a5 5 0 010 7.072M12 12h.01M3 3l18 18" />
          </svg>
        ) : (
          <svg className="w-5 h-5 text-tone" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
          </svg>
        )}
      </div>

      {/* Text */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-tone">
          {isOffline ? 'Backend Offline' : 'Something went wrong'}
        </p>
        <p className="text-xs mt-1 text-tone/70">
          {message ?? (isOffline
            ? 'The API server is not running. Start it with `make up` or `docker compose -f infra/docker-compose.yml up -d`.'
            : error?.message ?? 'An unexpected error occurred. Please try again.')}
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
