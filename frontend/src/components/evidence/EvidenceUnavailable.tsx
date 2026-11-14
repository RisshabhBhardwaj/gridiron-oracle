import { Link } from 'react-router-dom'
import { Hourglass } from 'lucide-react'

/** Designed "not available" state: calm, says what to do next, shows no raw error. */
export function EvidenceUnavailable({ onRetry }: { onRetry?: () => void }) {
  return (
    <div role="status" className="tone-info flex items-start gap-4 rounded-xl border border-tone/25 bg-tone/[0.06] p-5">
      <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-tone/15 text-tone">
        <Hourglass className="size-5" aria-hidden="true" />
      </span>
      <div className="flex flex-col gap-1.5">
        <p className="text-base font-semibold text-fg">Past results aren't available right now</p>
        <p className="text-sm text-fg-muted">
          We're refreshing the numbers for the current model. Please check back soon, or look at{' '}
          <Link to="/projections" className="text-accent underline-offset-4 hover:underline">
            this season's projections
          </Link>{' '}
          in the meantime.
        </p>
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="mt-2 w-fit rounded-lg border border-tone/30 bg-tone/15 px-3 py-1.5 text-xs font-semibold text-tone transition-colors hover:bg-tone/25 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            Try again
          </button>
        )}
      </div>
    </div>
  )
}
