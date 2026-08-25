import clsx from 'clsx'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const SIZE_CLASSES = { sm: 'w-4 h-4', md: 'w-8 h-8', lg: 'w-12 h-12' }
const BORDER_SIZES = { sm: 2, md: 3, lg: 4 }

export function LoadingSpinner({ size = 'md', className }: LoadingSpinnerProps) {
  const b = BORDER_SIZES[size]
  return (
    <div
      role="status"
      aria-label="Loading"
      className={clsx('rounded-full animate-spin', SIZE_CLASSES[size], className)}
      style={{
        border: `${b}px solid rgba(0,194,255,0.1)`,
        borderTopColor: '#00C2FF',
        boxShadow: '0 0 12px rgba(0,194,255,0.3)',
      }}
    />
  )
}

/**
 * Skeleton cards placeholder for loading states.
 *
 * NOTE: this component renders its OWN responsive grid. Call it once with
 * `count`; do not map over it inside another grid. CurrentSeason, DraftBoard
 * and GameDetail each wrapped it in a second grid and repeated it, producing a
 * stack of nested grids (8 x 6 = 48 cards on CurrentSeason) instead of one row
 * of placeholders.
 */
export function SkeletonCard({ count = 6 }: { count?: number }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {Array.from({ length: count }).map((_, i) => {
        // The stagger has to live on the elements that actually carry the
        // shimmer animation. It used to sit on this wrapper, which has no
        // animation of its own, so every card shimmered in lockstep.
        const delay = { animationDelay: `${i * 0.08}s` }
        return (
          <div
            key={i}
            className="rounded-xl p-5 flex flex-col gap-3"
            style={{
              background: 'rgba(13,27,48,0.8)',
              border: '1px solid rgba(255,255,255,0.05)',
            }}
          >
            <div className="flex justify-between items-center">
              <div className="skeleton h-4 w-28 rounded" style={delay} />
              <div className="skeleton h-5 w-8 rounded" style={delay} />
            </div>
            <div className="skeleton h-8 w-20 rounded" style={delay} />
            <div className="skeleton h-2 w-full rounded-full" style={delay} />
            <div className="flex gap-2">
              <div className="skeleton h-4 w-12 rounded-full" style={delay} />
              <div className="skeleton h-4 w-12 rounded-full" style={delay} />
            </div>
          </div>
        )
      })}
    </div>
  )
}
