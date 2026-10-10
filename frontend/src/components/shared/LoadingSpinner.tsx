import clsx from 'clsx'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const SIZE_CLASSES = { sm: 'w-4 h-4 border-2', md: 'w-8 h-8 border-[3px]', lg: 'w-12 h-12 border-4' }

export function LoadingSpinner({ size = 'md', className }: LoadingSpinnerProps) {
  return (
    <div
      role="status"
      aria-label="Loading"
      className={clsx(
        'rounded-full animate-spin border-info/10 border-t-info shadow-[0_0_12px_rgb(var(--info)/0.3)]',
        SIZE_CLASSES[size],
        className,
      )}
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
            className="rounded-xl p-5 flex flex-col gap-3 bg-surface-raised border border-white/5"
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
