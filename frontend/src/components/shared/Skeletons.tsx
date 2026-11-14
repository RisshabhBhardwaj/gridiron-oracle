import type { CSSProperties, ReactNode } from 'react'
import clsx from 'clsx'

/**
 * Loading placeholders, one per layout shape: page header, card grid, table rows
 * and chart frame. All are built from the global `.skeleton` class, which stops
 * shimmering when the visitor asks for reduced motion (see index.css).
 *
 * Each one holds the space the real content takes, so the page does not jump when
 * the data arrives. Wrap a set of them in `LoadingRegion` so a screen reader hears
 * one "Loading …" instead of silence.
 */

/** One shimmering block. Size it with classes. */
export function Skeleton({ className, style }: { className?: string; style?: CSSProperties }) {
  return <div className={clsx('skeleton rounded', className)} style={style} aria-hidden="true" />
}

/** Announces a loading area once and marks it busy. */
export function LoadingRegion({ label, className, children }: { label: string; className?: string; children: ReactNode }) {
  return (
    <div role="status" aria-busy="true" className={className}>
      <span className="sr-only">{label}</span>
      {children}
    </div>
  )
}

/** Page title and one line of subtitle: the shape of every page header. */
export function PageHeaderSkeleton() {
  return (
    <div className="flex flex-col gap-2 border-b border-line/60 pb-5" aria-hidden="true">
      <Skeleton className="h-8 w-56" />
      <Skeleton className="h-3.5 w-full max-w-md" />
    </div>
  )
}

/**
 * A responsive grid of cards. Renders its OWN grid: call it once with `count`,
 * never inside another grid. `compact` is the small tile (team win totals).
 */
export function CardGridSkeleton({
  count = 8,
  compact = false,
  wide = false,
}: {
  count?: number
  /** Small tiles (team win totals). */
  compact?: boolean
  /** Two-up wide cards (matchups). */
  wide?: boolean
}) {
  return (
    <div
      aria-hidden="true"
      className={
        compact
          ? 'grid grid-cols-2 gap-2 sm:grid-cols-4 md:grid-cols-8'
          : wide
            ? 'grid grid-cols-1 gap-4 md:grid-cols-2'
            : 'grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4'
      }
    >
      {Array.from({ length: count }).map((_, i) => {
        // The stagger sits on the elements that carry the shimmer.
        const delay = { animationDelay: `${i * 0.08}s` }
        if (wide) {
          return (
            <div key={i} className="flex h-[8.4rem] flex-col justify-between rounded-xl border border-line/50 bg-surface-raised/40 p-4">
              <Skeleton className="h-3 w-16" style={delay} />
              <div className="flex items-end justify-between">
                <Skeleton className="h-9 w-20" style={delay} />
                <Skeleton className="h-9 w-20" style={delay} />
              </div>
            </div>
          )
        }
        return compact ? (
          <div key={i} className="flex h-[4.75rem] flex-col items-center justify-center gap-1.5 rounded-lg border border-line/50 bg-surface-raised/60">
            <Skeleton className="h-3 w-8" style={delay} />
            <Skeleton className="h-5 w-10" style={delay} />
          </div>
        ) : (
          <div key={i} className="flex flex-col gap-3 rounded-xl border border-white/5 bg-surface-raised p-4">
            <div className="flex items-center justify-between">
              <Skeleton className="h-4 w-28" style={delay} />
              <Skeleton className="h-5 w-8" style={delay} />
            </div>
            <Skeleton className="h-9 w-20" style={delay} />
            <Skeleton className="h-1.5 w-full rounded-full" style={delay} />
            <div className="flex justify-between">
              <Skeleton className="h-3 w-12" style={delay} />
              <Skeleton className="h-3 w-12" style={delay} />
            </div>
          </div>
        )
      })}
    </div>
  )
}

/** Table-shaped rows inside the same bordered frame the data tables use. */
export function TableSkeleton({ rows = 6, columns = 5 }: { rows?: number; columns?: number }) {
  return (
    <div aria-hidden="true" className="overflow-hidden rounded-xl border border-line">
      <div className="flex gap-4 border-b border-line/60 bg-surface-overlay/40 px-3 py-3">
        {Array.from({ length: columns }).map((_, c) => (
          <Skeleton key={c} className={clsx('h-3', c === 0 ? 'w-16' : 'ml-auto w-14')} />
        ))}
      </div>
      <div className="divide-y divide-line/30">
        {Array.from({ length: rows }).map((_, r) => (
          <div key={r} className="flex gap-4 px-3 py-3.5">
            {Array.from({ length: columns }).map((_, c) => (
              <Skeleton key={c} className={clsx('h-4', c === 0 ? 'w-16' : 'ml-auto w-14')} style={{ animationDelay: `${r * 0.06}s` }} />
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

/** A chart's frame: the same height the chart will take, so nothing moves under it. */
export function ChartFrameSkeleton({ height = 280 }: { height?: number }) {
  return (
    <div aria-hidden="true" className="rounded-xl border border-white/[0.07] bg-panel-gradient p-5">
      <Skeleton className="mb-4 h-4 w-64 max-w-full" />
      <Skeleton className="w-full rounded-lg" style={{ height }} />
    </div>
  )
}
