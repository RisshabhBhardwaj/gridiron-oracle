/**
 * Loading state shaped like the page it stands in for: the hero figure with its range
 * bar, three supporting tiles, and the two panels. It is shown until the projection
 * request settles, so the page never flashes "no forecast" or empty tiles first.
 */
export function PlayerDetailSkeleton() {
  return (
    <div role="status" aria-label="Loading projection" className="flex flex-col gap-5">
      <div className="rounded-xl border border-white/[0.07] p-5 sm:p-6">
        <div className="skeleton h-3 w-40 rounded" />
        <div className="skeleton mt-4 h-14 w-44 rounded" />
        <div className="skeleton mt-8 h-3 w-full rounded-full" />
        <div className="mt-4 flex justify-between">
          <div className="skeleton h-9 w-16 rounded" />
          <div className="skeleton h-9 w-16 rounded" />
        </div>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="skeleton h-24 rounded-xl" />
        ))}
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="skeleton h-72 rounded-xl" />
        <div className="skeleton h-72 rounded-xl" />
      </div>
    </div>
  )
}
