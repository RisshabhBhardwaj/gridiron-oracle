import { cohortPlain } from '@/lib/evidence'

/** How to read the comparison: who is counted, and what each baseline is. */
export function BaselineGuide({ cohortRules }: { cohortRules: string[] }) {
  const items: [string, string][] = [
    ['Who is counted', cohortRules.map(cohortPlain).join(' ')],
    ['Simple method', 'Predict that a player repeats their last game.'],
    ['Trailing 3', 'Predict the average of their last three games.'],
    ['Marcel', 'A classic forecast that blends recent seasons and pulls extremes toward average.'],
    ['Fair test', 'Every prediction uses only what was known before that game.'],
  ]
  return (
    <dl className="grid grid-cols-1 gap-x-8 gap-y-3 text-sm sm:grid-cols-2">
      {items.map(([term, text]) => (
        <div key={term} className="flex flex-col gap-0.5">
          <dt className="text-xs font-semibold uppercase tracking-wider text-fg-muted">{term}</dt>
          <dd className="text-fg">{text}</dd>
        </div>
      ))}
    </dl>
  )
}
