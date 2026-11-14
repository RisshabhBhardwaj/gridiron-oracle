import { ArrowDown, ArrowUp, Download } from 'lucide-react'
import type { BacktestVerdict } from '@/types/api'
import { mixedMetricSentence, otherBaselinesSentence, toneGain, verdictSentence } from '@/lib/evidence'
import { cn } from '@/lib/utils'
import { toneClass, type Tone } from '@/lib/tokens'

/**
 * The page's one-line answer: better or not than the simple method, by how
 * much, with a link to the data behind it.
 */
export function EvidenceVerdict({
  verdict,
  noun,
  tableUrl,
  lead,
}: {
  verdict: BacktestVerdict
  /** What is counted, for example "season tests". */
  noun: string
  /** Path of the downloadable table, as the api returns it (no /api prefix). */
  tableUrl: string
  /** Optional small label above the sentence, for example "All stats, 2021 to 2025". */
  lead?: string
}) {
  const gain = toneGain(verdict)
  const mixed = mixedMetricSentence(verdict)
  const good = verdict.cell_seasons > 0 && verdict.beat_naive * 2 >= verdict.cell_seasons && (gain ?? 0) >= 0
  const tone: Tone = verdict.cell_seasons === 0 ? 'muted' : good ? 'positive' : 'warning'
  const Icon = good ? ArrowUp : ArrowDown
  return (
    <section
      aria-label="Verdict"
      className={cn(
        toneClass(tone),
        'flex flex-col gap-3 rounded-xl border border-tone/25 bg-tone/[0.06] p-5 sm:flex-row sm:items-start sm:gap-4',
      )}
    >
      {verdict.cell_seasons > 0 && (
        <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-tone/15 text-tone">
          <Icon className="size-5" aria-hidden="true" />
        </span>
      )}
      <div className="flex min-w-0 flex-1 flex-col gap-1.5">
        {lead && <p className="section-label">{lead}</p>}
        <p className="text-lg font-semibold leading-snug text-fg sm:text-xl">{verdictSentence(verdict, noun)}</p>
        {mixed && <p className="text-sm text-fg">{mixed}</p>}
        <p className="text-sm text-fg-muted">{otherBaselinesSentence(verdict)}</p>
        <a
          href={`/api${tableUrl}`}
          download
          className="mt-1 inline-flex w-fit items-center gap-1.5 rounded-md text-sm font-medium text-accent underline-offset-4 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          <Download className="size-4" aria-hidden="true" />
          Download all the numbers (CSV)
        </a>
      </div>
    </section>
  )
}
