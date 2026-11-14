import { BOARD_LABELS } from '@/lib/draft-board'

const ROWS = 10

// Same columns, alignment and phone visibility as the board, so nothing jumps when the data arrives.
const COLUMNS: Array<{ label: string; align: 'left' | 'right'; hideOnPhone?: boolean; bar: string }> = [
  { label: BOARD_LABELS.marketRank, align: 'right', bar: 'w-6' },
  { label: BOARD_LABELS.player, align: 'left', bar: 'w-28 sm:w-40' },
  { label: BOARD_LABELS.pos, align: 'left', hideOnPhone: true, bar: 'w-8' },
  { label: BOARD_LABELS.team, align: 'left', hideOnPhone: true, bar: 'w-8' },
  { label: BOARD_LABELS.avgPick, align: 'right', hideOnPhone: true, bar: 'w-10' },
  { label: BOARD_LABELS.ourRank, align: 'right', bar: 'w-6' },
  { label: BOARD_LABELS.ourPoints, align: 'right', hideOnPhone: true, bar: 'w-12' },
  { label: BOARD_LABELS.value, align: 'right', bar: 'w-12' },
]

/** Table-shaped placeholder for the draft board while it loads. */
export function DraftBoardSkeleton() {
  return (
    <div role="status" aria-label="Loading the draft board" className="overflow-clip rounded-xl border border-line">
      <table className="w-full text-left text-sm [&_td]:px-2 [&_th]:px-2 sm:[&_td]:px-3 sm:[&_th]:px-3" aria-hidden="true">
        <thead>
          <tr className="border-b border-line/60 bg-surface-overlay/40 text-fg-muted">
            {COLUMNS.map((c) => (
              <th
                key={c.label}
                className={`py-2 align-bottom text-xs font-semibold uppercase tracking-wider ${c.align === 'right' ? 'text-right' : 'text-left'} ${c.hideOnPhone ? 'hidden sm:table-cell' : ''}`}
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-line/30">
          {Array.from({ length: ROWS }).map((_, i) => (
            <tr key={i}>
              {COLUMNS.map((c) => (
                <td key={c.label} className={`py-3 ${c.hideOnPhone ? 'hidden sm:table-cell' : ''}`}>
                  <div
                    className={`skeleton h-4 rounded ${c.bar} ${c.align === 'right' ? 'ml-auto' : ''}`}
                    style={{ animationDelay: `${i * 0.06}s` }}
                  />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
