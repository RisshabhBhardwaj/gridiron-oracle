import { flexRender, type RowData } from '@tanstack/react-table'
import {
  getCoreRowModel,
  useLegacyTable,
  type LegacyColumnDef,
} from '@tanstack/react-table/legacy'
import { ArrowDown, ArrowUp, ChevronsUpDown } from 'lucide-react'
import { cn, FOCUS_RING } from '@/lib/utils'

/**
 * One themed table for every data grid. Headers, row dividers and the hover
 * wash come from the project tokens; cells keep full control through column
 * definitions, so numbers stay tabular and tones stay per-row.
 *
 * Built on the TanStack Table v8-style legacy API (fixed feature set, no
 * per-table `features` option needed for static grids).
 *
 * Usage:
 *   const columns: TableColumn<Row>[] = [
 *     { header: 'Position', accessorKey: 'position' },
 *     { header: 'Games', accessorKey: 'n_games', align: 'right', mono: true },
 *   ]
 *   <DataTable columns={columns} data={rows} />
 */
export type TableColumn<TData extends RowData> = LegacyColumnDef<TData> & {
  align?: 'left' | 'right' | 'center'
  mono?: boolean
  /** Hide this column on phones so the key columns fit without sideways scrolling. */
  hideOnPhone?: boolean
  /** Extra classes for this column's header and cells (for example tighter padding). */
  className?: string
  /** The header becomes a button that asks the table's `onSortChange` to sort by this column's `id`. */
  sortable?: boolean
}

/** Which column the data is sorted by, and which way. The caller sorts the data; the table shows the state. */
export interface TableSort {
  id: string
  desc: boolean
}

const PHONE_HIDDEN = 'hidden sm:table-cell'

function alignClass(align?: TableColumn<RowData>['align']): string {
  if (align === 'right') return 'text-right'
  if (align === 'center') return 'text-center'
  return 'text-left'
}

export function DataTable<TData extends RowData>({
  columns,
  data,
  className,
  stickyHeader,
  caption,
  sort,
  onSortChange,
}: {
  columns: TableColumn<TData>[]
  data: TData[]
  className?: string
  /**
   * Keep the header row in view while the page scrolls (it sticks under the
   * top nav). Needs a parent without `overflow-hidden/auto/scroll`; use
   * `overflow-clip` for rounded corners.
   */
  stickyHeader?: boolean
  /** Screen-reader name for the table (what it lists). */
  caption?: string
  /** Current sort, announced to screen readers through `aria-sort` on the header. */
  sort?: TableSort
  /** Called with a sortable column's `id` when its header is activated. */
  onSortChange?: (columnId: string) => void
}) {
  const table = useLegacyTable({
    columns,
    data,
    getCoreRowModel: getCoreRowModel(),
  })

  return (
    <table className={cn('w-full text-left text-sm', className)}>
      {caption && <caption className="sr-only">{caption}</caption>}
      <thead>
        {table.getHeaderGroups().map((group) => (
          <tr key={group.id} className="border-b border-line/60 bg-surface-overlay/40 text-fg-muted">
            {group.headers.map((header) => {
              const def = header.column.columnDef as TableColumn<TData>
              const sortable = Boolean(def.sortable && onSortChange)
              const active = sortable && sort?.id === header.column.id
              const content = flexRender(header.column.columnDef.header, header.getContext())
              const SortIcon = active ? (sort?.desc ? ArrowDown : ArrowUp) : ChevronsUpDown
              return (
                <th
                  key={header.id}
                  scope="col"
                  aria-sort={sortable ? (active ? (sort?.desc ? 'descending' : 'ascending') : 'none') : undefined}
                  className={cn(
                    'px-4 py-3 align-bottom text-xs font-semibold uppercase tracking-wider',
                    alignClass(def.align),
                    def.hideOnPhone && PHONE_HIDDEN,
                    stickyHeader && 'sticky top-13 z-10 border-b border-line/60 bg-surface-raised',
                    def.className,
                  )}
                >
                  {sortable ? (
                    <button
                      type="button"
                      onClick={() => onSortChange?.(header.column.id)}
                      className={cn(
                        'inline-flex items-center gap-1 rounded uppercase tracking-wider transition-colors hover:text-fg',
                        def.align === 'right' && 'flex-row-reverse',
                        active && 'text-fg',
                        FOCUS_RING,
                      )}
                    >
                      <span>{content}</span>
                      <SortIcon className={cn('size-3.5 shrink-0', !active && 'opacity-50')} aria-hidden="true" />
                    </button>
                  ) : (
                    content
                  )}
                </th>
              )
            })}
          </tr>
        ))}
      </thead>
      <tbody className="divide-y divide-line/30">
        {table.getRowModel().rows.map((row) => (
          <tr key={row.id} className="transition-colors hover:bg-surface-raised/40">
            {row.getVisibleCells().map((cell) => {
              const def = cell.column.columnDef as TableColumn<TData>
              return (
                <td
                  key={cell.id}
                  className={cn('px-4 py-3', alignClass(def.align), def.mono && 'font-mono tabular-nums', def.hideOnPhone && PHONE_HIDDEN, def.className)}
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              )
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
