import { flexRender, type RowData } from '@tanstack/react-table'
import {
  getCoreRowModel,
  useLegacyTable,
  type LegacyColumnDef,
} from '@tanstack/react-table/legacy'
import { cn } from '@/lib/utils'

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
              return (
                <th
                  key={header.id}
                  scope="col"
                  className={cn(
                    'px-4 py-3 align-bottom text-xs font-semibold uppercase tracking-wider',
                    alignClass(def.align),
                    def.hideOnPhone && PHONE_HIDDEN,
                    stickyHeader && 'sticky top-13 z-10 border-b border-line/60 bg-surface-raised',
                  )}
                >
                  {flexRender(header.column.columnDef.header, header.getContext())}
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
                  className={cn('px-4 py-3', alignClass(def.align), def.mono && 'font-mono tabular-nums', def.hideOnPhone && PHONE_HIDDEN)}
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
