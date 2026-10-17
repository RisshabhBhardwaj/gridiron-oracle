import { describe, expect, it } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import { DataTable, type TableColumn } from '../components/ui/data-table'

interface Row {
  position: string
  games: number
  miss: number
}

const COLUMNS: TableColumn<Row>[] = [
  { header: 'Position', accessorKey: 'position' },
  { header: 'Games', accessorKey: 'games', align: 'right', mono: true },
  {
    header: 'Average miss',
    accessorKey: 'miss',
    align: 'right',
    mono: true,
    cell: ({ row }) => <span>{row.original.miss.toFixed(2)}</span>,
  },
]

const ROWS: Row[] = [
  { position: 'WR', games: 120, miss: 8.42 },
  { position: 'RB', games: 98, miss: 7.11 },
]

describe('DataTable', () => {
  it('renders headers once and one row per record with formatted cells', () => {
    render(<DataTable columns={COLUMNS} data={ROWS} />)
    const table = screen.getByRole('table')
    const headers = within(table).getAllByRole('columnheader')
    expect(headers.map((h) => h.textContent)).toEqual(['Position', 'Games', 'Average miss'])
    expect(within(table).getAllByRole('row')).toHaveLength(3) // header + 2 records
    expect(screen.getByText('WR')).toBeInTheDocument()
    expect(screen.getByText('8.42')).toBeInTheDocument()
    expect(screen.getByText('98')).toBeInTheDocument()
  })

  it('right-aligns numeric columns in tabular figures', () => {
    render(<DataTable columns={COLUMNS} data={ROWS} />)
    expect(screen.getAllByRole('columnheader')[1]).toHaveClass('text-right')
    const cell = screen.getByText('120').closest('td')
    expect(cell).toHaveClass('text-right')
    expect(cell).toHaveClass('font-mono')
  })

  it('renders headers with no rows when data is empty', () => {
    render(<DataTable columns={COLUMNS} data={[]} />)
    expect(screen.getByRole('table')).toBeInTheDocument()
    expect(screen.queryByText('WR')).not.toBeInTheDocument()
  })
})
