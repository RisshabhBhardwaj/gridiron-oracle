import { POSITIONS } from '@/lib/constants'
import { positionTone } from '@/lib/tokens'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'

/** Position chips. A chip is disabled when the chosen stat has no results for it. */
export function PositionFilter({
  hasRows,
  value,
  onChange,
}: {
  hasRows: ReadonlySet<string>
  value: string[]
  onChange: (next: string[]) => void
}) {
  return (
    <ToggleGroup
      multiple
      aria-label="Positions"
      value={value}
      onValueChange={(next) => {
        if (next.length > 0) onChange(next)
      }}
    >
      {POSITIONS.map((pos) => {
        const enabled = hasRows.has(pos)
        return (
          <ToggleGroupItem
            key={pos}
            value={pos}
            tone={positionTone(pos)}
            disabled={!enabled}
            aria-label={enabled ? pos : `${pos}, no results for this stat`}
            title={enabled ? undefined : 'No results for this stat'}
          >
            {pos}
          </ToggleGroupItem>
        )
      })}
    </ToggleGroup>
  )
}
