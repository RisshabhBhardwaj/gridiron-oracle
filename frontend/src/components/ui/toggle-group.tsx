import { ToggleGroup as ToggleGroupPrimitive } from '@base-ui/react/toggle-group'
import { Toggle as TogglePrimitive } from '@base-ui/react/toggle'
import { cn } from '@/lib/utils'
import { toneClass, type Tone } from '@/lib/tokens'

/**
 * ToggleGroup on Base UI: a set of pressed/unpressed buttons with shared
 * state. `multiple` allows more than one on at a time (position filters).
 * Styled to the project tokens (docs/decisions/0004).
 *
 * Usage:
 *   <ToggleGroup multiple value={positions} onValueChange={setPositions} aria-label="Positions">
 *     <ToggleGroupItem value="WR">WR</ToggleGroupItem>
 *   </ToggleGroup>
 */
function ToggleGroup({ className, ...props }: ToggleGroupPrimitive.Props<string>) {
  return (
    <ToggleGroupPrimitive
      data-slot="toggle-group"
      className={cn('flex items-center gap-1.5', className)}
      {...props}
    />
  )
}

function ToggleGroupItem({
  className,
  tone,
  ...props
}: TogglePrimitive.Props & {
  /** Pressed background colour (position filters); default is the accent. */
  tone?: Tone
}) {
  return (
    <TogglePrimitive
      data-slot="toggle-group-item"
      className={cn(
        tone && toneClass(tone),
        'px-2.5 py-1 text-xs font-semibold rounded-md border transition-colors motion-reduce:transition-none select-none',
        'border-line text-fg-muted hover:border-line-strong hover:text-fg',
        'data-[pressed]:border-transparent data-[pressed]:text-accent-fg',
        tone ? 'data-[pressed]:bg-tone' : 'data-[pressed]:bg-accent',
        'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
        'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
        className,
      )}
      {...props}
    />
  )
}

export { ToggleGroup, ToggleGroupItem }
