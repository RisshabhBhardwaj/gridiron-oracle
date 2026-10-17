import { Switch as SwitchPrimitive } from '@base-ui/react/switch'
import { cn } from '@/lib/utils'
import { toneClass, type Tone } from '@/lib/tokens'

/**
 * Switch on Base UI: role=switch with arrow-key support from the primitive.
 * Styled to the project tokens (docs/decisions/0004).
 *
 * Usage:
 *   <Switch checked={dome} onCheckedChange={setDome} aria-label="Dome game" />
 */
function Switch({
  className,
  tone = 'accent',
  ...props
}: SwitchPrimitive.Root.Props & {
  /** Colour of the track and thumb when on. */
  tone?: Tone
}) {
  return (
    <SwitchPrimitive.Root
      data-slot="switch"
      className={cn(
        toneClass(tone),
        'relative flex h-5 w-9 shrink-0 items-center rounded-full border border-line-strong bg-surface-overlay p-0.5 select-none',
        'transition-colors motion-reduce:transition-none data-[checked]:border-tone data-[checked]:bg-tone/40',
        'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
        'data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50',
        className,
      )}
      {...props}
    >
      <SwitchPrimitive.Thumb
        data-slot="switch-thumb"
        className={cn(
          'block size-3.5 rounded-full bg-fg-muted',
          'transition-transform motion-reduce:transition-none data-[checked]:translate-x-4 data-[checked]:bg-tone',
        )}
      />
    </SwitchPrimitive.Root>
  )
}

export { Switch }
