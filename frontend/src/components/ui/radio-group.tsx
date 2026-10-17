import { RadioGroup as RadioGroupPrimitive } from '@base-ui/react/radio-group'
import { Radio as RadioPrimitive } from '@base-ui/react/radio'
import { cn } from '@/lib/utils'
import { toneClass } from '@/lib/tokens'

/**
 * RadioGroup on Base UI: role=radiogroup with arrow-key navigation and roving
 * focus from the primitive. Styled to the project tokens (docs/decisions/0004).
 *
 * Usage:
 *   <RadioGroup value={scoring} onValueChange={setScoring} aria-label="Scoring">
 *     <RadioGroupItem value="ppr" label="Full PPR" />
 *   </RadioGroup>
 */
function RadioGroup({ className, ...props }: RadioGroupPrimitive.Props) {
  return (
    <RadioGroupPrimitive
      data-slot="radio-group"
      className={cn('flex flex-col items-start gap-2', className)}
      {...props}
    />
  )
}

function RadioGroupItem({
  className,
  label,
  description,
  variant = 'row',
  ...props
}: RadioPrimitive.Root.Props & {
  label: React.ReactNode
  description?: React.ReactNode
  /** 'card' renders a large selectable card (scoring choice). */
  variant?: 'row' | 'card'
}) {
  if (variant === 'card') {
    return (
      <RadioPrimitive.Root
        data-slot="radio-group-item"
        className={cn(
          toneClass('info'),
          'group flex flex-col gap-1 p-4 rounded-xl text-left transition-all motion-reduce:transition-none border disabled:cursor-not-allowed',
          'data-[checked]:bg-tone/10 data-[checked]:border-tone/40 data-[checked]:shadow-[0_0_16px_rgb(var(--tone)/0.1)]',
          'data-[unchecked]:bg-surface-overlay/30 data-[unchecked]:border-white/[0.06] data-[unchecked]:hover:border-white/15',
          'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
          'disabled:opacity-60',
          className,
        )}
        {...props}
      >
        <span className="text-sm font-bold text-fg-muted group-data-[checked]:text-tone">{label}</span>
        {description && <span className="text-[10px] text-fg-muted">{description}</span>}
      </RadioPrimitive.Root>
    )
  }
  return (
    <label className="flex cursor-pointer items-center gap-2.5 text-sm text-fg select-none">
      <RadioPrimitive.Root
        data-slot="radio-group-item"
        className={cn(
          'flex size-4 shrink-0 items-center justify-center rounded-full border border-line-strong bg-surface-raised',
          'transition-colors motion-reduce:transition-none data-[checked]:border-accent',
          'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
          'data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50',
          className,
        )}
        {...props}
      >
        <RadioPrimitive.Indicator
          data-slot="radio-group-indicator"
          className="flex items-center justify-center data-[unchecked]:hidden before:size-2 before:rounded-full before:bg-accent"
        />
      </RadioPrimitive.Root>
      {label}
    </label>
  )
}

export { RadioGroup, RadioGroupItem }
