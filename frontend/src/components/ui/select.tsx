import { Select as SelectPrimitive } from '@base-ui/react/select'
import { Check, ChevronsUpDown } from 'lucide-react'
import { cn } from '@/lib/utils'

/**
 * Select on Base UI: a button that opens a listbox, with typeahead, arrow keys,
 * Home/End, Escape and screen-reader support from the primitive.
 *
 * Copied from the shadcn/ui Base UI select and adapted to this project's
 * Tailwind 4 and colour tokens (docs/decisions/0004). Scroll arrows and
 * separators are left out until something needs them.
 *
 * Usage:
 *   <Select items={[{ value: 'a', label: 'Option A' }]} value={v} onValueChange={setV}>
 *     <SelectTrigger aria-label="Stat"><SelectValue /></SelectTrigger>
 *     <SelectContent>{items.map(i => <SelectItem key={i.value} value={i.value}>{i.label}</SelectItem>)}</SelectContent>
 *   </Select>
 * Pass `items` so the closed trigger shows the label, not the raw value.
 */
const Select = SelectPrimitive.Root

function SelectGroup({ className, ...props }: SelectPrimitive.Group.Props) {
  return <SelectPrimitive.Group data-slot="select-group" className={cn('p-1', className)} {...props} />
}

function SelectLabel({ className, ...props }: SelectPrimitive.GroupLabel.Props) {
  return (
    <SelectPrimitive.GroupLabel
      data-slot="select-label"
      className={cn('px-2 py-1.5 text-xs font-semibold uppercase tracking-wider text-fg-muted', className)}
      {...props}
    />
  )
}

function SelectValue({ className, ...props }: SelectPrimitive.Value.Props) {
  return (
    <SelectPrimitive.Value
      data-slot="select-value"
      className={cn('truncate data-[placeholder]:text-fg-subtle', className)}
      {...props}
    />
  )
}

function SelectTrigger({ className, children, ...props }: SelectPrimitive.Trigger.Props) {
  return (
    <SelectPrimitive.Trigger
      data-slot="select-trigger"
      className={cn(
        'flex h-9 w-fit min-w-0 items-center justify-between gap-2 rounded-lg border border-line bg-surface-raised px-3 text-sm whitespace-nowrap text-fg select-none',
        'transition-colors motion-reduce:transition-none hover:border-line-strong data-[popup-open]:border-line-strong',
        'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
        'data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50',
        className,
      )}
      {...props}
    >
      {children}
      <SelectPrimitive.Icon className="flex-shrink-0 text-fg-muted">
        <ChevronsUpDown className="size-4" aria-hidden="true" />
      </SelectPrimitive.Icon>
    </SelectPrimitive.Trigger>
  )
}

function SelectContent({
  className,
  children,
  side = 'bottom',
  sideOffset = 4,
  align = 'start',
  ...props
}: SelectPrimitive.Popup.Props & Pick<SelectPrimitive.Positioner.Props, 'side' | 'sideOffset' | 'align'>) {
  return (
    <SelectPrimitive.Portal>
      <SelectPrimitive.Positioner
        side={side}
        sideOffset={sideOffset}
        align={align}
        // A dropdown that opens below its trigger is what people expect on the web.
        alignItemWithTrigger={false}
        className="isolate z-[70] outline-none"
      >
        <SelectPrimitive.Popup
          data-slot="select-content"
          className={cn(
            'max-h-[min(var(--available-height),20rem)] min-w-[var(--anchor-width)] origin-[var(--transform-origin)] overflow-y-auto rounded-lg border border-line-strong bg-surface-overlay text-fg outline-none',
            'shadow-[0_8px_32px_rgb(var(--shadow)/0.6)]',
            'transition-[opacity,transform] duration-100 ease-out motion-reduce:transition-none',
            'data-[starting-style]:scale-95 data-[starting-style]:opacity-0 data-[ending-style]:scale-95 data-[ending-style]:opacity-0',
            className,
          )}
          {...props}
        >
          <SelectPrimitive.List className="p-1">{children}</SelectPrimitive.List>
        </SelectPrimitive.Popup>
      </SelectPrimitive.Positioner>
    </SelectPrimitive.Portal>
  )
}

function SelectItem({ className, children, ...props }: SelectPrimitive.Item.Props) {
  return (
    <SelectPrimitive.Item
      data-slot="select-item"
      className={cn(
        'relative flex w-full cursor-default items-center gap-2 rounded-md py-1.5 pr-8 pl-2 text-sm text-fg outline-none select-none',
        'data-[highlighted]:bg-white/[0.08] data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
        className,
      )}
      {...props}
    >
      <SelectPrimitive.ItemText className="flex-1 truncate">{children}</SelectPrimitive.ItemText>
      <SelectPrimitive.ItemIndicator
        render={<span className="pointer-events-none absolute right-2 flex size-4 items-center justify-center text-accent" />}
      >
        <Check className="size-4" aria-hidden="true" />
      </SelectPrimitive.ItemIndicator>
    </SelectPrimitive.Item>
  )
}

export { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue }
