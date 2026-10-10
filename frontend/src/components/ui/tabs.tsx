import { Tabs as TabsPrimitive } from '@base-ui/react/tabs'
import { cn } from '@/lib/utils'

/**
 * Tabs on Base UI: role=tablist / tab / tabpanel, roving focus, arrow keys, Home
 * and End. Copied from the shadcn/ui Base UI tabs and adapted to this project's
 * Tailwind 3 and colour tokens (docs/decisions/0004).
 *
 * Not used by a page yet. It is meant to replace the hand-rolled tab switchers
 * (Current Season, Draft board mode) as those pages are migrated.
 */
function Tabs({ className, ...props }: TabsPrimitive.Root.Props) {
  return <TabsPrimitive.Root data-slot="tabs" className={cn('flex flex-col gap-4', className)} {...props} />
}

function TabsList({ className, ...props }: TabsPrimitive.List.Props) {
  return (
    <TabsPrimitive.List
      data-slot="tabs-list"
      // Panels render instantly, so arrowing to a tab selects it (the ARIA
      // authoring guide's "automatic activation").
      activateOnFocus
      className={cn('inline-flex w-fit items-center rounded-lg border border-line bg-surface-overlay p-1', className)}
      {...props}
    />
  )
}

function TabsTrigger({ className, ...props }: TabsPrimitive.Tab.Props) {
  return (
    <TabsPrimitive.Tab
      data-slot="tabs-trigger"
      className={cn(
        'inline-flex items-center justify-center rounded-md px-4 py-1.5 text-xs font-semibold whitespace-nowrap text-fg-muted select-none',
        'transition-colors motion-reduce:transition-none hover:text-fg',
        'data-[active]:bg-accent data-[active]:text-accent-fg data-[active]:shadow',
        'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
        'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
        className,
      )}
      {...props}
    />
  )
}

function TabsContent({ className, ...props }: TabsPrimitive.Panel.Props) {
  return (
    <TabsPrimitive.Panel
      data-slot="tabs-content"
      className={cn('outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent', className)}
      {...props}
    />
  )
}

export { Tabs, TabsContent, TabsList, TabsTrigger }
