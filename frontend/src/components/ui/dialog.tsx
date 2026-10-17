import type { ComponentProps } from 'react'
import { Dialog as DialogPrimitive } from '@base-ui/react/dialog'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'

/**
 * Modal dialog on Base UI. The primitive traps focus inside, returns focus to
 * the trigger on close, closes on Escape and on a click outside, locks page
 * scroll, and labels the dialog from DialogTitle / DialogDescription.
 *
 * Copied from the shadcn/ui Base UI dialog and adapted to this project's
 * Tailwind 4 and colour tokens (docs/decisions/0004).
 *
 * Every dialog needs a DialogTitle (use className="sr-only" to hide it).
 */
const Dialog = DialogPrimitive.Root
const DialogTrigger = DialogPrimitive.Trigger
const DialogClose = DialogPrimitive.Close

function DialogOverlay({ className, ...props }: DialogPrimitive.Backdrop.Props) {
  return (
    <DialogPrimitive.Backdrop
      data-slot="dialog-overlay"
      className={cn(
        'fixed inset-0 z-[60] bg-surface/80 backdrop-blur-sm',
        'transition-opacity duration-150 motion-reduce:transition-none',
        'data-[starting-style]:opacity-0 data-[ending-style]:opacity-0',
        className,
      )}
      {...props}
    />
  )
}

function DialogContent({
  className,
  children,
  showCloseButton = true,
  ...props
}: DialogPrimitive.Popup.Props & { showCloseButton?: boolean }) {
  return (
    <DialogPrimitive.Portal>
      <DialogOverlay />
      <DialogPrimitive.Popup
        data-slot="dialog-content"
        className={cn(
          'fixed left-1/2 top-1/2 z-[61] grid w-[calc(100%-2rem)] max-w-md -translate-x-1/2 -translate-y-1/2 gap-4 rounded-xl p-5 text-sm text-fg outline-none',
          'border border-line-strong bg-surface-overlay shadow-[0_24px_80px_rgb(var(--shadow)/0.8)]',
          'transition-[opacity,transform] duration-150 ease-out motion-reduce:transition-none',
          'data-[starting-style]:scale-95 data-[starting-style]:opacity-0 data-[ending-style]:scale-95 data-[ending-style]:opacity-0',
          className,
        )}
        {...props}
      >
        {children}
        {showCloseButton && (
          <DialogPrimitive.Close
            data-slot="dialog-close"
            aria-label="Close"
            className="absolute right-3 top-3 rounded-md p-1.5 text-fg-muted transition-colors hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            <X className="size-4" aria-hidden="true" />
          </DialogPrimitive.Close>
        )}
      </DialogPrimitive.Popup>
    </DialogPrimitive.Portal>
  )
}

function DialogHeader({ className, ...props }: ComponentProps<'div'>) {
  return <div data-slot="dialog-header" className={cn('flex flex-col gap-1.5', className)} {...props} />
}

function DialogTitle({ className, ...props }: DialogPrimitive.Title.Props) {
  return (
    <DialogPrimitive.Title
      data-slot="dialog-title"
      className={cn('text-base font-semibold text-fg', className)}
      {...props}
    />
  )
}

function DialogDescription({ className, ...props }: DialogPrimitive.Description.Props) {
  return (
    <DialogPrimitive.Description
      data-slot="dialog-description"
      className={cn('text-sm text-fg-muted', className)}
      {...props}
    />
  )
}

export { Dialog, DialogClose, DialogContent, DialogDescription, DialogHeader, DialogOverlay, DialogTitle, DialogTrigger }
