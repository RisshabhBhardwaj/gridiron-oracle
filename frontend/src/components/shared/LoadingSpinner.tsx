import clsx from 'clsx'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const SIZE_CLASSES = { sm: 'w-4 h-4 border-2', md: 'w-8 h-8 border-[3px]', lg: 'w-12 h-12 border-4' }

/**
 * A small spinner for work the visitor just started (a button that is saving).
 * Never use it for a page or panel that is loading: use the skeletons in
 * components/shared/Skeletons.tsx, which hold the layout.
 */
export function LoadingSpinner({ size = 'md', className }: LoadingSpinnerProps) {
  return (
    <div
      role="status"
      aria-label="Loading"
      className={clsx(
        'rounded-full animate-spin border-info/10 border-t-info shadow-[0_0_12px_rgb(var(--info)/0.3)]',
        SIZE_CLASSES[size],
        className,
      )}
    />
  )
}
