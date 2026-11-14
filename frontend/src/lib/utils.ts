import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * Join class names and let the last one win when two set the same property
 * (cn('px-2', 'px-4') -> 'px-4'). Every component in components/ui uses it so a
 * caller can override a default through className.
 *
 * tailwind-merge is pinned to ^2 because this project is on Tailwind 3. Move to
 * tailwind-merge 3 when the project moves to Tailwind 4.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** The visible keyboard-focus ring every button and link carries. */
export const FOCUS_RING =
  'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent'
