/**
 * Bridges the colour tokens in src/index.css to TypeScript.
 *
 * Components never contain hex or rgb() values. They use Tailwind classes
 * (text-accent, bg-negative/10) or, when the colour is chosen at run time, a
 * Tone: put `toneClass(tone)` on the root, then use text-tone / border-tone/20.
 * Charts draw SVG, which cannot use Tailwind classes, so they use `token()`.
 */

export type Tone = 'accent' | 'positive' | 'negative' | 'warning' | 'info' | 'highlight' | 'muted'

// Full literal class names so Tailwind and the reader can both find them.
const TONE_CLASS: Record<Tone, string> = {
  accent: 'tone-accent',
  positive: 'tone-positive',
  negative: 'tone-negative',
  warning: 'tone-warning',
  info: 'tone-info',
  highlight: 'tone-highlight',
  muted: 'tone-muted',
}

export function toneClass(tone: Tone): string {
  return TONE_CLASS[tone]
}

/** One colour per position, everywhere. */
const POSITION_TONE: Record<string, Tone> = {
  WR: 'info',
  RB: 'positive',
  TE: 'highlight',
  QB: 'warning',
}

export function positionTone(position: string | null | undefined): Tone {
  return (position && POSITION_TONE[position]) || 'muted'
}

type TokenName =
  | 'surface' | 'surface-raised' | 'surface-overlay'
  | 'line' | 'line-strong'
  | 'fg' | 'fg-muted' | 'fg-subtle'
  | 'accent' | 'positive' | 'negative' | 'warning' | 'info' | 'highlight'
  | 'shadow'

/** A colour from index.css as a CSS value, for SVG props and chart styles. */
export function token(name: TokenName, alpha?: number): string {
  return alpha === undefined ? `rgb(var(--${name}))` : `rgb(var(--${name}) / ${alpha})`
}
