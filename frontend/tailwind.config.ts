import type { Config } from 'tailwindcss'

/**
 * Colours have ONE source of truth: the CSS variables in src/index.css.
 * Each variable holds space-separated RGB channels so Tailwind opacity
 * modifiers keep working (bg-accent/15, border-line/40, ...).
 *
 * Use the semantic names below. Do not write hex or rgb() values in
 * components; add or change a variable in index.css instead.
 */
const token = (name: string) => `rgb(var(--${name}) / <alpha-value>)`

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Surfaces, from the page background up to hovered or floating layers
        surface: {
          DEFAULT: token('surface'),
          raised: token('surface-raised'),
          overlay: token('surface-overlay'),
        },
        // Borders and dividers
        line: {
          DEFAULT: token('line'),
          strong: token('line-strong'),
        },
        // Text ("foreground")
        fg: {
          DEFAULT: token('fg'),
          muted: token('fg-muted'),
          subtle: token('fg-subtle'),
        },
        // Brand and interactive emphasis (active nav, primary buttons, links)
        accent: {
          DEFAULT: token('accent'),
          fg: token('accent-fg'), // text on top of an accent background
        },
        // Meaning
        positive: token('positive'),
        negative: token('negative'),
        warning: token('warning'),
        info: token('info'),
        highlight: token('highlight'),
        // Whatever colour the nearest `.tone-*` ancestor set (see index.css).
        // Lets one component take a tone prop instead of a colour.
        tone: token('tone'),
      },
      fontFamily: {
        sans:    ['Space Grotesk', 'system-ui', 'sans-serif'],
        display: ['Bebas Neue', 'Impact', 'sans-serif'],
        mono:    ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      boxShadow: {
        card:         '0 4px 24px rgb(var(--shadow) / 0.4), 0 1px 0 rgb(255 255 255 / 0.04) inset',
        'card-hover': '0 8px 40px rgb(var(--shadow) / 0.6), 0 1px 0 rgb(255 255 255 / 0.06) inset',
      },
      backgroundImage: {
        // Faint white wash used on raised panels
        'panel-gradient': 'linear-gradient(135deg, rgb(255 255 255 / 0.04) 0%, rgb(255 255 255 / 0.01) 100%)',
        // Floor (bad week) to ceiling (great week) range bar
        'range-bar': 'linear-gradient(90deg, rgb(var(--negative) / 0.7) 0%, rgb(var(--info) / 0.8) 50%, rgb(var(--positive) / 0.7) 100%)',
      },
      keyframes: {
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'ping-slow': {
          '0%': { transform: 'scale(1)', opacity: '1' },
          '75%, 100%': { transform: 'scale(1.8)', opacity: '0' },
        },
      },
      animation: {
        'fade-in':   'fade-in 0.5s ease-out forwards',
        'ping-slow': 'ping-slow 2s cubic-bezier(0, 0, 0.2, 1) infinite',
      },
    },
  },
  plugins: [],
} satisfies Config
