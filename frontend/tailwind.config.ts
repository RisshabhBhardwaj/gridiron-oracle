import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        oracle: {
          // Base
          dark:    '#0D0D0D',
          deep:    '#101010',
          navy:    '#111111',
          card:    '#141414',
          surface: '#1A1A1A',
          border:  '#262626',
          line:    '#303030',
          // Accents
          blue:    '#59A6FF',
          cyan:    '#77D7FF',
          green:   '#42D889',
          emerald: '#2BBF75',
          amber:   '#FFB84D',
          gold:    '#FFD166',
          red:     '#FF5C77',
          pink:    '#FF7892',
          purple:  '#B789FF',
          violet:  '#8E6BFF',
          // Text
          white:   '#F3F5F7',
          muted:   '#8B93A1',
          subtle:  '#5F6875',
        },
      },
      fontFamily: {
        sans:    ['Space Grotesk', 'system-ui', 'sans-serif'],
        display: ['Bebas Neue', 'Impact', 'sans-serif'],
        mono:    ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      boxShadow: {
        'neon-blue':   '0 0 20px rgba(0,194,255,0.35), 0 0 60px rgba(0,194,255,0.12)',
        'neon-green':  '0 0 20px rgba(0,255,163,0.35), 0 0 60px rgba(0,255,163,0.12)',
        'neon-amber':  '0 0 20px rgba(255,184,0,0.35), 0 0 60px rgba(255,184,0,0.12)',
        'neon-red':    '0 0 20px rgba(255,59,92,0.35), 0 0 60px rgba(255,59,92,0.12)',
        'neon-purple': '0 0 20px rgba(168,85,247,0.35), 0 0 60px rgba(168,85,247,0.12)',
        'card':        '0 4px 24px rgba(0,0,0,0.4), 0 1px 0 rgba(255,255,255,0.04) inset',
        'card-hover':  '0 8px 40px rgba(0,0,0,0.6), 0 1px 0 rgba(255,255,255,0.06) inset',
        'inner-glow':  'inset 0 1px 0 rgba(255,255,255,0.06)',
      },
      backgroundImage: {
        'grid-pattern':    'linear-gradient(rgba(0,194,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,194,255,0.03) 1px, transparent 1px)',
        'hero-gradient':   'radial-gradient(ellipse at 30% 0%, rgba(0,194,255,0.15) 0%, transparent 60%), radial-gradient(ellipse at 70% 100%, rgba(0,255,163,0.08) 0%, transparent 60%)',
        'card-gradient':   'linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%)',
        'blue-fade':       'linear-gradient(135deg, rgba(0,194,255,0.2) 0%, rgba(0,194,255,0.05) 100%)',
        'green-fade':      'linear-gradient(135deg, rgba(0,255,163,0.2) 0%, rgba(0,255,163,0.05) 100%)',
        'amber-fade':      'linear-gradient(135deg, rgba(255,184,0,0.2) 0%, rgba(255,184,0,0.05) 100%)',
        'red-fade':        'linear-gradient(135deg, rgba(255,59,92,0.2) 0%, rgba(255,59,92,0.05) 100%)',
        'purple-fade':     'linear-gradient(135deg, rgba(168,85,247,0.2) 0%, rgba(168,85,247,0.05) 100%)',
        'range-bar':       'linear-gradient(90deg, rgba(255,59,92,0.5) 0%, rgba(0,194,255,0.6) 50%, rgba(0,255,163,0.5) 100%)',
      },
      backgroundSize: {
        'grid': '32px 32px',
      },
      keyframes: {
        'count-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-in-right': {
          '0%': { opacity: '0', transform: 'translateX(16px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 6px rgba(0,194,255,0.4)' },
          '50%':       { boxShadow: '0 0 20px rgba(0,194,255,0.8), 0 0 40px rgba(0,194,255,0.3)' },
        },
        'shimmer': {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        'bar-fill': {
          '0%': { width: '0%' },
          '100%': { width: '100%' },
        },
        'ping-slow': {
          '0%': { transform: 'scale(1)', opacity: '1' },
          '75%, 100%': { transform: 'scale(1.8)', opacity: '0' },
        },
      },
      animation: {
        'count-up':     'count-up 0.4s ease-out forwards',
        'fade-in':      'fade-in 0.5s ease-out forwards',
        'slide-right':  'slide-in-right 0.35s ease-out forwards',
        'pulse-glow':   'pulse-glow 2s ease-in-out infinite',
        'shimmer':      'shimmer 1.5s ease-in-out infinite',
        'ping-slow':    'ping-slow 2s cubic-bezier(0, 0, 0.2, 1) infinite',
      },
      transitionTimingFunction: {
        'spring': 'cubic-bezier(0.34, 1.56, 0.64, 1)',
      },
    },
  },
  plugins: [],
} satisfies Config
