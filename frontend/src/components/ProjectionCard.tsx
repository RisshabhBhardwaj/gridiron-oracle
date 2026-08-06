import { useEffect, useState } from 'react'

interface ProjectionCardProps {
  label: string
  value: number
  unit?: string
  colorClass?: string
  subtext?: string
}

type ColorKey = 'text-oracle-blue' | 'text-oracle-green' | 'text-oracle-amber' | 'text-oracle-red' | 'text-oracle-purple' | string

const GLOW_COLORS: Record<string, string> = {
  'text-oracle-blue':   'rgba(0,194,255,0.3)',
  'text-oracle-accent': 'rgba(0,194,255,0.3)',
  'text-oracle-cyan':   'rgba(0,194,255,0.3)',
  'text-oracle-green':  'rgba(0,255,163,0.3)',
  'text-oracle-emerald':'rgba(0,255,163,0.3)',
  'text-oracle-amber':  'rgba(255,184,0,0.3)',
  'text-oracle-yellow': 'rgba(255,184,0,0.3)',
  'text-oracle-red':    'rgba(255,59,92,0.3)',
  'text-oracle-pink':   'rgba(255,59,92,0.3)',
  'text-oracle-purple': 'rgba(168,85,247,0.3)',
  'text-oracle-violet': 'rgba(168,85,247,0.3)',
}

const HEX_COLORS: Record<string, string> = {
  'text-oracle-blue':   '#00C2FF',
  'text-oracle-accent': '#00C2FF',
  'text-oracle-green':  '#00FFA3',
  'text-oracle-emerald':'#00FFA3',
  'text-oracle-amber':  '#FFB800',
  'text-oracle-yellow': '#FFB800',
  'text-oracle-red':    '#FF3B5C',
  'text-oracle-pink':   '#FF3B5C',
  'text-oracle-purple': '#A855F7',
  'text-oracle-violet': '#A855F7',
}

function useCountUp(target: number, duration = 600): number {
  const [current, setCurrent] = useState(0)
  useEffect(() => {
    const start = performance.now()
    const step = (now: number) => {
      const elapsed = now - start
      const progress = Math.min(elapsed / duration, 1)
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3)
      setCurrent(target * eased)
      if (progress < 1) requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  }, [target, duration])
  return current
}

export function ProjectionCard({ label, value, unit, colorClass = 'text-oracle-blue', subtext }: ProjectionCardProps) {
  const animated = useCountUp(value)
  const glow = GLOW_COLORS[colorClass as ColorKey] ?? 'rgba(0,194,255,0.3)'
  const hex  = HEX_COLORS[colorClass as ColorKey]  ?? '#00C2FF'

  return (
    <div
      className="relative rounded-xl p-4 flex flex-col gap-2 overflow-hidden transition-all duration-300 group cursor-default"
      style={{
        background: 'linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%)',
        border: `1px solid ${glow.replace('0.3', '0.2')}`,
        boxShadow: `0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05)`,
      }}
    >
      {/* Corner glow */}
      <div
        className="absolute top-0 right-0 w-16 h-16 rounded-bl-full opacity-40"
        style={{ background: `radial-gradient(circle at top right, ${glow} 0%, transparent 70%)` }}
      />

      <span className="section-label">{label}</span>
      <div className="flex items-baseline gap-1.5">
        <span
          className="font-display text-4xl leading-none transition-all"
          style={{ color: hex, textShadow: `0 0 20px ${glow}` }}
        >
          {animated.toFixed(animated >= 10 ? 1 : 1)}
        </span>
        {unit && (
          <span className="text-xs font-medium" style={{ color: `${hex}80` }}>
            {unit}
          </span>
        )}
      </div>
      {subtext && <span className="text-xs text-oracle-muted">{subtext}</span>}
    </div>
  )
}
