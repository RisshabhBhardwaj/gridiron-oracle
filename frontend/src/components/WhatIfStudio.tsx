import { useState } from 'react'
import clsx from 'clsx'
import { FlaskConical, House, ShieldCheck, Thermometer, Wind, Zap, type LucideIcon } from 'lucide-react'
import { useScenario } from '@/hooks/useScenario'
import { LoadingSpinner } from '@/components/shared/LoadingSpinner'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import type { ScenarioOverrides } from '@/types/api'
import { toneClass, type Tone } from '@/lib/tokens'

interface WhatIfStudioProps {
  playerId: string
  week: number
  season: number
  stat: string
  baseProjection: number
}

interface SliderConfig {
  key: keyof Omit<ScenarioOverrides, 'dome_override'>
  label: string
  icon: LucideIcon
  min: number
  max: number
  step: number
  defaultValue: number
  format: (v: number) => string
  tone: Tone
}

const SLIDERS: SliderConfig[] = [
  {
    key: 'wind_speed_mph',
    label: 'Wind Speed',
    icon: Wind,
    min: 0, max: 60, step: 1, defaultValue: 5,
    format: (v) => `${v} mph`,
    tone: 'info',
  },
  {
    key: 'temperature_f',
    label: 'Temperature',
    icon: Thermometer,
    min: 20, max: 110, step: 1, defaultValue: 65,
    format: (v) => `${v}°F`,
    tone: 'warning',
  },
  {
    key: 'snap_share',
    label: 'Snap Share',
    icon: Zap,
    min: 0, max: 1, step: 0.01, defaultValue: 0.7,
    format: (v) => `${(v * 100).toFixed(0)}%`,
    tone: 'positive',
  },
  {
    key: 'primary_defender_grade',
    label: 'Defender Grade',
    icon: ShieldCheck,
    min: 0, max: 100, step: 1, defaultValue: 70,
    format: (v) => v.toFixed(0),
    tone: 'highlight',
  },
]

const INITIAL_OVERRIDES: ScenarioOverrides = {
  wind_speed_mph: 5,
  temperature_f: 65,
  snap_share: 0.7,
  primary_defender_grade: 70,
  dome_override: false,
}

export function WhatIfStudio({ playerId, week, season, stat, baseProjection }: WhatIfStudioProps) {
  const [overrides, setOverrides] = useState<ScenarioOverrides>(INITIAL_OVERRIDES)
  const { runScenario, data: scenarioData, isPending } = useScenario()

  function applyOverrides(updated: ScenarioOverrides) {
    setOverrides(updated)
    runScenario({ player_id: playerId, week, season, stat, overrides: updated })
  }

  function handleSliderChange(key: keyof Omit<ScenarioOverrides, 'dome_override'>, value: number) {
    applyOverrides({ ...overrides, [key]: value })
  }

  function handleDomeToggle(checked: boolean) {
    applyOverrides({ ...overrides, dome_override: checked })
  }

  const scenarioProjection = scenarioData?.scenario_projection ?? null
  const delta              = scenarioData?.delta ?? null
  const deltaPct           = scenarioData?.delta_pct ?? null
  const domeActive         = overrides.dome_override === true
  const deltaPositive      = delta !== null && delta >= 0
  const statFamily = stat.includes('receiving')
    ? 'coverage and defender strength'
    : stat.includes('passing')
      ? 'weather, volume, and coverage quality'
      : 'snap share and game environment'

  function handleReset() {
    applyOverrides(INITIAL_OVERRIDES)
  }

  return (
    <div className={clsx(toneClass('highlight'), 'rounded-xl overflow-hidden border border-tone/20 bg-panel-gradient shadow-card')}>
      {/* Header */}
      <div className="px-5 py-4 flex items-center justify-between border-b border-white/[0.06]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-tone/15 border border-tone/30">
            <FlaskConical className="size-4 text-tone" aria-hidden="true" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-fg">What-If Studio</h3>
            <p className="text-[10px] text-fg-muted">Adjust conditions to see projection impact</p>
          </div>
        </div>
        {isPending && <LoadingSpinner size="sm" />}
      </div>

      <div className="p-5 flex flex-col gap-5">
        {/* Live result banner */}
        <div
          className={clsx(
            'rounded-xl p-4 flex items-center gap-4 border',
            scenarioProjection === null
              ? 'bg-surface-overlay/50 border-white/5'
              : clsx(toneClass(deltaPositive ? 'positive' : 'negative'), 'bg-tone/[0.06] border-tone/20'),
          )}
        >
          {scenarioProjection !== null && delta !== null && deltaPct !== null ? (
            <>
              <div className="flex-1 grid grid-cols-3 gap-4 text-center">
                <div>
                  <p className="section-label mb-1">Base</p>
                  <p className="font-display text-2xl text-fg">{baseProjection.toFixed(1)}</p>
                </div>
                <div>
                  <p className="section-label mb-1">Scenario</p>
                  <p className={clsx(toneClass('info'), 'glow-text font-display text-2xl text-tone')}>
                    {scenarioProjection.toFixed(1)}
                  </p>
                </div>
                <div>
                  <p className="section-label mb-1">Delta</p>
                  <p className={clsx(toneClass(deltaPositive ? 'positive' : 'negative'), 'glow-text font-display text-2xl text-tone')}>
                    {deltaPositive ? '+' : ''}{delta.toFixed(1)}
                  </p>
                  <p className="text-[10px] text-fg-muted mt-0.5">
                    ({deltaPositive ? '+' : ''}{deltaPct.toFixed(1)}%)
                  </p>
                </div>
              </div>
            </>
          ) : (
            <p className="text-xs text-fg-muted text-center w-full">
              Adjust sliders below to simulate different game conditions
            </p>
          )}
        </div>

        <div className="flex items-center justify-between gap-3 text-[11px]">
          <p className="text-fg-muted">
            This scenario primarily shifts {statFamily} for <span className="text-fg">{stat.replaceAll('_', ' ')}</span>.
          </p>
          <button
            onClick={handleReset}
            className="px-3 py-1.5 rounded-lg text-[11px] font-semibold transition-colors bg-white/[0.04] border border-white/[0.08] text-fg-muted hover:text-fg"
          >
            Reset
          </button>
        </div>

        {/* Dome toggle */}
        <div
          className={clsx(
            'flex items-center justify-between rounded-xl px-4 py-3 border',
            domeActive ? clsx(toneClass('info'), 'bg-tone/[0.06] border-tone/20') : 'bg-surface-overlay/40 border-white/5',
          )}
        >
          <div className="flex items-center gap-3">
            <House className="size-5 text-fg-muted" aria-hidden="true" />
            <div>
              <p className="text-sm font-medium text-fg" id="dome-label">Indoor / Dome Game</p>
              <p className="text-[10px] text-fg-muted">Removes all weather penalties</p>
            </div>
          </div>
          <Switch
            checked={domeActive}
            onCheckedChange={handleDomeToggle}
            aria-labelledby="dome-label"
            tone="info"
          />
        </div>

        {/* Sliders */}
        <div
          className="flex flex-col gap-4 transition-opacity duration-200"
          style={{ opacity: domeActive ? 0.35 : 1, pointerEvents: domeActive ? 'none' : 'auto' }}
        >
          {SLIDERS.map((slider) => {
            const value = (overrides[slider.key] as number | undefined) ?? slider.defaultValue
            const Icon = slider.icon

            return (
              <div key={slider.key} className={clsx(toneClass(slider.tone), 'flex flex-col gap-2')}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Icon className="size-4 text-tone" aria-hidden="true" />
                    <span className="text-xs font-medium text-fg-muted">{slider.label}</span>
                  </div>
                  <span className="text-sm font-bold font-mono text-tone">
                    {slider.format(value)}
                  </span>
                </div>

                <Slider
                  tone={slider.tone}
                  min={slider.min}
                  max={slider.max}
                  step={slider.step}
                  value={value}
                  onValueChange={(next) => handleSliderChange(slider.key, next)}
                  aria-label={slider.label}
                />
              </div>
            )
          })}
        </div>

        <div className="rounded-xl px-4 py-3 text-[11px] text-fg-muted bg-white/[0.03] border border-white/[0.06]">
          <span className="text-fg font-medium">Assumptions:</span> snap share scales opportunity, wind and temperature apply passing/receiving penalties, defender grade uses 70 as neutral, and dome mode removes weather effects.
        </div>
      </div>
    </div>
  )
}
