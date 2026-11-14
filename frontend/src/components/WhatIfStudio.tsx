import { useState } from 'react'
import clsx from 'clsx'
import { FlaskConical, House, ShieldCheck, Thermometer, Wind, Zap, type LucideIcon } from 'lucide-react'
import { useScenario } from '@/hooks/useScenario'
import { LoadingSpinner } from '@/components/shared/LoadingSpinner'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import type { ScenarioOverrides } from '@/types/api'
import { toneClass, type Tone } from '@/lib/tokens'

interface WhatIfStudioProps {
  playerId: string
  week: number
  season: number
  stat: string
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

export function WhatIfStudio({ playerId, week, season, stat }: WhatIfStudioProps) {
  const [overrides, setOverrides] = useState<ScenarioOverrides>(INITIAL_OVERRIDES)
  const { runScenario, cancel, reset, data: scenarioData, isPending, isError, error } = useScenario()
  const touched = JSON.stringify(overrides) !== JSON.stringify(INITIAL_OVERRIDES)

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

  function handleReset() {
    cancel()
    reset()
    setOverrides(INITIAL_OVERRIDES)
  }

  const domeActive = overrides.dome_override === true
  const statLabel = stat.replaceAll('_', ' ')
  const hasResult = scenarioData != null && !isError
  const delta = scenarioData?.delta ?? 0
  const deltaTone: Tone = delta >= 0 ? 'positive' : 'negative'
  const sign = delta >= 0 ? '+' : ''

  return (
    <div className={clsx(toneClass('highlight'), 'overflow-hidden rounded-xl border border-tone/20 bg-panel-gradient shadow-card')}>
      <div className="flex items-center justify-between gap-3 border-b border-white/[0.06] px-5 py-4">
        <div className="flex items-center gap-3">
          <div className="flex size-8 items-center justify-center rounded-lg border border-tone/30 bg-tone/15">
            <FlaskConical className="size-4 text-tone" aria-hidden="true" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-fg">What-If Studio</h3>
            <p className="text-xs text-fg-muted">Change the game conditions and see what happens to {statLabel}</p>
          </div>
        </div>
        <button
          type="button"
          onClick={handleReset}
          disabled={!touched && !hasResult}
          className="shrink-0 rounded-lg border border-white/[0.08] bg-white/[0.04] px-3 py-1.5 text-xs font-semibold text-fg-muted transition-colors hover:text-fg disabled:opacity-40"
        >
          Reset
        </button>
      </div>

      <div className="grid gap-5 p-5 lg:grid-cols-[minmax(0,1fr)_16rem]">
        {/* Result sits beside the sliders on desktop and above them on a phone. */}
        <div className="lg:order-2">
          <div className="flex flex-col gap-3 lg:sticky lg:top-4">
            <div
              aria-live="polite"
              className={clsx(
                'flex min-h-28 flex-col justify-center gap-1 rounded-xl border p-4',
                hasResult ? clsx(toneClass(deltaTone), 'border-tone/25 bg-tone/[0.06]') : 'border-white/5 bg-surface-overlay/50',
              )}
            >
              {isError ? (
                <ErrorBanner error={error} onRetry={() => applyOverrides(overrides)} />
              ) : hasResult ? (
                <>
                  <p className="section-label">Change from projection</p>
                  <p className="glow-text font-display text-4xl leading-none text-tone">
                    {sign}{delta.toFixed(1)}
                  </p>
                  <p className="text-xs text-fg-muted">
                    {sign}{scenarioData.delta_pct.toFixed(1)}%, from {scenarioData.base_projection.toFixed(1)} to{' '}
                    <span className="font-semibold text-fg">{scenarioData.scenario_projection.toFixed(1)}</span>
                  </p>
                </>
              ) : (
                <p className="text-sm text-fg-muted">
                  {isPending ? 'Working it out…' : 'Move a slider to see how the projection changes.'}
                </p>
              )}
              {isPending && hasResult && <LoadingSpinner size="sm" />}
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-5 lg:order-1">
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
              <p className="text-xs text-fg-muted">Removes all weather penalties</p>
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

        <div className="rounded-xl px-4 py-3 text-xs text-fg-muted bg-white/[0.03] border border-white/[0.06]">
          <span className="text-fg font-medium">Assumptions:</span> snap share scales opportunity, wind and temperature apply passing/receiving penalties, defender grade uses 70 as neutral, and dome mode removes weather effects.
        </div>
        </div>
      </div>
    </div>
  )
}
