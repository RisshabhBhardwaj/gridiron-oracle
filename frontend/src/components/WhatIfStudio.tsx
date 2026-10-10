import { useState } from 'react'
import clsx from 'clsx'
import { useScenario } from '@/hooks/useScenario'
import { LoadingSpinner } from '@/components/shared/LoadingSpinner'
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
  icon: string
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
    icon: '💨',
    min: 0, max: 60, step: 1, defaultValue: 5,
    format: (v) => `${v} mph`,
    tone: 'info',
  },
  {
    key: 'temperature_f',
    label: 'Temperature',
    icon: '🌡',
    min: 20, max: 110, step: 1, defaultValue: 65,
    format: (v) => `${v}°F`,
    tone: 'warning',
  },
  {
    key: 'snap_share',
    label: 'Snap Share',
    icon: '⚡',
    min: 0, max: 1, step: 0.01, defaultValue: 0.7,
    format: (v) => `${(v * 100).toFixed(0)}%`,
    tone: 'positive',
  },
  {
    key: 'primary_defender_grade',
    label: 'Defender Grade',
    icon: '🛡',
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
            <svg className="w-4 h-4 text-tone" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
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
            <span className="text-2xl select-none">🏟️</span>
            <div>
              <p className="text-sm font-medium text-fg">Indoor / Dome Game</p>
              <p className="text-[10px] text-fg-muted">Removes all weather penalties</p>
            </div>
          </div>
          <button
            role="switch"
            aria-checked={domeActive}
            onClick={() => handleDomeToggle(!domeActive)}
            className={clsx(
              toneClass('info'),
              'relative inline-flex h-6 w-11 items-center rounded-full border transition-all duration-200',
              'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
              domeActive
                ? 'bg-tone/80 border-tone/50 shadow-[0_0_12px_rgb(var(--tone)/0.4)]'
                : 'bg-surface-overlay border-white/10',
            )}
            aria-label="Indoor / Dome Game toggle"
          >
            <span
              className="inline-block h-4 w-4 transform rounded-full bg-fg shadow-[0_1px_4px_rgb(var(--shadow)/0.4)] transition-transform duration-200"
              style={{ transform: `translateX(${domeActive ? '24px' : '4px'})` }}
            />
          </button>
        </div>

        {/* Sliders */}
        <div
          className="flex flex-col gap-4 transition-opacity duration-200"
          style={{ opacity: domeActive ? 0.35 : 1, pointerEvents: domeActive ? 'none' : 'auto' }}
        >
          {SLIDERS.map((slider) => {
            const value = (overrides[slider.key] as number | undefined) ?? slider.defaultValue
            const pct   = ((value - slider.min) / (slider.max - slider.min)) * 100

            return (
              <div key={slider.key} className={clsx(toneClass(slider.tone), 'flex flex-col gap-2')}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-base select-none">{slider.icon}</span>
                    <span className="text-xs font-medium text-fg-muted">{slider.label}</span>
                  </div>
                  <span className="text-sm font-bold font-mono text-tone">
                    {slider.format(value)}
                  </span>
                </div>

                {/* Custom slider with fill track */}
                <div className="relative h-2 rounded-full bg-surface-overlay/80">
                  {/* Fill */}
                  <div
                    className="absolute top-0 left-0 h-full rounded-full transition-all duration-150 bg-tone/40"
                    style={{ width: `${pct}%` }}
                  />
                  <input
                    type="range"
                    min={slider.min}
                    max={slider.max}
                    step={slider.step}
                    value={value}
                    onChange={(e) => handleSliderChange(slider.key, parseFloat(e.target.value))}
                    className="oracle-slider absolute inset-0 w-full opacity-0 h-full cursor-pointer"
                    aria-label={slider.label}
                    style={{ appearance: 'none', zIndex: 1 }}
                  />
                  {/* Custom thumb */}
                  <div
                    className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full pointer-events-none transition-all duration-150 bg-tone border-2 border-surface/80 shadow-[0_0_8px_rgb(var(--tone)/0.5)]"
                    style={{ left: `calc(${pct}% - 8px)` }}
                  />
                </div>
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
