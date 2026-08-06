import { useState, useEffect } from 'react'
import { useSettings, useUpdateSettings } from '@/hooks/useSettings'
import { LoadingSpinner } from '@/components/shared/LoadingSpinner'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import type { FeatureWeights, FantasyScoring } from '@/types/api'

const DEFAULT_WEIGHTS: FeatureWeights = {
  kalman_form: 35, seasonal_baseline: 20, matchup: 20,
  weather_venue: 10, team_context: 5, roster_injury: 5, rule_meta: 5,
}

const WEIGHT_META: Record<keyof FeatureWeights, { label: string; icon: string; color: string }> = {
  kalman_form:       { label: 'Kalman Form (Player Ability)', icon: '📈', color: '#00C2FF' },
  seasonal_baseline: { label: 'Seasonal Baseline',            icon: '📅', color: '#00FFA3' },
  matchup:           { label: 'Matchup Quality',              icon: '⚔️',  color: '#FFB800' },
  weather_venue:     { label: 'Weather & Venue',              icon: '🌦',  color: '#A855F7' },
  team_context:      { label: 'Team Context',                 icon: '🏈', color: '#00C2FF' },
  roster_injury:     { label: 'Roster & Injury Status',       icon: '🏥', color: '#FF3B5C' },
  rule_meta:         { label: 'Rule & Meta Trends',           icon: '📊', color: '#FFB800' },
}

const WEIGHT_KEYS = Object.keys(DEFAULT_WEIGHTS) as (keyof FeatureWeights)[]

function sumWeights(w: FeatureWeights): number {
  return WEIGHT_KEYS.reduce((acc, k) => acc + w[k], 0)
}

const SCORING_OPTIONS: { value: FantasyScoring; label: string; desc: string }[] = [
  { value: 'ppr',      label: 'PPR',      desc: '1pt per reception' },
  { value: 'half_ppr', label: 'Half-PPR', desc: '0.5pt per reception' },
  { value: 'standard', label: 'Standard', desc: 'No reception points' },
]

function GlassSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{
        background: 'linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%)',
        border: '1px solid rgba(255,255,255,0.07)',
        boxShadow: '0 4px 24px rgba(0,0,0,0.4)',
      }}
    >
      <div className="px-5 py-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <h2 className="text-sm font-semibold text-oracle-white">{title}</h2>
      </div>
      <div className="p-5">{children}</div>
    </div>
  )
}

export function Settings() {
  const { data, isLoading, isError, error } = useSettings()
  const updateSettings = useUpdateSettings()

  const [localWeights, setLocalWeights] = useState<FeatureWeights>(DEFAULT_WEIGHTS)
  const [scoring, setScoring]           = useState<FantasyScoring>('ppr')
  const [exposureCap, setExposureCap]   = useState(0.25)

  useEffect(() => {
    if (data) {
      setLocalWeights(data.weights)
      setScoring(data.fantasy_scoring)
      setExposureCap(data.engine_exposure_cap)
    }
  }, [data])

  const total      = sumWeights(localWeights)
  const totalValid = total >= 99.9 && total <= 100.1

  function handleWeightChange(key: keyof FeatureWeights, value: number) {
    setLocalWeights((prev) => ({ ...prev, [key]: value }))
  }

  function handleSave() {
    updateSettings.mutate({ weights: localWeights, fantasy_scoring: scoring, engine_exposure_cap: exposureCap })
  }

  function handleReset() {
    setLocalWeights(DEFAULT_WEIGHTS)
    setScoring('ppr')
    setExposureCap(0.25)
  }

  if (isLoading) return <div className="flex justify-center py-16"><LoadingSpinner size="lg" /></div>
  if (isError && error) return <ErrorBanner error={error} />

  return (
    <div className="flex flex-col gap-6 max-w-2xl">

      {/* Header */}
      <div>
        <h1
          className="font-display text-5xl tracking-widest"
          style={{
            background: 'linear-gradient(135deg, #FFB800 0%, #00C2FF 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          SETTINGS
        </h1>
        <p className="text-sm text-oracle-muted mt-1">Configure model weights and engine parameters</p>
      </div>

      {/* Feature Weights */}
      <GlassSection title="Feature Weights">
        <div className="flex flex-col gap-5">
          {/* Total indicator */}
          <div
            className="flex items-center justify-between p-3 rounded-lg"
            style={{
              background: totalValid ? 'rgba(0,255,163,0.06)' : 'rgba(255,59,92,0.06)',
              border: `1px solid ${totalValid ? 'rgba(0,255,163,0.2)' : 'rgba(255,59,92,0.2)'}`,
            }}
          >
            <span className="text-xs text-oracle-muted">Total weight</span>
            <span
              className="font-display text-2xl"
              style={{ color: totalValid ? '#00FFA3' : '#FF3B5C', textShadow: `0 0 12px ${totalValid ? 'rgba(0,255,163,0.5)' : 'rgba(255,59,92,0.5)'}` }}
            >
              {total.toFixed(1)}%
            </span>
            {!totalValid && <span className="text-xs text-oracle-red">Must equal 100%</span>}
          </div>

          {/* Sliders */}
          {WEIGHT_KEYS.map((key) => {
            const meta  = WEIGHT_META[key]
            const value = localWeights[key]
            const pct   = (value / 100) * 100

            return (
              <div key={key} className="flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-base select-none">{meta.icon}</span>
                    <span className="text-xs font-medium text-oracle-muted">{meta.label}</span>
                  </div>
                  <span className="text-sm font-bold font-mono" style={{ color: meta.color }}>
                    {value.toFixed(1)}%
                  </span>
                </div>

                <div className="relative h-2 rounded-full" style={{ background: 'rgba(26,47,78,0.8)' }}>
                  {/* Fill track */}
                  <div
                    className="absolute top-0 left-0 h-full rounded-full transition-all duration-150"
                    style={{ width: `${pct}%`, background: `${meta.color}50` }}
                  />
                  <input
                    type="range" min={0} max={100} step={0.5} value={value}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleWeightChange(key, parseFloat(e.target.value))}
                    className="absolute inset-0 w-full h-full cursor-pointer opacity-0"
                    style={{ appearance: 'none', zIndex: 1 }}
                    aria-label={meta.label}
                  />
                  {/* Custom thumb */}
                  <div
                    className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full pointer-events-none transition-all duration-150"
                    style={{
                      left: `calc(${pct}% - 8px)`,
                      background: meta.color,
                      boxShadow: `0 0 8px ${meta.color}80`,
                      border: '2px solid rgba(5,11,24,0.8)',
                    }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      </GlassSection>

      {/* Fantasy Scoring */}
      <GlassSection title="Fantasy Scoring Format">
        <div className="grid grid-cols-3 gap-3">
          {SCORING_OPTIONS.map(({ value, label, desc }) => (
            <button
              key={value}
              onClick={() => setScoring(value)}
              className="flex flex-col gap-1 p-4 rounded-xl text-left transition-all duration-150 border"
              style={scoring === value
                ? {
                    background: 'rgba(0,194,255,0.1)',
                    borderColor: 'rgba(0,194,255,0.4)',
                    boxShadow: '0 0 16px rgba(0,194,255,0.1)',
                  }
                : {
                    background: 'rgba(26,47,78,0.3)',
                    borderColor: 'rgba(255,255,255,0.06)',
                  }
              }
            >
              <span
                className="text-sm font-bold"
                style={{ color: scoring === value ? '#00C2FF' : '#607B9B' }}
              >
                {label}
              </span>
              <span className="text-[10px] text-oracle-muted">{desc}</span>
            </button>
          ))}
        </div>
      </GlassSection>

      {/* Engine Exposure Cap */}
      <GlassSection title="Engine Exposure Cap">
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-oracle-white font-medium">Max bankroll fraction per bet</p>
              <p className="text-[10px] text-oracle-muted mt-0.5">
                Fractional Kelly: default 25% — 94% log-growth, 50% vol of full Kelly
              </p>
            </div>
            <span
              className="font-display text-3xl"
              style={{ color: '#FFB800', textShadow: '0 0 16px rgba(255,184,0,0.5)' }}
            >
              {(exposureCap * 100).toFixed(0)}%
            </span>
          </div>

          <div className="relative h-2 rounded-full" style={{ background: 'rgba(26,47,78,0.8)' }}>
            <div
              className="absolute top-0 left-0 h-full rounded-full transition-all duration-150"
              style={{ width: `${exposureCap * 100}%`, background: 'rgba(255,184,0,0.5)' }}
            />
            <input
              type="range" min={1} max={100} step={1}
              value={Math.round(exposureCap * 100)}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setExposureCap(parseFloat(e.target.value) / 100)}
              className="absolute inset-0 w-full h-full cursor-pointer opacity-0"
              style={{ appearance: 'none', zIndex: 1 }}
              aria-label="Engine Exposure Cap"
            />
            <div
              className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full pointer-events-none"
              style={{
                left: `calc(${exposureCap * 100}% - 8px)`,
                background: '#FFB800',
                boxShadow: '0 0 8px rgba(255,184,0,0.7)',
                border: '2px solid rgba(5,11,24,0.8)',
              }}
            />
          </div>
        </div>
      </GlassSection>

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={handleSave}
          disabled={!totalValid || updateSettings.isPending}
          className="flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed"
          style={{
            background: totalValid ? 'rgba(0,194,255,0.15)' : 'rgba(96,123,155,0.1)',
            border: `1px solid ${totalValid ? 'rgba(0,194,255,0.4)' : 'rgba(255,255,255,0.06)'}`,
            color: totalValid ? '#00C2FF' : '#607B9B',
            boxShadow: totalValid ? '0 0 20px rgba(0,194,255,0.1)' : 'none',
          }}
        >
          {updateSettings.isPending && <LoadingSpinner size="sm" />}
          Save Settings
        </button>
        <button
          onClick={handleReset}
          className="px-6 py-3 rounded-xl text-sm font-semibold transition-all duration-150"
          style={{ background: 'rgba(26,47,78,0.4)', border: '1px solid rgba(255,255,255,0.06)', color: '#607B9B' }}
        >
          Reset to Defaults
        </button>
      </div>

      {/* Feedback */}
      {updateSettings.isSuccess && (
        <div
          className="flex items-center gap-2 px-4 py-3 rounded-xl text-sm"
          style={{ background: 'rgba(0,255,163,0.08)', border: '1px solid rgba(0,255,163,0.2)', color: '#00FFA3' }}
        >
          <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
          Settings saved successfully.
        </div>
      )}
      {updateSettings.isError && updateSettings.error && (
        <ErrorBanner error={updateSettings.error} />
      )}
    </div>
  )
}
