import { useState, useEffect } from 'react'
import clsx from 'clsx'
import { Check, CalendarDays, CloudSun, HeartPulse, Lock, Swords, TrendingUp, Users, ChartColumn, type LucideIcon } from 'lucide-react'
import { useSettings, useUpdateSettings } from '@/hooks/useSettings'
import { LoadingSpinner } from '@/components/shared/LoadingSpinner'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { Panel } from '@/components/shared/Panel'
import { Slider } from '@/components/ui/slider'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { toneClass, type Tone } from '@/lib/tokens'
import { isSettingsEditable } from '@/lib/config'
import { ApiError } from '@/lib/api-client'
import type { FeatureWeights, FantasyScoring } from '@/types/api'

const DEFAULT_WEIGHTS: FeatureWeights = {
  kalman_form: 35, seasonal_baseline: 20, matchup: 20,
  weather_venue: 10, team_context: 5, roster_injury: 5, rule_meta: 5,
}

const WEIGHT_META: Record<keyof FeatureWeights, { label: string; icon: LucideIcon; tone: Tone }> = {
  kalman_form:       { label: 'Recent form', icon: TrendingUp, tone: 'info' },
  seasonal_baseline: { label: 'Seasonal Baseline',            icon: CalendarDays, tone: 'positive' },
  matchup:           { label: 'Matchup Quality',              icon: Swords,  tone: 'warning' },
  weather_venue:     { label: 'Weather & Venue',              icon: CloudSun,  tone: 'highlight' },
  team_context:      { label: 'Team Context',                 icon: Users, tone: 'info' },
  roster_injury:     { label: 'Roster & Injury Status',       icon: HeartPulse, tone: 'negative' },
  rule_meta:         { label: 'Rule & Meta Trends',           icon: ChartColumn, tone: 'warning' },
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

/** Plain words for a failed save. Never show the server's own text. */
function saveFailureMessage(error: Error): string {
  if (error instanceof ApiError && error.status === 403) {
    return 'Your changes were not saved. This website does not allow changes to settings.'
  }
  return 'Your changes were not saved. Check your connection and try again.'
}

export function Settings() {
  const { data, isLoading, isError, error } = useSettings()
  const updateSettings = useUpdateSettings()
  // The public site cannot save settings. Say so up front and turn the controls
  // off, instead of letting someone change values and hit a refusal.
  const editable = isSettingsEditable()

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
    if (!editable) return
    updateSettings.mutate({ weights: localWeights, fantasy_scoring: scoring, engine_exposure_cap: exposureCap })
  }

  function handleReset() {
    if (!editable) return
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
        <h1 className="font-display text-5xl tracking-widest bg-gradient-to-br from-warning to-info bg-clip-text text-transparent">
          SETTINGS
        </h1>
        <p className="text-sm text-fg-muted mt-1">
          {editable ? 'Model weights and the bet-size limit' : 'The model weights and bet-size limit saved for this site'}
        </p>
      </div>

      {!editable && (
        <div
          id="settings-readonly-note"
          role="note"
          className="tone-info flex flex-col gap-1 rounded-xl p-4 text-sm bg-tone/[0.08] border border-tone/25"
        >
          <p className="flex items-center gap-2 font-semibold text-tone">
            <Lock className="size-4 flex-shrink-0" aria-hidden="true" />
            You can look at these settings, but not change them here
          </p>
          <p className="text-fg-muted leading-relaxed">
            Changes are turned off on this website so that nobody can change the numbers for everyone else.
            To try your own values, run your own copy of Gridiron Oracle. The setup steps are in the
            project&apos;s development guide.
          </p>
        </div>
      )}

      {/* Feature Weights */}
      <Panel title="Feature Weights">
        <div className="flex flex-col gap-5">
          {/* Total indicator */}
          <div className={clsx(toneClass(totalValid ? 'positive' : 'negative'), 'flex items-center justify-between p-3 rounded-lg bg-tone/[0.06] border border-tone/20')}>
            <span className="text-xs text-fg-muted">Total weight</span>
            <span className="glow-text font-display text-2xl text-tone">
              {total.toFixed(1)}%
            </span>
            {!totalValid && <span className="text-xs text-negative">Must equal 100%</span>}
          </div>

          {/* Sliders */}
          {WEIGHT_KEYS.map((key) => {
            const meta  = WEIGHT_META[key]
            const value = localWeights[key]

            return (
              <div key={key} className={clsx(toneClass(meta.tone), 'flex flex-col gap-2')}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <meta.icon className="size-4 text-tone" aria-hidden="true" />
                    <span className="text-xs font-medium text-fg-muted">{meta.label}</span>
                  </div>
                  <span className="text-sm font-bold font-mono text-tone">
                    {value.toFixed(1)}%
                  </span>
                </div>

                <Slider
                  tone={meta.tone}
                  min={0}
                  max={100}
                  step={0.5}
                  value={value}
                  onValueChange={(next) => handleWeightChange(key, next)}
                  disabled={!editable}
                  aria-label={meta.label}
                />
              </div>
            )
          })}
        </div>
      </Panel>

      {/* Fantasy Scoring */}
      <Panel title="Fantasy Scoring Format">
        <RadioGroup
          value={scoring}
          onValueChange={(v) => { if (editable && v) setScoring(v as FantasyScoring) }}
          disabled={!editable}
          aria-label="Fantasy scoring format"
          className="grid grid-cols-3 gap-3"
        >
          {SCORING_OPTIONS.map(({ value, label, desc }) => (
            <RadioGroupItem
              key={value}
              value={value}
              label={label}
              description={desc}
              variant="card"
            />
          ))}
        </RadioGroup>
      </Panel>

      {/* Engine Exposure Cap */}
      <Panel title="Max bet size">
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-fg font-medium">Largest slice of the bankroll allowed on one bet</p>
              <p className="text-[10px] text-fg-muted mt-0.5">
                Lower is safer. The default is 25%.
              </p>
            </div>
            <span className={clsx(toneClass('warning'), 'glow-text font-display text-3xl text-tone')}>
              {(exposureCap * 100).toFixed(0)}%
            </span>
          </div>

          <Slider
            tone="warning"
            min={1}
            max={100}
            step={1}
            value={Math.round(exposureCap * 100)}
            onValueChange={(next) => setExposureCap(next / 100)}
            disabled={!editable}
            aria-label="Engine Exposure Cap"
          />
        </div>
      </Panel>

      {/* Actions */}
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={handleSave}
          disabled={!editable || !totalValid || updateSettings.isPending}
          aria-describedby={editable ? undefined : 'settings-readonly-note'}
          className={clsx(
            toneClass('info'),
            'flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold transition-all duration-150 border disabled:opacity-40 disabled:cursor-not-allowed',
            totalValid
              ? 'bg-tone/15 border-tone/40 text-tone shadow-[0_0_20px_rgb(var(--tone)/0.1)]'
              : 'bg-fg-muted/10 border-white/[0.06] text-fg-muted',
          )}
        >
          {updateSettings.isPending && <LoadingSpinner size="sm" />}
          Save Settings
        </button>
        <button
          onClick={handleReset}
          disabled={!editable}
          className="px-6 py-3 rounded-xl text-sm font-semibold transition-all duration-150 bg-surface-overlay/40 border border-white/[0.06] text-fg-muted hover:text-fg disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:text-fg-muted"
        >
          Reset to Defaults
        </button>
        {!editable && (
          <p className="text-xs text-fg-muted">Saving is turned off on this website.</p>
        )}
      </div>

      {/* Feedback */}
      {updateSettings.isSuccess && (
        <div className="tone-positive flex items-center gap-2 px-4 py-3 rounded-xl text-sm bg-tone/[0.08] border border-tone/20 text-tone">
          <Check className="size-4 flex-shrink-0" aria-hidden="true" />
          Settings saved successfully.
        </div>
      )}
      {updateSettings.isError && updateSettings.error && (
        <ErrorBanner
          message={saveFailureMessage(updateSettings.error)}
          onRetry={updateSettings.error instanceof ApiError && updateSettings.error.status === 403 ? undefined : handleSave}
        />
      )}
    </div>
  )
}
