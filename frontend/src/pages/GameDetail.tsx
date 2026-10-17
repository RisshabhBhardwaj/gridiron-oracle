import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ChevronDown, ChevronUp, Dices } from 'lucide-react'
import { useGameDriveSim } from '@/hooks/useDriveSim'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { SkeletonCard } from '@/components/shared/LoadingSpinner'
import type { SimulatedDriveDTO, SimulatedPlayDTO } from '@/types/api'

const OUTCOME_STYLES: Record<string, { bg: string; text: string; label: string }> = {
  TOUCHDOWN: { bg: 'bg-emerald-500/20 border-emerald-500/40', text: 'text-emerald-400', label: 'Touchdown (+7)' },
  FIELD_GOAL: { bg: 'bg-teal-500/20 border-teal-500/40', text: 'text-teal-400', label: 'Field Goal (+3)' },
  MISSED_FIELD_GOAL: { bg: 'bg-amber-500/20 border-amber-500/40', text: 'text-amber-400', label: 'Missed FG' },
  PUNT: { bg: 'bg-blue-500/10 border-blue-500/30', text: 'text-blue-400', label: 'Punt' },
  TURNOVER: { bg: 'bg-rose-500/20 border-rose-500/40', text: 'text-rose-400', label: 'Turnover' },
  TURNOVER_ON_DOWNS: { bg: 'bg-rose-500/20 border-rose-500/40', text: 'text-rose-400', label: 'Turnover on Downs' },
  SAFETY: { bg: 'bg-purple-500/20 border-purple-500/40', text: 'text-purple-400', label: 'Safety' },
}

/**
 * Field position is stored as yards from the possessing team's own goal line
 * (0-100). Football notation flips at midfield, so 95 is "Opp 5", not the
 * "Own 95" the timeline was printing.
 */
function formatFieldPos(yardsFromOwnGoal: number): string {
  if (yardsFromOwnGoal >= 100) return 'Endzone'
  if (yardsFromOwnGoal > 50) return `Opp ${100 - yardsFromOwnGoal}`
  if (yardsFromOwnGoal === 50) return 'Midfield'
  return `Own ${yardsFromOwnGoal}`
}

export function GameDetail() {
  const { gameId } = useParams<{ gameId: string }>()
  const [seed, setSeed] = useState<number | undefined>(undefined)
  const [expandedDrives, setExpandedDrives] = useState<Set<number>>(new Set([1]))

  const { data, isLoading, error, refetch, isFetching } = useGameDriveSim(gameId, seed)

  const toggleDriveExpand = (driveNum: number) => {
    setExpandedDrives((prev) => {
      const next = new Set(prev)
      if (next.has(driveNum)) {
        next.delete(driveNum)
      } else {
        next.add(driveNum)
      }
      return next
    })
  }

  const handleResimulate = () => {
    const newSeed = Math.floor(Math.random() * 1_000_000_000)
    setSeed(newSeed)
  }

  return (
    <div className="flex flex-col gap-6 max-w-6xl mx-auto">
      {/* Navigation */}
      <div className="flex items-center justify-between">
        <Link
          to="/projections"
          className="inline-flex items-center gap-1.5 text-xs text-fg-muted hover:text-fg transition-colors"
        >
          ← Back to Current Season
        </Link>
      </div>

      {error && <ErrorBanner error={error} onRetry={() => refetch()} />}

      {isLoading && (
        <div>
          <SkeletonCard count={3} />
        </div>
      )}

      {!isLoading && data && (
        <>
          {/* Header Banner: Matchup & Simulated vs Anchor Score */}
          <div className="p-6 rounded-2xl flex flex-col gap-5 border border-line/60 bg-panel-gradient shadow-[0_8px_32px_rgb(var(--shadow)/0.3)]">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line/40 pb-4">
              <div>
                <span className="text-[11px] uppercase tracking-wider font-bold text-accent">
                  {data.season} Week {data.week} · Drive-by-drive simulation
                </span>
                <h1 className="text-2xl sm:text-3xl font-display font-bold text-fg mt-0.5">
                  {data.away_team} @ {data.home_team}
                </h1>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleResimulate}
                  disabled={isFetching}
                  className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-bold bg-accent text-accent-fg hover:brightness-110 shadow transition-all"
                >
                  <Dices className="size-3.5" aria-hidden="true" />
                  {isFetching ? 'Simulating…' : 'Resimulate'}
                </button>
              </div>
            </div>

            {/* Matchup Comparison Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
              {/* Away Team */}
              <div className="flex flex-col items-center md:items-start p-4 rounded-xl bg-surface-overlay/40 border border-line/40">
                <span className="text-xs font-semibold uppercase tracking-wider text-fg-muted">Away</span>
                <span className="text-2xl font-bold font-display text-fg mt-1">{data.away_team}</span>
                <div className="flex items-baseline gap-2 mt-2">
                  <span className="text-3xl font-bold text-fg font-mono">
                    {data.summary.simulated_away_points}
                  </span>
                  <span className="text-xs text-fg-muted">pts</span>
                </div>
                <span className="text-xs font-mono text-fg-muted mt-1">
                  Model: {data.anchor.away_points.toFixed(1)} pts · {data.anchor.away_yards.toFixed(0)} yds
                </span>
              </div>

              {/* Game Result Center */}
              <div className="flex flex-col items-center justify-center p-4 text-center">
                <span className="text-xs uppercase font-mono tracking-widest text-fg-muted">Simulated result</span>
                <span className="text-lg font-bold text-accent mt-1">
                  {data.summary.simulated_winner === 'TIE'
                    ? 'Tie game'
                    : `${data.summary.simulated_winner} win`}
                </span>
                <span className="text-xs font-mono text-fg-muted mt-2">
                  {data.summary.total_drives} drives · Home win chance: {(data.anchor.home_win_probability * 100).toFixed(0)}%
                </span>
              </div>

              {/* Home Team */}
              <div className="flex flex-col items-center md:items-end p-4 rounded-xl bg-surface-overlay/40 border border-line/40">
                <span className="text-xs font-semibold uppercase tracking-wider text-fg-muted">Home</span>
                <span className="text-2xl font-bold font-display text-fg mt-1">{data.home_team}</span>
                <div className="flex items-baseline gap-2 mt-2">
                  <span className="text-xs text-fg-muted">pts</span>
                  <span className="text-3xl font-bold text-fg font-mono">
                    {data.summary.simulated_home_points}
                  </span>
                </div>
                <span className="text-xs font-mono text-fg-muted mt-1">
                  Model: {data.anchor.home_points.toFixed(1)} pts · {data.anchor.home_yards.toFixed(0)} yds
                </span>
              </div>
            </div>
          </div>

          {/* Drive-by-Drive Timeline */}
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-bold font-display uppercase tracking-wider text-fg">
                Drives
              </h2>
              <span className="text-xs text-fg-muted">
                Click any drive to expand play-by-play details
              </span>
            </div>

            <div className="flex flex-col gap-3">
              {data.drives.map((drive: SimulatedDriveDTO) => {
                const style = OUTCOME_STYLES[drive.outcome] || {
                  bg: 'bg-surface-overlay border-line',
                  text: 'text-fg-muted',
                  label: drive.outcome,
                }
                const isExpanded = expandedDrives.has(drive.drive_number)
                const isHome = drive.possession_team === data.home_team

                return (
                  <div
                    key={drive.drive_number}
                    className="rounded-xl border border-line/60 bg-surface-raised/40 overflow-hidden transition-all"
                  >
                    {/* Drive Header Bar */}
                    <button
                      type="button"
                      onClick={() => toggleDriveExpand(drive.drive_number)}
                      className="w-full p-3.5 flex flex-wrap items-center justify-between gap-3 text-left hover:bg-surface-overlay/40 transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        <span className="font-mono text-xs font-bold text-fg-muted w-14">
                          Drive {drive.drive_number}
                        </span>
                        <span className="px-2 py-0.5 text-[10px] font-semibold rounded bg-surface-overlay border border-line text-fg-muted">
                          Q{drive.quarter}
                        </span>
                        <span
                          className={`text-sm font-bold ${
                            isHome ? 'text-fg' : 'text-fg'
                          }`}
                        >
                          {drive.possession_team}
                        </span>
                        <span className="text-xs text-fg-muted font-mono">
                          {formatFieldPos(drive.start_field_pos)} → {formatFieldPos(drive.end_field_pos)}
                        </span>
                      </div>

                      <div className="flex items-center gap-3">
                        <span className="text-xs font-mono text-fg-muted">
                          {drive.plays_count} plays · {drive.yards_gained.toFixed(0)} yds
                        </span>
                        <span
                          className={`px-2.5 py-0.5 rounded text-xs font-bold border ${style.bg} ${style.text}`}
                        >
                          {style.label}
                        </span>
                        <span className="font-mono text-xs font-bold text-fg pl-2 border-l border-line">
                          {data.away_team} {drive.away_score_after} - {drive.home_score_after} {data.home_team}
                        </span>
                        {isExpanded
                          ? <ChevronUp className="size-3.5 text-fg-muted" aria-hidden="true" />
                          : <ChevronDown className="size-3.5 text-fg-muted" aria-hidden="true" />}
                      </div>
                    </button>

                    {/* Expandable Play-by-Play Table */}
                    {isExpanded && drive.plays.length > 0 && (
                      <div className="border-t border-line/40 bg-surface/50 p-3 overflow-x-auto">
                        <table className="min-w-full text-left text-xs">
                          <thead className="text-[10px] uppercase text-fg-muted tracking-wider border-b border-line/40">
                            <tr>
                              <th className="pb-1.5 px-2">Play</th>
                              <th className="pb-1.5 px-2">Down & Dist</th>
                              <th className="pb-1.5 px-2">Field Pos</th>
                              <th className="pb-1.5 px-2">Type</th>
                              <th className="pb-1.5 px-2">Gain</th>
                              <th className="pb-1.5 px-2">Result</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-line/20 font-mono">
                            {drive.plays.map((play: SimulatedPlayDTO) => (
                              <tr key={play.play_number} className="hover:bg-surface-overlay/20">
                                <td className="py-1.5 px-2 text-fg-muted">#{play.play_number}</td>
                                <td className="py-1.5 px-2 text-fg">
                                  {play.down}&amp;{play.ytg}
                                </td>
                                <td className="py-1.5 px-2 text-fg-muted">{formatFieldPos(play.field_pos)}</td>
                                <td className="py-1.5 px-2 uppercase font-semibold text-fg-muted">
                                  {play.play_type}
                                </td>
                                <td
                                  className={`py-1.5 px-2 font-bold ${
                                    play.yards_gained > 0
                                      ? 'text-emerald-400'
                                      : play.yards_gained < 0
                                        ? 'text-rose-400'
                                        : 'text-fg-muted'
                                  }`}
                                >
                                  {play.yards_gained > 0 ? `+${play.yards_gained}` : play.yards_gained}
                                </td>
                                <td className="py-1.5 px-2">
                                  {play.is_touchdown && (
                                    <span className="text-emerald-400 font-bold">Touchdown</span>
                                  )}
                                  {play.is_first_down && !play.is_touchdown && (
                                    <span className="text-accent font-medium">1st Down</span>
                                  )}
                                  {play.is_turnover && (
                                    <span className="text-rose-400 font-bold">Turnover</span>
                                  )}
                                  {play.is_safety && (
                                    <span className="text-purple-400 font-bold">Safety</span>
                                  )}
                                  {!play.is_first_down && !play.is_touchdown && !play.is_turnover && !play.is_safety && (
                                    <span className="text-fg-muted">No 1st down</span>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
