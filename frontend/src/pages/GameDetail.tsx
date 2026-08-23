import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
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
    <div className="flex flex-col gap-6 p-4 sm:p-6 max-w-6xl mx-auto">
      {/* Navigation */}
      <div className="flex items-center justify-between">
        <Link
          to="/projections"
          className="inline-flex items-center gap-1.5 text-xs text-oracle-muted hover:text-oracle-white transition-colors"
        >
          ← Back to Current Season
        </Link>
        <span className="font-mono text-xs text-oracle-muted">Game ID: {gameId}</span>
      </div>

      {error && <ErrorBanner error={error} onRetry={() => refetch()} />}

      {isLoading && (
        <div className="grid gap-4">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      )}

      {!isLoading && data && (
        <>
          {/* Header Banner: Matchup & Simulated vs Anchor Score */}
          <div
            className="p-6 rounded-2xl flex flex-col gap-5 border border-oracle-border/60"
            style={{
              background: 'linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%)',
              boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
            }}
          >
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-oracle-border/40 pb-4">
              <div>
                <span className="text-[11px] uppercase tracking-wider font-bold text-oracle-green">
                  {data.season} Week {data.week} · Drive-By-Drive Simulation
                </span>
                <h1 className="text-2xl sm:text-3xl font-display font-bold text-oracle-white mt-0.5">
                  {data.away_team} @ {data.home_team}
                </h1>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleResimulate}
                  disabled={isFetching}
                  className="px-3.5 py-1.5 rounded-lg text-xs font-bold bg-oracle-green text-oracle-dark hover:brightness-110 shadow transition-all"
                >
                  {isFetching ? 'Simulating…' : '🎲 Resimulate Game'}
                </button>
                <span className="text-[10px] font-mono text-oracle-muted">Seed: {data.seed}</span>
              </div>
            </div>

            {/* Matchup Comparison Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
              {/* Away Team */}
              <div className="flex flex-col items-center md:items-start p-4 rounded-xl bg-oracle-surface/40 border border-oracle-border/40">
                <span className="text-xs font-semibold uppercase tracking-wider text-oracle-muted">Away</span>
                <span className="text-2xl font-bold font-display text-oracle-white mt-1">{data.away_team}</span>
                <div className="flex items-baseline gap-2 mt-2">
                  <span className="text-3xl font-bold text-oracle-white font-mono">
                    {data.summary.simulated_away_points}
                  </span>
                  <span className="text-xs text-oracle-muted">pts (Sim)</span>
                </div>
                <span className="text-xs font-mono text-oracle-muted mt-1">
                  Anchor Model: {data.anchor.away_points.toFixed(1)} pts · {data.anchor.away_yards.toFixed(0)} yds
                </span>
              </div>

              {/* Game Result Center */}
              <div className="flex flex-col items-center justify-center p-4 text-center">
                <span className="text-xs uppercase font-mono tracking-widest text-oracle-muted">Final Simulated Result</span>
                <span className="text-lg font-bold text-oracle-green mt-1">
                  {data.summary.simulated_winner === 'TIE'
                    ? 'Game Tied'
                    : `${data.summary.simulated_winner} Wins`}
                </span>
                <span className="text-xs font-mono text-oracle-muted mt-2">
                  Total Drives: {data.summary.total_drives} · Home Win Prob: {(data.anchor.home_win_probability * 100).toFixed(0)}%
                </span>
              </div>

              {/* Home Team */}
              <div className="flex flex-col items-center md:items-end p-4 rounded-xl bg-oracle-surface/40 border border-oracle-border/40">
                <span className="text-xs font-semibold uppercase tracking-wider text-oracle-muted">Home</span>
                <span className="text-2xl font-bold font-display text-oracle-white mt-1">{data.home_team}</span>
                <div className="flex items-baseline gap-2 mt-2">
                  <span className="text-xs text-oracle-muted">pts (Sim)</span>
                  <span className="text-3xl font-bold text-oracle-white font-mono">
                    {data.summary.simulated_home_points}
                  </span>
                </div>
                <span className="text-xs font-mono text-oracle-muted mt-1">
                  Anchor Model: {data.anchor.home_points.toFixed(1)} pts · {data.anchor.home_yards.toFixed(0)} yds
                </span>
              </div>
            </div>
          </div>

          {/* Drive-by-Drive Timeline */}
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-bold font-display uppercase tracking-wider text-oracle-white">
                Drive Sequence Timeline
              </h2>
              <span className="text-xs text-oracle-muted">
                Click any drive to expand play-by-play details
              </span>
            </div>

            <div className="flex flex-col gap-3">
              {data.drives.map((drive: SimulatedDriveDTO) => {
                const style = OUTCOME_STYLES[drive.outcome] || {
                  bg: 'bg-oracle-surface border-oracle-border',
                  text: 'text-oracle-muted',
                  label: drive.outcome,
                }
                const isExpanded = expandedDrives.has(drive.drive_number)
                const isHome = drive.possession_team === data.home_team

                return (
                  <div
                    key={drive.drive_number}
                    className="rounded-xl border border-oracle-border/60 bg-oracle-card/40 overflow-hidden transition-all"
                  >
                    {/* Drive Header Bar */}
                    <button
                      type="button"
                      onClick={() => toggleDriveExpand(drive.drive_number)}
                      className="w-full p-3.5 flex flex-wrap items-center justify-between gap-3 text-left hover:bg-oracle-surface/40 transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        <span className="font-mono text-xs font-bold text-oracle-muted w-14">
                          Drive {drive.drive_number}
                        </span>
                        <span className="px-2 py-0.5 text-[10px] font-semibold rounded bg-oracle-surface border border-oracle-border text-oracle-muted">
                          Q{drive.quarter}
                        </span>
                        <span
                          className={`text-sm font-bold ${
                            isHome ? 'text-oracle-white' : 'text-oracle-white'
                          }`}
                        >
                          {drive.possession_team}
                        </span>
                        <span className="text-xs text-oracle-muted font-mono">
                          Own {drive.start_field_pos} → {drive.end_field_pos >= 100 ? 'Endzone' : `Own ${drive.end_field_pos}`}
                        </span>
                      </div>

                      <div className="flex items-center gap-3">
                        <span className="text-xs font-mono text-oracle-muted">
                          {drive.plays_count} plays · {drive.yards_gained.toFixed(0)} yds
                        </span>
                        <span
                          className={`px-2.5 py-0.5 rounded text-xs font-bold border ${style.bg} ${style.text}`}
                        >
                          {style.label}
                        </span>
                        <span className="font-mono text-xs font-bold text-oracle-white pl-2 border-l border-oracle-border">
                          {data.away_team} {drive.away_score_after} - {drive.home_score_after} {data.home_team}
                        </span>
                        <span className="text-xs text-oracle-muted">{isExpanded ? '▲' : '▼'}</span>
                      </div>
                    </button>

                    {/* Expandable Play-by-Play Table */}
                    {isExpanded && drive.plays.length > 0 && (
                      <div className="border-t border-oracle-border/40 bg-oracle-dark/50 p-3 overflow-x-auto">
                        <table className="min-w-full text-left text-xs">
                          <thead className="text-[10px] uppercase text-oracle-muted tracking-wider border-b border-oracle-border/40">
                            <tr>
                              <th className="pb-1.5 px-2">Play</th>
                              <th className="pb-1.5 px-2">Down & Dist</th>
                              <th className="pb-1.5 px-2">Field Pos</th>
                              <th className="pb-1.5 px-2">Type</th>
                              <th className="pb-1.5 px-2">Gain</th>
                              <th className="pb-1.5 px-2">Result</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-oracle-border/20 font-mono">
                            {drive.plays.map((play: SimulatedPlayDTO) => (
                              <tr key={play.play_number} className="hover:bg-oracle-surface/20">
                                <td className="py-1.5 px-2 text-oracle-muted">#{play.play_number}</td>
                                <td className="py-1.5 px-2 text-oracle-white">
                                  {play.down}&amp;{play.ytg}
                                </td>
                                <td className="py-1.5 px-2 text-oracle-muted">Own {play.field_pos}</td>
                                <td className="py-1.5 px-2 uppercase font-semibold text-oracle-muted">
                                  {play.play_type}
                                </td>
                                <td
                                  className={`py-1.5 px-2 font-bold ${
                                    play.yards_gained > 0
                                      ? 'text-emerald-400'
                                      : play.yards_gained < 0
                                        ? 'text-rose-400'
                                        : 'text-oracle-muted'
                                  }`}
                                >
                                  {play.yards_gained > 0 ? `+${play.yards_gained}` : play.yards_gained}
                                </td>
                                <td className="py-1.5 px-2">
                                  {play.is_touchdown && (
                                    <span className="text-emerald-400 font-bold">TOUCHDOWN 🏈</span>
                                  )}
                                  {play.is_first_down && !play.is_touchdown && (
                                    <span className="text-oracle-green font-medium">1st Down ➔</span>
                                  )}
                                  {play.is_turnover && (
                                    <span className="text-rose-400 font-bold">TURNOVER ⚠️</span>
                                  )}
                                  {play.is_safety && (
                                    <span className="text-purple-400 font-bold">SAFETY 🛑</span>
                                  )}
                                  {!play.is_first_down && !play.is_touchdown && !play.is_turnover && !play.is_safety && (
                                    <span className="text-oracle-muted">No 1st down</span>
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
