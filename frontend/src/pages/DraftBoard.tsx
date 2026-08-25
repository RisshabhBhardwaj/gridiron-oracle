import { useMemo, useState, useEffect } from 'react'
import { useDraftBoard } from '@/hooks/useDraftBoard'
import { useMockDraftProfiles, useMockDraftPick } from '@/hooks/useMockDraft'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { SkeletonCard } from '@/components/shared/LoadingSpinner'
import { POSITIONS } from '@/lib/constants'
import type { DraftBoardPlayer, PickedItem } from '@/types/api'

type Mode = 'board' | 'mock'
type SortKey = 'adp' | 'model_rank' | 'value_vs_adp' | 'model_fantasy_ppr'

const POS_CLASS: Record<string, string> = {
  WR: 'pos-wr',
  RB: 'pos-rb',
  TE: 'pos-te',
  QB: 'pos-qb',
}

export function DraftBoard() {
  const [mode, setMode] = useState<Mode>('board')
  const [season, setSeason] = useState(2026)
  const [position, setPosition] = useState<string | null>(null)
  const [sortKey, setSortKey] = useState<SortKey>('adp')

  // Board Query
  const { data, isLoading, error, refetch, isFetching } = useDraftBoard({
    season,
    source: null,
    position,
  })

  // Mock Draft Query & Mutation
  const { data: profileData, isLoading: profilesLoading } = useMockDraftProfiles(season)
  const pickMutation = useMockDraftPick()

  // Mock Draft State
  const [draftOrder, setDraftOrder] = useState<string[]>([])
  const [userSlot, setUserSlot] = useState<number>(1)
  const [picksSoFar, setPicksSoFar] = useState<PickedItem[]>([])
  const [isDraftActive, setIsDraftActive] = useState<boolean>(false)

  // Initialize draft order from profiles
  useEffect(() => {
    if (profileData?.profiles && profileData.profiles.length > 0 && draftOrder.length === 0) {
      const sorted = [...profileData.profiles].sort((a, b) => (a.draft_slot ?? 99) - (b.draft_slot ?? 99))
      setDraftOrder(sorted.map((p) => p.owner_id))
    }
  }, [profileData, draftOrder])

  // Move manager in order
  const moveManager = (idx: number, dir: 'up' | 'down') => {
    const target = dir === 'up' ? idx - 1 : idx + 1
    if (target < 0 || target >= draftOrder.length) return
    const updated = [...draftOrder]
    const temp = updated[idx]
    updated[idx] = updated[target]
    updated[target] = temp
    setDraftOrder(updated)
  }

  // Available board rows
  const draftedPids = useMemo(() => {
    return new Set(picksSoFar.map((p) => p.player.player_id).filter(Boolean))
  }, [picksSoFar])

  const rows = useMemo(() => {
    let list = [...(data?.players ?? [])]
    if (mode === 'mock' && isDraftActive) {
      list = list.filter((p) => !draftedPids.has(p.player_id))
    }
    list.sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      if (av == null && bv == null) return 0
      if (av == null) return 1
      if (bv == null) return -1
      if (sortKey === 'value_vs_adp' || sortKey === 'model_fantasy_ppr') {
        return Number(bv) - Number(av)
      }
      return Number(av) - Number(bv)
    })
    return list
  }, [data, sortKey, mode, isDraftActive, draftedPids])

  // Current turn info in Mock Draft
  const currentPickNo = picksSoFar.length + 1
  const totalPicks = 8 * 17
  const currentRound = Math.floor((currentPickNo - 1) / 8) + 1
  const posInRound = (currentPickNo - 1) % 8
  const currentSlot = currentRound % 2 === 1 ? posInRound + 1 : 8 - posInRound
  const isUserTurn = isDraftActive && currentSlot === userSlot && currentPickNo <= totalPicks

  const currentManagerName = useMemo(() => {
    if (!draftOrder[currentSlot - 1]) return `Slot ${currentSlot}`
    const prof = profileData?.profiles.find((p) => p.owner_id === draftOrder[currentSlot - 1])
    return prof?.display_name || draftOrder[currentSlot - 1]
  }, [currentSlot, draftOrder, profileData])

  // Handlers for Mock Draft
  const handleStartDraft = () => {
    setIsDraftActive(true)
    setPicksSoFar([])
    if (userSlot !== 1) {
      // Simulate opponent picks before slot
      pickMutation.mutate(
        {
          season,
          draft_order: draftOrder,
          picks_so_far: [],
          user_slot: userSlot,
        },
        {
          onSuccess: (res) => {
            setPicksSoFar(res.new_picks)
          },
        },
      )
    }
  }

  const handleSimulateNextOpponents = () => {
    pickMutation.mutate(
      {
        season,
        draft_order: draftOrder,
        picks_so_far: picksSoFar,
        user_slot: userSlot,
      },
      {
        onSuccess: (res) => {
          setPicksSoFar((prev) => [...prev, ...res.new_picks])
        },
      },
    )
  }

  const handleUserPick = (player: DraftBoardPlayer) => {
    const userPickItem: PickedItem = {
      pick_no: currentPickNo,
      round: currentRound,
      slot: currentSlot,
      owner_id: draftOrder[userSlot - 1] || 'user',
      player,
    }
    const updated = [...picksSoFar, userPickItem]
    setPicksSoFar(updated)

    // Simulate following opponent picks until user turn again
    if (currentPickNo < totalPicks) {
      pickMutation.mutate(
        {
          season,
          draft_order: draftOrder,
          picks_so_far: updated,
          user_slot: userSlot,
        },
        {
          onSuccess: (res) => {
            setPicksSoFar((prev) => [...prev, ...res.new_picks])
          },
        },
      )
    }
  }

  const handleResetDraft = () => {
    setIsDraftActive(false)
    setPicksSoFar([])
  }

  return (
    <div className="flex flex-col gap-6 p-4 sm:p-6 max-w-7xl mx-auto">
      {/* Header & Mode Switcher */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-oracle-border/60 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold font-display tracking-wider text-oracle-white">
              {mode === 'board' ? 'Draft Board' : 'Mock Draft Simulator'}
            </h1>
            <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-oracle-green/10 text-oracle-green border border-oracle-green/20">
              {season} Season
            </span>
          </div>
          <p className="max-w-2xl text-xs text-oracle-muted mt-1">
            {mode === 'board'
              ? 'Preseason Full PPR projections vs available ADP source. Ranked via 8-team VOR.'
              : 'Simulate live snake drafts against real manager reach tendencies & historical timing.'}
          </p>
        </div>

        <div className="flex items-center rounded-lg bg-oracle-surface p-1 border border-oracle-border">
          <button
            onClick={() => setMode('board')}
            className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-colors ${
              mode === 'board'
                ? 'bg-oracle-green text-oracle-dark shadow'
                : 'text-oracle-muted hover:text-oracle-white'
            }`}
          >
            Draft Board
          </button>
          <button
            onClick={() => setMode('mock')}
            className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-colors ${
              mode === 'mock'
                ? 'bg-oracle-green text-oracle-dark shadow'
                : 'text-oracle-muted hover:text-oracle-white'
            }`}
          >
            Mock Draft Mode
          </button>
        </div>
      </div>

      {/* Mock Draft Manager Order Configuration */}
      {mode === 'mock' && (
        <div className="flex flex-col gap-4 p-4 rounded-xl bg-oracle-card/40 border border-oracle-border/60">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-xs font-bold uppercase tracking-wider text-oracle-white">
              8-Team Draft Order & Leaguemate Tendencies
            </h2>
            {!profilesLoading && draftOrder.length !== 8 && (
              <p className="text-[11px] text-oracle-red/80">
                Manager profiles unavailable — a mock draft needs all 8 leaguemates.
              </p>
            )}
            <div className="flex items-center gap-3">
              <label className="text-xs text-oracle-muted">Your Draft Slot:</label>
              <select
                value={userSlot}
                disabled={isDraftActive}
                onChange={(e) => setUserSlot(Number(e.target.value))}
                className="oracle-select bg-oracle-surface border border-oracle-border text-xs rounded-lg px-2.5 py-1 text-oracle-white"
              >
                {Array.from({ length: 8 }, (_, i) => i + 1).map((s) => (
                  <option key={s} value={s}>
                    Slot {s} {s === userSlot ? '(You)' : ''}
                  </option>
                ))}
              </select>

              {!isDraftActive ? (
                <button
                  onClick={handleStartDraft}
                  disabled={profilesLoading || draftOrder.length !== 8}
                  className="px-4 py-1.5 text-xs font-bold rounded-lg bg-oracle-green text-oracle-dark hover:brightness-110 shadow disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:brightness-100"
                >
                  Start Mock Draft
                </button>
              ) : (
                <button
                  onClick={handleResetDraft}
                  className="px-3 py-1.5 text-xs font-medium rounded-lg border border-oracle-red/50 text-oracle-red hover:bg-oracle-red/10"
                >
                  Reset Draft
                </button>
              )}
            </div>
          </div>

          {/* Manager order pills */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
            {draftOrder.map((oid, idx) => {
              const prof = profileData?.profiles.find((p) => p.owner_id === oid)
              const isUser = idx + 1 === userSlot
              return (
                <div
                  key={oid}
                  className={`p-2 rounded-lg border flex flex-col justify-between text-xs ${
                    isUser
                      ? 'bg-oracle-green/10 border-oracle-green/40 text-oracle-white'
                      : 'bg-oracle-surface/60 border-oracle-border/60 text-oracle-muted'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-mono text-[10px] font-bold">Slot {idx + 1}</span>
                    {!isDraftActive && (
                      <div className="flex gap-1">
                        <button
                          onClick={() => moveManager(idx, 'up')}
                          disabled={idx === 0}
                          className="hover:text-oracle-white disabled:opacity-20"
                        >
                          ‹
                        </button>
                        <button
                          onClick={() => moveManager(idx, 'down')}
                          disabled={idx === 7}
                          className="hover:text-oracle-white disabled:opacity-20"
                        >
                          ›
                        </button>
                      </div>
                    )}
                  </div>
                  <span className="font-semibold truncate text-oracle-white">
                    {prof?.display_name || oid} {isUser && '(You)'}
                  </span>
                  <span className="font-mono text-[9px] mt-1 text-oracle-muted">
                    ADP Δ: {prof?.adp_delta_mean_shrunk != null ? `${prof.adp_delta_mean_shrunk > 0 ? '+' : ''}${prof.adp_delta_mean_shrunk.toFixed(1)}` : '0.0'}
                  </span>
                </div>
              )
            })}
          </div>

          {/* Current Turn Notification */}
          {isDraftActive && (
            <div
              className={`p-3 rounded-lg flex items-center justify-between ${
                isUserTurn
                  ? 'bg-oracle-green/20 border border-oracle-green text-oracle-green animate-pulse'
                  : 'bg-oracle-surface border border-oracle-border text-oracle-white'
              }`}
            >
              <div className="flex items-center gap-2 text-xs">
                <span className="font-mono font-bold">Pick #{currentPickNo} (Round {currentRound}):</span>
                <span>
                  {isUserTurn
                    ? '🎯 YOUR TURN! Select a player below to draft.'
                    : `On the clock: ${currentManagerName} (Slot ${currentSlot})`}
                </span>
              </div>
              {!isUserTurn && currentPickNo <= totalPicks && (
                <button
                  onClick={handleSimulateNextOpponents}
                  disabled={pickMutation.isPending}
                  className="px-3 py-1 text-xs font-bold rounded bg-oracle-green text-oracle-dark hover:brightness-110"
                >
                  {pickMutation.isPending ? 'Simulating…' : 'Simulate Picks →'}
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {/* Controls Bar */}
      <div className="flex flex-wrap items-center gap-3">
        <label className="text-xs text-oracle-muted">
          Draft season
          <select
            className="ml-2 rounded-md border border-oracle-border bg-oracle-card px-2 py-1.5 text-sm text-oracle-white"
            aria-label="Draft season"
            value={season}
            onChange={(e) => setSeason(Number(e.target.value))}
          >
            {[2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019].map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>

        <div className="flex gap-1">
          <button
            type="button"
            onClick={() => setPosition(null)}
            className={`rounded-md px-3 py-1.5 text-xs font-medium ${
              position == null ? 'bg-oracle-green/20 text-oracle-green' : 'text-oracle-muted hover:text-oracle-white'
            }`}
          >
            ALL
          </button>
          {POSITIONS.map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => setPosition(p)}
              className={`rounded-md px-3 py-1.5 text-xs font-medium ${
                position === p ? 'bg-oracle-green/20 text-oracle-green' : 'text-oracle-muted hover:text-oracle-white'
              }`}
            >
              {p}
            </button>
          ))}
        </div>

        <label className="text-xs text-oracle-muted">
          Sort
          <select
            className="ml-2 rounded-md border border-oracle-border bg-oracle-card px-2 py-1.5 text-sm text-oracle-white"
            value={sortKey}
            onChange={(e) => setSortKey(e.target.value as SortKey)}
          >
            <option value="adp">ADP</option>
            <option value="model_rank">Model rank</option>
            <option value="value_vs_adp">Value vs ADP</option>
            <option value="model_fantasy_ppr">Model PPR</option>
          </select>
        </label>

        <button
          type="button"
          onClick={() => refetch()}
          className="rounded-md border border-oracle-border px-3 py-1.5 text-xs text-oracle-muted hover:text-oracle-white"
        >
          {isFetching ? 'Refreshing…' : 'Refresh'}
        </button>

        {data?.spearman_rho != null && (
          <span className="ml-auto font-mono text-xs text-oracle-green" aria-label="Draft projection metadata">
            {data.projection_source} · as of {data.as_of} · ρ={data.spearman_rho.toFixed(3)} · n={data.count}
          </span>
        )}
      </div>

      {error && <ErrorBanner error={error} onRetry={() => refetch()} />}

      {isLoading && (
        <div>
          <SkeletonCard count={8} />
        </div>
      )}

      {!isLoading && rows.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-oracle-border">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-oracle-surface text-[11px] uppercase tracking-wider text-oracle-muted">
              <tr>
                {mode === 'mock' && isDraftActive && <th className="px-3 py-2">Action</th>}
                <th className="px-3 py-2">ADP#</th>
                <th className="px-3 py-2">Player</th>
                <th className="px-3 py-2">Pos</th>
                <th className="px-3 py-2">Team</th>
                <th className="px-3 py-2">ADP</th>
                <th className="px-3 py-2">Model#</th>
                <th className="px-3 py-2">Model PPR</th>
                <th className="px-3 py-2">Value</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row: DraftBoardPlayer) => (
                <tr
                  key={`${row.player_name}-${row.adp_rank ?? row.model_rank}`}
                  className="border-t border-oracle-border/60 hover:bg-oracle-card/40"
                >
                  {mode === 'mock' && isDraftActive && (
                    <td className="px-3 py-2">
                      <button
                        onClick={() => handleUserPick(row)}
                        disabled={!isUserTurn}
                        className={`px-2.5 py-1 rounded text-xs font-bold ${
                          isUserTurn
                            ? 'bg-oracle-green text-oracle-dark hover:brightness-110'
                            : 'bg-oracle-border text-oracle-muted cursor-not-allowed opacity-40'
                        }`}
                      >
                        Draft
                      </button>
                    </td>
                  )}
                  <td className="px-3 py-2 font-mono text-oracle-muted">{row.adp_rank ?? '—'}</td>
                  <td className="px-3 py-2 font-medium text-oracle-white">
                    {row.player_name}
                    {row.board_tail && (
                      <span className="ml-1.5 px-1.5 py-0.5 text-[9px] rounded bg-oracle-border text-oracle-muted">
                        Tail
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2">
                    <span className={POS_CLASS[row.position ?? ''] ?? 'text-oracle-muted'}>
                      {row.position ?? '—'}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-oracle-muted">{row.team ?? '—'}</td>
                  <td className="px-3 py-2 font-mono">{row.adp != null ? row.adp.toFixed(1) : '—'}</td>
                  <td className="px-3 py-2 font-mono">{row.model_rank ?? '—'}</td>
                  <td className="px-3 py-2 font-mono">
                    {row.model_fantasy_ppr != null ? row.model_fantasy_ppr.toFixed(1) : '—'}
                  </td>
                  <td
                    className={`px-3 py-2 font-mono ${
                      (row.value_vs_adp ?? 0) > 0
                        ? 'text-oracle-green'
                        : (row.value_vs_adp ?? 0) < 0
                          ? 'text-oracle-red'
                          : 'text-oracle-muted'
                    }`}
                  >
                    {row.value_vs_adp != null
                      ? `${row.value_vs_adp > 0 ? '+' : ''}${row.value_vs_adp.toFixed(0)}`
                      : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
