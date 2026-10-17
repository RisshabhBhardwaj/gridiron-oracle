import { useMemo, useState, useEffect } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useDraftBoard } from '@/hooks/useDraftBoard'
import { useMockDraftProfiles, useMockDraftPick } from '@/hooks/useMockDraft'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { SkeletonCard } from '@/components/shared/LoadingSpinner'
import { POSITIONS } from '@/lib/constants'
import { PositionBadge } from '@/components/PositionBadge'
import { positionTone } from '@/lib/tokens'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import type { LegacyRow } from '@tanstack/react-table/legacy'
import { DataTable, type TableColumn } from '@/components/ui/data-table'
import type { DraftBoardPlayer, PickedItem } from '@/types/api'

type Mode = 'board' | 'mock'
type SortKey = 'adp' | 'model_rank' | 'value_vs_adp' | 'model_fantasy_ppr'

const SEASON_ITEMS = [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019].map((s) => ({
  value: String(s),
  label: String(s),
}))
const SLOT_ITEMS = Array.from({ length: 8 }, (_, i) => ({
  value: String(i + 1),
  label: `Slot ${i + 1}`,
}))
const SORT_ITEMS: Array<{ value: SortKey; label: string }> = [
  { value: 'adp', label: 'ADP' },
  { value: 'model_rank', label: 'Model rank' },
  { value: 'value_vs_adp', label: 'Value vs ADP' },
  { value: 'model_fantasy_ppr', label: 'Model PPR' },
]

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

  const showDraftAction = mode === 'mock' && isDraftActive

  const columns: TableColumn<DraftBoardPlayer>[] = useMemo(() => {
    const cols: TableColumn<DraftBoardPlayer>[] = []
    if (showDraftAction) {
      cols.push({
        header: 'Action',
        id: 'action',
        cell: ({ row }: { row: LegacyRow<DraftBoardPlayer> }) => (
          <button
            onClick={() => handleUserPick(row.original)}
            disabled={!isUserTurn}
            className={`px-2.5 py-1 rounded text-xs font-bold ${
              isUserTurn
                ? 'bg-accent text-accent-fg hover:brightness-110'
                : 'bg-line text-fg-muted cursor-not-allowed opacity-40'
            }`}
          >
            Draft
          </button>
        ),
      })
    }
    cols.push(
      {
        header: 'ADP#',
        accessorKey: 'adp_rank',
        mono: true,
        cell: ({ row }: { row: LegacyRow<DraftBoardPlayer> }) => (
          <span className="text-fg-muted">{row.original.adp_rank ?? '—'}</span>
        ),
      },
      {
        header: 'Player',
        accessorKey: 'player_name',
        cell: ({ row }: { row: LegacyRow<DraftBoardPlayer> }) => (
          <span className="font-medium text-fg">
            {row.original.player_name}
            {row.original.board_tail && (
              <span className="ml-1.5 px-1.5 py-0.5 text-[9px] rounded bg-line text-fg-muted">
                Tail
              </span>
            )}
          </span>
        ),
      },
      {
        header: 'Pos',
        accessorKey: 'position',
        cell: ({ row }: { row: LegacyRow<DraftBoardPlayer> }) => (
          <PositionBadge position={row.original.position} />
        ),
      },
      {
        header: 'Team',
        accessorKey: 'team',
        cell: ({ row }: { row: LegacyRow<DraftBoardPlayer> }) => (
          <span className="text-fg-muted">{row.original.team ?? '—'}</span>
        ),
      },
      {
        header: 'ADP',
        accessorKey: 'adp',
        mono: true,
        cell: ({ row }: { row: LegacyRow<DraftBoardPlayer> }) => (
          <>{row.original.adp != null ? row.original.adp.toFixed(1) : '—'}</>
        ),
      },
      {
        header: 'Model#',
        accessorKey: 'model_rank',
        mono: true,
        cell: ({ row }: { row: LegacyRow<DraftBoardPlayer> }) => (
          <>{row.original.model_rank ?? '—'}</>
        ),
      },
      {
        header: 'Model PPR',
        accessorKey: 'model_fantasy_ppr',
        mono: true,
        cell: ({ row }: { row: LegacyRow<DraftBoardPlayer> }) => (
          <>{row.original.model_fantasy_ppr != null ? row.original.model_fantasy_ppr.toFixed(1) : '—'}</>
        ),
      },
      {
        header: 'Value',
        accessorKey: 'value_vs_adp',
        mono: true,
        cell: ({ row }: { row: LegacyRow<DraftBoardPlayer> }) => {
          const v = row.original.value_vs_adp ?? 0
          return (
            <span className={v > 0 ? 'text-positive' : v < 0 ? 'text-negative' : 'text-fg-muted'}>
              {row.original.value_vs_adp != null
                ? `${row.original.value_vs_adp > 0 ? '+' : ''}${row.original.value_vs_adp.toFixed(0)}`
                : '—'}
            </span>
          )
        },
      },
    )
    return cols
    // handleUserPick closes over the current draft state, so the action
    // column always drafts the row it sits on.
  }, [showDraftAction, isUserTurn])
  return (
    <div className="flex flex-col gap-6">
      {/* Header & Mode Switcher */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-line/60 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold font-display tracking-wider text-fg">
              {mode === 'board' ? 'Draft Board' : 'Mock Draft Simulator'}
            </h1>
            <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-accent/10 text-accent border border-accent/20">
              {season} Season
            </span>
          </div>
          <p className="max-w-2xl text-xs text-fg-muted mt-1">
            {mode === 'board'
              ? 'Preseason Full PPR projections vs available ADP source. Ranked by value over a replacement player in an 8-team league.'
              : 'Simulate live snake drafts against real manager reach tendencies & historical timing.'}
          </p>
        </div>

        <Tabs value={mode} onValueChange={(v) => setMode(v as Mode)}>
          <TabsList aria-label="Draft mode">
            <TabsTrigger value="board">Draft Board</TabsTrigger>
            <TabsTrigger value="mock">Mock Draft Mode</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {/* Mock Draft Manager Order Configuration */}
      {mode === 'mock' && (
        <div className="flex flex-col gap-4 p-4 rounded-xl bg-surface-raised/40 border border-line/60">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-xs font-bold uppercase tracking-wider text-fg">
              8-Team Draft Order & Leaguemate Tendencies
            </h2>
            {!profilesLoading && draftOrder.length !== 8 && (
              <p className="text-[11px] text-negative/80">
                Manager profiles unavailable — a mock draft needs all 8 leaguemates.
              </p>
            )}
            <div className="flex items-center gap-3">
              <span className="text-xs text-fg-muted" id="slot-label">Your Draft Slot:</span>
              <Select
                items={SLOT_ITEMS.map((item) => ({
                  ...item,
                  label: item.value === String(userSlot) ? `${item.label} (You)` : item.label,
                }))}
                value={String(userSlot)}
                onValueChange={(v) => { if (v) setUserSlot(Number(v)) }}
                disabled={isDraftActive}
              >
                <SelectTrigger aria-labelledby="slot-label" className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {SLOT_ITEMS.map((item) => (
                    <SelectItem key={item.value} value={item.value}>
                      {item.value === String(userSlot) ? `${item.label} (You)` : item.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {!isDraftActive ? (
                <button
                  onClick={handleStartDraft}
                  disabled={profilesLoading || draftOrder.length !== 8}
                  className="px-4 py-1.5 text-xs font-bold rounded-lg bg-accent text-accent-fg hover:brightness-110 shadow disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:brightness-100"
                >
                  Start Mock Draft
                </button>
              ) : (
                <button
                  onClick={handleResetDraft}
                  className="px-3 py-1.5 text-xs font-medium rounded-lg border border-negative/50 text-negative hover:bg-negative/10"
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
                      ? 'bg-accent/10 border-accent/40 text-fg'
                      : 'bg-surface-overlay/60 border-line/60 text-fg-muted'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-mono text-[10px] font-bold">Slot {idx + 1}</span>
                    {!isDraftActive && (
                      <div className="flex gap-1">
                        <button
                          onClick={() => moveManager(idx, 'up')}
                          disabled={idx === 0}
                          aria-label={`Move ${prof?.display_name || oid} earlier`}
                          className="hover:text-fg disabled:opacity-20"
                        >
                          <ChevronLeft className="size-3.5" aria-hidden="true" />
                        </button>
                        <button
                          onClick={() => moveManager(idx, 'down')}
                          disabled={idx === 7}
                          aria-label={`Move ${prof?.display_name || oid} later`}
                          className="hover:text-fg disabled:opacity-20"
                        >
                          <ChevronRight className="size-3.5" aria-hidden="true" />
                        </button>
                      </div>
                    )}
                  </div>
                  <span className="font-semibold truncate text-fg">
                    {prof?.display_name || oid} {isUser && '(You)'}
                  </span>
                  <span className="font-mono text-[9px] mt-1 text-fg-muted">
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
                  ? 'bg-accent/20 border border-accent text-accent animate-pulse'
                  : 'bg-surface-overlay border border-line text-fg'
              }`}
            >
              <div className="flex items-center gap-2 text-xs">
                <span className="font-mono font-bold">Pick #{currentPickNo} (Round {currentRound}):</span>
                <span>
                  {isUserTurn
                    ? 'Your turn! Pick a player below to draft.'
                    : `On the clock: ${currentManagerName} (Slot ${currentSlot})`}
                </span>
              </div>
              {!isUserTurn && currentPickNo <= totalPicks && (
                <button
                  onClick={handleSimulateNextOpponents}
                  disabled={pickMutation.isPending}
                  className="px-3 py-1 text-xs font-bold rounded bg-accent text-accent-fg hover:brightness-110"
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
        <div className="flex items-center gap-2 text-xs text-fg-muted">
          <span id="draft-season-label">Draft season</span>
          <Select
            items={SEASON_ITEMS}
            value={String(season)}
            onValueChange={(v) => { if (v) setSeason(Number(v)) }}
          >
            <SelectTrigger aria-labelledby="draft-season-label" className="h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {SEASON_ITEMS.map((item) => (
                <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <ToggleGroup
          aria-label="Position filter"
          value={position === null ? ['ALL'] : [position]}
          onValueChange={(next) => {
            // Clicking the pressed filter keeps it (single-choice, always one on).
            if (next.length === 0) return
            setPosition(next[0] === 'ALL' ? null : next[0])
          }}
        >
          <ToggleGroupItem value="ALL">ALL</ToggleGroupItem>
          {POSITIONS.map((p) => (
            <ToggleGroupItem key={p} value={p} tone={positionTone(p)}>
              {p}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>

        <div className="flex items-center gap-2 text-xs text-fg-muted">
          <span id="sort-label">Sort</span>
          <Select
            items={SORT_ITEMS}
            value={sortKey}
            onValueChange={(v) => { if (v) setSortKey(v as SortKey) }}
          >
            <SelectTrigger aria-labelledby="sort-label" className="h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {SORT_ITEMS.map((item) => (
                <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <button
          type="button"
          onClick={() => refetch()}
          className="rounded-md border border-line px-3 py-1.5 text-xs text-fg-muted hover:text-fg"
        >
          {isFetching ? 'Refreshing…' : 'Refresh'}
        </button>

        {data?.spearman_rho != null && (
          <span className="ml-auto font-mono text-xs text-accent" aria-label="Draft projection metadata">
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
        <div className="overflow-x-auto rounded-xl border border-line">
          <DataTable
            columns={columns}
            data={rows}
            className="min-w-full [&_th]:px-3 [&_th]:py-2 [&_td]:px-3 [&_td]:py-2"
          />
        </div>
      )}
    </div>
  )
}
