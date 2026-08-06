import { useState, useCallback, useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSeasonProjections } from '@/hooks/useSeasonProjections'
import { useCurrentSeason } from '@/hooks/useCurrentSeason'
import { useSearch } from '@/context/SearchContext'
import { SkeletonCard } from '@/components/shared/LoadingSpinner'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { DataStalenessWarning } from '@/components/shared/DataStalenessWarning'
import { CURRENT_SEASON, POSITIONS } from '@/lib/constants'
import type { SeasonPlayerProjection } from '@/types/api'

type SortKey = 'passing_yards' | 'rushing_yards' | 'receiving_yards' | 'fantasy_ppr'
type SortDir = 'asc' | 'desc'

const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: 'fantasy_ppr',     label: 'Fantasy PPR' },
  { key: 'passing_yards',   label: 'Pass Yds' },
  { key: 'rushing_yards',   label: 'Rush Yds' },
  { key: 'receiving_yards', label: 'Rec Yds' },
]

const POS_CLASS: Record<string, string> = {
  WR: 'pos-wr', RB: 'pos-rb', TE: 'pos-te', QB: 'pos-qb',
}
const POS_COLORS: Record<string, string> = {
  WR: '#00C2FF', RB: '#00FFA3', TE: '#A855F7', QB: '#FFB800',
}

interface PlayerCardProps {
  row: SeasonPlayerProjection
  onClick: () => void
  statKey: SortKey
  style?: React.CSSProperties
}

function PlayerCard({ row, onClick, statKey, style }: PlayerCardProps) {
  const posColor = POS_COLORS[row.position] ?? '#607B9B'
  
  const statData = row[statKey]
  const projection = statData?.mean ?? 0
  const floor = statData?.p10 ?? 0
  const ceiling = statData?.p90 ?? 0

  if (!statData) return null

  return (
    <div
      className="relative rounded-xl overflow-hidden flex flex-col gap-3 p-4 group cursor-pointer transition-all duration-250"
      style={{
        background: 'linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.015) 100%)',
        border: '1px solid rgba(255,255,255,0.06)',
        boxShadow: '0 2px 12px rgba(0,0,0,0.3)',
        animation: 'fade-in 0.4s ease-out forwards',
        ...style,
      }}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter') onClick() }}
      onMouseEnter={(e) => {
        const el = e.currentTarget as HTMLDivElement
        el.style.transform = 'translateY(-2px)'
        el.style.boxShadow = `0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px ${posColor}30`
        el.style.borderColor = `${posColor}30`
      }}
      onMouseLeave={(e) => {
        const el = e.currentTarget as HTMLDivElement
        el.style.transform = ''
        el.style.boxShadow = '0 2px 12px rgba(0,0,0,0.3)'
        el.style.borderColor = 'rgba(255,255,255,0.06)'
      }}
    >
      {/* Name + position */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-oracle-white truncate leading-tight">{row.player_name}</p>
          <p className="text-[11px] text-oracle-muted mt-0.5">{row.team ?? 'UNK'}</p>
        </div>
        <span className={POS_CLASS[row.position] ?? 'pos-badge text-oracle-muted border-oracle-border'}>
          {row.position}
        </span>
      </div>

      {/* Projection */}
      <div className="flex items-baseline gap-1.5 mt-2">
        <span
          className="font-display text-4xl leading-none"
          style={{ color: posColor, textShadow: `0 0 16px ${posColor}60` }}
        >
          {projection.toFixed(1)}
        </span>
        <span className="text-xs text-oracle-muted">proj</span>
      </div>

      {/* Floor—Ceiling range bar */}
      <div className="flex flex-col gap-1.5 mt-2">
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(26,47,78,0.8)' }}>
          <div
            className="h-full rounded-full"
            style={{ width: '100%', background: `linear-gradient(90deg, rgba(255,59,92,0.6) 0%, ${posColor}80 50%, rgba(0,255,163,0.6) 100%)` }}
          />
        </div>
        <div className="flex justify-between text-[10px] font-mono">
          <span style={{ color: '#FF3B5C' }}>{floor.toFixed(0)}</span>
          <span className="text-oracle-muted">range (p10/p90)</span>
          <span style={{ color: '#00FFA3' }}>{ceiling.toFixed(0)}</span>
        </div>
      </div>
    </div>
  )
}

export function SeasonProjections() {
  const navigate = useNavigate()
  const { setDefaults } = useSearch()
  const { data: currentSeasonData } = useCurrentSeason()
  
  const season = currentSeasonData?.season ?? CURRENT_SEASON
  const startWeek = currentSeasonData?.week ? currentSeasonData.week + 1 : 1

  const [positions, setPositions] = useState<string[]>([...POSITIONS])
  const [sortKey, setSortKey] = useState<SortKey>('fantasy_ppr')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  const { data, isLoading, isError, error, refetch } = useSeasonProjections({
    season,
    startWeek,
    positions,
  })

  function togglePosition(pos: string) {
    setPositions((prev) => prev.includes(pos) ? prev.filter((p) => p !== pos) : [...prev, pos])
  }

  const sorted = useMemo(() => {
    if (!data?.projections) return []
    return [...data.projections]
      .filter((p) => p[sortKey] != null) // only show players that have a projection for this stat
      .sort((a, b) => {
        const aVal = a[sortKey]?.mean ?? 0
        const bVal = b[sortKey]?.mean ?? 0
        return sortDir === 'asc' ? aVal - bVal : bVal - aVal
      })
  }, [data?.projections, sortKey, sortDir])

  const handleRowClick = useCallback((row: SeasonPlayerProjection) => {
    navigate(`/player/${row.player_id}?name=${encodeURIComponent(row.player_name)}`)
  }, [navigate])

  const activeSortLabel = SORT_OPTIONS.find((option) => option.key === sortKey)?.label ?? sortKey

  useEffect(() => {
    setDefaults({ week: Math.min(startWeek, 18), season, stat: sortKey })
  }, [season, setDefaults, sortKey, startWeek])

  return (
    <div className="flex flex-col gap-6 p-2 sm:p-4">
      {/* Page Header */}
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-3 flex-wrap">
          <h1
            className="font-display text-5xl tracking-widest uppercase"
            style={{
              background: 'linear-gradient(135deg, #00C2FF 0%, #00FFA3 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Rest of Season
          </h1>
          <span className="px-3 py-1 rounded-full text-xs font-bold tracking-widest"
            style={{ background: 'rgba(0,194,255,0.15)', border: '1px solid rgba(0,194,255,0.3)', color: '#00C2FF' }}>
            SIMULATED
          </span>
          <span className="text-sm font-semibold text-oracle-purple bg-oracle-purple/10 border border-oracle-purple/30 px-3 py-1 rounded-full">
            W{startWeek}-18
          </span>
        </div>
        <p className="text-sm text-oracle-muted">Autoregressive Markov Chain simulations powering ROS rest-of-season totals</p>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-[11px] text-oracle-muted">
        <span className="px-2.5 py-1 rounded-full"
          style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
          Ranking by {activeSortLabel}
        </span>
        <span className="px-2.5 py-1 rounded-full"
          style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
          {positions.length === POSITIONS.length ? 'All positions' : positions.join('/')}
        </span>
        <span>
          Rest-of-season totals from Week {startWeek} through Week 18.
        </span>
      </div>

      {/* Filter Bar */}
      <div
        className="flex flex-wrap items-center gap-3 p-4 rounded-xl"
        style={{ background: 'rgba(13,27,48,0.8)', border: '1px solid rgba(255,255,255,0.06)', backdropFilter: 'blur(12px)' }}
      >
        {/* Position toggles */}
        <div className="flex items-center gap-1.5">
          {POSITIONS.map((pos) => (
            <button
              key={pos}
              onClick={() => togglePosition(pos)}
              className="px-3 py-1.5 rounded-lg text-xs font-bold tracking-wider transition-all duration-150 border"
              style={positions.includes(pos)
                ? { background: `${POS_COLORS[pos]}20`, borderColor: `${POS_COLORS[pos]}50`, color: POS_COLORS[pos], boxShadow: `0 0 10px ${POS_COLORS[pos]}20` }
                : { background: 'transparent', borderColor: 'rgba(255,255,255,0.07)', color: '#607B9B' }
              }
            >
              {pos}
            </button>
          ))}
        </div>

        <div className="w-px h-5 bg-oracle-border hidden sm:block" />

        {/* Sort */}
        <div className="flex items-center gap-1.5 flex-wrap flex-1 justify-end">
          <span className="text-[10px] font-semibold text-oracle-muted uppercase tracking-wider hidden sm:inline mr-1">Rank By</span>
          {SORT_OPTIONS.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => {
                if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
                else { setSortKey(key); setSortDir('desc') }
              }}
              className="px-2.5 py-1 rounded-lg text-[10px] font-semibold tracking-wide transition-all duration-150 border"
              style={sortKey === key
                ? { background: 'rgba(0,194,255,0.12)', borderColor: 'rgba(0,194,255,0.3)', color: '#00C2FF' }
                : { background: 'transparent', borderColor: 'rgba(255,255,255,0.06)', color: '#607B9B' }
              }
            >
              {label} {sortKey === key ? (sortDir === 'asc' ? '↑' : '↓') : ''}
            </button>
          ))}
        </div>
      </div>

      {/* Staleness */}
      {data && <DataStalenessWarning dataFreshness={data.data_freshness} />}

      {/* Loading */}
      {isLoading && <SkeletonCard count={8} />}

      {/* Error */}
      {isError && error && <ErrorBanner error={error} onRetry={() => void refetch()} />}

      {/* Player Grid */}
      {data && !isLoading && (
        <>
          <div className="flex flex-wrap items-center gap-2 text-xs text-oracle-muted">
            <span>{sorted.length} players projected</span>
            <span>·</span>
            <span>Updated {new Date(data.data_freshness).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}</span>
          </div>

          {sorted.length === 0 ? (
            <div className="rounded-xl p-12 text-center" style={{ background: 'rgba(13,27,48,0.6)', border: '1px solid rgba(255,255,255,0.05)' }}>
              <p className="text-oracle-muted text-sm">No ROS projections found for the selected filters.</p>
              <p className="text-oracle-muted text-xs mt-2">Try a different position mix or rank by a stat more relevant to the selected positions.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {sorted.map((row, idx) => (
                <PlayerCard
                  key={row.player_id}
                  row={row}
                  statKey={sortKey}
                  onClick={() => handleRowClick(row)}
                  style={{ animationDelay: `${Math.min(idx * 0.04, 0.6)}s`, opacity: 0 }}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}
