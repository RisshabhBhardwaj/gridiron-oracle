import { useEffect, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useWeekProjections } from '@/hooks/useWeekProjections'
import { useCurrentSeason } from '@/hooks/useCurrentSeason'
import { useOdds, findPlayerLine } from '@/hooks/useOdds'
import { useSearch } from '@/context/SearchContext'
import { calcEdge, useBetSlip } from '@/context/BetSlipContext'
import { AddToBetSlipButton } from '@/components/AddToBetSlipButton'
import { SkeletonCard } from '@/components/shared/LoadingSpinner'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { DataStalenessWarning } from '@/components/shared/DataStalenessWarning'
import {
  CURRENT_SEASON, CURRENT_WEEK, POSITIONS, STATS, STAT_LABELS,
} from '@/lib/constants'
import { formatPct } from '@/lib/formatters'
import type { WeekPlayerProjection } from '@/types/api'

type SortKey = keyof WeekPlayerProjection
type SortDir = 'asc' | 'desc'

const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: 'projection',         label: 'Projection' },
  { key: 'fantasy_projection', label: 'Fantasy' },
  { key: 'boom_probability',   label: 'Boom %' },
  { key: 'ceiling',            label: 'Ceiling' },
]

const POS_CLASS: Record<string, string> = {
  WR: 'pos-wr', RB: 'pos-rb', TE: 'pos-te', QB: 'pos-qb',
}
const POS_COLORS: Record<string, string> = {
  WR: '#00C2FF', RB: '#00FFA3', TE: '#A855F7', QB: '#FFB800',
}

interface PlayerCardProps {
  row: WeekPlayerProjection
  onClick: () => void
  bookLine: number | null
  week: number
  season: number
  stat: string
  style?: React.CSSProperties
}

function PlayerCard({ row, onClick, bookLine, week, season, stat, style }: PlayerCardProps) {
  const posColor   = POS_COLORS[row.position] ?? '#607B9B'
  const boomHigh   = (row.boom_probability ?? 0) >= 0.25
  const bustHigh   = (row.bust_probability ?? 0) >= 0.35
  const edge       = bookLine != null ? calcEdge(row.projection, bookLine) : null
  const edgePos    = edge !== null && edge > 0
  const edgeStr    = edge !== null ? `${edgePos ? '+' : ''}${edge.toFixed(1)}%` : null

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
      {/* Add to bet slip — top right */}
      <div className="absolute top-3 right-3 z-10" onClick={(e) => e.stopPropagation()}>
        <AddToBetSlipButton
          size="sm"
          leg={{
            player_id: row.player_id,
            player_name: row.player_name,
            position: row.position,
            team: row.team ?? null,
            stat,
            stat_label: STAT_LABELS[stat] ?? stat,
            week,
            season,
            our_projection: row.projection,
            floor: row.floor,
            ceiling: row.ceiling,
            book_line: bookLine,
            boom_probability: row.boom_probability ?? null,
            bust_probability: row.bust_probability ?? null,
            fantasy_projection: row.fantasy_projection ?? null,
          }}
        />
      </div>

      {/* Name + position */}
      <div className="flex items-start justify-between gap-2 pr-10">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-oracle-white truncate leading-tight">{row.player_name}</p>
          <p className="text-[11px] text-oracle-muted mt-0.5">{row.team ?? 'UNK'}</p>
        </div>
        <span className={POS_CLASS[row.position] ?? 'pos-badge text-oracle-muted border-oracle-border'}>
          {row.position}
        </span>
      </div>

      {/* Projection */}
      <div className="flex items-baseline gap-1.5">
        <span
          className="font-display text-4xl leading-none"
          style={{ color: posColor, textShadow: `0 0 16px ${posColor}60` }}
        >
          {row.projection.toFixed(1)}
        </span>
        <span className="text-xs text-oracle-muted">proj</span>
      </div>

      {/* Book line + edge */}
      {bookLine !== null && (
        <div
          className="flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs"
          style={{
            background: edgePos ? 'rgba(0,255,163,0.06)' : 'rgba(255,59,92,0.05)',
            border: `1px solid ${edgePos ? 'rgba(0,255,163,0.2)' : 'rgba(255,59,92,0.15)'}`,
          }}
        >
          <span className="text-oracle-muted">
            Line: <span className="text-oracle-white font-mono font-semibold">{bookLine.toFixed(1)}</span>
          </span>
          <span
            className="font-bold text-[11px]"
            style={{ color: edgePos ? '#00FFA3' : '#FF3B5C' }}
          >
            {edgeStr}
          </span>
        </div>
      )}

      {/* Floor—Ceiling range bar */}
      <div className="flex flex-col gap-1.5">
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(26,47,78,0.8)' }}>
          <div
            className="h-full rounded-full"
            style={{ width: '100%', background: `linear-gradient(90deg, rgba(255,59,92,0.6) 0%, ${posColor}80 50%, rgba(0,255,163,0.6) 100%)` }}
          />
        </div>
        <div className="flex justify-between text-[10px] font-mono">
          <span style={{ color: '#FF3B5C' }}>{row.floor == null ? '—' : row.floor.toFixed(0)}</span>
          <span className="text-oracle-muted">range</span>
          <span style={{ color: '#00FFA3' }}>{row.ceiling == null ? '—' : row.ceiling.toFixed(0)}</span>
        </div>
      </div>

      {/* Chips */}
      <div className="flex items-center gap-1.5 flex-wrap">
        {row.fantasy_projection != null && (
          <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full"
            style={{ background: 'rgba(168,85,247,0.15)', color: '#A855F7', border: '1px solid rgba(168,85,247,0.3)' }}>
            {row.fantasy_projection.toFixed(1)} PPR
          </span>
        )}
        {row.boom_probability != null && (
          <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full"
            style={{
              background: boomHigh ? 'rgba(0,255,163,0.12)' : 'rgba(26,47,78,0.6)',
              color: boomHigh ? '#00FFA3' : '#607B9B',
              border: `1px solid ${boomHigh ? 'rgba(0,255,163,0.3)' : 'rgba(255,255,255,0.06)'}`,
            }}>
            💥 {formatPct(row.boom_probability)}
          </span>
        )}
        {row.bust_probability != null && (
          <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full"
            style={{
              background: bustHigh ? 'rgba(255,59,92,0.12)' : 'rgba(26,47,78,0.6)',
              color: bustHigh ? '#FF3B5C' : '#607B9B',
              border: `1px solid ${bustHigh ? 'rgba(255,59,92,0.3)' : 'rgba(255,255,255,0.06)'}`,
            }}>
            📉 {formatPct(row.bust_probability)}
          </span>
        )}
      </div>
    </div>
  )
}

export function Dashboard() {
  const navigate = useNavigate()
  const [week, setWeek]       = useState(CURRENT_WEEK)
  const [season, setSeason]   = useState(CURRENT_SEASON)
  const [stat, setStat]       = useState('receiving_yards')
  const [positions, setPositions] = useState<string[]>([...POSITIONS])
  const [sortKey, setSortKey] = useState<SortKey>('projection')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const { setPlayers, setDefaults } = useSearch()
  const { state: slipState, setBookLine } = useBetSlip()

  const { data: currentSeasonData } = useCurrentSeason()
  useEffect(() => {
    if (currentSeasonData) {
      setWeek(currentSeasonData.week)
      setSeason(currentSeasonData.season)
    }
  }, [currentSeasonData])

  const { data, isLoading, isError, error, refetch } = useWeekProjections({ week, season, stat, positions })

  // Feed players into search context
  useEffect(() => {
    if (data?.projections) setPlayers(data.projections)
  }, [data?.projections, setPlayers])

  useEffect(() => {
    setDefaults({ week, season, stat })
  }, [season, setDefaults, stat, week])

  // Odds
  const playerNames = data?.projections.map((p) => p.player_name) ?? []
  const { data: oddsData } = useOdds({ playerNames, stat, enabled: playerNames.length > 0 })

  useEffect(() => {
    if (!oddsData?.player_props?.length || slipState.legs.length === 0) return
    for (const leg of slipState.legs) {
      if (leg.week !== week || leg.season !== season || leg.stat !== stat) continue
      const line = findPlayerLine(oddsData.player_props, leg.player_name)
      if (line != null && line !== leg.book_line) {
        setBookLine(leg.id, line)
      }
    }
  }, [oddsData?.player_props, season, setBookLine, slipState.legs, stat, week])

  function togglePosition(pos: string) {
    setPositions((prev) => prev.includes(pos) ? prev.filter((p) => p !== pos) : [...prev, pos])
  }

  const sorted = data
    ? [...data.projections].sort((a, b) => {
        const aVal = a[sortKey]; const bVal = b[sortKey]
        if (aVal == null && bVal == null) return 0
        if (aVal == null) return 1; if (bVal == null) return -1
        return sortDir === 'asc' ? Number(aVal) - Number(bVal) : Number(bVal) - Number(aVal)
      })
    : []

  const handleRowClick = useCallback((row: WeekPlayerProjection) => {
    navigate(`/player/${row.player_id}?week=${week}&season=${season}&stat=${stat}&name=${encodeURIComponent(row.player_name)}`)
  }, [navigate, week, season, stat])

  // Count players beating the line
  const edgedCount = oddsData
    ? sorted.filter((row) => {
        const line = findPlayerLine(oddsData.player_props, row.player_name)
        return line != null && calcEdge(row.projection, line) > 0
      }).length
    : 0

  return (
    <div className="flex flex-col gap-6">

      {/* Page Header */}
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-3 flex-wrap">
          <h1
            className="font-display text-5xl tracking-widest"
            style={{
              background: 'linear-gradient(135deg, #00C2FF 0%, #00FFA3 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            PROJECTIONS
          </h1>
          <span className="px-3 py-1 rounded-full text-xs font-bold tracking-widest"
            style={{ background: 'rgba(0,194,255,0.15)', border: '1px solid rgba(0,194,255,0.3)', color: '#00C2FF' }}>
            W{week} · {season}
          </span>
          {edgedCount > 0 && (
            <span className="px-3 py-1 rounded-full text-xs font-bold tracking-widest"
              style={{ background: 'rgba(0,255,163,0.12)', border: '1px solid rgba(0,255,163,0.3)', color: '#00FFA3' }}>
              ⚡ {edgedCount} beating the line
            </span>
          )}
          {oddsData && (
            <span className="px-3 py-1 rounded-full text-xs font-bold tracking-widest"
              style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: '#607B9B' }}>
              {oddsData.source === 'live' ? oddsData.provider : 'Mock demo lines'}
            </span>
          )}
        </div>
        <p className="text-sm text-oracle-muted">AI-powered projections vs live sportsbook lines · Click <span className="text-oracle-blue font-bold">+</span> to add to Bet Slip</p>
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

        {/* Stat */}
        <select value={stat} onChange={(e) => setStat(e.target.value)} className="oracle-select">
          {STATS.map((s) => <option key={s} value={s}>{STAT_LABELS[s]}</option>)}
        </select>

        {/* Week */}
        <select value={week} onChange={(e) => setWeek(Number(e.target.value))} className="oracle-select">
          {Array.from({ length: 18 }, (_, i) => i + 1).map((w) => (
            <option key={w} value={w}>Week {w}</option>
          ))}
        </select>

        {/* Season */}
        <select value={season} onChange={(e) => setSeason(Number(e.target.value))} className="oracle-select">
          {Array.from({ length: 7 }, (_, i) => 2019 + i).map((y) => (
            <option key={y} value={y}>{y}</option>
          ))}
        </select>

        {/* Sort */}
        <div className="flex items-center gap-1.5 ml-auto flex-wrap">
          <span className="text-[10px] font-semibold text-oracle-muted uppercase tracking-wider hidden sm:inline">Sort</span>
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

      {/* Odds disclaimer */}
      {oddsData && (
        <div className="flex items-center gap-2 text-[11px]"
          style={{ color: oddsData.source === 'live' ? '#A855F7' : '#607B9B' }}>
          <span>
            {oddsData.source === 'live' ? '📡 Live' : '🧪 Demo'} lines via {oddsData.provider}
          </span>
          {oddsData.source === 'mock' && (
            <span className="text-oracle-muted">· set `ODDS_API_KEY` on the server for sportsbook lines</span>
          )}
          {oddsData.source === 'live' && (
            <span className="text-oracle-muted">· updated {new Date(oddsData.last_updated).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</span>
          )}
        </div>
      )}

      {/* Staleness */}
      {data && <DataStalenessWarning dataFreshness={data.data_freshness} />}

      {/* Loading */}
      {isLoading && <SkeletonCard count={8} />}

      {/* Error */}
      {isError && error && <ErrorBanner error={error} onRetry={() => void refetch()} />}

      {/* Player Grid */}
      {data && !isLoading && (
        <>
          <p className="text-xs text-oracle-muted">{data.count} players · {STAT_LABELS[stat] ?? stat}</p>

          {sorted.length === 0 ? (
            <div className="rounded-xl p-12 text-center" style={{ background: 'rgba(13,27,48,0.6)', border: '1px solid rgba(255,255,255,0.05)' }}>
              <p className="text-oracle-muted text-sm">No projections found for the selected filters.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {sorted.map((row, idx) => {
                const bookLine = oddsData
                  ? findPlayerLine(oddsData.player_props, row.player_name)
                  : null

                return (
                  <PlayerCard
                    key={row.player_id}
                    row={row}
                    onClick={() => handleRowClick(row)}
                    bookLine={bookLine}
                    week={week}
                    season={season}
                    stat={stat}
                    style={{ animationDelay: `${Math.min(idx * 0.04, 0.6)}s`, opacity: 0 }}
                  />
                )
              })}
            </div>
          )}
        </>
      )}
    </div>
  )
}
