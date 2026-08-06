import { useRef, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSearch } from '@/context/SearchContext'
import { STAT_LABELS } from '@/lib/constants'
import clsx from 'clsx'

const POS_COLORS: Record<string, string> = {
  WR: '#00C2FF', RB: '#00FFA3', TE: '#A855F7', QB: '#FFB800',
}
const POS_CLASS: Record<string, string> = {
  WR: 'pos-wr', RB: 'pos-rb', TE: 'pos-te', QB: 'pos-qb',
}

export function PlayerSearch() {
  const { isOpen, close, query, setQuery, results, defaultWeek, defaultSeason, defaultStat } = useSearch()
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const [activeIndex, setActiveIndex] = useState(0)

  useEffect(() => {
    if (isOpen) setTimeout(() => inputRef.current?.focus(), 50)
  }, [isOpen])

  useEffect(() => {
    if (isOpen) setActiveIndex(0)
  }, [isOpen, query, results.length])

  if (!isOpen) return null

  const activePlayer = results[activeIndex] ?? null

  function handleSelect(player_id: string, name: string, stat = defaultStat) {
    navigate(`/player/${player_id}?week=${defaultWeek}&season=${defaultSeason}&stat=${stat}&name=${encodeURIComponent(name)}`)
    close()
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (results.length === 0) {
      if (e.key === 'Escape') close()
      return
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIndex((prev) => (prev + 1) % results.length)
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIndex((prev) => (prev - 1 + results.length) % results.length)
    } else if (e.key === 'Enter' && activePlayer) {
      e.preventDefault()
      handleSelect(activePlayer.player_id, activePlayer.player_name, activePlayer.stat || defaultStat)
    } else if (e.key === 'Escape') {
      e.preventDefault()
      close()
    }
  }

  const helperText = useMemo(() => {
    if (query.length === 0) return 'Start typing a first name, last name, team, or position.'
    if (results.length === 0) return `No players found for "${query}".`
    return `${results.length} match${results.length !== 1 ? 'es' : ''} for "${query}".`
  }, [query, results.length])

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-[200]"
        style={{ background: 'rgba(5,11,24,0.8)', backdropFilter: 'blur(8px)' }}
        onClick={close}
      />

      {/* Modal */}
      <div
        className="fixed top-20 left-1/2 -translate-x-1/2 z-[201] w-full max-w-2xl mx-auto px-4"
        style={{ animation: 'fade-in 0.15s ease-out' }}
      >
        <div
          className="rounded-2xl overflow-hidden"
          style={{
            background: 'rgba(8,15,32,0.98)',
            border: '1px solid rgba(0,194,255,0.3)',
            boxShadow: '0 0 0 1px rgba(0,194,255,0.1), 0 24px 80px rgba(0,0,0,0.8), 0 0 60px rgba(0,194,255,0.08)',
          }}
        >
          {/* Search input */}
          <div
            className="flex items-center gap-3 px-5 py-4"
            style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}
          >
            <svg className="w-5 h-5 flex-shrink-0" style={{ color: '#00C2FF' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search players, positions, teams…"
              className="flex-1 bg-transparent text-oracle-white placeholder-oracle-muted text-base outline-none"
              id="player-search-input"
            />
            <div className="flex items-center gap-1.5">
              <kbd
                className="px-2 py-0.5 text-[10px] font-mono rounded"
                style={{ background: 'rgba(255,255,255,0.08)', color: '#607B9B', border: '1px solid rgba(255,255,255,0.1)' }}
              >
                ESC
              </kbd>
            </div>
          </div>

          {/* Results */}
          <div className="max-h-80 overflow-y-auto">
            <div className="px-5 py-2 text-[11px] text-oracle-muted border-b"
              style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
              {helperText}
            </div>
            {results.length === 0 ? (
              <div className="px-5 py-8 text-center text-oracle-muted text-sm">
                {query.length > 0
                  ? 'Try a shorter prefix or search by team / position.'
                  : 'Start typing to search players…'}
              </div>
            ) : (
              <div className="py-2">
                {query.length === 0 && (
                  <p className="px-5 py-2 text-[10px] font-semibold text-oracle-muted uppercase tracking-widest">
                    All Players
                  </p>
                )}
                {results.map((player, idx) => {
                  const posColor = POS_COLORS[player.position] ?? '#607B9B'
                  const isActive = idx === activeIndex
                  return (
                    <button
                      key={`${player.player_id}-${idx}`}
                      onClick={() => handleSelect(player.player_id, player.player_name, player.stat)}
                      className="w-full flex items-center gap-4 px-5 py-3 text-left transition-colors group"
                      style={{
                        borderBottom: '1px solid rgba(255,255,255,0.03)',
                        background: isActive ? 'rgba(0,194,255,0.08)' : '',
                      }}
                      onMouseEnter={() => setActiveIndex(idx)}
                      >
                      {/* Position badge */}
                      <span className={clsx('flex-shrink-0', POS_CLASS[player.position] ?? 'pos-badge')}>
                        {player.position}
                      </span>

                      {/* Name + team */}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-oracle-white truncate">{player.player_name}</p>
                        <p className="text-[11px] text-oracle-muted">{player.team ?? 'UNK'}</p>
                      </div>

                      {/* Projection */}
                      <div className="text-right flex-shrink-0">
                        <p
                          className="font-display text-xl"
                          style={{ color: posColor, textShadow: `0 0 12px ${posColor}50` }}
                        >
                          {player.projection.toFixed(1)}
                        </p>
                        <p className="text-[10px] text-oracle-muted">
                          {STAT_LABELS[player.stat] ?? player.stat}
                        </p>
                      </div>

                      {/* Arrow */}
                      <svg
                        className={clsx('w-4 h-4 transition-opacity flex-shrink-0', isActive ? 'opacity-100' : 'opacity-0 group-hover:opacity-100')}
                        fill="none" stroke="currentColor" viewBox="0 0 24 24"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </button>
                  )
                })}
              </div>
            )}
          </div>

          {/* Footer hint */}
          <div
            className="px-5 py-2.5 flex items-center justify-between"
            style={{ borderTop: '1px solid rgba(255,255,255,0.04)' }}
          >
            <div className="flex items-center gap-3 text-[10px] text-oracle-muted">
              <span>↵ open highlighted player</span>
              <span>↑↓ move selection</span>
              <span>ESC to close</span>
            </div>
            <div className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 text-[10px] font-mono rounded" style={{ background: 'rgba(255,255,255,0.06)', color: '#607B9B', border: '1px solid rgba(255,255,255,0.08)' }}>
                ⌘K
              </kbd>
              <span className="text-[10px] text-oracle-muted">to toggle</span>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
