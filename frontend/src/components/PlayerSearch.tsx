import { useId, useRef, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import clsx from 'clsx'
import { ChevronRight, Search, X } from 'lucide-react'
import { useSearch } from '@/context/SearchContext'
import { weekFigureLabel } from '@/lib/forecast-provenance'
import { PositionBadge } from '@/components/PositionBadge'
import { positionTone, toneClass } from '@/lib/tokens'
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'

/**
 * Player search (Cmd+K or the Search button): a dialog holding a combobox and a
 * list of matching players.
 *
 * The dialog (focus trap, Escape, click outside, scroll lock, focus return) comes
 * from the shared Dialog. The input and list follow the ARIA combobox pattern:
 * focus stays in the input, arrow keys move the highlighted option, Enter opens it.
 */
export function PlayerSearch() {
  const { isOpen, close, query, setQuery, results, defaultWeek, defaultSeason, defaultStat } = useSearch()
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const listId = useId()
  const [activeIndex, setActiveIndex] = useState(0)

  useEffect(() => {
    if (isOpen) setActiveIndex(0)
  }, [isOpen, query, results.length])

  const helperText = useMemo(() => {
    if (query.length === 0) return 'Start typing a first name, last name, team, or position.'
    if (results.length === 0) return `No players found for "${query}".`
    return `${results.length} match${results.length !== 1 ? 'es' : ''} for "${query}".`
  }, [query, results.length])

  const activePlayer = results[activeIndex] ?? null
  const optionId = (index: number) => `${listId}-option-${index}`

  function handleSelect(player_id: string, name: string, stat = defaultStat) {
    navigate(`/player/${player_id}?week=${defaultWeek}&season=${defaultSeason}&stat=${stat}&name=${encodeURIComponent(name)}`)
    close()
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (results.length === 0) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIndex((prev) => (prev + 1) % results.length)
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIndex((prev) => (prev - 1 + results.length) % results.length)
    } else if (e.key === 'Enter' && activePlayer) {
      e.preventDefault()
      handleSelect(activePlayer.player_id, activePlayer.player_name, activePlayer.stat || defaultStat)
    }
  }

  // Keep the highlighted option in view while arrowing through a long list.
  useEffect(() => {
    if (isOpen) document.getElementById(optionId(activeIndex))?.scrollIntoView?.({ block: 'nearest' })
  }, [activeIndex, isOpen])

  return (
    <Dialog open={isOpen} onOpenChange={(open) => { if (!open) close() }}>
      <DialogContent
        showCloseButton={false}
        initialFocus={inputRef}
        className="top-20 max-w-2xl translate-y-0 gap-0 overflow-hidden rounded-2xl border-info/30 p-0 shadow-[0_0_0_1px_rgb(var(--info)/0.1),0_24px_80px_rgb(var(--shadow)/0.8),0_0_60px_rgb(var(--info)/0.08)]"
      >
        <DialogTitle className="sr-only">Search players</DialogTitle>
        <DialogDescription className="sr-only">
          Type a name, team or position. Use the up and down arrow keys to choose a player, then press Enter to open them.
        </DialogDescription>

        {/* Search input */}
        <div className="flex items-center gap-3 border-b border-white/[0.06] px-5 py-4">
          <Search className="size-5 flex-shrink-0 text-info" aria-hidden="true" />
          <input
            ref={inputRef}
            id="player-search-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search players or teams…"
            role="combobox"
            aria-expanded={results.length > 0}
            aria-controls={listId}
            aria-activedescendant={results.length > 0 ? optionId(activeIndex) : undefined}
            aria-autocomplete="list"
            aria-label="Search players, positions, teams"
            autoComplete="off"
            className="flex-1 bg-transparent text-base text-fg outline-none placeholder:text-fg-subtle"
          />
          <DialogClose
            aria-label="Close search"
            className="flex items-center rounded-md p-1 text-fg-muted transition-colors hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            <kbd className="hidden rounded border border-white/10 bg-white/[0.08] px-2 py-0.5 font-mono text-[10px] sm:inline">ESC</kbd>
            <X className="size-5 sm:hidden" aria-hidden="true" />
          </DialogClose>
        </div>

        {/* Results */}
        <div className="max-h-[min(20rem,60dvh)] overflow-y-auto">
          <div className="border-b border-white/[0.04] px-5 py-2 text-[11px] text-fg-muted" aria-live="polite">
            {helperText}
          </div>
          {results.length === 0 ? (
            <div className="px-5 py-8 text-center text-sm text-fg-muted">
              {query.length > 0
                ? 'Try a shorter prefix or search by team / position.'
                : 'Start typing to search players…'}
            </div>
          ) : (
            <div className="py-2">
              {query.length === 0 && (
                <p className="px-5 py-2 text-[10px] font-semibold uppercase tracking-widest text-fg-muted">
                  All Players
                </p>
              )}
              <div id={listId} role="listbox" aria-label="Matching players">
                {results.map((player, idx) => {
                  const isActive = idx === activeIndex
                  return (
                    <div
                      key={`${player.player_id}-${idx}`}
                      id={optionId(idx)}
                      role="option"
                      aria-selected={isActive}
                      onClick={() => handleSelect(player.player_id, player.player_name, player.stat)}
                      onMouseEnter={() => setActiveIndex(idx)}
                      className={clsx(
                        toneClass(positionTone(player.position)),
                        'group flex w-full cursor-pointer items-center gap-4 border-b border-white/[0.03] px-5 py-3 text-left transition-colors',
                        isActive && 'bg-info/[0.08]',
                      )}
                    >
                      <PositionBadge position={player.position} className="flex-shrink-0" />

                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-semibold text-fg">{player.player_name}</p>
                        <p className="text-[11px] text-fg-muted">{player.team ?? 'UNK'}</p>
                      </div>

                      <div className="flex-shrink-0 text-right">
                        {player.projection != null ? (
                          <>
                            <p className="glow-text font-display text-xl text-tone">{player.projection.toFixed(1)}</p>
                            <p className="text-[10px] text-fg-muted">{weekFigureLabel(defaultWeek, player.stat)}</p>
                          </>
                        ) : (
                          <p className="max-w-[9rem] text-[11px] leading-snug text-fg-muted">No Week {defaultWeek} forecast yet</p>
                        )}
                      </div>

                      <ChevronRight
                        className={clsx('size-4 flex-shrink-0 transition-opacity', isActive ? 'opacity-100' : 'opacity-0 group-hover:opacity-100')}
                        aria-hidden="true"
                      />
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>

        {/* Footer hint */}
        <div className="flex items-center justify-between border-t border-white/[0.04] px-5 py-2.5">
          <div className="flex items-center gap-3 text-[10px] text-fg-muted">
            <span>↵ open highlighted player</span>
            <span className="hidden sm:inline">↑↓ move selection</span>
            <span className="hidden sm:inline">ESC to close</span>
          </div>
          <div className="hidden items-center gap-1 sm:flex">
            <kbd className="rounded border border-white/[0.08] bg-white/[0.06] px-1.5 py-0.5 font-mono text-[10px] text-fg-muted">⌘K</kbd>
            <span className="text-[10px] text-fg-muted">to toggle</span>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
