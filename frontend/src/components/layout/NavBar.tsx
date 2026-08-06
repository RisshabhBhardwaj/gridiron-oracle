import { useState } from 'react'
import { Link, NavLink } from 'react-router-dom'
import clsx from 'clsx'
import { useSearch } from '@/context/SearchContext'
import { useBetSlip } from '@/context/BetSlipContext'

const NAV_LINKS = [
  { to: '/projections', label: 'All Projections', exact: false },
  { to: '/season', label: 'Season', exact: false },
  { to: '/draft', label: 'Draft', exact: false },
  { to: '/backtest', label: 'Backtesting', exact: false },
  { to: '/settings', label: 'Settings', exact: false },
]

export function NavBar() {
  const [menuOpen, setMenuOpen] = useState(false)
  const { open: openSearch } = useSearch()
  const { state: slipState, toggleOpen: toggleSlip } = useBetSlip()
  const legCount = slipState.legs.length

  return (
    <nav className="sticky top-0 z-50 border-b border-oracle-border bg-oracle-dark/95 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-[52px] gap-4">

          {/* Logo */}
          <Link to="/" className="flex flex-shrink-0 items-baseline gap-1.5 group">
            <span className="font-sans text-sm font-bold uppercase tracking-[0.16em] text-oracle-muted transition-colors group-hover:text-oracle-white">
              Gridiron
            </span>
            <span className="font-sans text-sm font-bold uppercase tracking-[0.16em] text-oracle-green">
              Oracle
            </span>
          </Link>

          <div className="hidden md:flex items-center gap-1 flex-1">
            <NavLink
              to="/"
              end
              className={({ isActive }) => clsx(
                'rounded-md border-b-2 px-3 py-1.5 text-sm font-medium transition-colors',
                isActive
                  ? 'border-oracle-green bg-oracle-surface text-oracle-white'
                  : 'border-transparent text-oracle-muted hover:bg-oracle-card hover:text-oracle-white',
              )}
            >
              Home
            </NavLink>
            {NAV_LINKS.map(({ to, label, exact }) => (
              <NavLink
                key={to}
                to={to}
                end={exact}
                className={({ isActive }) => clsx(
                  'rounded-md border-b-2 px-3 py-1.5 text-sm font-medium transition-colors',
                  isActive
                    ? 'border-oracle-green bg-oracle-surface text-oracle-white'
                    : 'border-transparent text-oracle-muted hover:bg-oracle-card hover:text-oracle-white',
                )}
              >
                {label}
              </NavLink>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={openSearch}
              id="nav-search-btn"
              className="hidden sm:flex items-center gap-2 rounded-md border border-oracle-border bg-oracle-card px-3 py-1.5 text-xs font-medium text-oracle-muted transition-colors hover:border-oracle-green/40 hover:text-oracle-green"
              title="Search players (Cmd+K)"
              aria-label="Search players"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <span>Search</span>
              <kbd className="rounded border border-oracle-border bg-oracle-dark px-1.5 py-0.5 font-mono text-[9px]">
                ⌘K
              </kbd>
            </button>

            <button
              onClick={toggleSlip}
              id="nav-bet-slip-btn"
              className={clsx(
                'relative flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-semibold transition-colors',
                slipState.isOpen
                  ? 'border-oracle-green/50 bg-oracle-green/10 text-oracle-green'
                  : 'border-oracle-border bg-oracle-card text-oracle-muted hover:border-oracle-green/40 hover:text-oracle-green',
              )}
              title="Bet Slip / Parlay Builder"
              aria-label="Toggle bet slip"
            >
              <span>Slip</span>
              {legCount > 0 && (
                <span className="flex h-4 min-w-4 items-center justify-center rounded-full bg-oracle-green px-1 font-mono text-[9px] font-bold text-oracle-dark">
                  {legCount}
                </span>
              )}
            </button>

            <div className="hidden lg:flex items-center gap-2 px-2 py-1">
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping-slow rounded-full bg-oracle-green opacity-75" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-oracle-green" />
              </span>
              <span className="font-mono text-[10px] font-medium uppercase tracking-wider text-oracle-muted">Live</span>
            </div>

            {/* Mobile hamburger */}
            <button
              className="md:hidden text-oracle-muted hover:text-oracle-white p-2 rounded-md border border-oracle-border transition-colors"
              onClick={() => setMenuOpen((prev) => !prev)}
              aria-label="Toggle navigation"
              aria-expanded={menuOpen}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {menuOpen
                  ? <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
                  : <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6h16M4 12h16M4 18h16" />}
              </svg>
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        {menuOpen && (
          <div className="md:hidden pb-4 border-t border-oracle-border pt-3 flex flex-col gap-1">
            {/* Mobile search */}
            <button
              onClick={() => { openSearch(); setMenuOpen(false) }}
              className="flex items-center gap-3 px-4 py-2.5 rounded-md text-sm font-medium text-oracle-muted hover:text-oracle-white hover:bg-oracle-card"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              Search Players
            </button>
            <NavLink
              to="/"
              end
              onClick={() => setMenuOpen(false)}
              className={({ isActive }) => clsx(
                'px-4 py-2.5 rounded-md text-sm font-medium transition-colors',
                isActive ? 'text-oracle-white bg-oracle-surface' : 'text-oracle-muted hover:text-oracle-white hover:bg-oracle-card',
              )}
            >
              Home
            </NavLink>
            {NAV_LINKS.map(({ to, label, exact }) => (
              <NavLink key={to} to={to} end={exact}
                onClick={() => setMenuOpen(false)}
                className={({ isActive }) => clsx(
                  'px-4 py-2.5 rounded-md text-sm font-medium transition-colors',
                  isActive ? 'text-oracle-white bg-oracle-surface' : 'text-oracle-muted hover:text-oracle-white hover:bg-oracle-card',
                )}
              >
                {label}
              </NavLink>
            ))}
          </div>
        )}
      </div>
    </nav>
  )
}
