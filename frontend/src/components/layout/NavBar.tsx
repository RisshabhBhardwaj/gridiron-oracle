import { useState } from 'react'
import { Link, NavLink } from 'react-router-dom'
import clsx from 'clsx'
import { Menu, Search } from 'lucide-react'
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
import { useSearch } from '@/context/SearchContext'

const NAV_LINKS = [
  { to: '/projections', label: 'Current Season', exact: false },
  { to: '/season', label: 'Season Review', exact: false },
  { to: '/draft', label: 'Draft', exact: false },
  { to: '/backtest', label: 'Backtesting', exact: false },
  { to: '/settings', label: 'Settings', exact: false },
]

export function NavBar() {
  const [menuOpen, setMenuOpen] = useState(false)
  const { open: openSearch } = useSearch()

  return (
    <nav className="sticky top-0 z-50 border-b border-line bg-surface/95 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-[52px] gap-4">

          {/* Logo */}
          <Link to="/" className="flex flex-shrink-0 items-baseline gap-1.5 group">
            <span className="font-sans text-sm font-bold uppercase tracking-[0.16em] text-fg-muted transition-colors group-hover:text-fg">
              Gridiron
            </span>
            <span className="font-sans text-sm font-bold uppercase tracking-[0.16em] text-accent">
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
                  ? 'border-accent bg-surface-overlay text-fg'
                  : 'border-transparent text-fg-muted hover:bg-surface-raised hover:text-fg',
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
                    ? 'border-accent bg-surface-overlay text-fg'
                    : 'border-transparent text-fg-muted hover:bg-surface-raised hover:text-fg',
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
              className="hidden sm:flex items-center gap-2 rounded-md border border-line bg-surface-raised px-3 py-1.5 text-xs font-medium text-fg-muted transition-colors hover:border-accent/40 hover:text-accent"
              title="Search players (Cmd+K)"
              aria-label="Search players"
            >
              <Search className="size-3.5" aria-hidden="true" />
              <span>Search</span>
              <kbd className="rounded border border-line bg-surface px-1.5 py-0.5 font-mono text-[9px]">
                ⌘K
              </kbd>
            </button>

            <div className="hidden lg:flex items-center gap-2 px-2 py-1">
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping-slow rounded-full bg-accent opacity-75" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-accent" />
              </span>
              <span className="font-mono text-[10px] font-medium uppercase tracking-wider text-fg-muted">Live</span>
            </div>

            {/* Mobile hamburger: opens the menu dialog (focus trap, Escape) */}
            <button
              className="md:hidden text-fg-muted hover:text-fg p-2 rounded-md border border-line transition-colors"
              onClick={() => setMenuOpen(true)}
              aria-label="Open navigation menu"
              aria-haspopup="dialog"
            >
              <Menu className="size-5" aria-hidden="true" />
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu dialog */}
      <Dialog open={menuOpen} onOpenChange={setMenuOpen}>
        <DialogContent className="max-w-xs">
          <DialogTitle>Menu</DialogTitle>
          <nav className="flex flex-col gap-1" aria-label="Mobile">
            <button
              onClick={() => { openSearch(); setMenuOpen(false) }}
              className="flex items-center gap-3 px-4 py-2.5 rounded-md text-sm font-medium text-fg-muted hover:text-fg hover:bg-surface-raised"
            >
              <Search className="size-4" aria-hidden="true" />
              Search Players
            </button>
            <NavLink
              to="/"
              end
              onClick={() => setMenuOpen(false)}
              className={({ isActive }) => clsx(
                'px-4 py-2.5 rounded-md text-sm font-medium transition-colors',
                isActive ? 'text-fg bg-surface-overlay' : 'text-fg-muted hover:text-fg hover:bg-surface-raised',
              )}
            >
              Home
            </NavLink>
            {NAV_LINKS.map(({ to, label, exact }) => (
              <NavLink key={to} to={to} end={exact}
                onClick={() => setMenuOpen(false)}
                className={({ isActive }) => clsx(
                  'px-4 py-2.5 rounded-md text-sm font-medium transition-colors',
                  isActive ? 'text-fg bg-surface-overlay' : 'text-fg-muted hover:text-fg hover:bg-surface-raised',
                )}
              >
                {label}
              </NavLink>
            ))}
          </nav>
        </DialogContent>
      </Dialog>
    </nav>
  )
}
