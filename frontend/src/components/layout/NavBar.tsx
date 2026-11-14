import { useState } from 'react'
import { Link, NavLink } from 'react-router-dom'
import clsx from 'clsx'
import { Menu, Search } from 'lucide-react'
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
import { useSearch } from '@/context/SearchContext'
import { PAGE_NAMES } from '@/lib/page-names'
import { ApiStatusBadge } from './ApiStatusBadge'

const NAV_LINKS = [
  { to: '/projections', label: PAGE_NAMES.projections, exact: false },
  { to: '/season', label: PAGE_NAMES.season, exact: false },
  { to: '/draft', label: PAGE_NAMES.draft, exact: false },
  { to: '/backtest', label: PAGE_NAMES.backtest, exact: false },
  { to: '/settings', label: PAGE_NAMES.settings, exact: false },
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

          <div className="hidden lg:flex items-center gap-1 flex-1">
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

            <ApiStatusBadge />

            {/* Mobile hamburger: opens the menu dialog (focus trap, Escape) */}
            <button
              className="lg:hidden text-fg-muted hover:text-fg p-2 rounded-md border border-line transition-colors"
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
