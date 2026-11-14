import { lazy, Suspense, useEffect, useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { NavBar } from './components/layout/NavBar'
import { PageShell } from './components/layout/PageShell'
import { SkipLink } from './components/layout/SkipLink'
import { SearchProvider, useSearch } from './context/SearchContext'
import { CardGridSkeleton, LoadingRegion, PageHeaderSkeleton } from './components/shared/Skeletons'

// Route-level chunks keep the landing experience small; analytics-heavy pages
// (charts, backtest tables, settings) are fetched only when the user opens them.
const Home = lazy(async () => ({ default: (await import('./pages/Home')).Home }))
const CurrentSeason = lazy(async () => ({ default: (await import('./pages/CurrentSeason')).CurrentSeason }))
const SeasonReview = lazy(async () => ({ default: (await import('./pages/SeasonReview')).SeasonReview }))
const GameDetail = lazy(async () => ({ default: (await import('./pages/GameDetail')).GameDetail }))
const DraftBoard = lazy(async () => ({ default: (await import('./pages/DraftBoard')).DraftBoard }))
const PlayerDetail = lazy(async () => ({ default: (await import('./pages/PlayerDetail')).PlayerDetail }))
const BacktestExplorer = lazy(async () => ({ default: (await import('./pages/BacktestExplorer')).BacktestExplorer }))
const Settings = lazy(async () => ({ default: (await import('./pages/Settings')).Settings }))
// The search dialog (and the dialog library behind it) is only needed once someone opens search.
const PlayerSearch = lazy(async () => ({ default: (await import('./components/PlayerSearch')).PlayerSearch }))

/** Shown while a route's code downloads: the shape every page starts with (header, then cards). */
function RouteFallback() {
  return (
    <LoadingRegion label="Loading page" className="flex flex-col gap-6">
      <PageHeaderSkeleton />
      <CardGridSkeleton count={4} />
    </LoadingRegion>
  )
}

/** Loads the search dialog the first time search is opened, then keeps it mounted. */
function SearchHost() {
  const { isOpen } = useSearch()
  const [wanted, setWanted] = useState(false)
  useEffect(() => {
    if (isOpen) setWanted(true)
  }, [isOpen])
  if (!wanted) return null
  return (
    <Suspense fallback={null}>
      <PlayerSearch />
    </Suspense>
  )
}

export function App() {
  return (
    <SearchProvider>
      <div className="min-h-screen bg-surface text-fg">
        <SkipLink />
        <NavBar />
        <SearchHost />
        <PageShell>
          <Suspense fallback={<RouteFallback />}>
            <Routes>
              <Route path="/"                element={<Home />} />
              <Route path="/projections"     element={<CurrentSeason />} />
              <Route path="/projections/game/:gameId" element={<GameDetail />} />
              <Route path="/season"          element={<SeasonReview />} />
              <Route path="/draft"           element={<DraftBoard />} />
              <Route path="/player/:player_id" element={<PlayerDetail />} />
              <Route path="/backtest"         element={<BacktestExplorer />} />
              <Route path="/settings"         element={<Settings />} />
              <Route path="*"               element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </PageShell>
      </div>
    </SearchProvider>
  )
}
