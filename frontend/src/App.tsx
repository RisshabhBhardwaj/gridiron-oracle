import { lazy, Suspense } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { NavBar } from './components/layout/NavBar'
import { PageShell } from './components/layout/PageShell'
import { PlayerSearch } from './components/PlayerSearch'
import { SearchProvider } from './context/SearchContext'

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

export function App() {
  return (
    <SearchProvider>
      <div className="min-h-screen bg-oracle-dark text-oracle-white">
        <NavBar />
        <PlayerSearch />
        <PageShell>
          <Suspense fallback={<div className="p-8 text-oracle-muted">Loading view…</div>}>
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
