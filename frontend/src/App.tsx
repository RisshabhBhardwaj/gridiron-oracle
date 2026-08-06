import { lazy, Suspense } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { NavBar } from './components/layout/NavBar'
import { PageShell } from './components/layout/PageShell'
import { PlayerSearch } from './components/PlayerSearch'
import { BetSlip } from './components/BetSlip'
import { BetSlipProvider } from './context/BetSlipContext'
import { SearchProvider } from './context/SearchContext'

// Route-level chunks keep the landing experience small; analytics-heavy pages
// (charts, backtest tables, settings) are fetched only when the user opens them.
const Home = lazy(async () => ({ default: (await import('./pages/Home')).Home }))
const Dashboard = lazy(async () => ({ default: (await import('./pages/Dashboard')).Dashboard }))
const SeasonProjections = lazy(async () => ({ default: (await import('./pages/SeasonProjections')).SeasonProjections }))
const PlayerDetail = lazy(async () => ({ default: (await import('./pages/PlayerDetail')).PlayerDetail }))
const BacktestExplorer = lazy(async () => ({ default: (await import('./pages/BacktestExplorer')).BacktestExplorer }))
const Settings = lazy(async () => ({ default: (await import('./pages/Settings')).Settings }))

export function App() {
  return (
    <BetSlipProvider>
      <SearchProvider>
        <div className="min-h-screen bg-oracle-dark text-oracle-white">
          <NavBar />
          <PlayerSearch />
          <BetSlip />
          <PageShell>
            <Suspense fallback={<div className="p-8 text-oracle-muted">Loading view…</div>}>
              <Routes>
                <Route path="/"                element={<Home />} />
                <Route path="/projections"     element={<Dashboard />} />
                <Route path="/season"          element={<SeasonProjections />} />
                <Route path="/player/:player_id" element={<PlayerDetail />} />
                <Route path="/backtest"         element={<BacktestExplorer />} />
                <Route path="/settings"         element={<Settings />} />
                <Route path="*"               element={<Navigate to="/" replace />} />
              </Routes>
            </Suspense>
          </PageShell>
        </div>
      </SearchProvider>
    </BetSlipProvider>
  )
}
