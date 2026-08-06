import { useMemo, useState } from 'react'
import { useDraftBoard, type DraftBoardPlayer } from '@/hooks/useDraftBoard'
import { useCurrentSeason } from '@/hooks/useCurrentSeason'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { SkeletonCard } from '@/components/shared/LoadingSpinner'
import { CURRENT_SEASON, POSITIONS } from '@/lib/constants'

type SortKey = 'adp' | 'model_rank' | 'value_vs_adp' | 'model_fantasy_ppr'

const POS_CLASS: Record<string, string> = {
  WR: 'pos-wr', RB: 'pos-rb', TE: 'pos-te', QB: 'pos-qb',
}

export function DraftBoard() {
  const { data: seasonMeta } = useCurrentSeason()
  const season = seasonMeta?.season ?? CURRENT_SEASON
  const [position, setPosition] = useState<string | null>(null)
  const [sortKey, setSortKey] = useState<SortKey>('adp')
  const [source, setSource] = useState('historical')

  const { data, isLoading, error, refetch, isFetching } = useDraftBoard({
    season,
    source,
    position,
  })

  const rows = useMemo(() => {
    const list = [...(data?.players ?? [])]
    list.sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      if (av == null && bv == null) return 0
      if (av == null) return 1
      if (bv == null) return -1
      if (sortKey === 'value_vs_adp' || sortKey === 'model_fantasy_ppr') {
        return Number(bv) - Number(av)
      }
      return Number(av) - Number(bv)
    })
    return list
  }, [data, sortKey])

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-2">
        <p className="text-[11px] uppercase tracking-[0.18em] text-oracle-muted">Draft</p>
        <h1 className="font-display text-3xl text-oracle-white">Draft Board</h1>
        <p className="max-w-2xl text-sm text-oracle-muted">
          Full PPR ADP vs model season ranks. Positive value = drafted later than model rank
          (ADP undervalues). Spearman ρ measures rank agreement with ADP.
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-3">
        <label className="text-xs text-oracle-muted">
          Source
          <select
            className="ml-2 rounded-md border border-oracle-border bg-oracle-card px-2 py-1.5 text-sm text-oracle-white"
            value={source}
            onChange={(e) => setSource(e.target.value)}
          >
            <option value="historical">historical</option>
            <option value="sleeper">sleeper</option>
            <option value="fantasypros">fantasypros</option>
          </select>
        </label>

        <div className="flex gap-1">
          <button
            type="button"
            onClick={() => setPosition(null)}
            className={`rounded-md px-3 py-1.5 text-xs font-medium ${
              position == null
                ? 'bg-oracle-green/20 text-oracle-green'
                : 'text-oracle-muted hover:text-oracle-white'
            }`}
          >
            ALL
          </button>
          {POSITIONS.map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => setPosition(p)}
              className={`rounded-md px-3 py-1.5 text-xs font-medium ${
                position === p
                  ? 'bg-oracle-green/20 text-oracle-green'
                  : 'text-oracle-muted hover:text-oracle-white'
              }`}
            >
              {p}
            </button>
          ))}
        </div>

        <label className="text-xs text-oracle-muted">
          Sort
          <select
            className="ml-2 rounded-md border border-oracle-border bg-oracle-card px-2 py-1.5 text-sm text-oracle-white"
            value={sortKey}
            onChange={(e) => setSortKey(e.target.value as SortKey)}
          >
            <option value="adp">ADP</option>
            <option value="model_rank">Model rank</option>
            <option value="value_vs_adp">Value vs ADP</option>
            <option value="model_fantasy_ppr">Model PPR</option>
          </select>
        </label>

        <button
          type="button"
          onClick={() => refetch()}
          className="rounded-md border border-oracle-border px-3 py-1.5 text-xs text-oracle-muted hover:text-oracle-white"
        >
          {isFetching ? 'Refreshing…' : 'Refresh'}
        </button>

        {data?.spearman_rho != null && (
          <span className="ml-auto font-mono text-xs text-oracle-green">
            Spearman ρ = {data.spearman_rho.toFixed(3)} · n={data.count}
          </span>
        )}
      </div>

      {error && <ErrorBanner message={(error as Error).message} onRetry={() => refetch()} />}

      {isLoading && (
        <div className="grid gap-3">
          {Array.from({ length: 8 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      )}

      {!isLoading && rows.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-oracle-border">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-oracle-surface text-[11px] uppercase tracking-wider text-oracle-muted">
              <tr>
                <th className="px-3 py-2">ADP#</th>
                <th className="px-3 py-2">Player</th>
                <th className="px-3 py-2">Pos</th>
                <th className="px-3 py-2">Team</th>
                <th className="px-3 py-2">ADP</th>
                <th className="px-3 py-2">Model#</th>
                <th className="px-3 py-2">Model PPR</th>
                <th className="px-3 py-2">Value</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row: DraftBoardPlayer) => (
                <tr
                  key={`${row.player_name}-${row.adp_rank}`}
                  className="border-t border-oracle-border/60 hover:bg-oracle-card/40"
                >
                  <td className="px-3 py-2 font-mono text-oracle-muted">{row.adp_rank ?? '—'}</td>
                  <td className="px-3 py-2 font-medium text-oracle-white">{row.player_name}</td>
                  <td className="px-3 py-2">
                    <span className={POS_CLASS[row.position ?? ''] ?? 'text-oracle-muted'}>
                      {row.position ?? '—'}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-oracle-muted">{row.team ?? '—'}</td>
                  <td className="px-3 py-2 font-mono">{row.adp.toFixed(1)}</td>
                  <td className="px-3 py-2 font-mono">{row.model_rank ?? '—'}</td>
                  <td className="px-3 py-2 font-mono">
                    {row.model_fantasy_ppr != null ? row.model_fantasy_ppr.toFixed(1) : '—'}
                  </td>
                  <td
                    className={`px-3 py-2 font-mono ${
                      (row.value_vs_adp ?? 0) > 0
                        ? 'text-oracle-green'
                        : (row.value_vs_adp ?? 0) < 0
                          ? 'text-oracle-red'
                          : 'text-oracle-muted'
                    }`}
                  >
                    {row.value_vs_adp != null
                      ? `${row.value_vs_adp > 0 ? '+' : ''}${row.value_vs_adp.toFixed(0)}`
                      : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
