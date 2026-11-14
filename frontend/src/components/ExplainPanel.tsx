import { SHAPChart } from '@/components/SHAPChart'
import { ErrorBanner } from '@/components/shared/ErrorBanner'
import { Panel } from '@/components/shared/Panel'
import { useExplain } from '@/hooks/useExplain'
import { explainFailure } from '@/lib/player-page'

interface ExplainPanelProps {
  playerId: string
  week: number
  season: number
  stat: string
}

function ExplainSkeleton() {
  return (
    <div role="status" aria-label="Loading what drives this projection" className="flex flex-col gap-3 py-1">
      {['w-3/4', 'w-full', 'w-2/3', 'w-5/6', 'w-1/2'].map((width) => (
        <div key={width} className={`skeleton h-5 rounded ${width}`} />
      ))}
    </div>
  )
}

/**
 * "What drives this projection". Three outcomes that must stay apart:
 * - the server has no breakdown for this stat (404): a plain not-available line;
 * - the request failed (5xx, network): an error with Retry, never "not available";
 * - a breakdown arrived: the chart.
 */
export function ExplainPanel({ playerId, week, season, stat }: ExplainPanelProps) {
  const { data, isLoading, error, refetch } = useExplain({ player_id: playerId, week, season, stat })

  return (
    <Panel title="What drives this projection">
      {isLoading && <ExplainSkeleton />}

      {!isLoading && error && explainFailure(error) === 'not-published' && (
        <p className="py-6 text-center text-sm text-fg-muted">
          We do not have a breakdown for this stat yet. Try another stat above.
        </p>
      )}

      {!isLoading && error && explainFailure(error) === 'failed' && (
        <ErrorBanner error={error} onRetry={() => void refetch()} />
      )}

      {!isLoading && data && data.top_factors.length > 0 && (
        <div className="flex flex-col gap-3">
          <SHAPChart factors={data.top_factors} height={280} />
          <p className="text-xs text-fg-muted">
            {data.attribution_source === 'real_shap' ? 'Full breakdown' : 'Simplified explanation, not the full breakdown'}
          </p>
        </div>
      )}

      {!isLoading && data && data.top_factors.length === 0 && (
        <p className="py-6 text-center text-sm text-fg-muted">
          No explanation available for this player and week yet.
        </p>
      )}
    </Panel>
  )
}
