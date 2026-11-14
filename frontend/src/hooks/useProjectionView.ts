import { useCallback, useState } from 'react'
import { readStoredView, writeStoredView, type ListView } from '@/lib/projection-table'

/**
 * The player list view (table or cards), remembered for this viewer in
 * localStorage. When storage is blocked it still works for the visit.
 */
export function useProjectionView(): [ListView, (view: ListView) => void] {
  const [view, setView] = useState<ListView>(readStoredView)
  const choose = useCallback((next: ListView) => {
    setView(next)
    writeStoredView(next)
  }, [])
  return [view, choose]
}
