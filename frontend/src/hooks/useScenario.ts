import { useRef } from 'react'
import { useMutation } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type { ScenarioRequest, ScenarioResponse } from '@/types/api'

const DEBOUNCE_MS = 300

export function useScenario() {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const mutation = useMutation<ScenarioResponse, Error, ScenarioRequest>({
    mutationFn: (req: ScenarioRequest) =>
      apiClient.post<ScenarioResponse>('/scenario', req),
  })

  function runScenario(req: ScenarioRequest): void {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current)
    }
    timerRef.current = setTimeout(() => {
      mutation.mutate(req)
    }, DEBOUNCE_MS)
  }

  return { runScenario, ...mutation }
}
