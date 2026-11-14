/**
 * Availability note (decision 0012 D3).
 *
 * The forward producer may serve an optional `availability_status` on weekly rows.
 * Questionable and Doubtful are served; Out and other excluded statuses never are.
 * `Unknown` means a manual run without injury data. Only that one value shows
 * anything yet: a plain note, so nobody reads a missing status as "healthy".
 * Questionable and Doubtful badges are a later item.
 */
export const AVAILABILITY_UNKNOWN_NOTE = 'Injury status not checked for this week'

export function isAvailabilityUnknown(status: string | null | undefined): boolean {
  return status?.trim().toLowerCase() === 'unknown'
}
