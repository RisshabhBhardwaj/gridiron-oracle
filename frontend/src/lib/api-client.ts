/**
 * Typed fetch wrapper for the Gridiron Oracle FastAPI backend.
 * Base URL is /api — Vite dev server proxies this to http://localhost:8000.
 */

const BASE_URL = '/api'

/** Structured `detail` payload shape used by "expected empty state" responses,
 * e.g. /predict's 404 when no approved forecast exists for a player/week/season. */
export interface StructuredErrorDetail {
  forecast_available?: boolean
  /** /explain: false when no breakdown is published for the stat (a 404, not an outage). */
  explanation_available?: boolean
  message?: string
  [key: string]: unknown
}

export class ApiError extends Error {
  status: number
  detail: string
  /** Present when the backend returned a JSON object (not a string) for `detail`.
   * Callers that only read `.detail`/`.message` are unaffected. */
  structuredDetail?: StructuredErrorDetail
  /** Machine-readable error code, when the server sent one. The /api proxy
   * sends "backend_unavailable" when the backend cannot be reached. */
  code?: string

  constructor(status: number, detail: string, structuredDetail?: StructuredErrorDetail, code?: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
    this.structuredDetail = structuredDetail
    this.code = code
  }
}

async function request<T>(
  method: 'GET' | 'POST' | 'PUT' | 'DELETE',
  path: string,
  body?: unknown,
): Promise<T> {
  const url = `${BASE_URL}${path}`
  const init: RequestInit = {
    method,
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
  }
  if (body !== undefined) {
    init.body = JSON.stringify(body)
  }

  const response = await fetch(url, init)

  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    let structuredDetail: StructuredErrorDetail | undefined
    let code: string | undefined
    try {
      const errorJson = (await response.json()) as { detail?: string | StructuredErrorDetail; code?: unknown }
      if (typeof errorJson.code === 'string') code = errorJson.code
      if (typeof errorJson.detail === 'string') {
        detail = errorJson.detail
      } else if (errorJson.detail && typeof errorJson.detail === 'object') {
        structuredDetail = errorJson.detail
        if (typeof structuredDetail.message === 'string') {
          detail = structuredDetail.message
        }
      }
    } catch {
      // ignore parse error — use fallback message
    }
    throw new ApiError(response.status, detail, structuredDetail, code)
  }

  return response.json() as Promise<T>
}

/**
 * True for the server's 404 "nothing here yet" answer (for example no approved
 * team-game predictions for a week). That is an expected empty state, not a
 * failure: show a designed no-data state with a next step, never the error
 * banner. 5xx, network errors and the proxy's backend_unavailable are not this.
 */
export function isNotReady(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404
}

export const apiClient = {
  get<T>(path: string): Promise<T> {
    return request<T>('GET', path)
  },
  /**
   * GET where a 404 means "not ready yet": resolves to null instead of throwing,
   * so the query neither retries nor lands in the error state. Any other failure
   * still throws.
   */
  async getOrNull<T>(path: string): Promise<T | null> {
    try {
      return await request<T>('GET', path)
    } catch (err) {
      if (isNotReady(err)) return null
      throw err
    }
  },
  post<T>(path: string, body: unknown): Promise<T> {
    return request<T>('POST', path, body)
  },
  put<T>(path: string, body: unknown): Promise<T> {
    return request<T>('PUT', path, body)
  },
}
