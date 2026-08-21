/**
 * Typed fetch wrapper for the Gridiron Oracle FastAPI backend.
 * Base URL is /api — Vite dev server proxies this to http://localhost:8000.
 */

const BASE_URL = '/api'

/** Structured `detail` payload shape used by "expected empty state" responses,
 * e.g. /predict's 404 when no approved forecast exists for a player/week/season. */
export interface StructuredErrorDetail {
  forecast_available?: boolean
  message?: string
  [key: string]: unknown
}

export class ApiError extends Error {
  status: number
  detail: string
  /** Present when the backend returned a JSON object (not a string) for `detail`.
   * Callers that only read `.detail`/`.message` are unaffected. */
  structuredDetail?: StructuredErrorDetail

  constructor(status: number, detail: string, structuredDetail?: StructuredErrorDetail) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
    this.structuredDetail = structuredDetail
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
    try {
      const errorJson = (await response.json()) as { detail?: string | StructuredErrorDetail }
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
    throw new ApiError(response.status, detail, structuredDetail)
  }

  return response.json() as Promise<T>
}

export const apiClient = {
  get<T>(path: string): Promise<T> {
    return request<T>('GET', path)
  },
  post<T>(path: string, body: unknown): Promise<T> {
    return request<T>('POST', path, body)
  },
  put<T>(path: string, body: unknown): Promise<T> {
    return request<T>('PUT', path, body)
  },
}
