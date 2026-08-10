/**
 * Typed fetch wrapper for the Gridiron Oracle FastAPI backend.
 * Base URL is /api — Vite dev server proxies this to http://localhost:8000.
 */

const BASE_URL = '/api'

export class ApiError extends Error {
  status: number
  detail: string

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
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
    try {
      const errorJson = (await response.json()) as { detail?: string }
      if (typeof errorJson.detail === 'string') {
        detail = errorJson.detail
      }
    } catch {
      // ignore parse error — use fallback message
    }
    throw new ApiError(response.status, detail)
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
