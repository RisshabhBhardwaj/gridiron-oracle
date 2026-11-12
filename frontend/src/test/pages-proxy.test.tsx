/**
 * Tests for the Cloudflare Pages /api proxy (frontend/functions/api/[[path]].js).
 *
 * Named .tsx because vite.config.ts collects only src/test/**\/*.test.tsx.
 * Runs under jsdom (the shared setup file needs window); Request, Response and
 * AbortSignal.timeout come from Node.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { onRequest } from '../../functions/api/[[path]].js'

const BACKEND = 'https://backend.example'
const KEY = 'test-key-not-real'

const fetchMock = vi.fn()

beforeEach(() => {
  fetchMock.mockReset()
  fetchMock.mockResolvedValue(new Response('{"ok":true}', { status: 200 }))
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => {
  vi.unstubAllGlobals()
})

type Env = { BACKEND_URL?: string; BACKEND_API_KEY?: string }

function call(path: string, init: RequestInit = {}, env: Env = { BACKEND_URL: BACKEND, BACKEND_API_KEY: KEY }) {
  const request = new Request(`https://site.example${path}`, init)
  return onRequest({ request, env })
}

const upstreamCall = () => {
  const [url, init] = fetchMock.mock.calls[0]
  return { url: url as string, init: init as RequestInit & { headers: Headers } }
}

describe('path and query mapping', () => {
  it('maps /api/x?q=1 to the backend', async () => {
    await call('/api/x?q=1')
    expect(upstreamCall().url).toBe(`${BACKEND}/x?q=1`)
  })

  it('keeps multi-segment paths and raw query encoding', async () => {
    await call('/api/draft/board/rb?a=b%20c&a=2')
    expect(upstreamCall().url).toBe(`${BACKEND}/draft/board/rb?a=b%20c&a=2`)
  })

  it('handles /api with no segments', async () => {
    await call('/api')
    expect(upstreamCall().url).toBe(`${BACKEND}/`)
    fetchMock.mockClear()
    await call('/api/')
    expect(upstreamCall().url).toBe(`${BACKEND}/`)
  })

  it('strips one trailing slash from BACKEND_URL', async () => {
    await call('/api/x', {}, { BACKEND_URL: `${BACKEND}/` })
    expect(upstreamCall().url).toBe(`${BACKEND}/x`)
  })

  it('uses the method of the incoming request', async () => {
    await call('/api/x', { method: 'HEAD' })
    expect(upstreamCall().init.method).toBe('HEAD')
  })
})

describe('request headers', () => {
  it('injects x-api-key from env', async () => {
    await call('/api/x')
    expect(upstreamCall().init.headers.get('x-api-key')).toBe(KEY)
  })

  it('replaces a client-sent x-api-key', async () => {
    await call('/api/x', { headers: { 'x-api-key': 'attacker' } })
    expect(upstreamCall().init.headers.get('x-api-key')).toBe(KEY)
  })

  it('drops a client-sent x-api-key when no key is configured', async () => {
    await call('/api/x', { headers: { 'x-api-key': 'attacker' } }, { BACKEND_URL: BACKEND })
    expect(upstreamCall().init.headers.has('x-api-key')).toBe(false)
  })

  it('strips hop-by-hop and Cloudflare headers, keeps the rest', async () => {
    await call('/api/x', {
      headers: {
        connection: 'keep-alive',
        'keep-alive': 'timeout=5',
        te: 'trailers',
        upgrade: 'websocket',
        'proxy-authorization': 'x',
        host: 'site.example',
        'cf-connecting-ip': '1.2.3.4',
        'cf-ray': 'abc',
        'cdn-loop': 'cloudflare',
        accept: 'application/json',
        'x-custom': 'keep',
      },
    })
    const h = upstreamCall().init.headers
    for (const name of ['connection', 'keep-alive', 'te', 'upgrade', 'proxy-authorization', 'host', 'cf-connecting-ip', 'cf-ray', 'cdn-loop']) {
      expect(h.has(name), name).toBe(false)
    }
    expect(h.get('accept')).toBe('application/json')
    expect(h.get('x-custom')).toBe('keep')
  })
})

describe('policy denials never reach the backend', () => {
  it.each([
    ['PUT', '/api/settings', 403],
    ['POST', '/api/settings', 403],
    ['GET', '/api/docs', 404],
    ['GET', '/api/%64ocs', 404],
    ['DELETE', '/api/players/1', 405],
    ['POST', '/api/players', 405],
    ['GET', '/api/%E0%A4%A', 400],
  ])('%s %s → %i', async (method, path, status) => {
    const res = await call(path, { method })
    expect(res.status).toBe(status)
    expect(await res.json()).toHaveProperty('detail')
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('allowed POST', () => {
  it.each(['scenario', 'mock-draft/pick'])('forwards the body for %s', async (path) => {
    const payload = JSON.stringify({ a: 1 })
    await call(`/api/${path}`, { method: 'POST', body: payload, headers: { 'content-type': 'application/json' } })
    const { url, init } = upstreamCall()
    expect(url).toBe(`${BACKEND}/${path}`)
    expect(init.method).toBe('POST')
    expect(new TextDecoder().decode(init.body as ArrayBuffer)).toBe(payload)
    expect(init.headers.get('content-type')).toBe('application/json')
  })

  it('sends {} for a POST with no body and defaults content-type', async () => {
    await call('/api/scenario', { method: 'POST' })
    const { init } = upstreamCall()
    expect(init.body).toBe('{}')
    expect(init.headers.get('content-type')).toBe('application/json')
  })

  it('does not send a body on GET', async () => {
    await call('/api/x')
    expect(upstreamCall().init.body).toBeUndefined()
  })
})

describe('errors', () => {
  it('returns 500 when BACKEND_URL is missing', async () => {
    const res = await call('/api/x', {}, { BACKEND_API_KEY: KEY })
    expect(res.status).toBe(500)
    expect(await res.json()).toEqual({ detail: 'BACKEND_URL is not configured' })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('returns 502 when fetch rejects, without leaking the key', async () => {
    fetchMock.mockRejectedValue(new TypeError('connect ECONNREFUSED'))
    const res = await call('/api/x')
    expect(res.status).toBe(502)
    const text = await res.text()
    expect(JSON.parse(text)).toEqual({ detail: 'Upstream unreachable: connect ECONNREFUSED' })
    expect(text).not.toContain(KEY)
  })

  it.each(['TimeoutError', 'AbortError'])('returns 504 on %s', async (name) => {
    fetchMock.mockRejectedValue(new DOMException('timed out', name))
    const res = await call('/api/x')
    expect(res.status).toBe(504)
    expect(await res.json()).toEqual({ detail: 'Upstream timed out after 60s' })
  })

  it('passes a timeout signal to fetch', async () => {
    await call('/api/x')
    expect(upstreamCall().init.signal).toBeInstanceOf(AbortSignal)
  })
})

describe('response', () => {
  it('passes status and body through', async () => {
    fetchMock.mockResolvedValue(new Response('{"detail":"nope"}', { status: 422 }))
    const res = await call('/api/x')
    expect(res.status).toBe(422)
    expect(await res.json()).toEqual({ detail: 'nope' })
  })

  it('drops content-encoding and hop-by-hop headers, keeps the rest', async () => {
    fetchMock.mockResolvedValue(
      new Response('ok', {
        status: 200,
        headers: {
          'content-encoding': 'gzip',
          'content-length': '2',
          'keep-alive': 'timeout=5',
          'content-type': 'text/plain',
          'x-request-id': 'r1',
        },
      }),
    )
    const res = await call('/api/x')
    expect(res.headers.has('content-encoding')).toBe(false)
    expect(res.headers.has('keep-alive')).toBe(false)
    expect(res.headers.get('content-type')).toBe('text/plain')
    expect(res.headers.get('x-request-id')).toBe('r1')
    expect(await res.text()).toBe('ok')
  })
})
