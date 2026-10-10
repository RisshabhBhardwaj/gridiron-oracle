/**
 * Tests for the public /api proxy policy (frontend/api/proxy-policy.js).
 *
 * Named .tsx because vite.config.ts collects only src/test/**\/*.test.tsx; a
 * .ts file here would be silently skipped.
 */
import { describe, expect, it } from 'vitest'
import { decide } from '../../api/proxy-policy.js'

const ALLOW = { allow: true }

describe('proxy policy: hidden routes', () => {
  it.each(['docs', 'redoc', 'openapi.json', 'integrity', 'metrics'])(
    '%s is 404 for every method',
    (name) => {
      for (const method of ['GET', 'HEAD', 'POST', 'PUT', 'DELETE']) {
        expect(decide(method, [name])).toMatchObject({ allow: false, status: 404 })
      }
    },
  )

  it('hides sub-paths under a hidden route', () => {
    expect(decide('GET', ['docs', 'oauth2-redirect'])).toMatchObject({ status: 404 })
  })

  it('does not let an encoded or query-suffixed name reach a hidden route', () => {
    expect(decide('GET', ['%64ocs'])).toMatchObject({ status: 404 })
    expect(decide('GET', ['openapi%2Ejson'])).toMatchObject({ status: 404 })
    expect(decide('GET', ['docs?x=1'])).toMatchObject({ status: 404 })
  })

  it('refuses malformed percent-encoding with 400', () => {
    expect(decide('GET', ['%E0%A4%A'])).toMatchObject({ allow: false, status: 400 })
  })
})

describe('proxy policy: settings is read-only', () => {
  it('allows GET and HEAD', () => {
    expect(decide('GET', ['settings'])).toEqual(ALLOW)
    expect(decide('HEAD', ['settings'])).toEqual(ALLOW)
  })

  it.each(['PUT', 'POST', 'PATCH', 'DELETE'])('refuses %s with 403', (method) => {
    expect(decide(method, ['settings'])).toEqual({
      allow: false,
      status: 403,
      detail: 'Settings are read-only on this site',
    })
  })

  it('refuses writes under an encoded settings name', () => {
    expect(decide('PUT', ['%73ettings'])).toMatchObject({ allow: false, status: 403 })
  })
})

describe('proxy policy: POST allow-list', () => {
  it('allows the POST paths the frontend calls', () => {
    expect(decide('POST', ['mock-draft', 'pick'])).toEqual(ALLOW)
    expect(decide('POST', ['scenario'])).toEqual(ALLOW)
  })

  it('refuses POST to any other path with 405', () => {
    expect(decide('POST', ['predict'])).toMatchObject({ allow: false, status: 405 })
    expect(decide('POST', ['alert-webhook'])).toMatchObject({ allow: false, status: 405 })
    expect(decide('POST', ['mock-draft'])).toMatchObject({ allow: false, status: 405 })
    expect(decide('POST', ['scenario', 'extra'])).toMatchObject({ allow: false, status: 405 })
  })
})

describe('proxy policy: other paths', () => {
  it('allows GET and HEAD', () => {
    expect(decide('GET', ['predict'])).toEqual(ALLOW)
    expect(decide('HEAD', ['season', 'current'])).toEqual(ALLOW)
    expect(decide('GET', [])).toEqual(ALLOW)
  })

  it.each(['PUT', 'PATCH', 'DELETE', 'OPTIONS'])('refuses %s with 405', (method) => {
    expect(decide(method, ['predict'])).toMatchObject({ allow: false, status: 405 })
  })
})
