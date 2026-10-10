/**
 * Access policy for the public /api proxy (see proxy.js).
 *
 * The browser is untrusted, so the proxy forwards only what the site itself
 * needs. This module is pure (no I/O, no env) so the rules can be unit-tested.
 *
 * decide(method, segments) returns {allow: true}, or {allow: false, status,
 * detail}. Rules run in order; the first match decides.
 */

/** @typedef {{allow: true}} Allow */
/** @typedef {{allow: false, status: number, detail: string}} Deny */

const ALLOW = { allow: true }

/** @param {number} status @param {string} detail @returns {Deny} */
const deny = (status, detail) => ({ allow: false, status, detail })

// Backend routes that describe the API or the deployment. Never served publicly.
const HIDDEN = new Set(['docs', 'redoc', 'openapi.json', 'integrity', 'metrics'])

// Reads that the backend answers without changing state.
const READ_METHODS = new Set(['GET', 'HEAD'])

// POST paths the frontend calls, matched exactly. Derived from frontend/src:
//   hooks/useMockDraft.ts:19  apiClient.post('/mock-draft/pick')
//   hooks/useScenario.ts:13   apiClient.post('/scenario')
const POST_ALLOWLIST = new Set(['mock-draft/pick', 'scenario'])

/**
 * Return the path the backend will route on, with percent-encoding decoded and
 * any "?" or "#" inside a segment treated as URL syntax. Without this, a name
 * such as "%64ocs" or "docs?x" would not match "docs" here but would still reach
 * the hidden route upstream. Returns null for malformed percent-encoding.
 *
 * @param {string[]} segments
 * @returns {string[] | null}
 */
function backendSegments(segments) {
  let pathname
  try {
    pathname = decodeURIComponent(new URL(`http://policy.invalid/${segments.join('/')}`).pathname)
  } catch {
    return null
  }
  return pathname.split('/').filter(Boolean)
}

/**
 * @param {string} method
 * @param {string[]} segments  path segments after /api/, as the proxy receives them
 * @returns {Allow | Deny}
 */
export function decide(method, segments = []) {
  const verb = String(method).toUpperCase()
  const path = backendSegments(segments)
  if (path === null) return deny(400, 'Malformed path')

  const head = path[0] ?? ''
  const full = path.join('/')

  // 1. Hidden routes return 404 for every method, so they do not reveal themselves.
  if (HIDDEN.has(head)) return deny(404, 'Not found')

  // 2. Settings is read-only on the public site. The proxy no longer attaches the
  //    admin key, and when the backend has no ADMIN_API_KEY the API key alone
  //    would accept the write, so this edge check is the one that must hold.
  if (head === 'settings' && !READ_METHODS.has(verb)) {
    return deny(403, 'Settings are read-only on this site')
  }

  // 3. POST reaches only the exact paths the frontend calls.
  if (verb === 'POST') {
    return POST_ALLOWLIST.has(full) ? ALLOW : deny(405, 'Method not allowed')
  }

  // 4. Reads are open on every other path.
  if (READ_METHODS.has(verb)) return ALLOW

  // 5. Everything else (PUT, PATCH, DELETE, OPTIONS, ...) is refused.
  return deny(405, 'Method not allowed')
}
