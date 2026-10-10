/**
 * Build-time switches. Vite reads them from the environment when the site is
 * built (frontend/.env.local for local work). Read them through these helpers,
 * at call time, so tests can stub them.
 */

/**
 * The public site refuses writes to /api/settings (frontend/api/proxy-policy.js
 * answers 403 "Settings are read-only on this site"), so the Settings page is
 * view-only by default. Set VITE_SETTINGS_EDITABLE=true for local development or
 * a private deployment whose API accepts the write.
 */
export function isSettingsEditable(): boolean {
  return import.meta.env.VITE_SETTINGS_EDITABLE === 'true'
}
