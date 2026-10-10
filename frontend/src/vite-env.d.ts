/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "true" lets the Settings page save changes. Unset on the public site. */
  readonly VITE_SETTINGS_EDITABLE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
