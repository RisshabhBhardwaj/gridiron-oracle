Type: Decision
Status: Accepted (owner approved 2026-10-10); migration in progress, see "Migration status"
Last verified against code: 2026-10-17 (`cd frontend && npm run typecheck && npm run test && npm run build`)

# Frontend design system

Decision record: [decisions/0004-shadcn-on-base-ui.md](decisions/0004-shadcn-on-base-ui.md). Principles: [UX_PRINCIPLES.md](UX_PRINCIPLES.md).

## What exists
- React 18, Vite, Tailwind 4 (via `@tailwindcss/vite`), Recharts, TanStack Query.
- **Tokens:** one set of colour variables in `frontend/src/index.css` (surface, surface-raised, surface-overlay, line, fg, fg-muted, fg-subtle, accent, positive, negative, warning, info, highlight, shadow), mapped to utilities in the same file's `@theme inline` block (`bg-surface-raised`, `text-fg-muted`, `border-line`, `text-positive`, `bg-accent/15`). No hex or `rgb(number)` in components (`grep -rnE "#[0-9A-Fa-f]{6}|rgba?\([0-9]" frontend/src` finds only tests). A **tone** (`.tone-info`, then `text-tone`, `border-tone/20`) lets one component take a tone prop; `lib/tokens.ts` has `Tone`, `positionTone` (one colour per position everywhere) and `token()` for SVG and chart props.
- **Primitives (shadcn/ui on Base UI, copied in):** `components/ui/slider.tsx`, `select.tsx`, `dialog.tsx`, `tabs.tsx`, `switch.tsx`, `radio-group.tsx` (row and card variants), `toggle-group.tsx` (single and multiple, optional tone); `lib/utils.ts` `cn`. Shared pieces: `components/shared/Panel.tsx`, `components/PositionBadge.tsx`, `ProjectionCard`, `ErrorBanner`, `NoForecastBanner`, `DataStalenessWarning`, `LoadingSpinner`.
- **Icons:** `lucide-react`. No inline SVG icons and no emoji remain in components.
- **Motion:** CSS transitions; `prefers-reduced-motion` is honoured globally in `index.css` and in `ProjectionCard`'s count-up.
- Fonts: Space Grotesk (text), Bebas Neue (display numbers), JetBrains Mono.

## Migration status (2026-10-17)
Done:
- Tokens unified (palette: the neutral dark set; the old neon values are gone).
- Tailwind 3 to 4 (`tailwindcss` 4.3.3, `@tailwindcss/vite`; `tailwind-merge` 3; `tailwind.config.ts` and `postcss.config.js` deleted; tokens live in `@theme inline`). Class names are unchanged, so rendered output is the same set of utilities.
- Settings: sliders use `Slider`; scoring choice is a `RadioGroup` (card variant); icons are lucide; view-only on the public site.
- Player page: stat chooser is a `Select` (was 15 pill buttons).
- Search (cmd+K): `Dialog` plus an ARIA combobox, loaded on first open so it stays out of the landing bundle (was a hand-built overlay with no focus trap that crashed on open).
- All native selects are now `Select` (`CurrentSeason`, `SeasonReview`, `BacktestExplorer`, `DraftBoard`); the `.oracle-select` CSS is gone. Tests use user-event, not `fireEvent.change`.
- Tab and mode switchers are now `Tabs` (`CurrentSeason` weekly/season/games, `DraftBoard` board/mock). Position filters are now `ToggleGroup` (multiple on season/backtest pages, single-choice with ALL on the draft board; per-position tones kept).
- `WhatIfStudio`: four `Slider`s and the dome `Switch`; icons are lucide. The `.oracle-slider` CSS is gone.
- Icons and emoji: `NavBar`, `ErrorBanner`, `NoForecastBanner`, `DataStalenessWarning`, `Home`, `PlayerDetail`, `WhatIfStudio`, `DraftBoard`, `GameDetail` use lucide; no emoji in components.

Still hand-rolled (follow-up):
1. Phone navigation menu in `components/layout/NavBar.tsx` and the inline search box in `pages/Home.tsx` (a second hand-built combobox). Both work and are tested; neither names a primitive, so they were left alone.
2. Tables (`DraftBoard`, `SeasonReview`, `BacktestExplorer`): TanStack Table. Chart wrapper: one themed Recharts component using `token()`.

## How to add a control
Check `components/ui/` first. If it is missing, copy it from shadcn (`npx shadcn@latest view <name>` with `style: "base-nova"` shows the source), read the Base UI doc for the part in `frontend/node_modules/@base-ui/react/docs/react/components/`, rewrite the class strings to Tailwind 4 and the project tokens, and add a test that uses the keyboard. Do not use shadcn's colour names; use the tokens above.
