Type: Decision
Status: Accepted (owner approved 2026-10-10); migration in progress, see "Migration status"
Last verified against code: 2026-10-10 (`cd frontend && npm run typecheck && npm run test && npm run build`)

# Frontend design system

Decision record: [decisions/0004-shadcn-on-base-ui.md](decisions/0004-shadcn-on-base-ui.md). Principles: [UX_PRINCIPLES.md](UX_PRINCIPLES.md).

## What exists
- React 18, Vite, Tailwind 3, Recharts, TanStack Query.
- **Tokens:** one set of colour variables in `frontend/src/index.css` (surface, surface-raised, surface-overlay, line, fg, fg-muted, fg-subtle, accent, positive, negative, warning, info, highlight, shadow), mapped to classes in `frontend/tailwind.config.ts` (`bg-surface-raised`, `text-fg-muted`, `border-line`, `text-positive`, `bg-accent/15`). No hex or `rgb(number)` in components (`grep -rnE "#[0-9A-Fa-f]{6}|rgba?\([0-9]" frontend/src` finds only tests). A **tone** (`.tone-info`, then `text-tone`, `border-tone/20`) lets one component take a tone prop; `lib/tokens.ts` has `Tone`, `positionTone` (one colour per position everywhere) and `token()` for SVG and chart props.
- **Primitives (shadcn/ui on Base UI, copied in):** `components/ui/slider.tsx`, `select.tsx`, `dialog.tsx`, `tabs.tsx`; `lib/utils.ts` `cn`. Shared pieces: `components/shared/Panel.tsx`, `components/PositionBadge.tsx`, `ProjectionCard`, `ErrorBanner`, `NoForecastBanner`, `DataStalenessWarning`, `LoadingSpinner`.
- **Icons:** `lucide-react`.
- **Motion:** CSS transitions; `prefers-reduced-motion` is honoured globally in `index.css` and in `ProjectionCard`'s count-up.
- Fonts: Space Grotesk (text), Bebas Neue (display numbers), JetBrains Mono.

## Migration status (2026-10-10)
Done:
- Tokens unified (palette: the neutral dark set; the old neon values are gone).
- Settings: sliders use `Slider`; icons are lucide; view-only on the public site.
- Player page: stat chooser is a `Select` (was 15 pill buttons).
- Search (cmd+K): `Dialog` plus an ARIA combobox (was a hand-built overlay with no focus trap that crashed on open).

Still hand-rolled (follow-up, in this order of value):
1. Native selects styled with `.oracle-select`: `pages/CurrentSeason.tsx:241,259,277`, `pages/SeasonReview.tsx:88,115`, `pages/BacktestExplorer.tsx:81`, `pages/DraftBoard.tsx:223,336,378`. Replace with `Select`; the tests that use `fireEvent.change` on them need user-event.
2. Tab and mode switchers made of buttons: `pages/CurrentSeason.tsx:201-221` (Weekly, Season-Long, Games), `pages/DraftBoard.tsx:187,197` (board, mock). Replace with `Tabs`, which exists and is tested but has no page using it yet. Position filter toggles (`CurrentSeason.tsx:301`, `DraftBoard.tsx:354,366`, `BacktestExplorer.tsx:90`, `SeasonReview`) are toggle buttons with `aria-pressed`; a ToggleGroup is the right primitive.
3. `components/WhatIfStudio.tsx`: four sliders (`:239`, a transparent range input over a drawn thumb, no visible focus ring) and the dome switch (`:190`). Use `Slider` and a Switch.
4. Settings scoring choice (PPR / Half-PPR / Standard): card buttons with `aria-pressed`; a RadioGroup is the right primitive.
5. Phone navigation menu in `components/layout/NavBar.tsx` and the inline search box in `pages/Home.tsx` (a second hand-built combobox).
6. Inline SVG icons (`NavBar`, `ErrorBanner`, `NoForecastBanner`, `DataStalenessWarning`, `Home`, `PlayerDetail`, `WhatIfStudio`) and emoji in `WhatIfStudio`, `DraftBoard`, `GameDetail`: switch to lucide.
7. Tables (`DraftBoard`, `SeasonReview`, `BacktestExplorer`): TanStack Table. Chart wrapper: one themed Recharts component using `token()`.
8. Upgrade Tailwind 3 to 4 once the above is stable (move `tailwind-merge` to 3, tokens into `@theme`).

## How to add a control
Check `components/ui/` first. If it is missing, copy it from shadcn (`npx shadcn@latest view <name>` with `style: "base-nova"` shows the source), read the Base UI doc for the part in `frontend/node_modules/@base-ui/react/docs/react/components/`, rewrite the class strings to Tailwind 3 and the project tokens, and add a test that uses the keyboard. Do not use shadcn's colour names; use the tokens above.
