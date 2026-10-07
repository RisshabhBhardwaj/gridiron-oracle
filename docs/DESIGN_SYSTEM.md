Type: Decision
Status: Proposed (no migration done)
Last verified against code: 2026-10-07 (frontend read; library status from a secondary web source, confirm before adopting)

# Frontend design system

## What exists
- React 18, Vite, Tailwind 3, `clsx`, Recharts, TanStack Query. No component library and no primitive library (no Radix, React Aria, Headless UI) in `frontend/package.json`.
- Tokens: Tailwind `oracle.*` colors, fonts (Space Grotesk, Bebas Neue, JetBrains Mono), neon shadows in `tailwind.config.ts`.
- Two palettes disagree: `src/index.css` CSS variables (`#00C2FF`, `#00FFA3`, ...) versus the Tailwind tokens (`#59A6FF`, `#42D889`, ...). Neon shadows and gradients use the old values.
- Hard-coded inline hex/rgba in components (e.g. `NoForecastBanner.tsx`, `Settings.tsx` colors) instead of tokens.
- Emoji icons in `Settings.tsx`.

## Hand-rolled and should be replaced
Command-style player search (`PlayerSearch.tsx`, cmd+K), sliders (`.oracle-slider`), selects (`.oracle-select`), toggle (WhatIfStudio), tabs/pills, panels (`GlassPanel`), banners, tables, loading and error states, page shell and nav.

## Recommendation
1. **Primitives and components: shadcn/ui** (copied into the repo, so we own the code), on accessible primitives. A web search on 2026-10-07 reports that shadcn now defaults to Base UI with Radix still supported, and that Base UI is stable. Verify this against shadcn docs, then pick Base UI or Radix (Radix has slower maintenance per the same source).
2. **Tokens:** one source of truth as CSS variables (Tailwind 4 `@theme`), semantic names (surface, text, positive, negative, accent), no raw hex in components.
3. **Charts:** keep Recharts, wrap in one themed chart component using the tokens.
4. **Motion:** CSS transitions for hover/focus; add the `motion` library only if layout animations are needed. Respect `prefers-reduced-motion`.
5. **Data tables:** TanStack Table for the draft board and backtest tables.
6. **Icons:** one icon set (e.g. lucide) replacing emoji.

Why: accessible keyboard and screen-reader behavior for combobox, dialog, slider and select is hard to hand-roll correctly and is the largest risk in the current code; owning copied components avoids lock-in; tokens remove the palette split.

## Migration order
1. Unify tokens (palette, remove duplicate CSS variables). 2. Add primitives for search/combobox, select, slider, toggle. 3. Replace panels, banners, loading/error with shared components. 4. Tables. 5. Chart wrapper. 6. Upgrade Tailwind 3 to 4 once the above is stable. Keep the 34 existing tests green at each step.
