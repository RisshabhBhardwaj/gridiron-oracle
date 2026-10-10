# 0004: Component primitives are shadcn/ui on Base UI, adapted to Tailwind 3

Type: Decision
Status: Accepted
Last verified against code: 2026-10-10 (`cd frontend && npm run typecheck && npm run test && npm run build`)
Date: 2026-10-10 (owner approved the stack proposed in `docs/DESIGN_SYSTEM.md`)

## Context
The frontend had no component library. Sliders, selects, the cmd+K overlay and tab switchers were hand-rolled, with no focus trap, no keyboard model and no screen-reader roles. One of them (`PlayerSearch`) crashed the app when opened. The design doc proposed shadcn/ui on an accessible primitive library and asked that the choice between Base UI and Radix be checked first.

## Choice
- **Primitives: Base UI (`@base-ui/react`, MIT, 1.9.0).** Checked on 2026-10-10: the shadcn changelog (July 2026, "Base UI as the Default") says `shadcn init` now defaults to Base UI, Radix stays supported, both can coexist per component, and existing Radix projects do not have to migrate. The `radix-ui` and Base UI registries are both live. Base UI is the library shadcn now builds first and it is maintained by the Radix, Floating UI and MUI authors. Radix has slower maintenance (the design doc's source).
- **Components are copied into the repo** (`frontend/src/components/ui/`), so the code is ours: `slider`, `select`, `dialog`, `tabs`, plus `lib/utils.ts` (`cn`).
- **Not the shadcn CLI output as-is.** The current registry is written for Tailwind 4 (`data-open:`, `ring-3`, `outline-hidden`, `w-(--anchor-width)`, `animate-in`); on Tailwind 3.4 most of those classes produce no CSS and no error. Each class string was rewritten to Tailwind 3 forms (`data-[open]:`, `w-[var(--anchor-width)]`, `focus-visible:outline`) and mapped onto this project's colour tokens. Open and close motion uses Base UI's `data-[starting-style]` / `data-[ending-style]` with CSS transitions, and `motion-reduce:transition-none`, so there is no animation plugin.
- **`cn` is `clsx` + `tailwind-merge@^2`.** tailwind-merge 3 targets Tailwind 4. The new `cn` package that the shadcn CLI installs (published by shadcn) was not adopted: it adds a dependency for a one-line helper.
- **Icons: `lucide-react`.**
- **Colour: one set of semantic tokens** in `frontend/src/index.css` (surface, line, fg, accent, positive, negative, warning, info, highlight), mapped to Tailwind classes in `frontend/tailwind.config.ts`. shadcn's own names (`background`, `muted-foreground`, `accent` as a hover wash) are not used: copied components are edited to the project's names so there is one vocabulary.

## Consequences
- New controls use these components, or a new one is copied the same way (read the Base UI doc in `node_modules/@base-ui/react/docs/react/components/`, then adapt the shadcn class strings to Tailwind 3 and the tokens). Do not hand-roll sliders, selects, dialogs, menus, tooltips or tabs.
- When the project moves to Tailwind 4 (design doc step 6), the Tailwind 3 bracket syntax still works; move `tailwind-merge` to 3 and the tokens into `@theme`.
- jsdom has no PointerEvent and never measures layout. `src/test/setup.ts` adds a minimal PointerEvent, and Base UI sliders report `visibility: hidden` to jsdom, so role queries for a slider need `{ hidden: true }`.
- Remaining hand-rolled controls are listed in `docs/DESIGN_SYSTEM.md`.
