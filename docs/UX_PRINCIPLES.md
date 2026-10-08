Type: Design
Status: Accepted
Last verified against code: not applicable (principles)

# UX principles

Canonical copy. Each repo carries the same file at `docs/UX_PRINCIPLES.md`. Source: owner's design notes, 2026-10-07.

## 1. Follow convention first (Jakob's Law)

People spend most of their time on other products and expect yours to work the same way. We are rarely first movers; we can be early followers. Established patterns and design systems have been tested on millions of users. Use them for the roughly 90% that is not what makes us different: navigation, layout, forms, tables, buttons, settings, onboarding, errors. Spend creativity on the remaining ~10%: brand, voice, color, illustration, motion, and the moments that make the product memorable. Do not reinvent the wheel; do not copy so closely that nothing is ours.

Uncertainty is the main killer of user experience. If a person does not know what to do next, the design has failed.

## 2. Know the user, then design from their side

For every product write down who uses it, what they care about, what they must see, what they want to see, and what they already use. Match the feel to them:

- Young learners: playful and fun is fine, but they know Duolingo, Khan Academy, and Clash Royale, so keep familiar patterns.
- Business owners, managers, founders: polished, professional, clean, easy to scan.
- People in difficult circumstances, for example incarcerated learners: simple and no-nonsense. Never make anyone feel stupid. Keep it uncomplicated.

Decide what the user needs to see, then the least needed to show it, in the best form: a line item, a table, a chart, an icon. Prefer tight, consistent spacing and type over many colors and loose layouts.

## 3. Always show the next step

Every screen has a clear hierarchy of actions and an obvious next step, including empty, loading, transition, error, success, and first-run states. Design each of these on purpose; none should feel like a leftover. Onboarding shows how to use the product and delivers an early "aha" moment. People should feel value quickly, because that is what keeps them.

## 4. Never leak the backend

No database ids, stack traces, internal names, raw errors, or system jargon in the interface. Errors say what happened in plain words and what to do.

## 5. Intentional polish

Everything feels custom and thought through, from error states to blank states: consistent spacing, type, icons, and motion. Plain words that anyone can read, including people reading in a second language.

## 6. The distracted-user test

Design for the least attentive, least expert user. If someone who is barely paying attention can still tell what is going on and what to do, an engaged expert will do even better.

## Checklist before shipping a screen

1. Who is the user, and what are they trying to do here?
2. Does it follow the convention that user already knows? Where do we deliberately differ, and why?
3. Is the next step obvious in every state (first run, empty, loading, error, success)?
4. Is anything shown that the user does not need? Anything internal leaked?
5. Would a distracted first-time user succeed?
6. Are we using the design system's primitives and components, not custom ones?
