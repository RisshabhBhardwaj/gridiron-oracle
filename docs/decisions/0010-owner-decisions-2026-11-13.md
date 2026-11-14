# 0010 Owner decisions of 2026-11-13: Claude takeover, data pull and rerun, UI review

Date: 2026-11-13 (recorded 2026-11-14). Status: accepted. These came from the owner, relayed by the risshabh.com coordinator.

## 1. Claude replaces Muse

The owner cancelled Muse. A Claude manager now runs Gridiron and owns the Box B host layer (`/srv/HOST.md`, systemd, packages, firewall) for Gridiron, ATS and statuta. Workers and leads are Claude only. Handover: `_internal-docs/gridiron/handoffs/muse-to-claude-2026-11-13.md`.

## 2. Pull fresh data and rerun what is needed

Owner's words: "we do need to pull in new data and we can re run the shit we need to for gridiron".

- Run the ETL by hand now instead of waiting for the Tuesday 13:00 UTC timer.
- Rerun whatever clears the degraded health: `alembic upgrade head` on a scratch DB first, the walk-forward causal rerun, the baseline re-freeze, per-cell causal evals and `scripts/summarize_causal_evals.py`. Store per-row bucket attributions during the rerun.
- Pin the `TRAIN_*_CMD` recipes actually used.
- Run it on Box B inside the Gridiron envelope (training slice 10 GB, 5 CPUs). Check `/srv/HOST.md` and free memory first so ATS and statuta are not starved, and stagger if needed.
- Back up the database before any write.
- Report honest numbers. The old 74/75 count is not reused unless the rerun reproduces it.
- The healthchecks.io account is still the owner's to create.

## 3. UI review and polish to a higher bar

The owner doubts the Muse-built UI meets the bar they want. The bar is risshabh.com: polished, convention-first, Liquid-Glass-level craft, nothing that looks unfinished. Every Muse-built screen counts as unproven.

1. Screenshot every screen at desktop and phone widths, in light and dark where they apply, plus the empty, loading, error and first-run states. Review the live site (gridiron.risshabh.com) end to end.
2. Write a hard, ranked critique (ux-reviewer style) to a handoff file.
3. Fix from the top with a Claude lead and Claude workers (frontend-implementer, then ux-reviewer on the result). Redo anything below the bar instead of patching it.
4. Run the gate, before and after sheets, and Lighthouse and axe.

Order: start after the data pull and rerun are underway, so screens are judged on fresh, healthy data. Run one heavy worker at a time, because the Mac is under load.

## Consequences

- The push of `main` is owner-authorised (handover). Secret-scan with gitleaks before every push; the repo is public.
- Decision 0008's note that a push deploys production is superseded; see the update note there.
