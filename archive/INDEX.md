# gridiron_oracle Archive Index
_Last organized: 2026-08-06_

Large run archives were moved out of the repository so the clone stays small.

| Former path | Current location | Size | Description |
|---|---|---|---|
| `archive/run_pre_2026/` | `/Users/risshabh/Projects/Active/_gridiron_archive/run_archives/run_pre_2026/` | ~23 GB | Pre-2026 ML runs (checkpoints, predictions, eval) |
| `archive/run_pre_20260313/` | `/Users/risshabh/Projects/Active/_gridiron_archive/run_archives/run_pre_20260313/` | ~2.1 GB | ML run archive from 2026-03-13 |

Also stored outside the tree (local recovery only — never push):

| Artifact | Location |
|---|---|
| Internal working docs (`remediation/`, adversarial audits) | `/Users/risshabh/Projects/Active/_gridiron_archive/internal-docs/` |
| Private git history bundle | `/Users/risshabh/Projects/Active/_gridiron_archive/gridiron-private-history.bundle` |
| Local `.claude/agents` backup | `/Users/risshabh/Projects/Active/_gridiron_archive/local-dotfiles/claude-agents/` |
| Local `.vscode/settings.json` backup | `/Users/risshabh/Projects/Active/_gridiron_archive/local-dotfiles/vscode-settings.json` |

## Still in-repo (small stubs)

| Folder | Description |
|--------|-------------|
| `scrapers/` | Old scraper scripts |
| `pipeline/` | Old pipeline configs |
| `backend/` | Old backend files |
| `frontend/` | Placeholder |

## Retired scripts (2026-10)

Moved out of `scripts/` on 2026-10-10. No Makefile target, CI job, or doc runs them.

| Folder | Description |
|--------|-------------|
| `archive/scripts-2026-10/` | One-time repairs and superseded tools. To restore one, `git mv` it back to `scripts/`. Their `ROOT` is `parents[1]`, so `PYTHONPATH=.` from the repo root is needed for imports, and default output paths (`ROOT / "reports"`) resolve under `archive/`. Do not run them against production without reading them first. |

## Restore private history (local only)

```bash
git clone /Users/risshabh/Projects/Active/_gridiron_archive/gridiron-private-history.bundle gridiron-private-history
```
