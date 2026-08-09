"""
Coaching / scheme seed data (manual, free).

**No seed CSV ships with this repository.** The previous `coaching_2026.csv` and
the `DEFAULT_ROWS` fallback that regenerated it both contained unverified staff
assignments — including a defensive coordinator who was never a coach and two
coordinators each holding the same role on two teams (audit C-24). A fallback
that writes unverified facts on demand is the same defect with an extra step, so
it is gone too.

Supply your own verified CSV: copy `coaching_TEMPLATE.csv`, fill it in with
`source` / `verified_by` / `verified_on`, and load it with
`scraper.adapters.coaching_adapter`, which validates before it upserts.
See `README.md` in this directory.
"""

from __future__ import annotations

from pathlib import Path

TEMPLATE_PATH = Path(__file__).with_name("coaching_TEMPLATE.csv")


def seed_path(season: int) -> Path:
    """Expected location of a verified coaching CSV for `season`."""
    return Path(__file__).with_name(f"coaching_{season}.csv")
