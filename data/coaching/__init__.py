"""
2026 coaching / scheme seed data (manual, free).

Update this CSV when HC/OC/DC changes land. Consumed by
`scraper.adapters.coaching_adapter`.
"""

from __future__ import annotations

from pathlib import Path

SEED_PATH = Path(__file__).with_name("coaching_2026.csv")

# Written once if missing so the repo has a usable starter file.
DEFAULT_ROWS = """season,team,head_coach,offensive_coordinator,defensive_coordinator,scheme_pass_rate_prior,notes
2026,KC,Andy Reid,Matt Nagy,Steve Spagnuolo,0.58,seed — verify before draft
2026,BUF,Sean McDermott,Joe Brady,Bobby Babich,0.55,seed — verify before draft
2026,PHI,Nick Sirianni,Kevin Patullo,Vic Fangio,0.52,seed — verify before draft
2026,DET,Dan Campbell,John Morton,Kelvin Sheppard,0.54,seed — verify before draft
2026,BAL,John Harbaugh,Todd Monken,Zach Orr,0.53,seed — verify before draft
2026,SF,Kyle Shanahan,Klay Kubiak,Robert Saleh,0.56,seed — verify before draft
"""


def ensure_seed_file() -> Path:
    if not SEED_PATH.exists():
        SEED_PATH.write_text(DEFAULT_ROWS)
    return SEED_PATH
