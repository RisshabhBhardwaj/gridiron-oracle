# Data sources

Inventory of every external data source this project reads, and its exact
status. Written because the phrase "PFR is gone" was circulating as a project
claim while `nflreadpy.load_pfr_advstats` was still being called on the
feature-engineering path (audit finding C-25). Keep this file honest: if a
statement here and the code disagree, the code is right and this file is a bug.

## Pro-Football-Reference — **partially live, commonly mis-stated**

| | |
|---|---|
| Direct scraping | **Retired.** `scraper/adapters/pro_football_ref.py` raises on construction and has zero callers. |
| PFR-derived ingestion | **Live.** `pipeline/pbp_pipeline.py::_load_pfr_drop_rates` calls `nflreadpy.load_pfr_advstats(seasons=..., stat_type="rec")`. |
| What it produces | `drop_rate` on `player_pbp_features`, consumed as the `drop_rate` field of the feature row. |
| Failure mode | Wrapped in `try/except`; a failed load returns an empty frame and leaves `drop_rate` NULL rather than crashing. |
| Superseded where available | FTN charting drop rate is preferred over the PFR value when present (`drop_rate_ftn`, later in the same module). |

The accurate one-line statement is: **"we no longer scrape
pro-football-reference.com; we still consume PFR advanced receiving stats
through nflverse's redistribution."** Anything shorter than that is wrong in one
direction or the other.

Retiring the remaining path means either dropping `drop_rate` or sourcing it
entirely from FTN — a feature-contract change, not a documentation change.

## nflverse / nflreadpy — primary, live

`scraper/adapters/nflreadpy_adapter.py` plus the pipeline modules. Free, no API
key. Supplies play-by-play, game logs, rosters, schedules, snap counts,
participation, NGS, depth charts, and (as above) PFR advanced stats.

## ADP

| Source | Access | Status |
|---|---|---|
| Fantasy Football Calculator | CSVs under `data/adp/historical/`, provenance in that directory's `PROVENANCE.md` | Primary historical ADP benchmark |
| Sleeper | Free public API, `scraper/adapters/sleeper_adp.py` | Live; aggregates completed public drafts |
| FantasyPros | **Manual CSV export only** — `scraper/adapters/fantasypros_adp_importer.py`. FantasyPros' terms prohibit scraping and this repo is public. | Live, import-only. The importer takes ADP from an `AVG`/`ADP` column and refuses a Rank-only export (audit C-23). |

## ESPN — unofficial endpoints

`scraper/adapters/espn_unofficial.py` reads ESPN's internal (undocumented)
endpoints for practice participation, depth charts, and inactives. These are not
a public API and may break without notice. `espn_adapter.py` is the companion
adapter. Treat availability as best-effort.

## Keyed third-party APIs (optional)

| Source | Key | Module |
|---|---|---|
| The Odds API | `ODDS_API_KEY` | `scraper/adapters/odds_adapter.py` (re-exported by `odds_api.py`) |
| OpenWeather | `OPENWEATHER_API_KEY` | `scraper/adapters/weather_adapter.py` (re-exported by `weather.py`) |

Both are unset by default in the `Makefile`. Nothing in the training or serving
path requires them.

## Manual seed data

| Source | Status |
|---|---|
| Coaching / scheme (`data/coaching/`) | **No seed ships.** The previous 2026 seed was provably wrong and unverified; see `data/coaching/README.md` (audit C-24). Supply a verified CSV; the adapter validates before upserting. |
| Rules changes (`scraper/adapters/rules_parser.py`) | Hand-catalogued per season; warns on an uncatalogued season. |

## Not used

No paid stats provider. No scraping of FantasyPros. No commercial projection
feed.
