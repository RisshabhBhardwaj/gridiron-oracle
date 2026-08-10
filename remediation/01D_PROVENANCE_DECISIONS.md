# 01D provenance decisions

Scope: the eleven `UNPROVEN` rows from the 01B as-of audit. No feature matrix,
OOF file, model artifact, or feature allowlist was changed in this session.

| Column | Source | Retroactive verdict | Evidence | Forward capture | 02 disposition |
|---|---|---|---|---|---|
| `temp_f` | `games.temp` from `nflreadpy.load_schedules` | UNOBTAINABLE | The schedule row carries game conditions and has no capture timestamp or forecast identity; it cannot establish pre-kickoff availability. | `weather_forecasts`: OpenWeather 5-day snapshot, `captured_at < kickoff_at`. | remove-and-recapture — no 2021–2025 forecast exists. |
| `wind_mph` | `games.wind` from schedules | UNOBTAINABLE | Same observed-condition field and absent source timestamp. | Same forecast snapshot stores `wind_mph`. | remove-and-recapture. |
| `temp_bucket` | deterministic child of `temp_f` | UNOBTAINABLE | Parent is not a pregame value. | Derive only from stored forecast `temp_f`. | remove-and-recapture. |
| `wind_bucket` | deterministic child of `wind_mph` | UNOBTAINABLE | Parent is not a pregame value. | Derive only from stored forecast `wind_mph`. | remove-and-recapture. |
| `wind_x_qb` | deterministic child of `wind_bucket` | UNOBTAINABLE | Parent is not a pregame value. | Derive only from timestamped forecast bucket. | remove-and-recapture. |
| `wind_x_wr` | deterministic child of `wind_bucket` | UNOBTAINABLE | Parent is not a pregame value. | Derive only from timestamped forecast bucket. | remove-and-recapture. |
| `precip_x_pass` | `games.precipitation_bucket` / weather adapter | UNOBTAINABLE | Historical bucket has no retained forecast capture timestamp and cannot be treated as known pregame. | Store forecast precipitation bucket in `weather_forecasts`. | remove-and-recapture. |
| `height` | nflreadpy season rosters | PROVEN SAFE | `player_season_profiles` has 24,633 backfilled `(player_id,effective_season)` records. The 2021–25 game-log join has 0 missing profiles and 94,735 height values. | Continue season-roster capture with source capture time. | keep — join `player_season_profiles.height_inches` by player and season. |
| `weight` | nflreadpy season rosters | PROVEN SAFE | Same dated roster backfill; 2021–25 join has 0 missing profiles and 94,735 weight values. | Continue season-roster capture with source capture time. | keep — join `player_season_profiles.weight_lbs` by player and season. |
| `depth_chart_rank` | nflreadpy depth charts | UNOBTAINABLE | 2,503,227 staged records have no `timestamp`, `published_at`, or `last_updated`; current `depth_charts.published_at` count is 0. A week label is not proof of a pregame snapshot. | Preserve source timestamps; accept a depth row only when `published_at < games.kickoff_at`. | remove-and-recapture. |
| `injury_status_encoded` | optional ESPN practice-report dataframe | MOOT | FeatureEngineer does not pass `injury_df`; audit and current matrix show 0 populated values. | Preserve report publication time and choose the latest pre-kickoff report before wiring it. | remove — it conveys no current data. |

## As-of contract hooks

`pipeline.provenance` exposes `assert_player_profiles_asof`,
`assert_depth_charts_pregame`, and `assert_weather_forecasts_pregame`. They fail
closed if effective-season/profile data, a depth publication time, or a
pre-kickoff forecast is absent. The feature-engineer roster query now uses the
season-profile join and invokes the profile assertion before writes.

## Scope change for 02

**2 of 11 are keepable** (`height`, `weight`). **9 are gone from the historical
rebuild**: 8 remove-and-recapture columns (seven weather descendants plus depth)
and 1 moot injury column. The decided list narrows the prior blanket-UNPROVEN
removal by two features; it does not make weather or depth safe by proxy.
