# Historical ADP CSVs (primary ADP benchmark source)
#
# One file per season, Full PPR. Columns: player_name,position,team,adp
# Provenance: see PROVENANCE.md (Fantasy Football Calculator API).
#
# Import:
#   python -m scripts.import_historical_adp --season 2024
#   python -m scripts.import_historical_adp --all
#
# Eval:
#   python -m ml.adp_eval --season 2024 --from-actuals
#
# Related ADP sources:
#   - FantasyPros manual CSV: scraper.adapters.fantasypros_adp_importer
#   - Sleeper draft aggregate: scraper.adapters.sleeper_adp
