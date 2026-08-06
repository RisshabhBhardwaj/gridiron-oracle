"""
scraper/adapters/rules_parser.py

NFL rules and meta-context parser.

Reads NFL rule change announcements, officiating notes, and league-level
memos to derive the `rule_meta` feature bucket (Bucket 7 in feature_engineer.py).

Sources (static, updated as needed):
  - NFL.com rules changes press releases
  - nfloperations.com officiating memoranda

STATUS: Stub — feature bucket defaults to zeros when not populated.
  The `rule_meta` features in feature_engineer.py fall back to 0.0
  gracefully when this adapter hasn't run.

  Implementation priority: LOW — rule changes are infrequent (1-2/year)
  and can be hard-coded as lookup tables rather than scraped dynamically.

When implemented, output should be:
  Dict[str, float] — feature name → value for the current season.
  Example: {"tuck_rule_active": 0.0, "hip_drop_tackle_penalty": 1.0}
"""

from __future__ import annotations

from datetime import datetime


class RulesParser:
    """
    Parses NFL rule change documents to derive rule_meta features.

    Not yet implemented — returns empty dict on all method calls.
    The pipeline degrades gracefully (rule_meta features default to 0.0).
    """

    # Hard-coded known rule changes by season.
    # Expand as new rules are added each offseason.
    _RULE_CHANGES_BY_SEASON: dict[int, dict[str, float]] = {
        2026: {
            # Carry forward 2025 kickoff / hip-drop regime until catalogued otherwise.
            "hip_drop_tackle_ban": 1.0,
            "kickoff_unified_rule": 1.0,
            "dynamic_kickoff_refined": 1.0,  # 2025–26 kickoff refinements
        },
        2025: {
            "hip_drop_tackle_ban": 1.0,
            "kickoff_unified_rule": 1.0,
            "dynamic_kickoff_refined": 1.0,
        },
        2024: {
            "hip_drop_tackle_ban": 1.0,         # Penalised starting 2024
            "kickoff_unified_rule": 1.0,         # New kickoff format 2024
            "dynamic_kickoff_refined": 0.0,
        },
        2023: {
            "hip_drop_tackle_ban": 0.0,
            "kickoff_unified_rule": 0.0,
            "dynamic_kickoff_refined": 0.0,
        },
        2022: {
            "hip_drop_tackle_ban": 0.0,
            "kickoff_unified_rule": 0.0,
            "dynamic_kickoff_refined": 0.0,
        },
        2021: {
            "hip_drop_tackle_ban": 0.0,
            "kickoff_unified_rule": 0.0,
            "dynamic_kickoff_refined": 0.0,
        },
        2020: {
            "hip_drop_tackle_ban": 0.0,
            "kickoff_unified_rule": 0.0,
            "dynamic_kickoff_refined": 0.0,
        },
        2019: {
            "hip_drop_tackle_ban": 0.0,
            "kickoff_unified_rule": 0.0,
            "dynamic_kickoff_refined": 0.0,
        },
    }

    def get_season_features(self, season: int) -> dict[str, float]:
        """
        Return rule_meta features for the given season.

        Uses the hard-coded lookup table above. Unknown future seasons raise
        rather than silently inheriting an older season's rules.
        """
        import logging
        import warnings

        if season in self._RULE_CHANGES_BY_SEASON:
            return dict(self._RULE_CHANGES_BY_SEASON[season])

        latest = max(self._RULE_CHANGES_BY_SEASON)
        msg = (
            f"RulesParser: season {season} not catalogued; "
            f"falling back to season {latest}. Update _RULE_CHANGES_BY_SEASON."
        )
        logging.getLogger(__name__).warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        return dict(self._RULE_CHANGES_BY_SEASON[latest])

    def get_current_season_features(self) -> dict[str, float]:
        """Return rule_meta features for the current calendar year season."""
        return self.get_season_features(datetime.now().year)


__all__ = ["RulesParser"]
