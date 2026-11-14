"""The history and forward writers share one code path (decision 0012, commit 2).

The golden file was generated from the old in-script code before the move; the
shared module must reproduce it byte for byte, and the script's old names must
still lead to the same output.
"""

from __future__ import annotations

import json
from pathlib import Path

import psycopg2.extras
import pytest

from backend.tests._serving_rows_probe import run_probe

GOLDEN = json.loads((Path(__file__).parent / "golden" / "history_writer.json").read_text())


@pytest.fixture
def batches(monkeypatch) -> list:
    captured: list = []
    monkeypatch.setattr(psycopg2.extras, "execute_batch",
                        lambda cur, sql, rows, page_size=100: captured.append((sql, list(rows))))
    return captured


def test_shared_module_output_is_byte_identical_to_the_old_history_writer(batches) -> None:
    from pipeline import serving_rows as sr

    got = run_probe(sr.attach_causal_conformal_intervals, sr.attach_derived_fantasy_projection,
                    sr.upsert_projections, lambda: batches)
    assert got == GOLDEN


def test_materializer_script_names_still_work_and_match(batches) -> None:
    from scripts import materialize_stack_projections as m

    got = run_probe(m._attach_causal_conformal_intervals, m._attach_derived_fantasy_projection,
                    m._upsert, lambda: batches)
    assert got == GOLDEN


def test_script_uses_the_shared_functions_not_copies() -> None:
    from pipeline import serving_rows as sr
    from scripts import materialize_stack_projections as m

    assert m._attach_causal_conformal_intervals is sr.attach_causal_conformal_intervals
    assert m._attach_derived_fantasy_projection is sr.attach_derived_fantasy_projection
    assert m._upsert is sr.upsert_projections


def test_conflict_update_clears_stale_uncertainty_fields() -> None:
    import inspect

    from pipeline import serving_rows as sr

    sql_source = inspect.getsource(sr)
    assert "boom_probability = EXCLUDED.boom_probability" in sql_source
    assert "bust_probability = EXCLUDED.bust_probability" in sql_source
    assert "interval_method = EXCLUDED.interval_method" in sql_source
