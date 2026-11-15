"""The serving-rows probe must agree across platforms and still catch real changes.

Background (2026-11-15): the golden matched on the Mac and failed on GitHub's x86-64
runner. Cause: ``Generator.normal(loc, scale)`` fuses ``loc + scale * z`` into one
rounding on arm64 and rounds twice on x86-64, so the synthetic input differed in the
last bit for about 8 % of values. See backend/tests/_serving_rows_probe.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2.extras
import pytest

from backend.tests import _serving_rows_probe as probe

GOLDEN = json.loads((Path(__file__).parent / "golden" / "history_writer.json").read_text())

#: sha256 of the synthetic frame's projection and y_true columns, as raw float64 bytes.
#: Identical on arm64 macOS and x86-64 Linux; a change means the generator changed.
SYNTHETIC_INPUT_SHA256 = "ce98cad93645f5775f2315bdf40249ed8c16599ba43e0e2422194b8a3f47dda3"


@pytest.fixture
def batches(monkeypatch) -> list:
    captured: list = []
    monkeypatch.setattr(psycopg2.extras, "execute_batch",
                        lambda cur, sql, rows, page_size=100: captured.append((sql, list(rows))))
    return captured


def _run(attach_intervals=None, upsert=None, batches=None) -> dict:
    from pipeline import serving_rows as sr

    return probe.run_probe(attach_intervals or sr.attach_causal_conformal_intervals,
                           sr.attach_derived_fantasy_projection,
                           upsert or sr.upsert_projections, lambda: batches)


# --- the input is the same on every platform ---------------------------------------

def test_synthetic_input_is_bit_identical_across_platforms() -> None:
    df, _ = probe.synthetic_history_frame()
    got = hashlib.sha256(df[["projection", "y_true"]].to_numpy().tobytes()).hexdigest()
    assert got == SYNTHETIC_INPUT_SHA256


def test_normal_helper_is_two_ufuncs_not_one_fused_expression() -> None:
    z = np.random.default_rng(1).standard_normal(500)
    assert np.array_equal(probe._normal(np.random.default_rng(1), 55.0, 55.0 / 3, 500),
                          np.float64(55.0) + np.float64(55.0 / 3) * z)


# --- the digest tolerates float noise but nothing larger ---------------------------

ROWS = [{"player_id": "p1", "game_id": "g1", "floor": 1.2345678901234, "stat": "receiving_yards"},
        {"player_id": "p2", "game_id": "g2", "floor": 20.5, "stat": "receiving_yards"}]


def _with_floor(rows: list, index: int, factor: float) -> list:
    out = [dict(r) for r in rows]
    out[index]["floor"] = out[index]["floor"] * factor
    return out


def test_last_bit_noise_does_not_change_the_rows_digest() -> None:
    noisy = _with_floor(ROWS, 0, 1 + 6.4e-14)   # worst platform difference measured (relative)
    assert probe.digest_rows(noisy) == probe.digest_rows(ROWS)


@pytest.mark.parametrize("factor", [1 + 1e-6, 1 + 1e-4, 1.01, 0.5])
def test_a_meaningful_value_change_changes_the_rows_digest(factor: float) -> None:
    assert probe.digest_rows(_with_floor(ROWS, 1, factor)) != probe.digest_rows(ROWS)


def test_a_dropped_row_changes_the_rows_digest() -> None:
    assert probe.digest_rows(ROWS[:1]) != probe.digest_rows(ROWS)


def test_reordered_rows_change_the_rows_digest() -> None:
    assert probe.digest_rows(list(reversed(ROWS))) != probe.digest_rows(ROWS)


def test_a_non_float_value_change_changes_the_rows_digest() -> None:
    changed = [dict(r) for r in ROWS]
    changed[0]["game_id"] = "g9"
    assert probe.digest_rows(changed) != probe.digest_rows(ROWS)


def test_a_string_that_looks_like_a_float_is_not_rounded() -> None:
    assert probe.digest_rows([{"x": "1.00000000001"}]) != probe.digest_rows([{"x": "1.0"}])


def test_negative_zero_and_nan_have_one_spelling() -> None:
    assert probe.canonical(-0.0) == probe.canonical(0.0)
    assert probe.canonical(float("nan")) == probe.canonical(np.float64("nan"))


def test_frame_digest_noise_and_change() -> None:
    df = pd.DataFrame({"a": [1.5, 2.25, 1e-9], "b": ["x", "y", "z"]})
    noisy = df.copy()
    noisy.loc[0, "a"] = 1.5 * (1 + 6.4e-14)
    assert probe.digest(noisy) == probe.digest(df)
    moved = df.copy()
    moved.loc[1, "a"] = 2.25 * (1 + 1e-6)
    assert probe.digest(moved) != probe.digest(df)
    assert probe.digest(df.iloc[:2]) != probe.digest(df)
    assert probe.digest(df.iloc[::-1].reset_index(drop=True)) != probe.digest(df)
    relabeled = df.copy()
    relabeled.loc[2, "b"] = "q"
    assert probe.digest(relabeled) != probe.digest(df)


# --- end to end: the real writer path against the golden ---------------------------

def test_unchanged_writer_matches_the_golden(batches) -> None:
    assert _run(batches=batches) == GOLDEN


def test_a_small_real_change_to_one_floor_fails_the_golden(batches) -> None:
    from pipeline import serving_rows as sr

    def nudged(df: pd.DataFrame) -> pd.DataFrame:
        out = sr.attach_causal_conformal_intervals(df)
        out.loc[out.index[-1], "floor"] = float(out["floor"].iloc[-1]) * (1 + 1e-6)
        return out

    assert _run(attach_intervals=nudged, batches=batches) != GOLDEN


def test_a_dropped_row_fails_the_golden(batches) -> None:
    from pipeline import serving_rows as sr

    def drops_one(conn, df, attributions=None):
        return sr.upsert_projections(conn, df.iloc[:-1], attributions)

    assert _run(upsert=drops_one, batches=batches) != GOLDEN


def test_reordered_rows_fail_the_golden(batches) -> None:
    from pipeline import serving_rows as sr

    def reversed_rows(conn, df, attributions=None):
        return sr.upsert_projections(conn, df.iloc[::-1].reset_index(drop=True), attributions)

    assert _run(upsert=reversed_rows, batches=batches) != GOLDEN


def test_a_changed_sql_statement_fails_the_golden(monkeypatch) -> None:
    captured: list = []
    monkeypatch.setattr(psycopg2.extras, "execute_batch",
                        lambda cur, sql, rows, page_size=100: captured.append((sql + " -- changed", list(rows))))
    assert _run(batches=captured) != GOLDEN
