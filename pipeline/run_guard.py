"""
Guard for materializers that write under a pending pipeline_run_id.

A pending run may insert new keys and overwrite rows that are not served (NULL,
or tagged with a run that is not approved). It must never overwrite a row that
another approved run serves: a failed refresh would otherwise change served
values before anything is promoted. Approved ids are the manifest's
projection_policy.approved_pipeline_run_ids, the same set the API serves.
"""

from __future__ import annotations

from pathlib import Path

from psycopg2.extensions import adapt

REPO_ROOT = Path(__file__).resolve().parents[1]


def approved_pipeline_run_ids() -> frozenset[str]:
    """Approved ids from the baseline manifest, resolved against the repo root.

    The projection service resolves a relative manifest path against the working
    directory. A cron job that starts elsewhere must not change which ids a
    materializer protects, so the writers resolve the path here instead.
    """
    from backend.app.core.config import settings
    from backend.app.services.projection import load_approved_pipeline_run_ids

    path = Path(settings.baseline_manifest_path)
    return load_approved_pipeline_run_ids(path if path.is_absolute() else REPO_ROOT / path)


def approved_array_literal(approved) -> str:
    """Embed the approved ids as a SQL array literal (psycopg2 quoting; ids come from the manifest)."""
    return adapt(sorted(approved)).getquoted().decode()


def skip_rows_served_by_another_approved_run(table: str, column: str, approved_literal: str) -> str:
    """ON CONFLICT ... WHERE clause: overwrite unless another approved run owns the row."""
    return (
        f"WHERE NOT (COALESCE({table}.{column} = ANY({approved_literal}), false) "
        f"AND {table}.{column} IS DISTINCT FROM EXCLUDED.{column})"
    )
