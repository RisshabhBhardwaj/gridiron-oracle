"""Turn skips into failures when ``FAIL_ON_SKIP=1``.

The integration tests skip when Postgres is unreachable. That is right on a laptop
without a database and wrong in the CI job whose only purpose is to reach one: a
skip there means the database was not exercised, and it must not pass. The CI
`integration` job sets ``FAIL_ON_SKIP=1``; nothing else does, so the offline job
(which relies on skips) and local runs are unchanged. The variable is explicit
rather than ``CI=true`` because GitHub sets ``CI`` for every job.

Expected failures (``xfail``) report as skipped too; they are left alone.
"""

from __future__ import annotations

import os

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if os.environ.get("FAIL_ON_SKIP") != "1":
        return
    if report.skipped and not hasattr(report, "wasxfail"):
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else report.longrepr
        report.outcome = "failed"
        report.longrepr = f"FAIL_ON_SKIP=1: skipped tests fail this run. Skip reason: {reason}"
