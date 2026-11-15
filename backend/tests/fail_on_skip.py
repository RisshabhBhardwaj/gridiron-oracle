"""Turn skips into failures when ``FAIL_ON_SKIP=1``.

The integration tests skip when Postgres is unreachable. That is right on a laptop
without a database and wrong in the CI job whose only purpose is to reach one: a
skip there means the database was not exercised, and it must not pass. The CI
`integration` job sets ``FAIL_ON_SKIP=1``; nothing else does, so the offline job
(which relies on skips) and local runs are unchanged. The variable is explicit
rather than ``CI=true`` because GitHub sets ``CI`` for every job.

Two places skip. A skip inside a test (``pytest.skip``, a skipping fixture) goes
through ``pytest_runtest_makereport``. A skip while collecting a file (module-level
``pytest.skip(allow_module_level=True)`` or ``pytest.importorskip``) goes through
``pytest_collectreport`` instead and never reaches a test report, so both hooks are
needed. A failed collect report aborts the run with a non-zero exit.

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


# A wrapper, not a plain hook: the session's own pytest_collectreport counts failures
# to decide the exit code, and it is registered after conftest plugins, so a plain
# hook (even tryfirst) would run after it and be too late. Code before the yield of
# a wrapper runs before every plain implementation.
@pytest.hookimpl(wrapper=True)
def pytest_collectreport(report):
    if os.environ.get("FAIL_ON_SKIP") == "1" and report.skipped:
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else report.longrepr
        report.outcome = "failed"
        report.longrepr = f"FAIL_ON_SKIP=1: a skipped file fails this run. Skip reason: {reason}"
    return (yield)
