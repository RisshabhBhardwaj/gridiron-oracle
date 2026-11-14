"""FAIL_ON_SKIP=1 makes a skipped test fail the run; unset, skips stay skips."""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]

CONFTEST = "from backend.tests.fail_on_skip import pytest_runtest_makereport  # noqa: F401\n"
TESTS = """
import pytest

def test_ok():
    pass

def test_skipped():
    pytest.skip("database unreachable")

@pytest.mark.xfail(reason="known")
def test_expected_failure():
    assert False
"""


@pytest.fixture
def project(pytester):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_sample=TESTS)
    return pytester


def test_skip_fails_the_run_when_enabled(project, monkeypatch):
    monkeypatch.setenv("FAIL_ON_SKIP", "1")
    result = project.runpytest("-p", "no:cacheprovider")
    assert result.ret != 0
    result.assert_outcomes(passed=1, failed=1, xfailed=1)
    result.stdout.fnmatch_lines(["*FAIL_ON_SKIP=1*database unreachable*"])


def test_skip_stays_a_skip_when_disabled(project, monkeypatch):
    monkeypatch.delenv("FAIL_ON_SKIP", raising=False)
    result = project.runpytest("-p", "no:cacheprovider")
    assert result.ret == 0
    result.assert_outcomes(passed=1, skipped=1, xfailed=1)
