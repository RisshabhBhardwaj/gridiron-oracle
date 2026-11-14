"""``--replace-week`` of the weekly refresh (decision 0012 D4, commit 7).

The producer never overwrites rows an approved run already serves. Replacing a served
week is therefore a manual, flagged action taken after a database dump. The delete and the
insert run in one transaction inside ``run_forward`` (tested in test_forward_serving.py), so a
failure keeps the old week. This file covers the command line: it must refuse without a dump
and never run from a timer.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


from scripts import weekly_refresh

ROOT = Path(__file__).resolve().parents[2]


def _main(argv):
    with patch.object(weekly_refresh, "run_weekly_refresh", return_value=0) as run:
        try:
            rc = weekly_refresh.main(argv)
        except SystemExit as exc:
            rc = exc.code
    return rc, run


def test_replace_week_without_a_dump_path_is_refused_before_anything_runs():
    rc, run = _main(["--database-url", "postgres://x", "--replace-week"])
    assert rc == 2
    run.assert_not_called()


def test_replace_week_with_a_missing_dump_file_is_refused(tmp_path):
    rc, run = _main(["--database-url", "postgres://x", "--replace-week", "--dump-path", str(tmp_path / "no.dump")])
    assert rc == 2
    run.assert_not_called()


def test_replace_week_with_an_empty_dump_file_is_refused(tmp_path):
    dump = tmp_path / "empty.dump"
    dump.write_bytes(b"")
    rc, run = _main(["--database-url", "postgres://x", "--replace-week", "--dump-path", str(dump)])
    assert rc == 2
    run.assert_not_called()


def test_a_dump_path_without_replace_week_is_refused(tmp_path):
    dump = tmp_path / "d.dump"
    dump.write_bytes(b"PGDMP")
    rc, run = _main(["--database-url", "postgres://x", "--dump-path", str(dump)])
    assert rc == 2
    run.assert_not_called()


def test_replace_week_with_a_real_dump_passes_the_flag_through(tmp_path):
    dump = tmp_path / "d.dump"
    dump.write_bytes(b"PGDMP")
    rc, run = _main(["--database-url", "postgres://x", "--replace-week", "--dump-path", str(dump)])
    assert rc == 0
    assert run.call_args.kwargs["replace_week"] is True


def test_the_default_run_does_not_replace_anything():
    rc, run = _main(["--database-url", "postgres://x"])
    assert rc == 0
    assert run.call_args.kwargs["replace_week"] is False


def test_no_systemd_unit_or_compose_file_ever_passes_replace_week():
    offenders = [
        str(p.relative_to(ROOT))
        for pattern in ("infra/**/*.service", "infra/**/*.timer", "infra/**/*.yml", "docker-compose*.yml", "Makefile")
        for p in ROOT.glob(pattern)
        if "--replace-week" in p.read_text()
    ]
    assert offenders == []
