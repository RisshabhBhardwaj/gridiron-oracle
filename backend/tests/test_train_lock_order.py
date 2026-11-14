"""The Friday lock race between the availability job and the full retrain (coordinator, option A).

Availability keeps Fri 22:00 UTC with a hard 20 minute limit. Train-full starts Fri 23:30 UTC and
waits up to 30 minutes for the lock; if it still cannot get it, it alerts (healthchecks ``/fail``)
and fails instead of silently skipping. Train-small keeps its short wait and its quiet exit 75.

The Mac has no ``flock``, so these tests put a small ``flock`` shim on PATH that takes a REAL
``fcntl.flock`` on a real lock file in a temp dir. Holders are real processes holding the same file.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from backend.tests.test_retrain_window_scripts import FAKE_PY

REPO = Path(__file__).resolve().parents[2]
VPS1 = REPO / "infra/hosts/vps1"
BASH = shutil.which("bash")

FLOCK_SHIM = f"""#!{sys.executable}
import fcntl, os, sys, time
args = sys.argv[1:]
wait, nonblock, conflict = None, False, 1
while args and args[0].startswith("-"):
    flag = args.pop(0)
    if flag == "-w": wait = float(args.pop(0))
    elif flag == "-E": conflict = int(args.pop(0))
    elif flag == "-n": nonblock = True
    else: sys.exit(64)
target = args.pop(0)
fd = int(target) if target.isdigit() else os.open(target, os.O_CREAT | os.O_RDWR)
deadline = time.monotonic() + (0 if nonblock else wait if wait is not None else 1e9)
while True:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
    except OSError:
        if time.monotonic() >= deadline:
            sys.exit(conflict)
        time.sleep(0.05)
if args:
    os.set_inheritable(fd, True)
    os.execvp(args[0], args)
"""

HOLDER = """
import fcntl, sys, time
fd = open(sys.argv[1], "a")
fcntl.flock(fd, fcntl.LOCK_EX)
print("held", flush=True)
time.sleep(float(sys.argv[2]))
"""


@pytest.fixture
def box(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def script(name: str, body: str) -> Path:
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
        return path

    script("flock", FLOCK_SHIM)
    script("fakepy", FAKE_PY)
    # curl records every URL it is asked to fetch; the availability job's python stub records a run.
    script("curl", '#!/bin/sh\nfor a; do case "$a" in http*) echo "$a" >>"$PING_LOG";; esac; done\n')
    script("python3", '#!/bin/sh\necho ran >>"$AVAIL_LOG"\n')
    # Stands in for `docker compose run --rm training <cmd>`: runs the recipe directly.
    script("fakedocker", f'#!/bin/sh\ncd {REPO}\nexec {BASH} scripts/train_window.sh "${{MODE:-full}}"\n')
    out_root = tmp_path / "train_out"
    out_root.mkdir()
    environ = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PY": str(bin_dir / "fakepy"),
        "REAL_PY": sys.executable,
        "FAKE_LOG": str(tmp_path / "calls.log"),
        "FINAL_LOG": str(tmp_path / "final_calls.log"),
        "TRAIN_OUT_ROOT": str(out_root),
        "SKIP_CONTRACT": "1",
        "PYTHONPATH": str(REPO),
        "SEASONS": "2019-2025",
        "PING_LOG": str(tmp_path / "pings.log"),
        "AVAIL_LOG": str(tmp_path / "avail.log"),
    }
    for name in ("ONLY_CELLS", "TRAIN_LOCK_WAIT_SECONDS", "TRAIN_FULL_LOCK_WAIT_SECONDS", "HEALTHCHECKS_TRAIN_FULL_URL"):
        environ.pop(name, None)
    return environ, out_root, tmp_path


def _hold(lock: Path, seconds: float) -> subprocess.Popen:
    holder = subprocess.Popen([sys.executable, "-c", HOLDER, str(lock), str(seconds)],
                              stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "held"
    return holder


def _lines(path: Path) -> list[str]:
    return path.read_text().split() if path.exists() else []


def _availability_command(out_root: Path) -> str:
    """The compose command, as the container shell receives it (compose turns ``$$`` into ``$``)."""
    compose = yaml.safe_load((VPS1 / "compose.yml").read_text())
    command = compose["services"]["availability"]["command"][2]
    return command.replace("$$", "\0").replace("/train_out", str(out_root)).replace("\0", "$")


def _run_availability(environ: dict, out_root: Path, **extra) -> subprocess.CompletedProcess:
    env = {**environ, "HEALTHCHECKS_AVAILABILITY_URL": "http://hc.test/avail", **extra}
    return subprocess.run(["/bin/sh", "-c", _availability_command(out_root)], env=env,
                          capture_output=True, text=True, timeout=60)


# ── order 1: availability holds the lock, train-full waits and then runs ───────────────────
def test_train_full_waits_for_the_availability_job_and_then_runs(box) -> None:
    environ, out_root, tmp_path = box
    environ["TRAIN_FULL_LOCK_WAIT_SECONDS"] = "20"
    environ["TRAIN_LOCK_WAIT_SECONDS"] = "0"  # the short wait must not be what full uses
    holder = _hold(out_root / ".train.lock", 1.5)
    started = time.monotonic()
    try:
        done = subprocess.run([BASH, str(REPO / "scripts/train_window.sh"), "full"], env=environ,
                              capture_output=True, text=True, timeout=120)
    finally:
        holder.kill()
    assert done.returncode == 0, done.stderr
    assert time.monotonic() - started >= 1.0, "it should have waited for the holder"
    assert (out_root / "last_full_stamp").exists()


# ── order 2: train-full holds the lock, availability times out and fails without a ping ────
def test_availability_times_out_behind_train_full_and_sends_no_success_ping(box) -> None:
    environ, out_root, _ = box
    holder = _hold(out_root / ".train.lock", 30)
    try:
        result = _run_availability(environ, out_root, AVAILABILITY_LOCK_WAIT_SECONDS="1")
    finally:
        holder.kill()
    assert result.returncode == 75, result.stderr
    assert _lines(Path(environ["AVAIL_LOG"])) == [], "the refresh must not run without the lock"
    assert _lines(Path(environ["PING_LOG"])) == [], "a timeout must not send the success ping"


def test_availability_runs_and_pings_once_when_the_lock_is_free(box) -> None:
    environ, out_root, _ = box
    result = _run_availability(environ, out_root, AVAILABILITY_LOCK_WAIT_SECONDS="5")
    assert result.returncode == 0, result.stderr
    assert _lines(Path(environ["AVAIL_LOG"])) == ["ran"]
    assert _lines(Path(environ["PING_LOG"])) == ["http://hc.test/avail"]


def test_a_failing_refresh_sends_no_success_ping(box) -> None:
    environ, out_root, tmp_path = box
    (tmp_path / "bin/python3").write_text('#!/bin/sh\necho ran >>"$AVAIL_LOG"\nexit 1\n')
    result = _run_availability(environ, out_root, AVAILABILITY_LOCK_WAIT_SECONDS="5")
    assert result.returncode != 0
    assert _lines(Path(environ["PING_LOG"])) == []


def test_a_hung_refresh_is_killed_so_it_cannot_hold_the_lock_past_the_job_limit(box) -> None:
    environ, out_root, tmp_path = box
    (tmp_path / "bin/python3").write_text('#!/bin/sh\necho ran >>"$AVAIL_LOG"\nexec sleep 30\n')
    started = time.monotonic()
    result = _run_availability(environ, out_root, AVAILABILITY_LOCK_WAIT_SECONDS="5", AVAILABILITY_RUN_SECONDS="1")
    assert time.monotonic() - started < 20
    assert result.returncode != 0
    assert _lines(Path(environ["PING_LOG"])) == []


# ── train-full that still cannot get the lock: alert and fail, never a silent skip ─────────
def test_train_full_that_cannot_get_the_lock_logs_why_and_exits_non_zero(box) -> None:
    environ, out_root, _ = box
    environ["TRAIN_FULL_LOCK_WAIT_SECONDS"] = "1"
    holder = _hold(out_root / ".train.lock", 30)
    try:
        result = subprocess.run([BASH, str(REPO / "scripts/train_window.sh"), "full"], env=environ,
                                capture_output=True, text=True, timeout=60)
    finally:
        holder.kill()
    assert result.returncode not in (0, 75), "75 means a quiet skip or a STOP pause; this must be distinct"
    assert ".train.lock" in result.stderr and "1s" in result.stderr
    assert not (out_root / "full_stamp").exists(), "no rebuild may start without the lock"


def test_the_full_launcher_pings_fail_when_the_lock_wait_times_out(box) -> None:
    environ, out_root, _ = box
    environ.update(TRAIN_FULL_LOCK_WAIT_SECONDS="1", TRAIN_FULL_CMD="bash scripts/train_window.sh full",
                   HEALTHCHECKS_TRAIN_FULL_URL="http://hc.test/full", DOCKER=str(Path(environ["PATH"].split(":")[0]) / "fakedocker"),
                   COMPOSE_DIR=str(VPS1))
    holder = _hold(out_root / ".train.lock", 30)
    try:
        result = subprocess.run(["sh", str(VPS1 / "scripts/train_full.sh")], env=environ,
                                capture_output=True, text=True, timeout=60)
    finally:
        holder.kill()
    assert result.returncode != 0
    assert _lines(Path(environ["PING_LOG"])) == ["http://hc.test/full/fail"]


def test_the_full_launcher_pings_success_once_and_no_fail_on_a_clean_run(box) -> None:
    environ, out_root, _ = box
    environ.update(TRAIN_FULL_CMD="bash scripts/train_window.sh full", HEALTHCHECKS_TRAIN_FULL_URL="http://hc.test/full",
                   DOCKER=str(Path(environ["PATH"].split(":")[0]) / "fakedocker"), COMPOSE_DIR=str(VPS1))
    result = subprocess.run(["sh", str(VPS1 / "scripts/train_full.sh")], env=environ,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert _lines(Path(environ["PING_LOG"])) == ["http://hc.test/full"]


def test_the_full_launcher_without_a_url_still_fails_on_a_lock_timeout(box) -> None:
    environ, out_root, _ = box
    environ.update(TRAIN_FULL_LOCK_WAIT_SECONDS="1", TRAIN_FULL_CMD="bash scripts/train_window.sh full",
                   DOCKER=str(Path(environ["PATH"].split(":")[0]) / "fakedocker"), COMPOSE_DIR=str(VPS1))
    holder = _hold(out_root / ".train.lock", 30)
    try:
        result = subprocess.run(["sh", str(VPS1 / "scripts/train_full.sh")], env=environ,
                                capture_output=True, text=True, timeout=60)
    finally:
        holder.kill()
    assert result.returncode != 0
    assert _lines(Path(environ["PING_LOG"])) == []


# ── train-small keeps its short wait and its quiet skip ────────────────────────────────────
def test_train_small_keeps_the_short_wait_and_exit_75_with_no_alert(box) -> None:
    environ, out_root, _ = box
    environ["TRAIN_LOCK_WAIT_SECONDS"] = "1"
    environ["TRAIN_FULL_LOCK_WAIT_SECONDS"] = "30"  # must not apply to small
    environ["HEALTHCHECKS_TRAIN_FULL_URL"] = "http://hc.test/full"
    holder = _hold(out_root / ".train.lock", 30)
    started = time.monotonic()
    try:
        result = subprocess.run([BASH, str(REPO / "scripts/train_window.sh"), "small"], env=environ,
                                capture_output=True, text=True, timeout=60)
    finally:
        holder.kill()
    assert result.returncode == 75
    assert time.monotonic() - started < 10
    assert _lines(Path(environ["PING_LOG"])) == []


# ── the units ──────────────────────────────────────────────────────────────────────────────
def _unit_value(path: str, key: str) -> str:
    text = (VPS1 / "systemd" / path).read_text()
    match = re.search(rf"^{key}=(.*)$", text, re.M)
    assert match, f"{path} has no {key}="
    return match.group(1).strip()


def test_train_full_starts_friday_2330_and_still_ends_monday_0600() -> None:
    assert _unit_value("gridiron-train-full.timer", "OnCalendar") == "Fri *-*-* 23:30:00"
    # Fri 23:30 to Mon 06:00 is 54 h 30 min.
    assert _unit_value("gridiron-train-full.service", "RuntimeMaxSec") == "54h 30min"
    assert "56h" not in (VPS1 / "systemd/gridiron-train-full.service").read_text()


def test_availability_keeps_friday_2200_with_a_hard_20_minute_unit_timeout() -> None:
    assert _unit_value("gridiron-availability.timer", "OnCalendar") == "Fri *-*-* 22:00:00"
    assert _unit_value("gridiron-availability.service", "TimeoutStartSec") == "20min"


def test_the_in_container_waits_fit_inside_the_unit_timeout() -> None:
    command = yaml.safe_load((VPS1 / "compose.yml").read_text())["services"]["availability"]["command"][2]
    wait = int(re.search(r"AVAILABILITY_LOCK_WAIT_SECONDS:-(\d+)", command).group(1))
    run = int(re.search(r"AVAILABILITY_RUN_SECONDS:-(\d+)", command).group(1))
    assert wait + run < 20 * 60
    assert "timeout" in command and "-E 75" in command


def test_the_compose_command_escapes_the_shell_variable_for_compose_interpolation() -> None:
    command = yaml.safe_load((VPS1 / "compose.yml").read_text())["services"]["availability"]["command"][2]
    assert "$$HEALTHCHECKS_AVAILABILITY_URL" in command
    assert not re.search(r"(?<!\$)\$HEALTHCHECKS_AVAILABILITY_URL", command)


# ── L4: the availability job gets only what it needs, not the whole .env ──────────────────
def test_the_availability_service_does_not_load_the_env_file() -> None:
    service = yaml.safe_load((VPS1 / "compose.yml").read_text())["services"]["availability"]
    assert "env_file" not in service  # .env holds R2 keys and every other secret on the box


def test_the_availability_service_passes_only_the_variables_the_job_reads() -> None:
    service = yaml.safe_load((VPS1 / "compose.yml").read_text())["services"]["availability"]
    assert set(service["environment"]) == {
        "PYTHONPATH", "DATABASE_URL", "PRODUCT_MODE", "AVAILABILITY_LOCK_WAIT_SECONDS",
        "AVAILABILITY_RUN_SECONDS", "HEALTHCHECKS_AVAILABILITY_URL",
    }
    assert service["environment"]["PRODUCT_MODE"] == "artifact_backed"
    for name in service["environment"]:
        assert not re.search(r"R2|SECRET|TOKEN|PASSWORD|KEY", name), name
