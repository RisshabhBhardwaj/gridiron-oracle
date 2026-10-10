"""
backend/tests/test_ml_cov_backend_ipc.py

Coverage tests for backend/app/services/ipc_client.py (persistent Unix socket
client for the C++ engine).

No real socket is opened. asyncio.open_unix_connection is replaced with fake
reader/writer objects, which lets each path (connect, EOF, read error, dispatch,
broken pipe, stop) run deterministically. IpcClient is a process singleton, so
each test swaps in a fresh instance and restores the original afterwards.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from backend.app.services import ipc_client as ipc


@pytest.fixture
def fresh_client(monkeypatch):
    """A brand-new IpcClient singleton for one test, restored afterwards."""
    saved = ipc.IpcClient._instance
    ipc.IpcClient._instance = None
    client = ipc.IpcClient(socket_path="/tmp/unused-test.sock")
    yield client
    ipc.IpcClient._instance = saved


class _FakeWriter:
    def __init__(self, fail_drain=None, fail_write=None):
        self.written: list[bytes] = []
        self.closed = False
        self._fail_drain = fail_drain
        self._fail_write = fail_write

    def write(self, data):
        if self._fail_write:
            raise self._fail_write
        self.written.append(data)

    async def drain(self):
        if self._fail_drain:
            raise self._fail_drain

    def is_closing(self):
        return self.closed

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None


class _FakeReader:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def read(self, n):
        if self._chunks:
            item = self._chunks.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return b""


def _run(coro_fn, timeout: float = 5.0):
    async def wrapper():
        return await asyncio.wait_for(coro_fn(), timeout)

    return asyncio.run(wrapper())


# ── singleton and basic state ─────────────────────────────────────────────────

class TestSingletonAndState:
    def test_constructor_returns_same_instance(self, fresh_client) -> None:
        assert ipc.IpcClient() is fresh_client
        assert ipc.IpcClient(socket_path="/other.sock") is fresh_client
        assert fresh_client.socket_path == "/tmp/unused-test.sock"

    def test_default_socket_path(self, monkeypatch) -> None:
        saved = ipc.IpcClient._instance
        ipc.IpcClient._instance = None
        try:
            assert ipc.IpcClient().socket_path == "/tmp/gridiron.sock"
        finally:
            ipc.IpcClient._instance = saved

    def test_not_connected_until_writer_exists(self, fresh_client) -> None:
        assert fresh_client.is_connected() is False
        fresh_client._writer = _FakeWriter()
        assert fresh_client.is_connected() is True
        fresh_client._writer.close()
        assert fresh_client.is_connected() is False

    def test_disconnect_callback_is_stored(self, fresh_client) -> None:
        reasons: list[str] = []
        fresh_client.set_disconnect_callback(reasons.append)
        fresh_client._on_disconnect("engine down")
        assert reasons == ["engine down"]


# ── send_event ────────────────────────────────────────────────────────────────

class TestSendEvent:
    def test_refused_when_not_running(self, fresh_client) -> None:
        assert fresh_client.send_event({"kind": "bet"}) is False
        assert fresh_client._queue.qsize() == 0

    def test_enqueues_json_when_running(self, fresh_client) -> None:
        fresh_client._running = True
        assert fresh_client.send_event({"kind": "bet", "stake": 12.5}) is True
        queued = fresh_client._queue.get_nowait()
        assert json.loads(queued) == {"kind": "bet", "stake": 12.5}

    def test_full_queue_drops_event(self, fresh_client) -> None:
        fresh_client._running = True
        for i in range(100):
            assert fresh_client.send_event({"i": i}) is True
        assert fresh_client.send_event({"i": 100}) is False

    def test_unserializable_payload_is_rejected(self, fresh_client) -> None:
        fresh_client._running = True
        assert fresh_client.send_event({"bad": object()}) is False
        assert fresh_client._queue.qsize() == 0


# ── dispatch loop ─────────────────────────────────────────────────────────────

class TestDispatch:
    def test_queued_event_is_written_newline_terminated(self, fresh_client) -> None:
        writer = _FakeWriter()

        async def scenario():
            fresh_client._running = True
            fresh_client._writer = writer
            fresh_client.send_event({"kind": "signal", "edge": 0.07})
            task = asyncio.create_task(fresh_client._dispatch_loop())
            for _ in range(50):
                if writer.written:
                    break
                await asyncio.sleep(0.01)
            task.cancel()  # the loop treats cancellation as a clean exit
            await task

        _run(scenario)
        assert len(writer.written) == 1
        assert writer.written[0].endswith(b"\n")
        assert json.loads(writer.written[0].rstrip(b"\n")) == {"kind": "signal", "edge": 0.07}

    def test_event_dropped_when_disconnected(self, fresh_client) -> None:
        async def scenario():
            fresh_client._running = True
            fresh_client._writer = None
            fresh_client.send_event({"kind": "stale"})
            task = asyncio.create_task(fresh_client._dispatch_loop())
            await asyncio.sleep(0.05)
            await fresh_client._queue.join()  # dropped events still mark task_done
            task.cancel()
            await task

        _run(scenario)
        assert fresh_client._queue.qsize() == 0

    def test_broken_pipe_clears_writer(self, fresh_client) -> None:
        writer = _FakeWriter(fail_drain=BrokenPipeError())

        async def scenario():
            fresh_client._running = True
            fresh_client._writer = writer
            fresh_client._reader = object()
            fresh_client.send_event({"kind": "x"})
            task = asyncio.create_task(fresh_client._dispatch_loop())
            await asyncio.sleep(0.05)
            await fresh_client._queue.join()
            task.cancel()
            await task

        _run(scenario)
        assert writer.closed is True
        assert fresh_client._writer is None
        assert fresh_client._reader is None

    def test_generic_write_error_is_logged_and_loop_continues(self, fresh_client) -> None:
        writer = _FakeWriter(fail_write=ValueError("encoder bug"))

        async def scenario():
            fresh_client._running = True
            fresh_client._writer = writer
            fresh_client.send_event({"kind": "x"})
            task = asyncio.create_task(fresh_client._dispatch_loop())
            await asyncio.sleep(0.05)
            await fresh_client._queue.join()  # task_done is still called on the error path
            task.cancel()
            await task

        _run(scenario)
        assert writer.written == []
        assert fresh_client._writer is writer  # not cleared for a non-pipe error


# ── reconnect loop ────────────────────────────────────────────────────────────

class TestReconnectLoop:
    def test_connects_then_detects_eof_and_reports_disconnect(self, fresh_client, monkeypatch) -> None:
        reasons: list[str] = []
        reader = _FakeReader([b""])          # first read is EOF
        writer = _FakeWriter()

        async def fake_open(path):
            assert path == "/tmp/unused-test.sock"
            return reader, writer

        monkeypatch.setattr(ipc.asyncio, "open_unix_connection", fake_open)
        fresh_client.set_disconnect_callback(reasons.append)

        async def scenario():
            fresh_client._running = True
            task = asyncio.create_task(fresh_client._reconnect_loop())
            for _ in range(100):
                if reasons:
                    break
                await asyncio.sleep(0.01)
            fresh_client._running = False
            await task

        _run(scenario, timeout=10)
        assert reasons == ["C++ engine closed the connection (EOF)"]
        assert writer.closed is True
        assert fresh_client._writer is None

    def test_read_error_reports_connection_error(self, fresh_client, monkeypatch) -> None:
        reasons: list[str] = []
        reader = _FakeReader([ConnectionResetError("reset by peer")])

        async def fake_open(path):
            return reader, _FakeWriter()

        monkeypatch.setattr(ipc.asyncio, "open_unix_connection", fake_open)
        fresh_client.set_disconnect_callback(reasons.append)

        async def scenario():
            fresh_client._running = True
            task = asyncio.create_task(fresh_client._reconnect_loop())
            for _ in range(100):
                if reasons:
                    break
                await asyncio.sleep(0.01)
            fresh_client._running = False
            await task

        _run(scenario, timeout=10)
        assert reasons and reasons[0].startswith("IPC connection error: reset by peer")

    def test_connect_failure_backs_off_and_retries(self, fresh_client, monkeypatch) -> None:
        attempts: list[int] = []
        delays: list[float] = []

        async def failing_open(path):
            attempts.append(1)
            raise FileNotFoundError("no socket yet")

        async def fake_sleep(delay):
            delays.append(delay)
            if len(attempts) >= 3:
                fresh_client._running = False

        monkeypatch.setattr(ipc.asyncio, "open_unix_connection", failing_open)
        monkeypatch.setattr(ipc.asyncio, "sleep", fake_sleep)

        async def scenario():
            fresh_client._running = True
            await fresh_client._reconnect_loop()

        _run(scenario)
        assert len(attempts) == 3
        assert delays == [1.0, 2.0, 4.0]  # exponential backoff doubles each failure

    def test_backoff_is_capped_at_thirty_seconds(self, fresh_client, monkeypatch) -> None:
        delays: list[float] = []

        async def failing_open(path):
            raise OSError("down")

        async def fake_sleep(delay):
            delays.append(delay)
            if len(delays) >= 8:
                fresh_client._running = False

        monkeypatch.setattr(ipc.asyncio, "open_unix_connection", failing_open)
        monkeypatch.setattr(ipc.asyncio, "sleep", fake_sleep)

        async def scenario():
            fresh_client._running = True
            await fresh_client._reconnect_loop()

        _run(scenario)
        assert max(delays) == pytest.approx(30.0)
        assert delays[:5] == [1.0, 2.0, 4.0, 8.0, 16.0]


# ── start / stop lifecycle ────────────────────────────────────────────────────

class TestLifecycle:
    def test_start_is_idempotent_and_stop_closes_the_connection(self, fresh_client, monkeypatch) -> None:
        writer = _FakeWriter()

        async def fake_open(path):
            return _FakeReader([]), writer  # reader returns EOF immediately

        monkeypatch.setattr(ipc.asyncio, "open_unix_connection", fake_open)

        async def scenario():
            await fresh_client.start()
            first_reconnect = fresh_client._reconnect_task
            await fresh_client.start()  # second start must not spawn new tasks
            assert fresh_client._reconnect_task is first_reconnect
            await asyncio.sleep(0.05)
            fresh_client._writer = writer  # simulate an established connection
            await fresh_client.stop()
            return first_reconnect

        task = _run(scenario)
        assert fresh_client._running is False
        assert task.cancelled() or task.done()
        assert writer.closed is True
        assert fresh_client._writer is None
        assert fresh_client._reader is None

    def test_stop_without_connection_is_safe(self, fresh_client) -> None:
        async def scenario():
            await fresh_client.stop()

        _run(scenario)
        assert fresh_client._running is False
        assert fresh_client._writer is None


def test_module_level_singleton_is_an_ipc_client() -> None:
    assert isinstance(ipc.ipc_client, ipc.IpcClient)
    assert ipc.ipc_client is ipc.IpcClient()
