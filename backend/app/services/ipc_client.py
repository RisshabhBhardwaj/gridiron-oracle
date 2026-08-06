"""
backend/app/services/ipc_client.py

Robust async IPC client that connects to the C++ execution engine via Unix socket.
Maintains a persistent connection and automatically reconnects on disconnects
using exponential backoff.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class IpcClient:
    """
    Singleton persistent Unix socket client for pushing events to the C++ engine.
    """
    _instance: Optional["IpcClient"] = None

    def __new__(cls, socket_path: str = "/tmp/gridiron.sock") -> "IpcClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.socket_path = socket_path
            cls._instance._writer: Optional[asyncio.StreamWriter] = None
            cls._instance._reader: Optional[asyncio.StreamReader] = None
            cls._instance._running = False
            cls._instance._reconnect_task: Optional[asyncio.Task] = None
            cls._instance._lock = asyncio.Lock()  # Requires Python ≥3.10 (Lock is not loop-bound at creation)
            # Queue to store a small amount of events if temporarily disconnected
            cls._instance._queue = asyncio.Queue(maxsize=100)
            cls._instance._dispatcher_task: Optional[asyncio.Task] = None
            cls._instance._on_disconnect: Optional[Callable[[str], None]] = None
        return cls._instance

    def set_disconnect_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback invoked whenever the engine disconnects.

        The callback receives a reason string. Use this to surface engine
        failures to the alert feed without creating a circular import.
        """
        self._on_disconnect = callback

    async def start(self) -> None:
        """Start the background connection loop."""
        if self._running:
            return
        self._running = True
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())
        logger.info(f"IPC client started, targeting {self.socket_path}")

    async def stop(self) -> None:
        """Stop the client and close connections."""
        self._running = False
        if self._reconnect_task:
            self._reconnect_task.cancel()
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
        
        async with self._lock:
            if self._writer:
                self._writer.close()
                try:
                    await self._writer.wait_closed()
                except Exception:
                    pass
                self._writer = None
                self._reader = None
        logger.info("IPC client stopped")

    def is_connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def _reconnect_loop(self) -> None:
        """Maintain persistent connection to the C++ Unix socket."""
        backoff = 1.0
        max_backoff = 30.0

        while self._running:
            # Check if currently connected
            async with self._lock:
                is_connected = self._writer is not None and not self._writer.is_closing()
            
            if not is_connected:
                try:
                    reader, writer = await asyncio.open_unix_connection(self.socket_path)
                    async with self._lock:
                        self._reader = reader
                        self._writer = writer
                    logger.info(f"IPC client connected to {self.socket_path}")
                    backoff = 1.0  # reset backoff
                except Exception as e:
                    logger.debug(f"IPC connect failed: {e}. Retrying in {backoff}s")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)
                    continue

            # Check if connection died (by attempting to read EOF)
            try:
                if self._reader:
                    # Wait for EOF or data (which we ignore as it's unidirectional C++ -> Python typically handled elsewhere, or we just want connection status)
                    data = await self._reader.read(1)
                    if not data:
                        reason = "C++ engine closed the connection (EOF)"
                        logger.warning("IPC connection closed by remote host")
                        async with self._lock:
                            if self._writer:
                                self._writer.close()
                                self._writer = None
                                self._reader = None
                        if self._on_disconnect:
                            self._on_disconnect(reason)
            except Exception as e:
                reason = f"IPC connection error: {e}"
                logger.warning(reason)
                async with self._lock:
                    if self._writer:
                        self._writer.close()
                        self._writer = None
                        self._reader = None
                if self._on_disconnect:
                    self._on_disconnect(reason)
            
            await asyncio.sleep(1) # check connection health periodically

    async def _dispatch_loop(self) -> None:
        """Dequeue events and send them to the socket."""
        while self._running:
            try:
                # Get event from queue
                event_json: str = await self._queue.get()
                
                async with self._lock:
                    if self._writer is None or self._writer.is_closing():
                        # Drop event if disconnected (live betting signals are stale quickly)
                        logger.debug("IPC disconnected, dropping queued event")
                        self._queue.task_done()
                        continue
                    
                    try:
                        self._writer.write(event_json.encode('utf-8') + b'\n')
                        await self._writer.drain()
                        self._queue.task_done()
                    except BrokenPipeError:
                        logger.warning("IPC broken pipe, event dropped")
                        self._writer.close()
                        self._writer = None
                        self._reader = None
                        self._queue.task_done()
                    except Exception as e:
                        logger.error(f"Failed to send IPC event: {e}")
                        self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dispatch loop error: {e}")
                await asyncio.sleep(1)

    def send_event(self, event_dict: dict) -> bool:
        """
        Queue an event to be sent over IPC. 
        Returns True if enqueued, False if queue is full or client is stopped.
        """
        if not self._running:
            logger.warning("IPC client is not running, dropping event")
            return False
        
        try:
            event_json = json.dumps(event_dict)
            self._queue.put_nowait(event_json)
            return True
        except asyncio.QueueFull:
            logger.warning("IPC queue is full, dropping event")
            return False
        except Exception as e:
            logger.error(f"Failed to encode/queue event: {e}")
            return False

# Initialize a global instance for the application to interact with
ipc_client = IpcClient()
