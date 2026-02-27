"""
PID lockfile guard to prevent duplicate bot instances.

Usage:
    lock = PidLock("/tmp/ga_mssr_live.pid")
    lock.acquire()  # raises RuntimeError if another instance is running
    # ... bot runs ...
    lock.release()  # called automatically via atexit
"""
from __future__ import annotations

import atexit
import logging
import os
import signal
from pathlib import Path

logger = logging.getLogger(__name__)


class PidLock:
    """Simple PID lockfile to prevent duplicate bot instances."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._locked = False

    def acquire(self) -> None:
        """Acquire the lock. Raises RuntimeError if another instance is alive."""
        if self._path.exists():
            try:
                old_pid = int(self._path.read_text().strip())
                # Check if process is still running
                os.kill(old_pid, 0)
                raise RuntimeError(
                    f"Another instance is already running (PID {old_pid}). "
                    f"Lock file: {self._path}"
                )
            except ProcessLookupError:
                # Old process is dead, stale lock file
                logger.warning(
                    "Removing stale lock file (PID %s no longer running)",
                    self._path.read_text().strip(),
                )
            except ValueError:
                logger.warning("Corrupt lock file, overwriting")

        self._path.write_text(str(os.getpid()))
        self._locked = True
        atexit.register(self.release)
        logger.info("PID lock acquired: %s (PID %d)", self._path, os.getpid())

    def release(self) -> None:
        """Release the lock (remove the PID file)."""
        if self._locked and self._path.exists():
            try:
                # Only remove if it's our PID
                stored_pid = int(self._path.read_text().strip())
                if stored_pid == os.getpid():
                    self._path.unlink()
                    logger.info("PID lock released: %s", self._path)
            except (ValueError, OSError):
                pass
            self._locked = False
