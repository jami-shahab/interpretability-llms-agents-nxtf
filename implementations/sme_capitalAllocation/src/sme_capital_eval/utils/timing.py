"""Timing utilities: timed() context manager and ISO timestamp helper."""

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator


class _Timer:
    """Holds elapsed milliseconds after a timed() block completes."""

    elapsed_ms: float = 0.0


@contextmanager
def timed() -> Generator[_Timer, None, None]:
    """Context manager that measures wall-clock time in milliseconds.

    Yields
    ------
    _Timer
        Object whose ``elapsed_ms`` attribute is set after the block exits.
    """
    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed_ms = (time.perf_counter() - start) * 1000.0


def iso_now() -> str:
    """Return current UTC time as an ISO-8601 string.

    Returns
    -------
    str
        UTC timestamp in ISO format (e.g. '2026-04-30T17:00:00.000000+00:00').
    """
    return datetime.now(tz=timezone.utc).isoformat()
