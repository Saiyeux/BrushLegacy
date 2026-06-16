"""
log.py — Lightweight session timer for robot debug logging.

Each tlog() call prints the total elapsed time since tlog_reset() plus the
delta (Δ) since the previous call.  The Δ column shows how long the previous
operation took, making slow steps immediately visible.

Usage:
    from log import tlog, tlog_reset
    tlog_reset()           # call once at the top of a session
    tlog("DIP → 大红")     # [00:00.001  +0.001s]  DIP → 大红
    ...
    tlog("PAINT 1/300")    # [00:04.523  +4.522s]  PAINT 1/300
"""
from __future__ import annotations
import time as _time

_t0: float | None = None
_tl: float | None = None


def tlog(msg: str) -> None:
    """Print msg with [MM:SS.sss  +Δs] prefix."""
    global _t0, _tl
    now = _time.perf_counter()
    if _t0 is None:
        _t0 = _tl = now
    elapsed = now - _t0
    delta   = now - _tl
    _tl     = now
    m, s = divmod(elapsed, 60)
    print(f"[{int(m):02d}:{s:06.3f}  +{delta:5.3f}s]  {msg}", flush=True)


def tlog_reset() -> None:
    """Reset the session clock and print a separator line."""
    global _t0, _tl
    _t0 = _tl = _time.perf_counter()
    tlog("─── session start ───")
