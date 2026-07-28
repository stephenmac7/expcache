"""Opt-out switch for cache reads and writes.

Within a ``no_cache()`` block the ``memo()`` and ``arrays()`` wrappers
compute and return their function's result without consulting or writing
the store — for throwaway experiments whose keys would otherwise pollute
a shared cache. The flag is a ``ContextVar`` so it stays correct across
async boundaries and concurrent tasks.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Iterator

_bypassed: ContextVar[bool] = ContextVar("expcache_bypassed", default=False)


def is_bypassed() -> bool:
    """True when the current context is inside a ``no_cache()`` block."""
    return _bypassed.get()


@contextlib.contextmanager
def no_cache() -> Iterator[None]:
    """Disable cache reads and writes for the duration of the block."""
    token = _bypassed.set(True)
    try:
        yield
    finally:
        _bypassed.reset(token)
