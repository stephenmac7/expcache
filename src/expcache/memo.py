"""diskcache-backed memoization storage."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

import diskcache

from .bypass import is_bypassed
from .repair import warn_dropped

_MISSING = object()


class MemoFn:
    """Disk-memoized function with one entry per call key and no eviction."""

    def __init__(
        self,
        fn: Callable[..., Any],
        key_fn: Callable[[tuple, dict], str],
        dir: Path,
    ) -> None:
        self._fn = fn
        self._key_fn = key_fn
        self._dir = dir
        self._data = diskcache.Index(str(dir))
        self.hits = 0
        self.misses = 0
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__wrapped__ = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if is_bypassed():
            return self._fn(*args, **kwargs)
        key = self._key_fn(args, kwargs)
        value = self._read(key)
        if value is not _MISSING:
            self.hits += 1
            return value
        value = self._fn(*args, **kwargs)
        self._data[key] = value
        self.misses += 1
        return value

    def _read(self, key: str) -> Any:
        """Discard entries that fail to deserialize."""
        try:
            return self._data.get(key, _MISSING)
        except (EOFError, pickle.UnpicklingError):
            del self._data[key]
            warn_dropped(
                f"{self.__name__}: dropped a cached result that could not be "
                f"read back (cache {self._dir}); recomputing it."
            )
            return _MISSING

    def clear(self) -> None:
        """Delete every stored entry for this function."""
        self._data.clear()
