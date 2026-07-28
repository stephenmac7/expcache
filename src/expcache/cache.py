"""Cache roots and the memo()/arrays() decorators."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Iterable

from .arrays import ArrayFn
from .fingerprint import Fingerprinter, fingerprint
from .memo import MemoFn


class _CallKey:
    """Builds a cache key from a call's bound arguments.

    The key covers the function's cache name, the explicit version, and
    the fingerprints of all non-ignored arguments (defaults applied) — it
    never includes code, so bump ``version=`` after logic changes.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str,
        version: int | str,
        ignore: Iterable[str],
        fp: Fingerprinter,
    ) -> None:
        self._sig = inspect.signature(fn)
        self._ignore = frozenset(ignore)
        unknown = self._ignore - set(self._sig.parameters)
        if unknown:
            raise TypeError(
                f"ignore= names not in {name}'s signature: {sorted(unknown)}"
            )
        self._name = name
        self._version = version
        self._fp = fp

    def __call__(self, args: tuple, kwargs: dict) -> str:
        bound = self._sig.bind(*args, **kwargs)
        bound.apply_defaults()
        payload = {
            k: v for k, v in bound.arguments.items() if k not in self._ignore
        }
        return self._fp(("call", self._name, self._version, payload))


class Cache:
    """Root of an on-disk experiment cache.

    ``memo()`` stores one pickled entry per call — for derived results of
    any picklable type. ``arrays()`` stores results in stacked mmap
    shards — for heavy per-item functions returning numpy arrays (e.g.
    encoders).

    Entries are stored under ``root/<memo|arrays>/<name>/``, so functions
    sharing a root must have distinct cache names (default: the function
    name; override with ``name=``).
    """

    def __init__(self, root, fp: Fingerprinter = fingerprint) -> None:
        self.root = Path(root).expanduser()
        self._fp = fp

    def memo(
        self,
        *,
        version: int | str,
        ignore: Iterable[str] = (),
        name: str | None = None,
    ) -> Callable[[Callable[..., Any]], MemoFn]:
        def deco(fn: Callable[..., Any]) -> MemoFn:
            n = name or fn.__name__
            key = _CallKey(fn, n, version, ignore, self._fp)
            return MemoFn(fn, key, self.root / "memo" / n)

        return deco

    def arrays(
        self,
        *,
        version: int | str,
        ignore: Iterable[str] = (),
        name: str | None = None,
    ) -> Callable[[Callable[..., Any]], ArrayFn]:
        def deco(fn: Callable[..., Any]) -> ArrayFn:
            n = name or fn.__name__
            key = _CallKey(fn, n, version, ignore, self._fp)
            return ArrayFn(fn, key, self.root / "arrays" / n)

        return deco
