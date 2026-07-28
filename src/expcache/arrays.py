"""Block storage for functions returning numpy arrays.

Results are stacked along axis 0 into large shard files, one shard per
(dtype, trailing-shape) signature, with a diskcache-backed index mapping
call keys to row ranges. Reads are zero-copy slices of a read-only
memory map, so loading a whole corpus of cached features is one
sequential mmap rather than per-entry deserialization.

Single-writer: concurrent writes from multiple processes are not
supported.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import diskcache
import numpy as np

from .bypass import is_bypassed


class ArrayStore:
    """Append-only store of variable-length arrays stacked along axis 0."""

    def __init__(self, dir: Path) -> None:
        self._dir = dir
        self._mmaps: dict[str, np.memmap] = {}
        self._index = diskcache.Index(str(dir / "index"))
        self._shards: dict[str, dict] = {}
        self._entries: dict[str, list] = {}
        for key, value in self._index.items():
            kind, _, name = key.partition(":")
            assert kind in ("s", "e"), key
            (self._shards if kind == "s" else self._entries)[name] = value

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> np.ndarray:
        shard, start, stop = self._entries[key]
        meta = self._shards[shard]
        if start == stop:
            return np.empty((0, *meta["trailing"]), dtype=np.dtype(meta["dtype"]))
        return self._mmap(shard, stop)[start:stop]

    def put_many(self, items: Sequence[tuple[str, np.ndarray]]) -> None:
        """Append (key, array) pairs, saving the index once at the end."""
        if not items:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        by_shard: dict[str, list[tuple[str, np.ndarray]]] = {}
        for key, value in items:
            by_shard.setdefault(self._shard_for(value), []).append((key, value))
        for shard, group in by_shard.items():
            meta = self._shards[shard]
            rowbytes = np.dtype(meta["dtype"]).itemsize * math.prod(meta["trailing"])
            path = self._dir / f"{shard}.bin"
            with path.open("r+b" if path.exists() else "wb") as f:
                f.seek(meta["rows"] * rowbytes)
                for key, value in group:
                    start = meta["rows"]
                    if value.shape[0]:
                        arr = np.ascontiguousarray(value)
                        f.write(arr.reshape(-1).data)
                        meta["rows"] = start + value.shape[0]
                    self._entries[key] = [shard, start, meta["rows"]]
        # Index commit comes after the byte writes: a crash in between
        # leaves orphaned tail rows that the next append overwrites.
        with self._index.transact():
            for key, _ in items:
                self._index[f"e:{key}"] = self._entries[key]
            for shard in by_shard:
                self._index[f"s:{shard}"] = self._shards[shard]

    def _shard_for(self, value: np.ndarray) -> str:
        sig = (value.dtype.str, list(value.shape[1:]))
        for name, meta in self._shards.items():
            if (meta["dtype"], meta["trailing"]) == sig:
                return name
        name = f"s{len(self._shards)}"
        self._shards[name] = {"dtype": sig[0], "trailing": sig[1], "rows": 0}
        return name

    def _mmap(self, shard: str, rows: int) -> np.memmap:
        # Appends never move existing rows, so a map stays valid for the
        # rows it covers; remap only when a read reaches past its end.
        # Each live np.memmap holds one file descriptor.
        meta = self._shards[shard]
        cached = self._mmaps.get(shard)
        if cached is not None and cached.shape[0] >= rows:
            return cached
        mm = np.memmap(
            self._dir / f"{shard}.bin",
            dtype=np.dtype(meta["dtype"]),
            mode="r",
            shape=(meta["rows"], *meta["trailing"]),
        )
        self._mmaps[shard] = mm
        return mm

    def clear(self) -> None:
        """Drop every entry and shard file; the store stays usable."""
        self._index.clear()
        self._mmaps.clear()
        for shard in self._shards:
            (self._dir / f"{shard}.bin").unlink(missing_ok=True)
        self._shards = {}
        self._entries = {}


class ArrayFn:
    """Disk-cached function returning numpy arrays, stored in stacked shards.

    Results are read-only — hits are zero-copy views into a shard memory
    map, misses return the freshly computed array; copy before mutating.
    """

    def __init__(
        self,
        fn: Callable[..., np.ndarray],
        key_fn: Callable[[tuple, dict], str],
        dir: Path,
    ) -> None:
        self._fn = fn
        self._key_fn = key_fn
        self._dir = dir
        self.store = ArrayStore(dir)
        self.hits = 0
        self.misses = 0
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__wrapped__ = fn

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        if is_bypassed():
            view = self._validate(self._fn(*args, **kwargs)).view()
            view.setflags(write=False)
            return view
        key = self._key_fn(args, kwargs)
        if key in self.store:
            self.hits += 1
            return self.store.get(key)
        value = self._validate(self._fn(*args, **kwargs))
        self.store.put_many([(key, value)])
        self.misses += 1
        # Returning the in-memory result (not store.get) keeps cold-loop
        # reads from remapping the shard once per appended entry.
        view = value.view()
        view.setflags(write=False)
        return view

    def batch(self, calls: Iterable[tuple | dict]) -> list[np.ndarray]:
        """Compute or fetch many calls, appending all misses in one write.

        ``calls`` is a sequence of positional-argument tuples or
        keyword-argument dicts. Returns results in call order.
        """
        normalized: list[tuple[tuple, dict]] = []
        for call in calls:
            if isinstance(call, tuple):
                normalized.append((call, {}))
            elif isinstance(call, dict):
                normalized.append(((), call))
            else:
                raise TypeError(
                    "batch() calls must be tuples (positional args) or "
                    f"dicts (keyword args), got {type(call).__name__}"
                )
        if is_bypassed():
            return [
                self._validate(self._fn(*args, **kwargs))
                for args, kwargs in normalized
            ]
        keys = [self._key_fn(args, kwargs) for args, kwargs in normalized]
        pending: dict[str, np.ndarray] = {}
        for key, (args, kwargs) in zip(keys, normalized):
            if key in self.store or key in pending:
                self.hits += 1
            else:
                pending[key] = self._validate(self._fn(*args, **kwargs))
                self.misses += 1
        self.store.put_many(list(pending.items()))
        return [self.store.get(key) for key in keys]

    def clear(self) -> None:
        """Delete every stored entry for this function."""
        self.store.clear()

    def _validate(self, value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray) or value.ndim < 1:
            raise TypeError(
                f"{self.__name__} must return a numpy array with ndim >= 1, "
                f"got {type(value).__name__}"
            )
        return value
