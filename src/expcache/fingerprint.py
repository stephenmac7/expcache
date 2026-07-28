"""Stable fingerprints for arbitrary Python objects.

A fingerprint is a hex digest identifying an object for cache-key
purposes. Built-in support covers primitives, containers, numpy arrays,
and paths. Other types opt in by defining ``__fingerprint__`` (returning
any fingerprintable value) or by registering a handler with
``fingerprint.register``. Unknown types raise TypeError rather than
being pickled blindly.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path, PurePath
from typing import Any, Callable

import numpy as np


def _pack_len(n: int) -> bytes:
    return struct.pack("<Q", n)


class Fingerprinter:
    def __init__(self) -> None:
        self._registry: dict[type, Callable[[Any], Any]] = {}

    def register(self, cls: type, fn: Callable[[Any], Any] | None = None):
        """Register ``fn(obj) -> fingerprintable`` for instances of ``cls``.

        Handlers apply to subclasses too (nearest class in the MRO wins).
        Usable directly or as a decorator::

            @fingerprint.register(WavLMEncoder)
            def _(enc):
                return ("wavlm", enc.size, enc.layer)
        """
        if fn is None:
            def deco(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
                self._registry[cls] = f
                return f

            return deco
        self._registry[cls] = fn
        return fn

    def __call__(self, obj: Any) -> str:
        h = hashlib.blake2b(digest_size=16)
        self._feed(h, obj)
        return h.hexdigest()

    def _feed(self, h, obj: Any) -> None:
        tp = type(obj)

        # An object's fingerprint is the fingerprint of its declared
        # identity payload, so payloads compose transparently.
        for base in tp.__mro__:
            if base in self._registry:
                self._feed(h, self._registry[base](obj))
                return

        if hasattr(tp, "__fingerprint__"):
            self._feed(h, obj.__fingerprint__())
            return

        if obj is None:
            h.update(b"none")
        elif isinstance(obj, bool):
            h.update(b"T" if obj else b"F")
        elif isinstance(obj, int):
            data = str(obj).encode()
            h.update(b"int:" + _pack_len(len(data)) + data)
        elif isinstance(obj, float):
            h.update(b"flt:" + struct.pack("<d", obj))
        elif isinstance(obj, str):
            data = obj.encode()
            h.update(b"str:" + _pack_len(len(data)) + data)
        elif isinstance(obj, (bytes, bytearray)):
            h.update(b"byt:" + _pack_len(len(obj)) + bytes(obj))
        elif isinstance(obj, PurePath):
            # Identity is the resolved location, not the contents; use
            # file_tag() when the contents matter (e.g. model weights).
            data = str(Path(obj).resolve()).encode()
            h.update(b"pth:" + _pack_len(len(data)) + data)
        elif isinstance(obj, np.ndarray):
            arr = np.ascontiguousarray(obj)
            dt = arr.dtype.str.encode()
            h.update(b"arr:" + _pack_len(len(dt)) + dt)
            h.update(_pack_len(arr.ndim) + b"".join(_pack_len(s) for s in arr.shape))
            h.update(arr.reshape(-1).data)
        elif isinstance(obj, np.generic):
            dt = obj.dtype.str.encode()
            h.update(b"npg:" + _pack_len(len(dt)) + dt + obj.tobytes())
        elif isinstance(obj, np.dtype):
            data = obj.str.encode()
            h.update(b"dtp:" + _pack_len(len(data)) + data)
        elif isinstance(obj, (tuple, list)):
            h.update(b"seq:" + _pack_len(len(obj)))
            for item in obj:
                self._feed(h, item)
        elif isinstance(obj, dict):
            items = sorted((self(k), k, v) for k, v in obj.items())
            h.update(b"map:" + _pack_len(len(items)))
            for _, k, v in items:
                self._feed(h, k)
                self._feed(h, v)
        elif isinstance(obj, (set, frozenset)):
            digests = sorted(self(item) for item in obj)
            h.update(b"set:" + _pack_len(len(digests)))
            for d in digests:
                h.update(d.encode())
        else:
            raise TypeError(
                f"Cannot fingerprint object of type {tp.__module__}.{tp.__qualname__}. "
                "Define __fingerprint__ on the class, register a handler with "
                "fingerprint.register(cls, fn), or exclude the argument with ignore=."
            )


fingerprint = Fingerprinter()


def file_tag(path) -> tuple:
    """Cheap identity for a file's contents: resolved path + mtime + size.

    Use inside __fingerprint__ / registered handlers for things like model
    checkpoints, where the fingerprint should change when the file is
    replaced.
    """
    p = Path(path)
    st = p.stat()
    return ("file", str(p.resolve()), st.st_mtime_ns, st.st_size)
