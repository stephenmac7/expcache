"""Stable fingerprints for arbitrary Python objects.

A fingerprint is a hex digest identifying an object for cache-key
purposes. Built-in support covers primitives, containers, numpy arrays,
paths, and torch tensors. Other types opt in by defining
``__fingerprint__`` (returning any fingerprintable value) or by
registering a handler with ``fingerprint.register``. Unknown types raise
TypeError rather than being pickled blindly.

Arrays carrying a provenance tag (see ``tracked``) are identified by that
tag rather than by their contents, so a cached corpus can be a cache-key
argument without being hashed byte by byte.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path, PurePath
from typing import Any, Callable

import numpy as np

#: Attribute holding an array's provenance tag. Any ndarray subclass may
#: set it; ``None`` means "identify me by my contents".
PROVENANCE_ATTR = "expcache_provenance"


def _pack_len(n: int) -> bytes:
    return struct.pack("<Q", n)


class TrackedArray(np.ndarray):
    """A numpy array that remembers which cached call produced it.

    Fingerprinting one hashes its provenance tag instead of its bytes,
    which is what makes a multi-gigabyte cached corpus usable as a
    cache-key argument.

    Any array *derived* from a tracked one -- a slice, a ufunc result, a
    reshape, an ``astype`` that actually converts -- drops the tag and
    falls back to hashing contents, because its bytes are no longer what
    the tagged call returned. Only an untouched result stays tagged.
    """

    def __array_finalize__(self, obj: Any) -> None:
        # Runs for every array built from this one as a template. The tag
        # is never inherited; ``tracked`` re-applies it explicitly.
        setattr(self, PROVENANCE_ATTR, None)

    def __reduce__(self):
        fn, args, state = super().__reduce__()
        return fn, args, (state, getattr(self, PROVENANCE_ATTR, None))

    def __setstate__(self, state):
        base, provenance = state
        super().__setstate__(base)
        setattr(self, PROVENANCE_ATTR, provenance)


def tracked(array: np.ndarray, provenance: Any) -> TrackedArray:
    """Return a read-only view of ``array`` tagged with ``provenance``.

    ``provenance`` must itself be fingerprintable, and must identify the
    contents *completely*: two arrays sharing a tag are treated as equal
    by every cache key that sees them. A cached call's key is the natural
    tag; anything weaker silently conflates results.
    """
    if provenance is None:
        raise ValueError("provenance must not be None")
    view = array.view(TrackedArray)
    setattr(view, PROVENANCE_ATTR, provenance)
    view.setflags(write=False)
    return view


def _is_torch_tensor(obj: Any) -> bool:
    # If torch was never imported, obj cannot be a tensor -- so this stays
    # a soft dependency.
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(obj, torch.Tensor)


def _torch_payload(obj: Any) -> tuple:
    """Identity of a tensor: its dtype, shape, and bytes.

    Device and ``requires_grad`` are deliberately excluded -- the same
    weights on CPU and GPU are the same weights.
    """
    torch = sys.modules["torch"]
    tensor = obj.detach().cpu().contiguous()
    try:
        data = tensor.numpy()
    except TypeError:
        # dtypes numpy has no equivalent for (bfloat16, the float8s).
        data = tensor.view(torch.uint8).numpy()
    return ("torch.Tensor", str(tensor.dtype), tuple(tensor.shape), data)


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
            provenance = getattr(obj, PROVENANCE_ATTR, None)
            if provenance is not None:
                h.update(b"prv:")
                self._feed(h, provenance)
                return
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
        elif _is_torch_tensor(obj):
            self._feed(h, _torch_payload(obj))
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
