# expcache

`expcache` caches computations on disk.

Cache results are keyed by a name, a `version` number, and hashes of the arguments.
Function code is not hashed, so caches survive refactors and change only when an
argument changes or you bump `version=`.

```python
from expcache import Cache

cache = Cache("~/.cache/my-project")
```

## Choose a cache

Use `memo()` for picklable results. Each call is stored in a SQLite-backed
[diskcache](https://grantjenks.com/docs/diskcache/) store with no eviction.

```python
@cache.memo(version=1)
def teacher_boundaries(teacher, source_key):
    ...
    return raw, snapped
```

Use `arrays()` for functions that return one NumPy array with `ndim >= 1` and
a variable first axis. Arrays with the same dtype and trailing shape share a
shard. Normal results are read-only; cache hits are zero-copy memory-map views.

```python
@cache.arrays(version=1)
def wavlm_feats(model, waveform):
    return model.extract_features(waveform)

feats = wavlm_feats(model, waveform)
all_feats = wavlm_feats.batch(
    [(model, waveform) for waveform in waveforms]
)
```

`batch()` accepts positional-argument tuples or keyword-argument dictionaries,
computes each missing key once, and returns results in input order. Copy an
array before modifying it.

## Fingerprints

`fingerprint(obj)` returns a stable hex digest. It supports primitives,
containers, paths, NumPy arrays and dtypes, and PyTorch tensors. Paths are
identified by resolved location. Arrays and tensors include their dtype, shape,
and contents; tensor device, layout, and `requires_grad` are ignored. PyTorch is
optional and is only inspected after it has been imported.

Unknown types raise `TypeError`. Define their identity on your own class or
register a handler for a third-party class:

```python
from expcache import file_tag, fingerprint

class Teacher:
    def __fingerprint__(self):
        return (file_tag(self.model_path), self.hparams)

@fingerprint.register(WavLMEncoder)
def _(encoder):
    return ("wavlm", encoder.size, encoder.layer)
```

Fingerprint payloads can contain any other supported value. `file_tag(path)`
returns the resolved path, modification time, and size, which is a cheap way to
notice replaced model files. A bare `Path` records only its location.

## Keys and cache control

A call key contains the cache name, version, and fingerprints of its bound
arguments after defaults are applied. Equivalent positional and keyword calls
therefore share an entry.

- `version=` is required. Bump it after changing the function's behavior.
- `ignore=("param",)` removes named arguments from the key. Use it only when
  those arguments cannot change the result for the remaining key.
- The cache name defaults to the function name. Set `name=` when functions
  sharing a root would otherwise collide.

Cached functions expose `.hits`, `.misses`, `.clear()`, and `.__wrapped__`.
Use `no_cache()` to compute without reading, writing, or updating counters:

```python
from expcache import no_cache

with no_cache():
    fresh = teacher_boundaries(teacher, source_key)
```

## Array provenance

Results from `arrays()` remember the call that produced them. Passing cached
arrays into another cached function fingerprints the call metadata instead of
the array bytes, which keeps large derived keys cheap. The cache root is omitted
from this metadata, so moving a cache does not invalidate downstream entries.

Slices, reshapes, ufunc results, transposes, and dtype conversions drop the
metadata and fall back to hashing their contents. Results computed inside
`no_cache()` are also untagged.

Use `tracked()` to attach the same metadata to an external array:

```python
import numpy as np

from expcache import file_tag, tracked

feats = tracked(np.load(path), ("corpus-v3", file_tag(path)))
```

The tag must identify the contents completely. Every cache key treats arrays
with the same tag as equal.

## Storage limits

```text
root/
  memo/<name>/cache.db
  arrays/<name>/index/cache.db, s0.bin, s1.bin, ...
```

The array store supports concurrent readers but only one writer. Shards are
append-only, so entries invalidated by a version bump keep using disk space
until you call `.clear()`.

## Crash recovery

Array stores check shard sizes on open and discard entries past the end of
missing or truncated files. Memo entries that fail to deserialize are discarded
on read. Discarded entries emit `CacheRepairWarning` and recompute on demand.
Missing memo files are already treated as misses by diskcache.

Recovery does not validate contents: same-size corruption and truncated raw
memo bytes or strings can go undetected. Cache writes are not guaranteed to
survive machine failure.

## Development

```sh
uv run pytest
```
