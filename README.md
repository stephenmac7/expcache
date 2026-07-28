# expcache

A small caching framework for experiment notebooks: one key scheme, two
storage tiers. Keys are built from **explicit versions** plus
**fingerprints of the arguments** — code is never hashed, so caches
survive refactors and invalidate only when you bump `version=`.

```python
from expcache import Cache, fingerprint, file_tag

cache = Cache("~/.cache/my-project")
```

## Fingerprints

`fingerprint(obj)` maps an object to a stable hex digest. Primitives,
tuples/lists, dicts, sets, `Path`s (by resolved location), numpy arrays
(by dtype/shape/contents), numpy dtypes, and torch tensors work out of
the box. Tensors are identified by dtype, shape, and contents — device,
layout, and `requires_grad` are excluded, so the same weights on CPU and
GPU fingerprint alike, and a `state_dict` composes through the dict
branch. Torch is a soft dependency: if it was never imported, nothing
here touches it.

**Unknown types raise `TypeError`** instead of being pickled blindly —
teach the fingerprinter about your types one of two ways:

```python
class Teacher:
    def __fingerprint__(self):                     # on your own classes
        return (file_tag(self.model_path), self.hparams)

@fingerprint.register(WavLMEncoder)                # for third-party types
def _(enc):
    return ("wavlm", enc.size, enc.layer)
```

An object's fingerprint is the fingerprint of its declared payload, so
payloads compose (e.g. a teacher's payload can include
`fingerprint(model)`). `file_tag(path)` captures a file's *contents*
cheaply (resolved path + mtime + size) — use it for model checkpoints;
bare `Path` arguments are identified by location only.

## `memo`: derived results

One pickled entry per call, stored in a single SQLite-backed
[diskcache](https://grantjenks.com/docs/diskcache/) index (no eviction).
For alignments, boundaries, metrics, fitted models — anything picklable.

```python
@cache.memo(version=1)
def teacher_boundaries(teacher, source_key):
    ...
    return raw, snapped
```

## `arrays`: heavy encoders

For per-item functions returning a single numpy array (`ndim >= 1`,
variable first axis). Results are stacked along axis 0 into large shard
files — one shard per (dtype, trailing-shape) — with a diskcache index
of row ranges. Cache hits are **zero-copy read-only views of a memory map**, so
sweeping a warm corpus is one sequential read, not per-file
deserialization.

```python
@cache.arrays(version=1, ignore=("load_waveform",))
def wavlm_feats(model, source_key, load_waveform):
    return model.extract_features(load_waveform())

feats = wavlm_feats(phone_model, utt.key, utt.loader)          # (T, 1024) view
all_feats = wavlm_feats.batch(                                 # one index write,
    [(phone_model, u.key, u.loader) for u in utterances]       # computes only misses
)
```

`batch()` takes tuples (positional args) or dicts (kwargs) and returns
results in order. Copy before mutating a hit — views are read-only.

## Provenance: caching functions of cached things

Results from `arrays()` remember the call that produced them, so they
can be *arguments* to another cached function without being hashed:

```python
@cache.memo(version=1)
def train(corpus, seed):        # corpus is 12 GiB of cached features
    ...

train([wavlm_feats(enc, u.key, u.loader) for u in utterances], seed=0)
```

The key here costs a few hundred bytes of hashing, not 12 GiB, because
each array fingerprints as *"the result of `wavlm_feats` v1 on this
utterance"* rather than as its bytes. The tag omits the cache root, so
relocating a cache directory doesn't invalidate anything derived from it.

Any array derived from a tagged one — a slice, a ufunc result, a
reshape, an `astype` that actually converts — silently drops the tag and
falls back to hashing contents, so a tag can never outlive its truth.
Inside `no_cache()` there is no key, so results come back untagged too.

Use `tracked(array, provenance)` to tag arrays from elsewhere. The tag
must identify the contents *completely* — two arrays sharing a tag are
treated as equal by every key that sees them:

```python
feats = tracked(np.load(path), ("corpus-v3", file_tag(path)))
```

## Keys, versions, ignore

The key for a call is `(cache name, version, fingerprints of bound
arguments)`, with defaults applied — `fn(1)`, `fn(1, 10)`, and
`fn(a=1, b=10)` share an entry.

- `version=` is required. Bump it when the function's logic changes;
  there is no automatic code hashing by design.
- `ignore=("param",)` excludes arguments from the key — for loader
  callables and other things that are determined by the real key
  arguments. The contract is yours to keep: ignored args must not affect
  the result given the remaining ones.
- The cache name defaults to the function name (`name=` overrides).
  Functions sharing a root must have distinct names.

Every cached function exposes `.hits` / `.misses` counters, `.clear()`,
and the original function as `.__wrapped__`.

## Layout & limitations

```
root/
  memo/<name>/cache.db
  arrays/<name>/index/cache.db, s0.bin, s1.bin, ...
```

Single-writer: don't append from multiple processes at once (concurrent
readers of a warm cache are fine). Shards are append-only; entries
orphaned by a version bump keep their disk space until you `clear()`.

## Development

```sh
uv run --with pytest pytest
```
