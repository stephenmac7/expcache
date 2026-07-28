import numpy as np
import pytest

from expcache import Cache


def make_encode(cache, calls, version=1):
    @cache.arrays(version=version)
    def encode(key, length):
        calls.append(key)
        return np.full((length, 4), float(length), dtype=np.float32) + np.arange(
            length, dtype=np.float32
        ).reshape(-1, 1)

    return encode


def test_roundtrip_ragged(tmp_path):
    calls = []
    encode = make_encode(Cache(tmp_path), calls)
    a = encode("a", 5)
    b = encode("b", 3)
    assert a.shape == (5, 4) and b.shape == (3, 4)
    np.testing.assert_array_equal(encode("a", 5), a)
    assert calls == ["a", "b"]
    assert (encode.hits, encode.misses) == (1, 2)


def test_hits_are_memmap_views(tmp_path):
    encode = make_encode(Cache(tmp_path), [])
    encode("a", 5)
    result = encode("a", 5)
    assert isinstance(result.base, np.memmap)
    with pytest.raises(ValueError):
        result[0, 0] = 99.0  # read-only


def test_persists_across_reopen(tmp_path):
    calls = []
    encode = make_encode(Cache(tmp_path), calls)
    expected = np.asarray(encode("a", 5))

    calls2 = []
    encode2 = make_encode(Cache(tmp_path), calls2)
    np.testing.assert_array_equal(encode2("a", 5), expected)
    assert calls2 == []
    assert encode2("a", 5).dtype == np.float32


def test_batch_computes_only_missing(tmp_path):
    calls = []
    encode = make_encode(Cache(tmp_path), calls)
    encode("a", 5)
    results = encode.batch([("a", 5), ("b", 3), ("c", 2), ("b", 3)])
    assert [r.shape[0] for r in results] == [5, 3, 2, 3]
    assert calls == ["a", "b", "c"]
    np.testing.assert_array_equal(results[1], results[3])
    np.testing.assert_array_equal(results[1], encode("b", 3))
    # kwargs-style calls hit the same entries.
    np.testing.assert_array_equal(
        encode.batch([{"key": "c", "length": 2}])[0], results[2]
    )
    assert calls == ["a", "b", "c"]


def test_batch_rejects_bare_args(tmp_path):
    encode = make_encode(Cache(tmp_path), [])
    with pytest.raises(TypeError, match="tuples"):
        encode.batch(["a"])


def test_mixed_signatures_get_separate_shards(tmp_path):
    cache = Cache(tmp_path)

    @cache.arrays(version=1)
    def feats(key, length, dim, dtype):
        return np.arange(length * dim, dtype=dtype).reshape(length, dim)

    f32, i64 = np.dtype(np.float32), np.dtype(np.int64)
    wide = feats("a", 3, 8, f32)
    narrow = feats("b", 5, 2, f32)
    ints = feats("c", 4, 2, i64)
    np.testing.assert_array_equal(feats("a", 3, 8, f32), wide)
    np.testing.assert_array_equal(feats("b", 5, 2, f32), narrow)
    np.testing.assert_array_equal(feats("c", 4, 2, i64), ints)
    assert feats("c", 4, 2, i64).dtype == np.int64


def test_one_dimensional_and_empty(tmp_path):
    cache = Cache(tmp_path)

    @cache.arrays(version=1)
    def boundaries(key, n):
        return np.arange(n, dtype=np.int64)

    np.testing.assert_array_equal(boundaries("a", 4), np.arange(4))
    empty = boundaries("b", 0)
    assert empty.shape == (0,) and empty.dtype == np.int64
    np.testing.assert_array_equal(boundaries("a", 4), np.arange(4))
    np.testing.assert_array_equal(boundaries("b", 0), empty)


def test_non_array_return_raises(tmp_path):
    cache = Cache(tmp_path)

    @cache.arrays(version=1)
    def bad(x):
        return [1, 2, 3]

    with pytest.raises(TypeError, match="numpy array"):
        bad(1)

    @cache.arrays(version=1)
    def scalar(x):
        return np.float32(1.0)

    with pytest.raises(TypeError, match="ndim"):
        scalar(1)


def test_version_bump_invalidates(tmp_path):
    cache = Cache(tmp_path)
    calls = []
    v1 = make_encode(cache, calls, version=1)
    v1("a", 5)
    v2 = make_encode(cache, calls, version=2)
    v2("a", 5)
    assert calls == ["a", "a"]


def test_clear(tmp_path):
    calls = []
    encode = make_encode(Cache(tmp_path), calls)
    encode("a", 5)
    encode.clear()
    np.testing.assert_array_equal(encode("a", 5)[:, 0], np.arange(5) + 5.0)
    assert calls == ["a", "a"]


def test_results_carry_provenance(tmp_path):
    from expcache import fingerprint

    cache = Cache(tmp_path)
    calls = []
    encode = make_encode(cache, calls)

    miss = encode("a", 5)
    hit = encode("a", 5)
    # A miss and a hit for the same call are interchangeable as cache-key
    # arguments, even though one is fresh memory and one is a memmap view.
    assert fingerprint(miss) == fingerprint(hit)
    assert fingerprint(encode("b", 5)) != fingerprint(miss)
    # Provenance follows the call, not the contents: "b" and "c" here
    # produce identical arrays but must not share downstream keys.
    np.testing.assert_array_equal(encode("b", 5), encode("c", 5))
    assert fingerprint(encode("b", 5)) != fingerprint(encode("c", 5))
    assert fingerprint(encode.batch([("a", 5)])[0]) == fingerprint(miss)


def test_provenance_survives_version_and_root_moves(tmp_path):
    from expcache import fingerprint

    first = make_encode(Cache(tmp_path / "one"), [])("a", 5)
    # Same call, same version, a different cache root: relocating a cache
    # must not invalidate keys derived from what it holds.
    assert fingerprint(make_encode(Cache(tmp_path / "two"), [])("a", 5)) == fingerprint(
        first
    )
    bumped = make_encode(Cache(tmp_path / "three"), [], version=2)("a", 5)
    assert fingerprint(bumped) != fingerprint(first)


def test_no_cache_results_are_untagged(tmp_path):
    from expcache import fingerprint, no_cache

    encode = make_encode(Cache(tmp_path), [])
    with no_cache():
        bypassed = encode("a", 5)
    # No key exists inside no_cache(), so results fall back to contents.
    assert fingerprint(bypassed) == fingerprint(np.asarray(bypassed))
    assert fingerprint(bypassed) != fingerprint(encode("a", 5))
