import numpy as np

from expcache import Cache, no_cache


def test_memo_bypass_skips_read_and_write(tmp_path):
    cache = Cache(tmp_path)
    calls = []

    @cache.memo(version=1)
    def add(a, b):
        calls.append((a, b))
        return a + b

    add(1, 2)  # populate
    assert (add.hits, add.misses) == (0, 1)

    with no_cache():
        assert add(1, 2) == 3  # recomputed, not a cache hit
        assert add(4, 5) == 9  # novel key, not stored
    assert (add.hits, add.misses) == (0, 1)  # counters untouched inside block
    assert calls == [(1, 2), (1, 2), (4, 5)]

    # The novel key computed under bypass was never persisted.
    assert add(4, 5) == 9
    assert add.misses == 2


def test_arrays_bypass_recomputes_and_stays_readonly(tmp_path):
    cache = Cache(tmp_path)
    calls = []

    @cache.arrays(version=1)
    def feat(n):
        calls.append(n)
        return np.arange(n, dtype=np.float32).reshape(-1, 1)

    stored = feat(3)
    assert (feat.hits, feat.misses) == (0, 1)

    with no_cache():
        got = feat(3)
        assert np.array_equal(got, stored)
        assert not got.flags.writeable
    assert (feat.hits, feat.misses) == (0, 1)
    assert calls == [3, 3]


def test_arrays_batch_bypass(tmp_path):
    cache = Cache(tmp_path)
    calls = []

    @cache.arrays(version=1)
    def feat(n):
        calls.append(n)
        return np.arange(n, dtype=np.float32).reshape(-1, 1)

    with no_cache():
        out = feat.batch([(2,), (4,)])
    assert [o.shape[0] for o in out] == [2, 4]
    assert calls == [2, 4]
    # Nothing was persisted, so a real call afterwards still misses.
    feat(2)
    assert feat.misses == 1


def test_bypass_scope_restores_on_exit(tmp_path):
    cache = Cache(tmp_path)
    calls = []

    @cache.memo(version=1)
    def fn(x):
        calls.append(x)
        return x

    fn(1)
    with no_cache():
        fn(1)
    fn(1)  # back to normal: served from cache
    assert calls == [1, 1]  # only the pre- and in-bypass calls recomputed
