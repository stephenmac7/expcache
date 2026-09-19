"""Crash recovery: entries whose bytes were lost become misses, loudly."""

import math

import numpy as np
import pytest

from expcache import Cache, CacheRepairWarning
from expcache.arrays import ArrayStore


def make_encode(cache, calls, width=4):
    @cache.arrays(version=1)
    def encode(key, length):
        calls.append(key)
        return np.full((length, width), float(len(calls)), dtype=np.float32)

    return encode


def truncate_shard(dir, shard, rows_lost):
    """Simulate a crash that lost the tail of a shard's appended bytes."""
    store = ArrayStore(dir)
    meta = store._shards[shard]
    rowbytes = np.dtype(meta["dtype"]).itemsize * math.prod(meta["trailing"])
    path = dir / f"{shard}.bin"
    with open(path, "r+b") as f:
        f.truncate(path.stat().st_size - rows_lost * rowbytes)


# --- arrays ---------------------------------------------------------------


def test_lost_shard_tail_is_recomputed(tmp_path):
    calls = []
    encode = make_encode(Cache(tmp_path), calls)
    for i in range(10):
        encode(f"k{i}", 5)
    assert len(calls) == 10

    truncate_shard(tmp_path / "arrays" / "encode", "s0", rows_lost=10)

    calls.clear()
    with pytest.warns(CacheRepairWarning, match="dropped 2 cached arrays"):
        encode = make_encode(Cache(tmp_path), calls)  # the sweep runs on open

    # The two entries in the lost rows recompute; the other eight still hit.
    for i in range(10):
        encode(f"k{i}", 5)
    assert calls == ["k8", "k9"]


def test_intact_entries_in_a_damaged_shard_still_read(tmp_path):
    calls = []
    encode = make_encode(Cache(tmp_path), calls)
    expected = {f"k{i}": np.array(encode(f"k{i}", 5)) for i in range(10)}

    truncate_shard(tmp_path / "arrays" / "encode", "s0", rows_lost=10)

    with pytest.warns(CacheRepairWarning):
        encode = make_encode(Cache(tmp_path), [])
    # Without the sweep, mapping the shard at the row count the index still
    # claimed would raise for every entry in it, including untouched ones.
    for i in range(8):
        np.testing.assert_array_equal(encode(f"k{i}", 5), expected[f"k{i}"])


def test_repair_reuses_the_lost_region(tmp_path):
    encode = make_encode(Cache(tmp_path), [])
    for i in range(10):
        encode(f"k{i}", 5)
    size_before = (tmp_path / "arrays" / "encode" / "s0.bin").stat().st_size

    truncate_shard(tmp_path / "arrays" / "encode", "s0", rows_lost=10)
    with pytest.warns(CacheRepairWarning):
        encode = make_encode(Cache(tmp_path), [])
    for i in range(10):
        encode(f"k{i}", 5)

    # Recomputed rows land in the reclaimed space, not past it.
    assert (tmp_path / "arrays" / "encode" / "s0.bin").stat().st_size == size_before


def test_repair_runs_once_and_is_silent_when_healthy(tmp_path):
    import warnings

    encode = make_encode(Cache(tmp_path), [])
    for i in range(10):
        encode(f"k{i}", 5)

    truncate_shard(tmp_path / "arrays" / "encode", "s0", rows_lost=10)
    with pytest.warns(CacheRepairWarning):
        ArrayStore(tmp_path / "arrays" / "encode")

    # The sweep committed, so the damage is gone and later opens say nothing --
    # whether or not the dropped entries are ever recomputed.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ArrayStore(tmp_path / "arrays" / "encode")
        encode = make_encode(Cache(tmp_path), [])
        for i in range(10):
            encode(f"k{i}", 5)
        ArrayStore(tmp_path / "arrays" / "encode")


def test_dropped_entry_is_not_resurrected_by_later_appends(tmp_path):
    """A dropped row must not outlive the sweep.

    If it did, appends that grow the shard back past the row range it names
    would leave it pointing at another entry's bytes.
    """
    values = {}

    @Cache(tmp_path).arrays(version=1)
    def encode(key):
        return np.full((10, 4), float(key), dtype=np.float32)

    for i in range(100):
        values[i] = np.array(encode(i))
    truncate_shard(tmp_path / "arrays" / "encode", "s0", rows_lost=50)  # loses 95..99

    # This run needs only one of the lost entries; 96..99 stay dropped.
    with pytest.warns(CacheRepairWarning, match="dropped 5 cached arrays"):
        encode = Cache(tmp_path).arrays(version=1)(encode.__wrapped__)
    encode(95)

    # Unrelated work grows the shard back past where 96..99 used to live.
    for i in range(100, 130):
        encode(i)

    # A fresh open must still recompute them, not read the new rows.
    encode = Cache(tmp_path).arrays(version=1)(encode.__wrapped__)
    for i in range(100):
        np.testing.assert_array_equal(encode(i), values[i])


def test_missing_shard_file_drops_everything_in_it(tmp_path):
    calls = []
    encode = make_encode(Cache(tmp_path), calls)
    for i in range(5):
        encode(f"k{i}", 5)
    (tmp_path / "arrays" / "encode" / "s0.bin").unlink()

    calls.clear()
    with pytest.warns(CacheRepairWarning, match="dropped 5 cached arrays"):
        encode = make_encode(Cache(tmp_path), calls)
    for i in range(5):
        encode(f"k{i}", 5)
    assert calls == [f"k{i}" for i in range(5)]


def test_untouched_shards_survive_a_damaged_neighbour(tmp_path):
    calls = []

    def build():
        @Cache(tmp_path).arrays(version=1)
        def encode(key, width):
            calls.append(key)
            return np.ones((5, width), dtype=np.float32)

        return encode

    encode = build()
    encode("a", 4)
    encode("b", 8)  # different trailing shape -> its own shard
    assert len(encode.store._shards) == 2

    truncate_shard(tmp_path / "arrays" / "encode", "s0", rows_lost=5)

    calls.clear()
    with pytest.warns(CacheRepairWarning, match="dropped 1 cached array"):
        encode = build()
    encode("a", 4)
    encode("b", 8)
    assert calls == ["a"]  # only the damaged shard's entry was lost


# --- memo -----------------------------------------------------------------


def big_value():
    # Over disk_min_file_size (32 KiB), so diskcache gives it its own file.
    return list(range(20_000))


def value_files(dir):
    return sorted(dir.rglob("*.val"))


def test_large_memo_values_get_their_own_file(tmp_path):
    @Cache(tmp_path).memo(version=1)
    def compute(key):
        return big_value()

    compute("a")
    assert len(value_files(tmp_path / "memo" / "compute")) == 1


def test_truncated_memo_value_is_recomputed(tmp_path):
    calls = []

    @Cache(tmp_path).memo(version=1)
    def compute(key):
        calls.append(key)
        return big_value()

    compute("a")
    compute("b")
    assert calls == ["a", "b"]

    # Lose one value file's contents, as an eviction mid-write would.
    (files := value_files(tmp_path / "memo" / "compute"))[0].write_bytes(b"")
    assert len(files) == 2

    calls.clear()
    with pytest.warns(CacheRepairWarning, match="could not be read back"):
        compute("a"), compute("b")
    assert len(calls) == 1  # only the damaged one recomputed

    # The row was dropped and rewritten, so the next call is a clean hit.
    calls.clear()
    compute("a"), compute("b")
    assert calls == []


def test_missing_memo_value_file_is_already_a_miss(tmp_path):
    """diskcache turns a vanished value file into a miss on its own.

    Only a file that exists but cannot be unpickled reaches our handler,
    so this case recomputes without a warning.
    """
    import warnings

    calls = []

    @Cache(tmp_path).memo(version=1)
    def compute(key):
        calls.append(key)
        return big_value()

    compute("a")
    value_files(tmp_path / "memo" / "compute")[0].unlink()

    calls.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", CacheRepairWarning)
        assert compute("a") == big_value()
    assert calls == ["a"]


def test_small_memo_values_are_unaffected(tmp_path):
    @Cache(tmp_path).memo(version=1)
    def compute(key):
        return {"k": key}

    compute("a")
    assert value_files(tmp_path / "memo" / "compute") == []


def test_reads_a_cache_written_by_the_previous_implementation(tmp_path):
    """The on-disk format is unchanged, so existing caches carry over."""
    import diskcache

    from expcache.cache import _CallKey
    from expcache.fingerprint import fingerprint

    @Cache(tmp_path).memo(version=1)
    def compute(key):
        raise AssertionError("should have hit the entry written by Index")

    key = _CallKey(compute.__wrapped__, "compute", 1, (), fingerprint)(("a",), {})
    diskcache.Index(str(tmp_path / "memo" / "compute"))[key] = big_value()
    assert compute("a") == big_value()


def test_shard_stat_failure_preserves_entries(tmp_path, monkeypatch):
    from pathlib import Path

    store = ArrayStore(tmp_path)
    store.put_many([("a", np.ones((2, 4)))])
    original_stat = Path.stat

    def stat(path, *args, **kwargs):
        if path.suffix == ".bin":
            raise PermissionError("access denied")
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", stat)
        with pytest.raises(PermissionError, match="access denied"):
            ArrayStore(tmp_path)
    assert "a" in ArrayStore(tmp_path)


def test_memo_deletion_failure_propagates(tmp_path, monkeypatch):
    import diskcache

    @Cache(tmp_path).memo(version=1)
    def compute():
        return big_value()

    compute()
    value_files(tmp_path)[0].write_bytes(b"")

    def fail_delete(self, key):
        raise PermissionError("cannot delete")

    monkeypatch.setattr(diskcache.Index, "__delitem__", fail_delete)
    with pytest.raises(PermissionError, match="cannot delete"):
        compute()
