"""Properties of fingerprints, persistence, and cache recovery."""

from __future__ import annotations

import copy
import math
import shutil
import tempfile
import uuid
import warnings
from pathlib import Path, PurePath

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from expcache import Cache, CacheRepairWarning, fingerprint, no_cache
from expcache.arrays import ArrayStore

# --- fingerprints ---------------------------------------------------------

DTYPES = ["float32", "float64", "int16", "uint8"]


def np_arrays():
    shapes = st.sampled_from([(0,), (1,), (3,), (2, 2), (2, 0)])
    return st.builds(
        lambda dtype, shape, seed: (
            np.arange(math.prod(shape), dtype=np.int64).reshape(shape) + seed
        ).astype(dtype),
        st.sampled_from(DTYPES),
        shapes,
        st.integers(min_value=0, max_value=3),
    )


def values(depth=2):
    scalars = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(max_size=8),
        st.binary(max_size=8),
        st.builds(PurePath, st.sampled_from(["a", "a/b", "c"])),
        np_arrays(),
    )
    if depth <= 0:
        return scalars
    inner = values(depth - 1)
    return st.one_of(
        scalars,
        st.lists(inner, max_size=3),
        st.tuples(inner, inner),
        st.dictionaries(st.text(max_size=4), inner, max_size=3),
    )


@given(values())
def test_fingerprint_is_deterministic(value):
    assert fingerprint(value) == fingerprint(value)


@given(values())
def test_copying_a_value_preserves_its_fingerprint(value):
    assert fingerprint(copy.deepcopy(value)) == fingerprint(value)


@given(values())
def test_wrapping_a_value_changes_its_fingerprint(value):
    assert fingerprint(value) != fingerprint([value])


@given(st.lists(values(), max_size=3), values())
def test_extending_a_sequence_changes_its_fingerprint(xs, extra):
    assert fingerprint(xs) != fingerprint(xs + [extra])


@given(st.lists(values(), max_size=3), st.lists(values(), max_size=3))
def test_concatenation_is_not_confusable_with_nesting(xs, ys):
    assert fingerprint(xs + ys) != fingerprint([xs, ys])


@given(st.text(min_size=1, max_size=8), st.data())
def test_adjacent_strings_cannot_borrow_characters(text, data):
    i = data.draw(st.integers(min_value=0, max_value=len(text)))
    j = data.draw(st.integers(min_value=0, max_value=len(text)))
    assume(i != j)
    assert fingerprint([text[:i], text[i:]]) != fingerprint([text[:j], text[j:]])


@given(st.binary(min_size=1, max_size=8), st.data())
def test_adjacent_bytes_cannot_borrow_bytes(blob, data):
    i = data.draw(st.integers(min_value=0, max_value=len(blob)))
    j = data.draw(st.integers(min_value=0, max_value=len(blob)))
    assume(i != j)
    assert fingerprint([blob[:i], blob[i:]]) != fingerprint([blob[:j], blob[j:]])


@given(st.lists(values(), max_size=3), values())
def test_nesting_depth_is_unambiguous(xs, y):
    assert fingerprint([xs, y]) != fingerprint([xs + [y]])


@given(np_arrays())
def test_reshaping_an_array_changes_its_fingerprint(arr):
    assume(arr.ndim > 1 or arr.size > 1)
    assert fingerprint(arr) != fingerprint(arr.reshape(-1, 1))


# --- the array store ------------------------------------------------------

KEY_NAMES = [f"k{i}" for i in range(6)]
KEYS = st.sampled_from(KEY_NAMES)


def payloads():
    return st.builds(
        lambda dtype, trailing, rows, seed: (
            np.arange(rows * math.prod(trailing) if trailing else rows, dtype=np.int64)
            + seed
        )
        .reshape((rows, *trailing))
        .astype(dtype),
        st.sampled_from(DTYPES),
        st.sampled_from([(), (1,), (3,), (0,)]),
        st.integers(min_value=0, max_value=4),
        st.integers(min_value=0, max_value=5),
    )


def damage_array_files(directory, data, operation):
    """Storage-specific fault injection; offsets are arbitrary bytes."""
    paths = sorted(directory.glob("*.bin"))
    if not paths:
        return
    path = data.draw(st.sampled_from(paths))
    if operation == "remove":
        path.unlink()
    elif operation == "truncate":
        keep = data.draw(st.integers(0, path.stat().st_size))
        with path.open("r+b") as stream:
            stream.truncate(keep)
    else:
        with path.open("ab") as stream:
            stream.write(data.draw(st.binary(min_size=1, max_size=100)))


class ArrayStoreMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.dir = Path(tempfile.mkdtemp(prefix="expcache-prop-"))
        self.store = ArrayStore(self.dir)
        self.model: dict[str, np.ndarray] = {}
        self.required: set[str] = set()

    def teardown(self):
        shutil.rmtree(self.dir)

    def _restart(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CacheRepairWarning)
            self.store = ArrayStore(self.dir)

    @rule(items=st.lists(st.tuples(KEYS, payloads()), min_size=1, max_size=3))
    def write(self, items):
        deduped = dict(items)
        self.store.put_many(list(deduped.items()))
        self.model.update(deduped)
        self.required.update(deduped)

    @rule()
    def restart(self):
        self._restart()

    @rule(data=st.data(), operation=st.sampled_from(["truncate", "remove", "append"]))
    def crash(self, data, operation):
        damage_array_files(self.dir, data, operation)
        self._restart()
        if operation != "append":
            # A destructive crash may lose entries; survivors must persist afterward.
            self.required = {key for key in self.model if key in self.store}

    @invariant()
    def required_entries_remain_available(self):
        for key in self.required:
            assert key in self.store, key

    @invariant()
    def reads_are_never_silently_wrong(self):
        for key, expected in self.model.items():
            if key in self.store:
                got = self.store.get(key)
                assert got.dtype == expected.dtype, key
                np.testing.assert_array_equal(got, expected, err_msg=key)

    @invariant()
    def the_store_invents_nothing(self):
        present = {key for key in KEY_NAMES if key in self.store}
        assert present <= self.model.keys()
        assert len(self.store) == len(present)


TestArrayStore = ArrayStoreMachine.TestCase
TestArrayStore.settings = settings(
    max_examples=150,
    stateful_step_count=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# --- the cached decorators ------------------------------------------------


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(calls=st.lists(st.tuples(KEYS, payloads()), min_size=1, max_size=6))
def test_arrays_decorator_computes_each_key_once(tmp_path, calls):
    # tmp_path is shared across Hypothesis examples.
    table = dict(calls)
    computed = []

    @Cache(tmp_path / uuid.uuid4().hex).arrays(version=1)
    def encode(key):
        computed.append(key)
        return table[key]

    for key in list(table) * 2:
        np.testing.assert_array_equal(encode(key), table[key])
    assert computed == list(table)
    assert (encode.hits, encode.misses) == (len(table), len(table))


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(value=values(), key=st.text(max_size=5))
def test_memo_round_trips_any_fingerprintable_value(tmp_path, value, key):
    calls = []

    @Cache(tmp_path / uuid.uuid4().hex).memo(version=1)
    def compute(k):
        calls.append(k)
        return value

    first, second = compute(key), compute(key)
    assert calls == [key]
    assert fingerprint(first) == fingerprint(value)
    assert fingerprint(second) == fingerprint(value)


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(a=values(), b=values())
def test_distinct_arguments_do_not_share_a_memo_entry(tmp_path, a, b):
    assume(fingerprint(a) != fingerprint(b))
    calls = []

    @Cache(tmp_path / uuid.uuid4().hex).memo(version=1)
    def compute(x):
        calls.append(x)
        return len(calls)

    assert compute(a) != compute(b)


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(calls=st.lists(KEYS, min_size=1, max_size=5))
def test_no_cache_never_touches_the_store(tmp_path, calls):
    computed = []

    @Cache(tmp_path / uuid.uuid4().hex).arrays(version=1)
    def encode(key):
        computed.append(key)
        return np.ones((2, 2), dtype=np.float32)

    with no_cache():
        for key in calls:
            encode(key)
    assert computed == calls
    assert (encode.hits, encode.misses) == (0, 0)
    assert len(encode.store) == 0
