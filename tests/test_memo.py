import pytest

from expcache import Cache


def test_memoizes_and_persists(tmp_path):
    calls = []

    def make(cache):
        @cache.memo(version=1)
        def add(a, b):
            calls.append((a, b))
            return a + b

        return add

    add = make(Cache(tmp_path))
    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert add(1, 3) == 4
    assert calls == [(1, 2), (1, 3)]
    assert (add.hits, add.misses) == (1, 2)

    # A fresh Cache over the same root hits without recomputing.
    add2 = make(Cache(tmp_path))
    assert add2(1, 2) == 3
    assert calls == [(1, 2), (1, 3)]
    assert (add2.hits, add2.misses) == (1, 0)


def test_version_bump_invalidates(tmp_path):
    cache = Cache(tmp_path)
    calls = []

    def compute(x):
        calls.append(x)
        return x * 2

    v1 = cache.memo(version=1, name="compute")(compute)
    v1(5)
    v2 = cache.memo(version=2, name="compute")(compute)
    assert v2(5) == 10
    assert calls == [5, 5]


def test_ignore_excludes_argument(tmp_path):
    cache = Cache(tmp_path)

    @cache.memo(version=1, ignore=("loader",))
    def load(key, loader):
        return loader()

    assert load("a", lambda: 1) == 1
    # Different loader, same key: cached value wins.
    assert load("a", lambda: 2) == 1
    assert load("b", lambda: 2) == 2


def test_ignore_unknown_param_raises(tmp_path):
    cache = Cache(tmp_path)
    with pytest.raises(TypeError, match="typo"):

        @cache.memo(version=1, ignore=("typo",))
        def fn(x):
            return x


def test_defaults_and_call_styles_share_keys(tmp_path):
    cache = Cache(tmp_path)
    calls = []

    @cache.memo(version=1)
    def fn(a, b=10):
        calls.append((a, b))
        return a + b

    assert fn(1) == fn(1, 10) == fn(a=1, b=10) == 11
    assert calls == [(1, 10)]


def test_arbitrary_return_types(tmp_path):
    cache = Cache(tmp_path)

    @cache.memo(version=1)
    def pair(x):
        return {"raw": [x, x + 1], "snapped": (x,)}

    first = pair(3)
    again = pair(3)
    assert first == again == {"raw": [3, 4], "snapped": (3,)}


def test_clear(tmp_path):
    cache = Cache(tmp_path)
    calls = []

    @cache.memo(version=1)
    def fn(x):
        calls.append(x)
        return x

    fn(1)
    fn.clear()
    fn(1)
    assert calls == [1, 1]
