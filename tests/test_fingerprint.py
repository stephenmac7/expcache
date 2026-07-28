import pickle
from pathlib import Path

import numpy as np
import pytest

from expcache import Fingerprinter, file_tag, fingerprint, tracked


def test_primitives_stable_and_distinct():
    assert fingerprint(1) == fingerprint(1)
    assert fingerprint(1) != fingerprint(2)
    assert fingerprint(True) != fingerprint(1)
    assert fingerprint(1) != fingerprint(1.0)
    assert fingerprint("1") != fingerprint(1)
    assert fingerprint("ab") != fingerprint(b"ab")
    assert fingerprint(None) != fingerprint(0)


def test_containers():
    assert fingerprint((1, 2)) == fingerprint([1, 2])
    assert fingerprint((1, 2)) != fingerprint((2, 1))
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1, "b": 2}) != fingerprint({"a": 2, "b": 1})
    assert fingerprint({1, 2, 3}) == fingerprint({3, 2, 1})
    # Nesting doesn't flatten ambiguously.
    assert fingerprint([[1], [2]]) != fingerprint([[1, 2]])


def test_ndarray():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert fingerprint(a) == fingerprint(a.copy())
    assert fingerprint(a) != fingerprint(a.astype(np.float64))
    assert fingerprint(a) != fingerprint(a.reshape(3, 2))
    b = a.copy()
    b[0, 0] += 1
    assert fingerprint(a) != fingerprint(b)
    # Non-contiguous views fingerprint by content.
    assert fingerprint(a.T) == fingerprint(np.ascontiguousarray(a.T))
    assert fingerprint(np.float32(1.5)) == fingerprint(np.float32(1.5))
    assert fingerprint(np.float32(1.5)) != fingerprint(np.float64(1.5))


def test_tracked_identifies_by_tag_not_contents():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.zeros((100, 3), dtype=np.float32)
    # The whole point: contents are never read, so a tag stands in for
    # arrays too large to hash.
    assert fingerprint(tracked(a, ("call", "x"))) == fingerprint(
        tracked(b, ("call", "x"))
    )
    assert fingerprint(tracked(a, ("call", "x"))) != fingerprint(
        tracked(a, ("call", "y"))
    )
    # A tag is not the same thing as the contents it stands for.
    assert fingerprint(tracked(a, ("call", "x"))) != fingerprint(a)
    # Tags compose like any other payload, including nested in containers.
    assert fingerprint({"f": tracked(a, ("call", "x"))}) == fingerprint(
        {"f": tracked(b, ("call", "x"))}
    )


def test_tracked_arrays_are_read_only():
    t = tracked(np.arange(4, dtype=np.float32), ("call", "x"))
    with pytest.raises(ValueError):
        t[0] = 1.0
    with pytest.raises(ValueError, match="provenance"):
        tracked(np.arange(4), None)


def test_derived_arrays_drop_the_tag():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = tracked(a, ("call", "x"))
    # Anything whose bytes are no longer the tagged result falls back to
    # hashing contents, so a tag can never outlive its truth.
    for derived in (t[1:], t + 1, t.reshape(3, 2), t.astype(np.float64), t.T):
        assert fingerprint(derived) == fingerprint(np.asarray(derived))
    # A no-op astype returns the array itself, so the tag survives.
    assert t.astype(np.float32, copy=False) is t


def test_tracked_survives_pickling():
    t = tracked(np.arange(6, dtype=np.float32), ("call", "x"))
    restored = pickle.loads(pickle.dumps(t))
    assert fingerprint(restored) == fingerprint(t)
    np.testing.assert_array_equal(restored, t)


def test_torch_tensors():
    torch = pytest.importorskip("torch")
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert fingerprint(a) == fingerprint(a.clone())
    assert fingerprint(a) != fingerprint(a + 1)
    assert fingerprint(a) != fingerprint(a.reshape(3, 2))
    assert fingerprint(a) != fingerprint(a.to(torch.float64))
    # A tensor is not its numpy equivalent.
    assert fingerprint(a) != fingerprint(a.numpy())
    # Layout, autograd, and device are not part of the identity.
    assert fingerprint(a.T) == fingerprint(a.T.contiguous())
    assert fingerprint(torch.ones(3, requires_grad=True)) == fingerprint(torch.ones(3))
    # State dicts compose through the dict branch.
    layer = torch.nn.Linear(3, 2)
    assert fingerprint(layer.state_dict()) == fingerprint(layer.state_dict())


def test_path_resolution(tmp_path):
    assert fingerprint(tmp_path / "x") == fingerprint(tmp_path / "sub" / ".." / "x")
    assert fingerprint(tmp_path / "x") != fingerprint(str(tmp_path / "x"))


def test_fingerprint_protocol():
    class Encoder:
        def __init__(self, slug):
            self.slug = slug

        def __fingerprint__(self):
            return ("encoder", self.slug)

    assert fingerprint(Encoder("wavlm")) == fingerprint(Encoder("wavlm"))
    assert fingerprint(Encoder("wavlm")) != fingerprint(Encoder("hubert"))
    # Identity is the payload, not the class.
    assert fingerprint(Encoder("wavlm")) == fingerprint(("encoder", "wavlm"))


def test_register_and_mro():
    fp = Fingerprinter()

    class Base:
        def __init__(self, x):
            self.x = x

    class Sub(Base):
        pass

    fp.register(Base, lambda obj: ("base", obj.x))
    assert fp(Base(1)) == fp(Sub(1))
    # Nearest class in the MRO wins.
    fp.register(Sub, lambda obj: ("sub", obj.x))
    assert fp(Base(1)) != fp(Sub(1))


def test_unknown_type_raises():
    class Mystery:
        pass

    with pytest.raises(TypeError, match="register"):
        fingerprint(Mystery())
    with pytest.raises(TypeError, match="register"):
        fingerprint(lambda: None)


def test_file_tag(tmp_path):
    p = tmp_path / "model.pt"
    p.write_bytes(b"weights-v1")
    tag1 = file_tag(p)
    assert fingerprint(tag1) == fingerprint(file_tag(p))
    p.write_bytes(b"weights-v2-longer")
    assert fingerprint(tag1) != fingerprint(file_tag(p))
    with pytest.raises(FileNotFoundError):
        file_tag(tmp_path / "missing.pt")
