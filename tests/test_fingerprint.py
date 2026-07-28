from pathlib import Path

import numpy as np
import pytest

from expcache import Fingerprinter, file_tag, fingerprint


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
