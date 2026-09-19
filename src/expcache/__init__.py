"""expcache: a small caching framework for experiment notebooks.

Two storage tiers behind one key scheme:

- ``Cache.memo``: one pickled entry per call, for derived results.
- ``Cache.arrays``: stacked mmap block storage, for heavy functions
  returning numpy arrays (encoders and the like).

Keys are built from explicit versions plus argument fingerprints; teach
the fingerprinter about your own types via ``__fingerprint__`` or
``fingerprint.register``. Code is never hashed — bump ``version=`` to
invalidate.

Results from ``Cache.arrays`` carry their call key as provenance, so
they can be passed to another cached function without being hashed;
``tracked`` applies the same treatment to arrays from elsewhere.

Detected missing array data and invalid memo pickles are discarded
with ``CacheRepairWarning`` and recomputed on demand.
"""

from .bypass import no_cache
from .cache import Cache
from .fingerprint import (
    PROVENANCE_ATTR,
    Fingerprinter,
    TrackedArray,
    file_tag,
    fingerprint,
    tracked,
)
from .repair import CacheRepairWarning

__all__ = [
    "PROVENANCE_ATTR",
    "Cache",
    "CacheRepairWarning",
    "Fingerprinter",
    "TrackedArray",
    "file_tag",
    "fingerprint",
    "no_cache",
    "tracked",
]
