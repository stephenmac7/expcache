"""expcache: a small caching framework for experiment notebooks.

Two storage tiers behind one key scheme:

- ``Cache.memo``: one pickled entry per call, for derived results.
- ``Cache.arrays``: stacked mmap block storage, for heavy functions
  returning numpy arrays (encoders and the like).

Keys are built from explicit versions plus argument fingerprints; teach
the fingerprinter about your own types via ``__fingerprint__`` or
``fingerprint.register``. Code is never hashed — bump ``version=`` to
invalidate.
"""

from .bypass import no_cache
from .cache import Cache
from .fingerprint import Fingerprinter, file_tag, fingerprint

__all__ = ["Cache", "Fingerprinter", "file_tag", "fingerprint", "no_cache"]
