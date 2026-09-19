"""Warnings for discarded cache entries."""

import warnings


class CacheRepairWarning(UserWarning):
    """A cache entry was discarded because it could not be read."""


def warn_dropped(message: str) -> None:
    warnings.warn(message, CacheRepairWarning, stacklevel=3)
