"""Argument checks shared by the package (private).

Python and numpy integers and reals are accepted alike (``np.int64(3)`` is an
integer count, ``np.float32(0.05)`` a number); booleans are neither.
"""

from __future__ import annotations

import numbers


def is_integer(value: object) -> bool:
    """Whether ``value`` is an integer (Python or numpy), not a boolean."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def is_number(value: object) -> bool:
    """Whether ``value`` is a real number (Python or numpy), not a boolean."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def as_count(value: object, name: str, minimum: int = 1) -> int:
    """``value`` as a Python ``int`` of at least ``minimum``, or a ``ValueError`` naming it."""
    if not is_integer(value) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}, not {value!r}")
    return int(value)


def check_seed(value: object, name: str = "random_state") -> int:
    """A random seed: an integer of at least 0 (numpy integers accepted), as a Python ``int``."""
    if not is_integer(value) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer seed, not {value!r}")
    return int(value)


__all__ = ["as_count", "check_seed", "is_integer", "is_number"]
