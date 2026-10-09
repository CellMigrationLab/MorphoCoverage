"""Argument checks shared by the package (private).

Python and numpy integers and reals are accepted alike (``np.int64(3)`` is an
integer count, ``np.float32(0.05)`` a number); booleans are neither.
"""

from __future__ import annotations

import math
import numbers

import numpy as np


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


def is_share(value: object, *, closed_high: bool = False) -> bool:
    """Whether ``value`` is a real number in (0, 1) (``closed_high``: in (0, 1]); strings,
    booleans, NaN and infinities are not."""
    if not is_number(value) or not math.isfinite(value):
        return False
    return 0 < value <= 1 if closed_high else 0 < value < 1


def is_positive_finite(value: object) -> bool:
    """Whether ``value`` is a real number above 0 and finite (not a string, boolean or NaN)."""
    return is_number(value) and math.isfinite(value) and value > 0


def check_flag(value: object, name: str) -> bool:
    """A mode flag: a real boolean (numpy's too), never generic truthiness such as ``1`` or
    ``"no"``; ``TypeError`` naming it otherwise."""
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be True or False, not {value!r}")
    return bool(value)


def check_column(value: object, name: str) -> str | None:
    """``None`` or one non-empty column name; a sequence, mapping or empty text is refused."""
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise TypeError(f"{name} must be one column name (text), or None, not {value!r}")
    return value


__all__ = [
    "as_count",
    "check_column",
    "check_flag",
    "check_seed",
    "is_integer",
    "is_number",
    "is_positive_finite",
    "is_share",
]
