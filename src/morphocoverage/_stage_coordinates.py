"""One strict contract for stage coordinates (private; issue 232).

Stage coordinates place fields on the microscope stage, so they decide acquisition order,
neighbours and spatial statistics. Every path that reads them goes through this module:

- two distinct column names (one axis cannot stand for two);
- real numeric columns: booleans, complex numbers, text and untyped (object) columns are
  refused, never converted (no ``float()`` of text, no ``errors="coerce"``);
- values a float cannot hold refused: too large (as infinite) or an integer beyond 2**53
  that would lose digits, so two positions never become one (#538);
- infinite values always refused; missing values refused (``"refuse"``) or kept as NaN where an
  analysis handles missing positions explicitly (``"keep"``).

``check_xy`` is the same contract for an ``(n, 2)`` array of positions.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from .metadata import column_names

Missing = Literal["refuse", "keep"]
# Every integer up to this magnitude is held exactly by a float64 (its 53-bit significand).
_EXACT_INTEGERS = 2**53


def stage_names(stage_cols: object, name: str = "stage_cols") -> tuple[str, str]:
    """``stage_cols`` as two distinct column names (x, y), read once."""
    names = column_names(stage_cols, name)
    if names is None or len(names) != 2 or not all(isinstance(c, str) and c for c in names):
        raise TypeError(f"{name} must be two column names (x, y)")
    if names[0] == names[1]:
        raise ValueError(f"{name} must be two different columns (x, y), not one axis twice")
    return names[0], names[1]


_REAL_KINDS = ("integer", "floating", "mixed-integer-float", "empty")


def _check_axis(values: pd.Series, column: str) -> None:
    """A coordinate column is real numbers: never booleans, complex numbers or text. An untyped
    (object) column is accepted only when every value in it is itself a real number, as a table
    built from rows of Python numbers is; numeric text in it is refused, not converted."""
    dtype = values.dtype
    if pd.api.types.is_bool_dtype(dtype):
        raise TypeError(f"stage column {column!r} holds booleans, which are not coordinates")
    if pd.api.types.is_complex_dtype(dtype):
        raise TypeError(f"stage column {column!r} holds complex numbers, not real coordinates")
    if dtype == object:
        kind = pd.api.types.infer_dtype(values, skipna=True)
        if kind in _REAL_KINDS:
            return
        raise TypeError(
            f"stage column {column!r} holds {kind} values; coordinates must be real numbers "
            "(text is never converted: fix the column)"
        )
    if not pd.api.types.is_numeric_dtype(dtype):
        raise TypeError(
            f"stage column {column!r} holds {dtype} values; coordinates must be real numbers"
        )


def stage_matrix(
    table: pd.DataFrame,
    stage_cols: object,
    *,
    missing: Missing = "refuse",
    name: str = "stage_cols",
) -> np.ndarray:
    """The ``(n, 2)`` float positions of the rows of ``table`` (see the module)."""
    x_col, y_col = stage_names(stage_cols, name)
    absent = [c for c in (x_col, y_col) if c not in table.columns]
    if absent:
        raise ValueError(
            f"stage columns missing from the FOV table: {absent} (not a column of the FOV table)"
        )
    for column in (x_col, y_col):
        _check_axis(table[column], column)
    xy = np.column_stack([_axis_floats(table[column], column) for column in (x_col, y_col)])
    _check_values(xy, missing)
    return xy


def _axis_floats(values: pd.Series, column: str) -> np.ndarray:
    """A checked coordinate column as floats: a value too large for a float is refused as
    not finite, and an integer a float cannot hold exactly (beyond 2**53) is refused, so
    two positions never become one (#538)."""
    try:
        floats = values.to_numpy(dtype=float, na_value=np.nan)
    except OverflowError:
        raise ValueError(
            f"stage coordinates must be finite: column {column!r} holds a value too large "
            "for a float"
        ) from None
    _check_exact(values.to_numpy(dtype=object), floats, f"stage column {column!r}")
    return floats


def _check_exact(raw: np.ndarray, floats: np.ndarray, name: str) -> None:
    """Refuse an integer of ``raw`` that its float in ``floats`` does not hold exactly; only
    magnitudes from 2**53 can lose digits, so only those are compared."""
    for index in np.flatnonzero(np.abs(np.nan_to_num(floats)) >= _EXACT_INTEGERS):
        value = raw.flat[index]
        if isinstance(value, int | np.integer) and int(floats.flat[index]) != int(value):
            raise ValueError(
                f"{name} holds {value}, which a float cannot hold exactly (integers beyond "
                "2**53), so distinct positions could coincide; express the coordinates in "
                "larger units"
            )


def _check_values(xy: np.ndarray, missing: Missing) -> None:
    if np.isinf(xy).any():
        raise ValueError("stage coordinates must be finite: correct or drop the infinite ones")
    if missing == "refuse" and np.isnan(xy).any():
        raise ValueError("stage coordinates contain missing values")


def check_xy(xy: Any, name: str = "xy", *, missing: Missing = "refuse") -> np.ndarray:
    """``xy`` as a float ``(n, 2)`` array of real, finite positions (booleans, complex numbers
    and text are refused, not converted; integers a float cannot hold exactly too)."""
    array = np.asarray(xy)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must hold real numbers, not {array.dtype}")
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2), not {array.shape}")
    floats = array.astype(float)
    if array.dtype.kind in "iu":
        _check_exact(array, floats, name)
    _check_values(floats, missing)
    return floats


__all__ = ["Missing", "check_xy", "stage_matrix", "stage_names"]
