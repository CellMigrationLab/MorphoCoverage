"""One strict contract for stage coordinates (private; issue 232).

Stage coordinates place fields on the microscope stage, so they decide acquisition order,
neighbours and spatial statistics. Every path that reads them goes through this module:

- two distinct column names (one axis cannot stand for two);
- real numeric columns: booleans, complex numbers, text and untyped (object) columns are
  refused, never converted (no ``float()`` of text, no ``errors="coerce"``);
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
    xy = table[[x_col, y_col]].to_numpy(dtype=float, na_value=np.nan)
    _check_values(xy, missing)
    return xy


def _check_values(xy: np.ndarray, missing: Missing) -> None:
    if np.isinf(xy).any():
        raise ValueError("stage coordinates must be finite: correct or drop the infinite ones")
    if missing == "refuse" and np.isnan(xy).any():
        raise ValueError("stage coordinates contain missing values")


def check_xy(xy: Any, name: str = "xy", *, missing: Missing = "refuse") -> np.ndarray:
    """``xy`` as a float ``(n, 2)`` array of real, finite positions (booleans, complex numbers
    and text are refused, not converted)."""
    array = np.asarray(xy)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must hold real numbers, not {array.dtype}")
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2), not {array.shape}")
    array = array.astype(float)
    _check_values(array, missing)
    return array


__all__ = ["Missing", "check_xy", "stage_matrix", "stage_names"]
