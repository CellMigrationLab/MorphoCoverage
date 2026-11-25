"""The one check of an FOV table's imaged-area column (private).

Shared-private: ``area_problem`` is used by ``benchmark`` (effort at the stop,
which raises on a problem) and ``strategies.baselines.ImagedAreaEstimator``
(which reports the problem as ``not_applicable``), so both judge the same
areas unusable for the same reason.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def area_problem(fovs: pd.DataFrame, area_col: str, fov_key: Sequence[str]) -> str | None:
    """Why the areas in ``fovs[area_col]`` cannot be summed, or ``None`` when they can.

    An imaged area is a positive, finite number, recorded for every field, and their
    total must be finite too (one that overflows a float would read as an infinite
    effort, #544): a missing area summed as zero would understate the effort, and an
    infinite, zero or negative one is impossible. The problem names up to
    three such fields by ``fov_key``; each caller raises it or reports it.
    Areas that are not numbers (text such as ``"100"`` would be concatenated,
    not added) are a configuration error and raise here.
    """
    areas = fovs[area_col]
    if areas.isna().any():
        missing = fovs.loc[areas.isna(), list(fov_key)].head(3).to_dict("records")
        return f"area_col {area_col!r} is missing for some fields, e.g. {missing}"
    if not pd.api.types.is_numeric_dtype(areas) or pd.api.types.is_bool_dtype(areas):
        raise ValueError(
            f"area_col {area_col!r} holds {areas.dtype} values, not numbers; convert it "
            f"first, for example with pd.to_numeric(fovs[{area_col!r}])"
        )
    values = areas.to_numpy(dtype=float)
    invalid = ~np.isfinite(values) | (values <= 0)
    if invalid.any():
        where = fovs.loc[invalid, list(fov_key)].head(3).to_dict("records")
        return (
            f"area_col {area_col!r} has infinite, zero or negative areas for some fields, "
            f"e.g. {where}; an imaged area is a positive, finite number"
        )
    if len(values) and not math.isfinite(_total(values)):
        return (
            f"area_col {area_col!r}: the fields' areas are each finite but their total "
            "overflows a float; express them in larger units"
        )
    return None


def _total(values: np.ndarray) -> float:
    """The sum of positive, finite ``values``, infinite when it overflows a float (#544).

    Summed as the largest value times the sum of the values divided by it (each at most
    1), so the sum itself cannot overflow and only the final product, a Python float, can
    become infinite, without a numpy overflow warning.
    """
    largest = float(values.max())
    return largest * float((values / largest).sum())


__all__ = ["area_problem"]
