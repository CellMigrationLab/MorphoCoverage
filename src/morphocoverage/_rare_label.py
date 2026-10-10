"""The one contract of a rare-state label column (private; issue 344).

A label marks the observations of a rare state: real booleans or 0/1, with a missing value
meaning *unknown*, never "not the state". Every analysis that reads a label (the rare-incidence
estimator, the workflows' label check, populations and share, the inspection) goes through
``check_rare_labels`` so that a value such as ``2`` or ``"false"`` is refused everywhere alike
instead of being truth-cast to True by some of them, and through ``known_flags`` so that a missing
label is handled in the open: ``known`` says which observations carry a label, ``flags`` which
of those carry the state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def check_rare_labels(values: pd.Series, name: str) -> None:
    """Refuse a label column holding anything but booleans, 0/1 or missing values;
    ``name`` words the column in the message (``rare_label 'mitotic'``)."""
    present = values.dropna()
    if pd.api.types.is_bool_dtype(present) or set(present.unique()) <= {0, 1}:
        return
    shown = sorted(map(str, present.unique()))[:5]
    raise ValueError(f"{name} must hold booleans or 0/1, got {shown}")


def known_flags(values: pd.Series, name: str) -> tuple[np.ndarray, np.ndarray]:
    """``(known, flags)`` of a label column: which observations carry a label, and which of
    them carry the state (False where the label is unknown). The column is checked first."""
    check_rare_labels(values, name)
    known = values.notna().to_numpy()
    flags = known & values.where(values.notna(), False).astype(bool).to_numpy()
    return known, flags


__all__ = ["check_rare_labels", "known_flags"]
