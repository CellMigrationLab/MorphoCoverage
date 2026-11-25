"""The robust SD of a sample or of each column, under one of two rules (private; re-exported
by ``statistics``; owner, 2026-10-12, after audit 12).

Every robust scale of the package is the MAD times a constant that makes it a Gaussian SD.
Where the MAD is no scale, a fallback takes over, and which fallback is safe depends on the
job of the scale:

- ``"detector"``, for a scale that flags the few far values (the acquisition anomalies,
  ``sampling_unit.acquisition``; the fields that sway the result,
  ``retrospective.inspection.stands_out``): the MAD, and the SD only where the MAD is 0
  (most values tied). A fallback triggered by a ratio to the SD would defeat itself there:
  the far value inflates the SD, the MAD falls under the floor, and the SD then hides the
  very value to flag (one far field among 12 hid itself, audit 10).
- ``"geometry"``, for a scale that equalises the columns before distances, PCA, HDBSCAN and
  MFA (``embedding.robust_scale`` and its callers): the MAD, and the SD where the MAD is
  under ``MAD_FLOOR`` of the SD; 0 for a column that is constant up to rounding
  (``constant_columns``), which is then left out. A column with just under half its values
  tied, as a ReLU output, has a tiny non-zero MAD and would otherwise carry thousands of
  times the weight of the others (audit 7B: 830,000 times).

The novelty transform (``transforms.RobustStandardize``) keeps its own rule, the SD where
the MAD is at rounding level, until benchmark module P10 decides between the floor and the
alternatives.
"""

from __future__ import annotations

import numpy as np

# The MAD times this estimates a Gaussian SD (1 / Phi^-1(0.75)); 1.4826 where rounded
# (embedding.robust_scale, sampling_unit.acquisition), kept so that no result moves.
MAD_TO_SD = 1.482602218505602
MAD_TO_SD_ROUNDED = 1.4826
# Below this share of a column's SD, its MAD is no scale for the geometry rule.
MAD_FLOOR = 0.25
# The spread, relative to a feature's magnitude, below which it is rounding noise:
# about the precision of single-precision floats (2**-23 = 1.2e-7), in which feature
# tables are often stored, so noise of 1e-9 is never standardised to full weight.
CONSTANT_RELATIVE_SPREAD = 1e-7
DETECTOR = "detector"
GEOMETRY = "geometry"


def constant_columns(matrix: np.ndarray) -> np.ndarray:
    """Which columns do not vary: identical values, or a spread that is rounding noise.

    A constant such as 0.1 gives an SD of about 1e-17 from rounding rather than 0,
    so ``sd > 0`` alone would standardise by noise; and a value stored in single
    precision carries rounding of about 1e-7 of its magnitude, which
    standardising would blow up to full weight. A column counts as constant
    when its range is 0 or its SD is at most ``CONSTANT_RELATIVE_SPREAD``
    (1e-7) of its absolute mean.
    """
    spread = np.ptp(matrix, axis=0)
    # matrix.shape[1:]: one value per column, or one value for a 1-D sample (#418).
    sd = matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros(matrix.shape[1:])
    scale = np.abs(matrix.mean(axis=0))
    return (spread == 0) | (sd <= CONSTANT_RELATIVE_SPREAD * scale)


def robust_sd(values: np.ndarray, rule: str, mad_to_sd: float = MAD_TO_SD) -> np.ndarray | float:
    """The robust SD of ``values`` (a float for one sample, one per column of a matrix
    without missing values): ``mad_to_sd`` times the median absolute deviation, with the
    ``rule``'s fallback, ``DETECTOR`` or ``GEOMETRY`` (see the module). Each caller keeps
    the constant it had, ``MAD_TO_SD`` or ``MAD_TO_SD_ROUNDED``: changing only the constant
    moved the populations of a column of many tied counts (audit 12)."""
    values = np.asarray(values, dtype=float)
    mad = mad_to_sd * np.median(np.abs(values - np.median(values, axis=0)), axis=0)
    sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mad)
    if rule == DETECTOR:
        scale = np.where(mad > 0, mad, sd)
    elif rule == GEOMETRY:
        scale = np.where(mad >= MAD_FLOOR * sd, mad, sd)
        if len(values):
            scale = np.where(constant_columns(values), 0.0, scale)
    else:
        raise ValueError(f"rule must be {DETECTOR!r} or {GEOMETRY!r}, not {rule!r}")
    return float(scale) if scale.ndim == 0 else scale


__all__ = [
    "CONSTANT_RELATIVE_SPREAD",
    "DETECTOR",
    "GEOMETRY",
    "MAD_FLOOR",
    "MAD_TO_SD",
    "MAD_TO_SD_ROUNDED",
    "constant_columns",
    "robust_sd",
]
