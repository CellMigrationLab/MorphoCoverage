"""The values biological unusualness reads (private; owner, 2026-10-06, plan v2).

Unusual observations are split in two (``docs/common/what_happens.md``): an acquisition
anomaly is read on the raw values, as quality control (``sampling_unit.acquisition``), and
biological unusualness on values made relative to each observation's own control in its
repeat, against that control, as the normalised analyses read them:

- with ``normalize`` declared, the values the normalised analyses read (each feature divided
  by its own control's mean in the same repeat, the other tables centred per repeat): the
  caller passes them;
- without it, each feature centred on its own control's mean in the same repeat
  (``centred_on_own_control``), on the scale the main test reads it (``log_scale``: the log
  ratio to that mean when every value of the feature is positive, else the difference): a
  repeat imaged brighter or a staining or plating batch is then not read as unusual biology.

An observation whose repeat holds no own control has no such value and is not scored (it is
counted among the observations with a missing value). The threshold is today's (the share
``quantile`` of the control's observations; its choice is held for module P10 of the
validation protocol), and the result is exploratory.

Shared-private: used by ``workflows._shared``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._control import control_mask
from .._repeat_statistics import log_scale
from ..data_check import validate_input_tables
from ..metadata import CONDITION_COLUMN

if TYPE_CHECKING:
    from .setup import Setup

# Said in the unusual observations' definition: the values they are judged on.
NORMALISED_WORDS = "each relative to its own control in its repeat, as normalised"
CENTRED_WORDS = (
    "each relative to its own control's mean in its repeat, as a log ratio when every value is "
    "positive, else as a difference"
)


def _own_mask(attached: pd.DataFrame, setup: Setup) -> dict[Any, tuple[np.ndarray, np.ndarray]]:
    """By control, the rows it centres and the rows that are that control: each condition's
    own control with several (``own_references``), else the one control for every row."""
    own = setup.own_references
    if own is None:
        reference = setup.reference_control
        assert reference is not None
        return {None: (np.ones(len(attached), dtype=bool), control_mask(attached, reference))}
    conditions = attached[CONDITION_COLUMN].to_numpy()
    return {
        control: (
            np.isin(conditions, [c for c, r in own.items() if r == control]),
            conditions == control,
        )
        for control in dict.fromkeys(own.values())
    }


def centred_on_own_control(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, columns: Sequence[str]
) -> pd.DataFrame:
    """A copy of ``objects`` with ``columns`` relative to the mean of each observation's own
    control in its repeat, a log ratio for a feature whose every value is positive, else a
    difference (NaN where the repeat holds none); rows and index kept."""
    data = validate_input_tables(objects, fovs=fovs, feature_cols=list(columns), **setup.common)
    attached = data.objects
    if len(attached) != len(objects):
        raise ValueError("validation changed the number of object rows; cannot align the values")
    repeat = setup.repeat_col(fovs)
    repeats = attached[repeat].to_numpy() if repeat else np.zeros(len(attached))
    values = objects[list(columns)].to_numpy(dtype=float)
    centres = np.full(values.shape, np.nan)
    for centred, control in _own_mask(attached, setup).values():
        for value in pd.unique(repeats[centred]):
            here = repeats == value
            reference = values[here & control]
            if len(reference):
                with warnings.catch_warnings():  # a feature with no control value: NaN
                    warnings.simplefilter("ignore", RuntimeWarning)
                    centres[here & centred] = np.nanmean(reference, axis=0)
    # The main test's scale per feature (``log_scale``): a log ratio to the control's mean
    # when every value is positive, so a repeat imaged brighter is not wider either.
    logged = np.array([log_scale(values[:, j]) for j in range(values.shape[1])], dtype=bool)
    relative = values - centres
    with np.errstate(divide="ignore", invalid="ignore"):
        relative[:, logged] = np.log(values[:, logged] / centres[:, logged])
    result = objects.copy()
    result[list(columns)] = relative
    return result


def biological_values(
    values: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, columns: Sequence[str]
) -> tuple[pd.DataFrame, str]:
    """The values biological unusualness reads and how, in words (see the module): ``values``
    as given with ``normalize`` declared (the caller passes the normalised ones), else
    centred on each own control in its repeat."""
    if setup.normalize is not None:
        return values, NORMALISED_WORDS
    return centred_on_own_control(values, fovs, setup, columns), CENTRED_WORDS


__all__ = ["CENTRED_WORDS", "NORMALISED_WORDS", "biological_values", "centred_on_own_control"]
