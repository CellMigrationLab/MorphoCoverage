"""Controls apart in intensity: an acquisition gap between experiments (private; ``report``;
owner, 2026-10-05).

With several controls (``Setup.control`` naming several, ``_contrasts``), each experiment is
normalised to its own control, but raw comparisons across experiments (the ``vs_master_raw``
and ``across_normalised`` rows, a control against its parent) and the maps on raw values
assume the experiments were imaged and processed alike. Intensities show when they were not:
on the owner's ECM data the untransfected controls were about 10 times dimmer than the siRNA
controls. So the controls are compared with each other on every declared column whose name
contains "intensity" (any case) and those listed in ``Setup.intensity_features``: when two
controls' medians, over every observation of each, differ ``GAP_FOLD``-fold or more, the
report says so once where the data are described (a check of the values) and in "Do the
conditions differ?", and beside each secondary row that joins two such controls (the gaps
in ``contrast.settings["acquisition_gaps"]``, ``across_text``). A median
that is not positive has no fold and is not compared. With one control nothing changes.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._caveats import caveat_lines
from .._style import Status
from .._wording import number
from ..data_check import validate_input_tables
from ..metadata import CONDITION_COLUMN
from ..retrospective.report import MicroscopistReport
from ..schema import fov_positions
from ._document import with_text

if TYPE_CHECKING:
    from .setup import Setup

# Two controls whose medians differ this many times or more were likely imaged or processed
# differently (owner, 2026-10-05): an engineering threshold for quality control, not a
# statistical test (plan v2, owner, 2026-10-06).
GAP_FOLD = 2.0
NAME = "Controls apart in intensity"
_PAIRS = 4  # the pairs named; the rest are counted


def gap_features(setup: Setup) -> list[str]:
    """The intensity-like columns: every declared column whose name contains "intensity",
    then those of ``Setup.intensity_features``."""
    named = [c for c in setup.columns if "intensity" in c.lower()]
    return list(dict.fromkeys([*named, *setup.intensity_features]))


def acquisition_gaps(objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup) -> list[dict]:
    """Each pair of controls whose medians differ ``GAP_FOLD``-fold or more in an
    intensity-like column, with the column of the largest fold: ``{"controls": (a, b),
    "feature": ..., "fold": ...}`` (raw values; empty with one control)."""
    own = setup.own_references
    features = gap_features(setup)
    if own is None or not features:
        return []
    controls = list(dict.fromkeys(own.values()))
    data = validate_input_tables(objects, fovs=fovs, feature_cols=features, **setup.common)
    if CONDITION_COLUMN not in data.fovs:
        return []
    positions = fov_positions(data, data.fovs)
    labels = data.fovs[CONDITION_COLUMN].to_numpy()[np.maximum(positions, 0)]
    labels = np.where(positions >= 0, labels, None)
    medians: dict[tuple[Any, str], float] = {}
    for feature in features:
        values = data.feature_values(feature)
        for control in controls:
            mine = values[(labels == control) & ~np.isnan(values)]
            medians[control, feature] = float(np.median(mine)) if len(mine) else np.nan
    gaps = []
    for first, second in combinations(controls, 2):
        folds = []
        for feature in features:
            low, high = sorted((medians[first, feature], medians[second, feature]))
            if low > 0:  # a fold needs two positive medians (NaN fails too)
                folds.append((high / low, feature))
        if folds:
            fold, feature = max(folds, key=lambda item: item[0])
            if fold >= GAP_FOLD:
                gaps.append({"controls": (first, second), "feature": feature, "fold": fold})
    return gaps


def _fold(fold: float) -> str:
    """``13-fold`` or ``2.4-fold``."""
    return f"{fold:.0f}-fold" if fold >= 10 else f"{number(fold)}-fold"


def gap_caution(gaps: list[dict]) -> str | None:
    """The caution in words (see the module), ``None`` without a gap."""
    if not gaps:
        return None
    first = gaps[0]
    pair = " and ".join(map(str, first["controls"]))
    parts = [f"{pair} differ {_fold(first['fold'])} in median {first['feature']}"]
    for gap in gaps[1:_PAIRS]:
        other = "" if gap["feature"] == first["feature"] else f" in {gap['feature']}"
        parts.append(f"{' and '.join(map(str, gap['controls']))} {_fold(gap['fold'])}{other}")
    if len(gaps) > _PAIRS:
        parts.append(f"{len(gaps) - _PAIRS} more pairs of controls")
    return (
        f"{'; '.join(parts)}: the experiments may have been imaged or processed differently, "
        "so raw comparisons across them, and maps on raw values, may show acquisition rather "
        f"than biology. {_fold(GAP_FOLD)} is an engineering threshold for quality control, "
        "not a test."
    )


def with_gaps(
    results: dict[str, Any], objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup
) -> list[dict]:
    """The gaps (``acquisition_gaps``), said in ``results``: a check of the values in the
    overview, and in the contrast's settings for the rows that join such controls."""
    gaps = acquisition_gaps(objects, fovs, setup)
    caution = gap_caution(gaps)
    if caution is None:
        return gaps
    overview = results.get("overview")
    if overview is not None:
        checks = (*overview.value_checks, f"{NAME}: {caution}")
        results["overview"] = replace(overview, value_checks=checks)
    contrast = results.get("contrast")
    if contrast is not None:
        settings = {**contrast.settings, "acquisition_gaps": gaps}
        results["contrast"] = replace(contrast, settings=settings)
    return gaps


def with_gap_caution(document: MicroscopistReport, gaps: list[dict]) -> MicroscopistReport:
    """The report with the gaps' caution in "Do the conditions differ?" (unchanged without
    a gap, or without that answer)."""
    caution = gap_caution(gaps)
    if caution is None or "conditions" not in document.sections:
        return document
    short = "\n".join(caveat_lines([caution], indent=2))
    return with_text(document, "conditions", caution, status=Status.CAUTION, short_text=short)


__all__ = [
    "GAP_FOLD",
    "NAME",
    "acquisition_gaps",
    "gap_caution",
    "gap_features",
    "with_gap_caution",
    "with_gaps",
]
