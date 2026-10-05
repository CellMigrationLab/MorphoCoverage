"""The rare phenotype's label as a positive control (private; ``retrospective``, ``_document``).

"Can these features detect what I care about?" needs a condition known to change the
phenotype. A labelled rare state is such a check already (owner, 2026-10-02): its cells
were told apart by another channel, so the features should place them together. For every
labelled cell, the nearest cell of another field (on the declared features, each robustly
standardized: median and MAD over all cells) is looked up: the share of those neighbours
that are labelled says how well the features see the state, against the share labelled
overall (what a feature set blind to it gives). A cell whose label is missing is unknown: it
is in neither share (owner, 2026-10-05). Another field, so that cells of one
field, alike for reasons of that field, do not recognize each other. The check uses
``Setup.minimum_detection`` as a positive control does. The features that separate the
state most are given with their Cohen's d (labelled against the other cells).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .._style import Status, wrap
from .._wording import count, listing, percent, plural
from ..schema import fov_positions, validate_input_tables

if TYPE_CHECKING:
    from ..retrospective import MicroscopistReport
    from .setup import Setup

# Neighbours first looked up per labelled cell for the nearest one of another field.
NEIGHBOURS = 64
MIN_LABELLED = 5


@dataclass(frozen=True)
class LabelControl:
    """The label check (see the module)."""

    label: str
    phenotype: str
    recognized: float
    chance: float
    labelled: int
    observations: int
    separation: pd.Series  # Cohen's d per feature, largest |d| first
    unit: str  # one observation: "cell"
    features: int
    minimum: float


def _robust(matrix: np.ndarray) -> np.ndarray:
    median = np.nanmedian(matrix, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(matrix - median), axis=0)
    spread = np.nanstd(matrix, axis=0)
    scale = np.where(scale > 0, scale, np.where(spread > 0, spread, 1.0))
    return (matrix - median) / scale


def label_control(objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup) -> LabelControl | None:
    """The check, or ``None`` without a label, with too few labelled cells, or with missing
    feature values."""
    label = setup.rare_label
    if label is None or label not in objects.columns or setup.rare is None:
        return None
    features = list(setup.columns)
    data = validate_input_tables(objects, fovs=fovs, feature_cols=features, **setup.common)
    matrix = data.feature_matrix()
    if np.isnan(matrix).any():
        return None
    # A missing label is unknown: the cell is in no share, nor anyone's neighbour (owner,
    # 2026-10-05); the standardization, which reads no label, keeps every cell.
    known = data.objects[label].notna().to_numpy()
    marked = data.objects[label][known].to_numpy(dtype=bool)
    if marked.sum() < MIN_LABELLED or marked.all():
        return None
    positions = fov_positions(data, data.fovs)[known]
    points = _robust(matrix)[known]
    tree = cKDTree(points)
    own = positions[marked]
    nearest = np.full(len(own), -1)
    todo = np.arange(len(own))
    k = min(NEIGHBOURS, len(points))
    # A cell with k closer cells of its own field is looked up again with twice k, so
    # crowded fields keep their labelled cells (every cell of one field: none is found).
    while len(todo):
        found = np.atleast_2d(tree.query(points[marked][todo], k=k)[1])
        other = positions[found] != own[todo][:, None]
        hit = other.any(axis=1)
        nearest[todo[hit]] = found[hit, other[hit].argmax(axis=1)]
        todo = todo[~hit]
        if k == len(points):
            break
        k = min(2 * k, len(points))
    neighbours = nearest[nearest >= 0]
    recognized = float(marked[neighbours].mean()) if len(neighbours) else np.nan
    values = pd.DataFrame(matrix[known], columns=features)
    inside, outside = values[marked], values[~marked]
    pooled = np.sqrt((inside.var(ddof=1) + outside.var(ddof=1)) / 2)
    separation = ((inside.mean() - outside.mean()) / pooled.replace(0, np.nan)).dropna()
    separation = separation.reindex(separation.abs().sort_values(ascending=False).index)
    return LabelControl(
        label=label,
        phenotype=setup.rare.phenotype,
        recognized=recognized,
        chance=float(marked.mean()),
        labelled=int(marked.sum()),
        observations=len(marked),
        separation=separation,
        unit=setup.observation_unit or "observation",
        features=len(features),
        minimum=float(setup.minimum_detection),
    )


def label_answer(check: LabelControl) -> tuple[Status, str, str]:
    """Question 1's status, headline and text (unindented) from the label check."""
    unit, units = check.unit, plural(check.unit)
    seen = check.recognized >= check.minimum
    status = Status.OK if seen else Status.CAUTION
    named = [f"{f} (d {d:+.1f})" for f, d in check.separation.head(3).items()]
    headline = (
        f"{check.label} {units} recognized by their nearest neighbour in "
        f"{percent(check.recognized)} of cases ({percent(check.chance)} by chance)"
    )
    verdict = (
        "the features see this state"
        if seen
        else "the features see this state only in part: a phenotype like it may be missed"
    )
    text = (
        f"Can the features see the labelled state? {check.label.capitalize()} {units}, "
        f"{check.phenotype}: {count(check.labelled, 'labelled ' + unit)} of "
        f"{check.observations}. For {percent(check.recognized)} of them, the nearest "
        f"{unit} of another field on the {count(check.features, 'declared feature')} "
        f"(robustly standardized) is labelled too, against {percent(check.chance)} labelled "
        f"overall; {percent(check.minimum)} is asked (minimum_detection): {verdict}. The "
        f"features that separate it most: {listing(named, 3)}. No other positive control was "
        "declared, so this is the check of the representation."
    )
    return status, headline, text


__all__ = ["LabelControl", "label_answer", "label_control", "with_label_check"]


def with_label_check(
    document: MicroscopistReport, check: LabelControl | None, results: dict[str, Any]
) -> MicroscopistReport:
    """The report with question 1 answered by the label when no positive control ran."""
    if check is None or results.get("controls") is not None:
        return document
    from ._document import with_text  # the assembly imports this module

    status, headline, text = label_answer(check)
    return with_text(
        document,
        "controls",
        text,
        status=status,
        headline=headline,
        short_text=wrap(text, 2, hang=4),
    )
