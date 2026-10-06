"""Populations of cells on a ``design.Layout``: the outcome generator of the population and
rare-target modules (P6, P7; exploratory until validated).

Each cell belongs to a group (label 1) or not (0). A field's chance of a group cell is
``expit(logit(p_c) + prevalence_sd_repeat z_repeat + prevalence_sd_field z_field)``:
``p_c`` the condition's prevalence (``prevalence`` in the control, its odds multiplied by
``enrichment`` in the other conditions), heterogeneous over repeats and fields (a
beta-binomial-like spread; the field term reuses the layout's field effect, so a crowded
field can hold more of the group with informative cluster size). The ``features``
explainable features are normal with repeat and field effects (``icc_repeat``,
``icc_field``); a group cell is moved by ``separation`` SDs along the first two features.
``structure`` sets what the groups are (``STRUCTURES``):

- ``none``: no group (``prevalence`` ignored); ``clear`` (``separation`` 4) and ``weak``
  (``separation`` 1.5): one group; ``rare``: one clear group at ``prevalence`` (0.5-5%);
- ``split``: one group in the first half of the repeats, split in two directions in the
  others (a population that is not the same across repeats);
- ``batch``: a group in the first repeat only, in every condition (a batch-only cluster).

``embedding_dims`` adds an embedding table carrying the same labels through a random linear
map. ``truth`` holds each cell's label and each condition's expected prevalence.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from morphocoverage.metadata import REPEAT_COLUMN

from .design import Design, Layout, layout, tables

STRUCTURES = ("none", "clear", "weak", "rare", "split", "batch")


@dataclass(frozen=True)
class PopulationSpec:
    """The populations of one dataset (see the module)."""

    structure: str = "clear"
    features: int = 4
    prevalence: float = 0.2
    enrichment: float = 1.0
    separation: float = 4.0
    prevalence_sd_repeat: float = 0.3
    prevalence_sd_field: float = 0.5
    icc_repeat: float = 0.1
    icc_field: float = 0.1
    embedding_dims: int = 0

    def __post_init__(self) -> None:
        if self.structure not in STRUCTURES:
            raise ValueError(f"structure must be one of {STRUCTURES}")
        if not 0 < self.prevalence < 1:
            raise ValueError("prevalence must be between 0 and 1")
        if self.features < 2:
            raise ValueError("populations need at least 2 features")


def population_from(params: dict[str, Any]) -> PopulationSpec:
    """A ``PopulationSpec`` from the outcome keys of ``params`` (the others ignored)."""
    names = {f.name for f in fields(PopulationSpec)}
    return PopulationSpec(**{k: v for k, v in params.items() if k in names})


@dataclass(frozen=True)
class PopulationData:
    """One simulated dataset: the tables, the explainable features, the embedding columns
    (empty without one), the truth (``labels`` per cell of ``objects``, ``prevalence`` by
    condition) and the layout."""

    objects: pd.DataFrame
    fovs: pd.DataFrame
    features: tuple[str, ...]
    embedding: tuple[str, ...]
    truth: dict[str, Any]
    layout: Layout


def _prevalence(spec: PopulationSpec, design: Design) -> dict[str, float]:
    """Each condition's prevalence before the spread over repeats and fields."""
    if spec.structure == "none":
        return dict.fromkeys(design.conditions, 0.0)
    control = design.conditions[0]
    odds = spec.prevalence / (1 - spec.prevalence)
    out = {}
    for condition in design.conditions:
        here = odds * (1.0 if condition == control else spec.enrichment)
        out[condition] = here / (1 + here)
    return out


def generate(design: Design, spec: PopulationSpec, rng: np.random.Generator) -> PopulationData:
    """One dataset of ``spec`` on a fresh layout of ``design`` (see the module)."""
    lay = layout(design, rng)
    frame, rows = lay.fields, lay.cells()
    prevalence = _prevalence(spec, design)
    repeats = frame[REPEAT_COLUMN].to_numpy()
    base = np.array([prevalence[c] for c in frame["condition"]])
    chance = np.zeros(len(frame))
    if spec.structure != "none":
        chance = expit(
            logit(np.clip(base, 1e-9, 1 - 1e-9))
            + spec.prevalence_sd_repeat * frame["z_repeat"].to_numpy()
            + spec.prevalence_sd_field * frame["z_field"].to_numpy()
        )
    if spec.structure == "batch":
        chance = np.where(repeats == "R1", chance, 0.0)
    labels = (rng.random(len(rows)) < chance[rows]).astype(int)
    rest = 1 - spec.icc_repeat - spec.icc_field
    codes = np.unique(repeats, return_inverse=True)[1]
    values = (
        np.sqrt(spec.icc_repeat) * rng.standard_normal((codes.max() + 1, spec.features))[codes]
        + np.sqrt(spec.icc_field) * rng.standard_normal((len(frame), spec.features))
    )[rows] + np.sqrt(rest) * rng.standard_normal((len(rows), spec.features))
    separation = {"weak": 1.5}.get(spec.structure, spec.separation)
    direction = np.zeros(spec.features)
    direction[:2] = 1 / np.sqrt(2)
    other = np.zeros(spec.features)
    other[0], other[1] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    late = np.isin(repeats[rows], sorted(set(repeats))[len(set(repeats)) // 2 :])
    split = (spec.structure == "split") & late & (rng.random(len(rows)) < 0.5)
    moved = np.where(split[:, None], other, direction)
    values = values + separation * labels[:, None] * moved
    names = tuple(f"f{j}" for j in range(spec.features))
    table = pd.DataFrame(values, columns=list(names))
    embedding: tuple[str, ...] = ()
    if spec.embedding_dims:
        mixing = rng.standard_normal((spec.features, spec.embedding_dims))
        noise = 0.5 * rng.standard_normal((len(rows), spec.embedding_dims))
        embedded = values @ mixing / np.sqrt(spec.features) + noise
        embedding = tuple(f"e{j}" for j in range(spec.embedding_dims))
        table = pd.concat([table, pd.DataFrame(embedded, columns=list(embedding))], axis=1)
    objects, fovs = tables(lay, table)
    truth = {"labels": labels, "prevalence": prevalence, "structure": spec.structure}
    return PopulationData(objects, fovs, names, embedding, truth, lay)


__all__ = ["STRUCTURES", "PopulationData", "PopulationSpec", "generate", "population_from"]
