"""The data of a benchmark run (private): the dataset loaded and validated, its analysis
groups, the streams replayed and the groups scored against, and the imaged area.

Shared-private: ``_area``, ``_group_key``, ``_check_metrics``, ``_loaded_groups``,
``_streams`` and ``_Streams``, used by ``benchmark``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ._areas import area_problem
from .data_check import validate_input_tables
from .evaluator import iter_analysis_groups
from .outcomes import OutcomeMetric
from .reservoir import ReservoirSplit
from .schema import ValidatedInput


@dataclass(frozen=True)
class _Streams:
    """Per analysis group: what the strategies replay (``groups``), what outcomes are scored
    against (``references``) and the fields recorded about that reference; and the
    ``(objects, fovs)`` tables replayed."""

    groups: dict[tuple, ValidatedInput]
    references: dict[tuple, ValidatedInput]
    tables: tuple[pd.DataFrame, pd.DataFrame]
    reference_fields: dict[tuple, dict[str, Any]]


def _group_key(group: Mapping[str, Any]) -> tuple:
    """Hashable identity of an analysis group, for matching replay results."""
    return tuple(group.items())


def _area(data: ValidatedInput, area_col: str | None) -> float | None:
    """Total imaged area of the FOVs, or ``None`` when the dataset declares no area column.

    Areas are checked by ``_areas.area_problem``, the check
    ``ImagedAreaEstimator`` uses too: a field with no recorded area, or an
    infinite, zero or negative one, raises here (summing it would misstate the
    effort at the stop), and so do areas that are not numbers.
    """
    if area_col is None:
        return None
    problem = area_problem(data.fovs, area_col, data.fov_key)
    if problem is not None:
        raise ValueError(problem)
    return float(data.fovs[area_col].to_numpy(dtype=float).sum())


def _check_metrics(metrics: Sequence[OutcomeMetric], dataset: Any) -> None:
    """Distinct outcome metric names, and the dataset's area column not a feature."""
    names = [metric.name for metric in metrics]
    if len(set(names)) != len(names):
        raise ValueError(
            f"outcome metric names must be distinct: {names}; give two settings of one "
            "metric different names with label=, e.g. MMDError(..., label='mmd_wide')"
        )
    if dataset.area_col is not None and dataset.area_col in dataset.feature_cols:
        raise ValueError("the area column is acquisition metadata, not a phenotype feature")


def _loaded_groups(
    dataset: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[tuple, ValidatedInput]]:
    """The dataset (a ``BenchmarkDataset``) loaded and validated: its tables, the column
    arguments of the replay and its analysis groups by ``_group_key``."""
    objects, fovs = dataset.load()
    columns = {
        "feature_cols": list(dataset.feature_cols),
        "fov_col": dataset.fov_col,
        "group_cols": None if dataset.group_cols is None else list(dataset.group_cols),
        "observation_unit": dataset.observation_unit,
    }
    data = validate_input_tables(objects, fovs=fovs, **columns)
    if dataset.area_col is not None and dataset.area_col not in data.fovs.columns:
        raise ValueError(f"area_col {dataset.area_col!r} is not a column of the FOV table")
    _area(data, dataset.area_col)  # refuses unusable areas before any replay
    full_groups = {
        _group_key(group): group_data for group, group_data in iter_analysis_groups(data)
    }
    return objects, fovs, columns, full_groups


def _streams(
    reservoir: ReservoirSplit | None,
    full_groups: dict[tuple, ValidatedInput],
    tables: tuple[pd.DataFrame, pd.DataFrame],
) -> _Streams:
    """The full groups replayed and scored against; with ``reservoir``, each group's stream
    replayed and its held-out reservoir scored against."""
    if reservoir is None:
        fields: dict[tuple, dict[str, Any]] = {key: {} for key in full_groups}
        return _Streams(full_groups, full_groups, tables, fields)
    if not isinstance(reservoir, ReservoirSplit):
        raise TypeError("reservoir must be a ReservoirSplit")
    splits = {key: reservoir.split(dict(key), group) for key, group in full_groups.items()}
    streams = {key: stream for key, (stream, _) in splits.items()}
    references = {key: held for key, (_, held) in splits.items()}
    replay_tables = (
        pd.concat([stream.objects for stream in streams.values()]),
        pd.concat([stream.fovs for stream in streams.values()]),
    )
    reference_fields = {
        key: {
            "reference": "reservoir",
            "reservoir_fovs": len(held.fovs),
            "reservoir": reservoir.parameters,
        }
        for key, held in references.items()
    }
    return _Streams(streams, references, replay_tables, reference_fields)
