"""The plots workflow: PlotsOfData-style figures of the declared features, from a ``Setup``."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from ..plots import plots_of_data
from ..sampling_unit import FeatureRanking
from ..sampling_unit.ranking import TOP_FEATURES
from .retrospective import _check_tables
from .setup import Setup


def plots(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    *,
    features: Sequence[str] | None = None,
    folder: str | Path | None = None,
    ranking: FeatureRanking | None = None,
    top: int = TOP_FEATURES,
    **settings: Any,
) -> dict[str, list[Figure]]:
    """PlotsOfData-style figures of the readout (or ``features``), on the prepared table.

    Without ``features``, the readout is plotted and, given the report's
    ``ranking`` (``Report.ranking``), the ``top`` best-ranked other features:
    those that separate the conditions beyond chance, largest |d| first.

    Within each condition the biological repeats sit side by side; every
    object is a point coloured by technical repeat (a well), with each
    biological repeat's median and its
    interval from resampling whole fields (``plots.plots_of_data``, which
    also takes ``settings`` such as ``summary`` or ``max_conditions``).
    With ``folder`` each figure is also saved there as
    ``<feature>[_<part>].png``. Returns the figures by feature.
    """
    _check_tables(objects, fovs, setup)
    if features is None:
        chosen = [setup.readout]
        if ranking is not None:
            if not isinstance(ranking, FeatureRanking):
                raise TypeError("ranking must be the report's FeatureRanking")
            chosen += [f for f in ranking.top(top) if f != setup.readout]
    else:
        chosen = list(features)
    undeclared = sorted(set(chosen) - set(setup.columns))
    if undeclared:
        raise ValueError(f"features to plot must be declared in the setup: {undeclared}")
    figures = plots_of_data(
        setup.prepared(objects, fovs),
        fovs,
        features=chosen,
        random_state=settings.pop("random_state", setup.random_state),
        **settings,
        **setup.common,
    )
    if folder is not None:
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        for feature, parts in figures.items():
            for index, figure in enumerate(parts, start=1):
                suffix = f"_{index}" if len(parts) > 1 else ""
                figure.savefig(target / f"{feature}{suffix}.png", dpi=150)
    return figures


__all__ = ["plots"]
