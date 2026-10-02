"""The plots workflow: PlotsOfData-style figures of the declared features, from a ``Setup``."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from ..embedding import MIN_FEATURES, embed
from ..metadata import CONDITION_COLUMN
from ..plots import plots_of_data
from ..plots_embedding import embedding_figures
from ..plots_pairwise import add_pairwise, pairwise_contrasts
from ..sampling_unit import FeatureDiscovery, FeatureRanking
from ..sampling_unit.ranking import TOP_FEATURES
from ._rare_share import compared_columns, with_share
from ._shared import check_workflow_inputs
from .setup import Setup


def plots(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    *,
    features: Sequence[str] | None = None,
    folder: str | Path | None = None,
    ranking: FeatureDiscovery | FeatureRanking | None = None,
    top: int = TOP_FEATURES,
    pairwise: bool = True,
    **settings: Any,
) -> dict[str, list[Figure]]:
    """PlotsOfData-style figures of the readout (or ``features``), on the prepared table.

    Without ``features``, the readout is plotted or, given a ranking from the report
    (``Report.discovery``, the cell-level gate, or ``Report.ranking``, the test over whole
    technical repeats), the ``top`` best-ranked features: those that pass its gate, largest
    |d| first, the readout among them only when it ranks (and alone when none does).

    Within each condition the biological repeats sit side by side; every
    object is a point coloured by technical repeat (a well), with each
    biological repeat's median and its interval from resampling its whole
    technical repeats with their fields as acquired, not reliable with only
    2-3 technical repeats, which the legend says (``plots.plots_of_data``, which
    also takes ``settings`` such as ``summary`` or ``max_conditions``). Normalised
    values (``Setup.normalize``) say so on the value axis. With ``pairwise`` (the default)
    and two or more conditions, every pair of the figure's conditions is drawn under it, as
    Plot&Stats does (``plots_pairwise``): Cohen's d and its randomization test, and
    each figure carries the table as ``pairwise``.
    With ``folder`` each figure is also saved there as
    ``<feature>[_<part>].png``. Returns the figures by feature.
    """
    objects, fovs, _ = check_workflow_inputs(objects, fovs, setup)
    if features is None:
        chosen = [setup.readout]
        if ranking is not None:
            if not isinstance(ranking, (FeatureDiscovery, FeatureRanking)):
                raise TypeError("ranking must be the report's discovery or ranking")
            # The report is neutral (owner, 2026-10-02): the readout is plotted only when it
            # ranks, or when no feature does.
            chosen = list(ranking.top(len(ranking.ranking)))[:top] or chosen
    else:
        chosen = list(features)
    undeclared = sorted(set(chosen) - set(compared_columns(setup)))
    if undeclared:
        raise ValueError(f"features to plot must be declared in the setup: {undeclared}")
    prepared = with_share(setup.prepared(objects, fovs), setup)
    figures = plots_of_data(
        prepared,
        fovs,
        features=chosen,
        random_state=settings.pop("random_state", setup.random_state),
        value_note=settings.pop(
            "value_note", None if setup.normalize is None else "relative to control"
        ),
        control=settings.pop("control", setup.control),
        **settings,
        **setup.common,
    )
    if pairwise:
        _add_pairwise(figures, prepared, fovs, setup)
    if folder is not None:
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        for feature, parts in figures.items():
            for index, figure in enumerate(parts, start=1):
                suffix = f"_{index}" if len(parts) > 1 else ""
                figure.savefig(target / f"{feature}{suffix}.png", dpi=150)
    return figures


def _add_pairwise(
    figures: dict[str, list[Figure]], prepared: pd.DataFrame, fovs: pd.DataFrame, setup: Setup
) -> None:
    """Every pair of each figure's conditions under it (see ``plots``)."""
    control = (setup.control or {}).get(CONDITION_COLUMN)
    for feature, parts in figures.items():
        table = pairwise_contrasts(
            prepared,
            fovs,
            feature=feature,
            control=control,
            random_state=setup.random_state,
            **setup.common,
        )
        for figure in parts:
            summaries = getattr(figure, "cluster_summaries", None)
            if summaries is None or summaries.empty:
                continue
            conditions = list(dict.fromkeys(summaries["condition"]))
            shown = table[table["first"].isin(conditions) & table["second"].isin(conditions)]
            figure.pairwise = shown.reset_index(drop=True)
            add_pairwise(figure, shown, conditions)


def embedding_plots(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    *,
    discovery: FeatureDiscovery | None = None,
    folder: str | Path | None = None,
    **settings: Any,
) -> dict[str, Figure]:
    """The PCA biplot and the UMAP of the declared features (``embedding.embed``), coloured
    by condition, on the values the plots read (normalised with ``Setup.normalize``).

    The discovery's best-ranked features (``Report.discovery``) are the PCA's arrows.
    ``settings`` go to ``embed`` (``umap=False``, ``max_points``, ``n_neighbors``,
    ``min_dist``). Needs ``embedding.MIN_FEATURES`` declared features. With ``folder``,
    saved as ``pca.png`` and ``umap.png``. Returns the figures by name.
    """
    objects, fovs, _ = check_workflow_inputs(objects, fovs, setup)
    if len(setup.columns) < MIN_FEATURES:
        raise ValueError(
            f"an embedding needs at least {MIN_FEATURES} declared features; the setup has "
            f"{len(setup.columns)}"
        )
    if discovery is not None and not isinstance(discovery, FeatureDiscovery):
        raise TypeError("discovery must be the report's FeatureDiscovery")
    embedding = embed(
        setup.prepared(objects, fovs),
        fovs,
        features=setup.columns,
        top=discovery.top() if discovery is not None else (),
        random_state=settings.pop("random_state", setup.random_state),
        **settings,
        **setup.common,
    )
    figures = embedding_figures(embedding, control=setup.control)
    if folder is not None:
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(target / f"{name}.png", dpi=150)
    return figures


__all__ = ["embedding_plots", "plots"]
