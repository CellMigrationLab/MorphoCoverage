"""The plots workflow: PlotsOfData-style figures of the declared features, from a ``Setup``."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from .._palette import COLOURS, GREY
from ..embedding import MIN_FEATURES, embed
from ..metadata import CONDITION_COLUMN
from ..plots import plots_of_data
from ..plots_embedding import embedding_figures
from ..plots_pairwise import add_pairwise, pairwise_contrasts
from ..sampling_unit import FeatureDiscovery, FeatureRanking
from ..sampling_unit.ranking import TOP_FEATURES
from ._rare_share import compared_columns, plotted_values, share_column, with_share
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
    object is a point in its condition's colour, shaped by technical repeat (a
    well), with each biological repeat's median and its interval from resampling its whole
    technical repeats with their fields as acquired, not reliable with only
    2-3 technical repeats, which the legend says (``plots.plots_of_data``, which
    also takes ``settings`` such as ``summary`` or ``max_conditions``). Normalised
    values (``Setup.normalize``) say so on the value axis. With ``pairwise`` (the default)
    and two or more conditions, every pair of the figure's conditions is drawn under it, as
    Plot&Stats does (``plots_pairwise``): Cohen's d and its randomization test, and
    each figure carries the table as ``pairwise``.
    With ``folder`` each figure is also saved there as
    ``<feature>[_<part>].png``, a path separator in the name written ``_`` and a name
    taken twice given a counter (``_file_name``). Returns the figures by feature.
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
    options = {
        "random_state": settings.pop("random_state", setup.random_state),
        "value_note": settings.pop(
            "value_note", None if setup.normalize is None else "relative to control"
        ),
        "control": settings.pop("control", setup.figure_control),
        **settings,
    }
    share = share_column(setup)
    # Past 12 conditions in one figure, the hits keep a colour (owner, 2026-10-05).
    options.setdefault("highlight", highlighted(ranking)[0])
    figures = (
        plots_of_data(
            prepared,
            fovs,
            features=[f for f in chosen if f != share],
            **options,
            **setup.common,
        )
        if any(f != share for f in chosen)
        else {}
    )
    if share in chosen:
        # The labelled share, one point per field: the field's share of its cells.
        figures[share] = plots_of_data(
            plotted_values(prepared, fovs, setup, share),
            fovs,
            features=[share],
            **options,
            **{**setup.common, "observation_unit": "field"},
        )[share]
        figures = {f: figures[f] for f in chosen}
    if pairwise:
        _add_pairwise(figures, prepared, fovs, setup, ranking)
    if folder is not None:
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        taken: set[str] = set()
        for feature, parts in figures.items():
            for index, figure in enumerate(parts, start=1):
                suffix = f"_{index}" if len(parts) > 1 else ""
                name = _file_name(f"{feature}{suffix}", taken)
                figure.savefig(target / f"{name}.png", dpi=150)
    return figures


def _file_name(name: str, taken: set[str]) -> str:
    """``name`` with no path separator, so the figure stays in the folder, made unique
    among ``taken`` (compared without case, as some file systems do) with a counter: ``x``
    in two parts and a feature ``x_1`` are three files; added to ``taken``."""
    stem = "".join("_" if c in "/\\" else c for c in name) or "feature"
    unique, n = stem, 1
    while unique.casefold() in taken:
        n += 1
        unique = f"{stem}_{n}"
    taken.add(unique.casefold())
    return unique


def verdicts(ranking: Any, feature: str, conditions: Sequence[Any]) -> dict[Any, float]:
    """The verdict's q of each of ``conditions`` against its control on ``feature``: the
    discovery's paired q, adjusted over the feature's family (audit 10, owner, 2026-10-06);
    empty without a discovery, and only the q that exist."""
    tests = getattr(ranking, "tests", None)
    if not isinstance(ranking, FeatureDiscovery) or tests is None or not len(tests):
        return {}
    mine = tests[tests["feature"].eq(feature) & tests["q_paired"].notna()]
    found = dict(zip(mine["condition"], mine["q_paired"].astype(float), strict=True))
    return {c: found[c] for c in conditions if c in found}


def _add_pairwise(
    figures: dict[str, list[Figure]],
    prepared: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    ranking: Any = None,
) -> None:
    """Every pair of each figure's conditions under it (see ``plots``), tested on that
    figure's conditions alone, so the q-values are adjusted over the pairs it shows, as its
    legend says, and a split figure costs its own pairs only; with the report's discovery
    (``ranking``), the verdict's q beside them (``verdicts``, audit 10, owner, 2026-10-06).
    With several controls, the normalised values compare only the conditions normalised to
    the same control (owner, 2026-10-05)."""
    control = (setup.figure_control or {}).get(CONDITION_COLUMN)
    own = setup.own_references
    references = own if setup.normalize is not None else None
    for feature, parts in figures.items():
        for figure in parts:
            summaries = getattr(figure, "cluster_summaries", None)
            if summaries is None or summaries.empty:
                continue
            conditions = list(dict.fromkeys(summaries["condition"]))
            table = pairwise_contrasts(
                prepared,
                fovs,
                feature=feature,
                conditions=conditions,
                control=control,
                random_state=setup.random_state,
                references=references,
                scale=setup.scale_of(feature),
                **setup.common,
            )
            figure.pairwise = table.reset_index(drop=True)
            figure.pairwise_verdicts = verdicts(ranking, feature, conditions)
            add_pairwise(
                figure,
                table,
                conditions,
                several=own is not None,
                verdicts=figure.pairwise_verdicts,
            )


def highlighted(ranking: Any, most: int = len(COLOURS) - 1) -> tuple[list[Any], str]:
    """The conditions a figure of more than 12 keeps in colour beside the control (owner,
    2026-10-05), at most ``most``, and the words that say which: those significant across
    repeats on any feature, else the cell-level screen's candidates, else those with the
    largest effects, each by its largest |d|. None without a discovery."""
    tests = getattr(ranking, "tests", None)
    if not isinstance(ranking, FeatureDiscovery) or tests is None or not len(tests):
        return [], ""
    order = tests.assign(_size=tests["d"].abs()).sort_values(
        "_size", ascending=False, kind="stable"
    )
    for column, words in (
        ("confirmed", "significant across repeats"),
        ("passes", "best-ranked candidates of the cell-level screen"),
        ("_size", "with the largest effects"),
    ):
        mask = order[column].notna() if column == "_size" else order[column].fillna(False)
        names = list(dict.fromkeys(order.loc[mask.astype(bool), "condition"]))
        if names:
            return names[:most], words
    return [], ""


def colour_note(colours: dict[Any, str], words: str) -> str:
    """What a figure past 12 conditions colours, in a sentence; empty when every condition
    has its own colour."""
    greyed = sum(colour == GREY for colour in colours.values())
    if not greyed:
        return ""
    shown = len(colours) - greyed - 1
    which = f"the {shown} conditions {words}" if shown else "no other condition"
    return (
        f"With {len(colours)} conditions, the control and {which} are in colour; "
        f"the other {greyed} are grey."
    )


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
    figures = embedding_figures(
        embedding, control=setup.figure_control, highlight=highlighted(discovery)[0]
    )
    if folder is not None:
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(target / f"{name}.png", dpi=150)
    return figures


__all__ = ["colour_note", "embedding_plots", "highlighted", "plots"]
