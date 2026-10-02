"""The data behind each figure of the HTML report, as CSV beside its PNG and PDF (private;
``html_report``), so the same figure can be drawn again outside (owner, 2026-10-02).

- ``feature_data``: per feature figure, every observation plotted (long: one row per
  observation and kind of values, raw or normalised), each biological repeat's summary and
  interval (the figure's ``cluster_summaries``) and every pair of conditions (``pairwise``).
- ``embedding_data``: the PCA and UMAP coordinates of every point, with its condition,
  repeat, field, population and standardised features, and the PCA loadings.
- ``heatmap_data``: Cohen's d and the q-values of every feature and condition.

Each is written in ``folder`` (the report's ``figures``) by ``write_csv``, so the page
carries it (``_files``) and the results folder and its zip hold it; ``download`` is the
button that saves a file the page carries, ``data_buttons`` one per file.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ._html_plotly import _observations

# The columns of a feature's observations, named as the MorphoBricks tables name them.
_NAMES = {"repeat": "biological_repeat", "technical": "technical_repeat"}


def write_csv(frame: pd.DataFrame, folder: Path, name: str) -> str:
    """``frame`` as ``folder/name`` (6 significant digits, enough to draw it again); its
    path as the page carries it."""
    frame.to_csv(folder / name, index=False, float_format="%.6g")
    return f"figures/{name}"


def download(path: str, label: str) -> str:
    """A button that saves the file the page carries at ``path`` (``html_report._files``)."""
    return (
        f'<button type="button" class="dl" data-file="{html.escape(path)}">'
        f"{html.escape(label)}</button>"
    )


def data_buttons(files: Sequence[tuple[str, str]]) -> str:
    """A save button per ``(path, label)``."""
    return " ".join(download(path, label) for path, label in files)


def feature_data(
    name: str,
    feature: str,
    prepared: Mapping[str, pd.DataFrame],
    figures: Mapping[str, Sequence[Any]],
    fovs: pd.DataFrame,
    setups: Mapping[str, Any],
    folder: Path,
) -> list[tuple[str, str]]:
    """The observations, repeat summaries and pairs of one feature's figures, by kind of
    values (``prepared``, ``figures`` and ``setups`` keyed alike); the files and labels."""
    points, repeats, pairs = [], [], []
    for values, table in prepared.items():
        frame = _observations(table, fovs, feature, setups[values]).rename(columns=_NAMES)
        points.append(frame.rename(columns={"value": feature}).assign(values=values))
        for figure in figures.get(values, ()):
            summaries = getattr(figure, "cluster_summaries", None)
            if summaries is not None and len(summaries):
                repeats.append(summaries.rename(columns=_NAMES).assign(values=values))
            table_ = getattr(figure, "pairwise", None)
            if table_ is not None and len(table_):
                pairs.append(table_.assign(values=values))
    files = []
    if points:
        frame = pd.concat(points, ignore_index=True)
        order = ["values", *[c for c in frame.columns if c != "values"]]
        files.append((write_csv(frame[order], folder, f"{name}_data.csv"), "Data (CSV)"))
    if repeats:
        frame = pd.concat(repeats, ignore_index=True).drop_duplicates(
            ["values", "condition", "biological_repeat"]
        )
        files.append((write_csv(frame, folder, f"{name}_repeats.csv"), "Repeat summaries (CSV)"))
    if pairs:
        frame = pd.concat(pairs, ignore_index=True).drop_duplicates(["values", "first", "second"])
        files.append((write_csv(frame, folder, f"{name}_pairs.csv"), "Pairs (CSV)"))
    return files


def embedding_data(embeddings: Mapping[str, Any], folder: Path) -> list[tuple[str, str]]:
    """Every point of the PCA and UMAP, and the loadings, per kind of values."""
    files = []
    for values, embedding in embeddings.items():
        points = embedding.points.rename(
            columns={"pc1": "PC1", "pc2": "PC2", "umap1": "UMAP1", "umap2": "UMAP2"}
        )
        first = [c for c in ("PC1", "PC2", "UMAP1", "UMAP2") if c in points]
        points = points[first + [c for c in points.columns if c not in first]]
        label = f"Points, {values} (CSV)" if len(embeddings) > 1 else "Points (CSV)"
        files.append((write_csv(points, folder, f"embedding_{values}.csv"), label))
    for values, embedding in embeddings.items():
        loadings = embedding.loadings.rename(columns={"pc1": "PC1", "pc2": "PC2", "top": "arrow"})
        label = f"Loadings, {values} (CSV)" if len(embeddings) > 1 else "Loadings (CSV)"
        files.append((write_csv(loadings, folder, f"pca_loadings_{values}.csv"), label))
    return files


HEATMAP_COLUMNS = [
    "feature",
    "condition",
    "control_condition",
    "d",
    "q_cells",
    "passes",
    "repeats",
    "lower",
    "higher",
    "q_paired",
    "confirmed",
]


def heatmap_data(discovery: Any, folder: Path) -> list[tuple[str, str]]:
    """The heatmap's numbers, its rows in its order (best-ranked first)."""
    tests = getattr(discovery, "tests", None)
    if tests is None or not len(tests):
        return []
    order = {f: i for i, f in enumerate(discovery.ranking["feature"])}
    frame = tests.reindex(columns=HEATMAP_COLUMNS)
    frame = frame.assign(_rank=frame["feature"].map(order)).sort_values("_rank", kind="stable")
    frame = frame.drop(columns="_rank").rename(
        columns={"passes": "candidate", "confirmed": "significant_across_repeats"}
    )
    return [(write_csv(frame, folder, "heatmap_d.csv"), "Data (CSV)")]


__all__ = [
    "data_buttons",
    "download",
    "embedding_data",
    "feature_data",
    "heatmap_data",
    "write_csv",
]
