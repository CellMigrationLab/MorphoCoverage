"""The data behind each figure of the HTML report, as CSV beside its PNG and PDF (private;
``html_report``), so the same figure can be drawn again outside (owner, 2026-10-02).

- ``feature_data``: per feature figure, every observation plotted (long: one row per
  observation and kind of values, raw or normalised), each biological repeat's summary and
  interval (the figure's ``cluster_summaries``) and every pair of conditions (``pairwise``).
- ``embedding_data``: the PCA and UMAP coordinates of every point, with its condition,
  repeat, field, observation (its index in the observation table), ``label_id`` when the
  table has one, population and standardised features, and the PCA loadings.
- ``heatmap_data``: Cohen's d and the q-values of every feature and condition.
- ``across_data``: with several controls, the secondary rows of every feature.

Each is written in ``folder`` (the report's ``figures``) by ``write_csv``, so the page
carries it (``page_files``, up to ``EMBED_BUDGET``; past it the page links to it) and the
results folder and its zip hold it; ``download`` is the button that saves a file the page
carries or opens one it links to, ``data_buttons`` one per file.
"""

from __future__ import annotations

import base64
import html
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ..sampling_unit.across_text import ACROSS_CSV
from ._html_plotly import _observations

# The columns of a feature's observations, named as the MorphoBricks tables name them.
_NAMES = {"repeat": "biological_repeat", "technical": "technical_repeat"}


def write_csv(frame: pd.DataFrame, folder: Path, name: str) -> str:
    """``frame`` as ``folder/name`` (6 significant digits, enough to draw it again); its
    path as the page carries it."""
    frame.to_csv(folder / name, index=False, float_format="%.6g")
    return f"figures/{name}"


_TYPES = {
    ".png": "image/png",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".json": "application/json",
}


# The most the page carries inside it, in bytes (owner, 2026-10-05: "embed up to a
# budget"): past it, the largest of the figures folder's files are linked, not embedded.
EMBED_BUDGET = 25 * 1024 * 1024
# A file the page links to rather than carries, and a PNG it carries once, as its image.
LINKED, IN_IMAGE = "", "img"


def page_files(
    folder: Path,
    texts: dict[str, str],
    attach: Sequence[Path],
    since: float,
    budget: int = EMBED_BUDGET,
) -> tuple[dict[str, list[Any]], dict[str, str]]:
    """Every file the page carries or links to, by the path it is saved under: the figures
    this run wrote in ``folder/figures`` (modified at or after ``since``, so an older run's
    are left out), the ``texts`` written here, and the ``attach`` files; each
    ``[type, base64, bytes]``. The texts and attachments are always carried, an attachment
    whose name is taken as ``<name>_2`` (``_3``, ...), a missing one not (``files_card`` says
    so); the figures'
    files, smallest first, while all of it stays within ``budget``, the others linked
    (``LINKED``: the page's button opens ``figures/<name>``). A PNG carried is written once,
    as its image's data (``IN_IMAGE``); the second value maps those paths to their base64."""
    files: dict[str, list[Any]] = {}
    for name, text in texts.items():
        mime = _TYPES.get(Path(name).suffix, "text/plain")
        data = text.encode("utf-8")
        files[name] = [mime, base64.b64encode(data).decode(), len(data)]
    for path in dict.fromkeys(Path(p).resolve() for p in attach if Path(p).is_file()):
        mime = _TYPES.get(path.suffix, "application/octet-stream")
        data = path.read_bytes()
        files[_free_name(path, files)] = [mime, base64.b64encode(data).decode(), len(data)]
    used = sum(entry[2] for entry in files.values())
    figures = [
        path
        for path in (folder / "figures").glob("*")
        if path.suffix in _TYPES and path.stat().st_mtime >= since
    ]
    images: dict[str, str] = {}
    carried: dict[str, list[Any]] = {}
    for path in sorted(figures, key=lambda p: (p.stat().st_size, p.name)):
        name, size = f"figures/{path.name}", path.stat().st_size
        if used + size > budget:
            carried[name] = [_TYPES[path.suffix], LINKED, size]
            continue
        used += size
        data = base64.b64encode(path.read_bytes()).decode()
        if path.suffix == ".png":
            images[name] = data
            data = IN_IMAGE
        carried[name] = [_TYPES[path.suffix], data, size]
    return {**dict(sorted(carried.items())), **files}, images


def _free_name(path: Path, files: dict[str, list[Any]]) -> str:
    """``path``'s name, or ``<stem>_2<suffix>`` (``_3``, ...) when a file carried has it, so
    two attachments of one name are both carried."""
    name, number = path.name, 1
    while name in files:
        number += 1
        name = f"{path.stem}_{number}{path.suffix}"
    return name


def inline_images(markup: str, images: dict[str, str]) -> str:
    """Each carried PNG's ``<img src="figures/...">`` given its data, so the page shows it
    without the folder; a PNG past the budget keeps its relative path."""
    for name, data in images.items():
        markup = markup.replace(f'src="{name}"', f'src="data:image/png;base64,{data}"')
    return markup


def files_card(files: dict[str, list[Any]], attach: Sequence[str | Path] = ()) -> str:
    """The card listing every file the page carries or links to, each with its size and
    save button; a line names the ``attach`` files that were not found."""
    rows = "".join(
        f'<tr><td>{html.escape(name)}</td><td class="num">'
        f"{entry[2] / 1024:.0f} kB</td><td>{download(name, 'Save')}</td></tr>"
        for name, entry in files.items()
    )
    linked = [entry[2] for entry in files.values() if entry[1] == LINKED]
    note = (
        "Everything this report was made with and from is inside this page: save any file "
        "with its button, no folder needed. The same files are in the results folder and "
        "its zip."
        if not linked
        else f"The page carries its files up to {EMBED_BUDGET // 1024**2} MB in all; the "
        f"{len(linked)} largest, {sum(linked) / 1024**2:.0f} MB, open from the figures folder "
        "beside it, which the zip holds too, so keep the page with its folder."
    )
    missing = list(dict.fromkeys(str(p) for p in attach if not Path(p).is_file()))
    if missing:
        names = html.escape(", ".join(missing))
        note += f'</p><p class="note">Not found, so not attached: {names}.'
    return (
        f'<section class="card" id="files"><h2>Files</h2><p class="note">{note}</p>'
        '<div class="table-wrap"><table class="sortable"><thead><tr><th>file</th><th>size</th>'
        f"<th></th></tr></thead><tbody>{rows}</tbody></table></div></section>"
    )


def json_island(data: Any) -> str:
    """``data`` as JSON to sit inside a ``<script type="application/json">``: ``<``, ``>``
    and ``&`` written as JSON escapes, so no name (a feature called ``x<!--<script>``) can
    end or confuse the script element."""
    text = json.dumps(data, separators=(",", ":"), default=str)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


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


def _stem(key: str) -> str:
    """``raw, DINO`` as a file name: ``raw_DINO``."""
    return re.sub(r"[^\w.-]+", "_", key)


def embedding_data(embeddings: Mapping[str, Any], folder: Path) -> list[tuple[str, str]]:
    """Every point of the PCA and UMAP, and the loadings, per kind of values (and feature
    table, ``raw, DINO``)."""
    files = []
    for values, embedding in embeddings.items():
        points = embedding.points.rename(
            columns={"pc1": "PC1", "pc2": "PC2", "umap1": "UMAP1", "umap2": "UMAP2"}
        )
        first = [c for c in ("PC1", "PC2", "UMAP1", "UMAP2") if c in points]
        points = points[first + [c for c in points.columns if c not in first]]
        label = f"Points, {values} (CSV)" if len(embeddings) > 1 else "Points (CSV)"
        files.append((write_csv(points, folder, f"embedding_{_stem(values)}.csv"), label))
    for values, embedding in embeddings.items():
        loadings = embedding.loadings.rename(columns={"pc1": "PC1", "pc2": "PC2", "top": "arrow"})
        label = f"Loadings, {values} (CSV)" if len(embeddings) > 1 else "Loadings (CSV)"
        files.append((write_csv(loadings, folder, f"pca_loadings_{_stem(values)}.csv"), label))
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
    # Each feature's table after it, with explainable tables (owner, 2026-10-06).
    named = ["feature", *(["table"] if "table" in tests else []), *HEATMAP_COLUMNS[1:]]
    frame = tests.reindex(columns=named)
    frame = frame.assign(_rank=frame["feature"].map(order)).sort_values("_rank", kind="stable")
    frame = frame.drop(columns="_rank").rename(
        columns={"passes": "candidate", "confirmed": "significant_across_repeats"}
    )
    return [(write_csv(frame, folder, "heatmap_d.csv"), "Data (CSV)")]


def across_data(contrast: Any, folder: Path) -> list[tuple[str, str]]:
    """The secondary rows of every feature (``ACROSS_CSV``, owner 2026-10-05): kind, label,
    feature, the biological repeats each used (named), the change (a fold change, or the
    difference where a mean is not positive), p and q; empty without them."""
    every = getattr(contrast, "settings", {}).get("across_features")
    if every is None:
        across = getattr(contrast, "across", None)
        if across is None or not len(across):
            return []
        every = across.assign(feature=contrast.settings["feature"])
    folded = every["fold_change"].notna()
    frame = pd.DataFrame(
        {
            "kind": every["kind"],
            "label": every["comparison"],
            "feature": every["feature"],
            "biological_repeats": every["held"].map(
                lambda held: "; ".join(map(str, held)) if isinstance(held, list) else ""
            ),
            "repeats": every["repeats"],
            "values": every["values"],
            "change": every["fold_change"].where(folded, every["effect"]),
            "change_is": folded.map({True: "fold change", False: "difference"}),
            "lower": every["lower"],
            "higher": every["higher"],
            "p_paired_t": every["p_paired_t"],
            "q_paired_t": every["q_paired_t"],
            "status": every["status"],
        }
    )
    return [(write_csv(frame, folder, ACROSS_CSV), "Data (CSV)")]


__all__ = [
    "EMBED_BUDGET",
    "across_data",
    "data_buttons",
    "download",
    "embedding_data",
    "feature_data",
    "heatmap_data",
    "inline_images",
    "json_island",
    "page_files",
    "write_csv",
]
