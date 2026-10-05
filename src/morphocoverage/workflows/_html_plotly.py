"""The HTML report's interactive feature figures (private; ``html_report``), drawn by
plotly.js in the page from the numbers of the static figures.

- ``plotly_js``: the plotly.js bundle the page embeds once (``assets/``, MIT licence), so the
  figures work offline.
- ``feature_specs``: per static figure of ``workflows.plots`` (one feature, one part of its
  conditions), the data the page's script draws: every observation as a point at its
  cluster (a condition and biological repeat, side by side as in the static figure), in
  its condition's colour (``_palette.COLOURS``, as the PCA and UMAP), with its repeat,
  technical repeat and field on hover; each cluster's summary and interval, the very
  numbers of the static figure (``Figure.cluster_summaries``); and every pair of conditions
  (``Figure.pairwise``) for the matrix under it. Beyond ``MAX_POINTS`` observations a
  figure shows a seeded sample of each cluster in proportion (the summaries and tests keep
  every observation), and says so.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._palette import COLOURS
from ..data_check import validate_input_tables
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN
from ..plots_pairwise import ALPHA, D_CAP, format_p, p_shade
from ..sampling_unit.contrast_repeats import MIN_REPEATS
from ..schema import fov_positions
from ._rare_share import plotted_values

ASSET = Path(__file__).with_name("assets") / "plotly-cartesian.min.js"
MAX_POINTS = 4000
JITTER = 0.36


@cache
def plotly_js() -> str:
    """The plotly.js bundle, as text for a ``<script>`` element."""
    return ASSET.read_text(encoding="utf-8").replace("</script", "<\\/script")


def _number(value: Any, digits: int = 4) -> float | None:
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def _observations(
    prepared: pd.DataFrame, fovs: pd.DataFrame, feature: str, setup: Any
) -> pd.DataFrame:
    """Every observation's value and labels, on the values the static figures plot (the
    labelled share per field, ``_rare_share.plotted_values``)."""
    prepared = plotted_values(prepared, fovs, setup, feature)
    data = validate_input_tables(prepared, fovs=fovs, feature_cols=[feature], **setup.common)
    meta = data.fovs.iloc[fov_positions(data, data.fovs)].reset_index(drop=True)
    frame = pd.DataFrame({"value": data.feature_values(feature)})
    for name, column in (
        ("condition", CONDITION_COLUMN),
        ("repeat", REPEAT_COLUMN),
        ("technical", TECHNICAL_REPEAT_COLUMN),
        ("field", data.fov_col),
    ):
        frame[name] = meta[column].to_numpy() if column in meta else None
    return frame[np.isfinite(frame["value"])]


def _pairwise(table: pd.DataFrame, names: list[Any]) -> dict[str, Any] | None:
    """The matrix of every pair: d in the lower-left triangle, the corrected p above."""
    n = len(names)
    if n < 2 or table is None or table.empty:
        return None
    lookup = {(r.first, r.second): r for r in table.itertuples()}
    d = [[None] * n for _ in range(n)]
    p = [[None] * n for _ in range(n)]
    text = [[""] * n for _ in range(n)]
    hover = [[""] * n for _ in range(n)]
    for i, row in enumerate(names):
        for j, column in enumerate(names):
            pair = lookup.get((row, column)) or lookup.get((column, row))
            if i == j or pair is None:
                continue
            if i > j:
                value = pair.d if pair.first == row else -pair.d
                d[i][j] = _number(value, 3)
                apart = pair.shared_repeats == 0
                text[i][j] = "–" if d[i][j] is None else f"{value:+.2f}" + ("†" if apart else "")
                # A cell without a value is still drawn, at the middle colour, to show "–".
                d[i][j] = 0.0 if d[i][j] is None else d[i][j]
                mine, theirs = (pair.observations, pair.second_observations)
                if pair.first != row:
                    mine, theirs = theirs, mine
                hover[i][j] = (
                    f"{row} minus {column}: Cohen's d {text[i][j]} ({mine} vs {theirs} "
                    "observations)"
                    + ("; imaged in different repeats, so d holds the batch too" if apart else "")
                )
            else:
                q = float(pair.q)
                apart = pair.shared_repeats == 0
                p[i][j] = round(p_shade(q), 3) if np.isfinite(q) else 0.0
                text[i][j] = "–" if not np.isfinite(q) else (format_p(q) + ("†" if apart else ""))
                paired = (
                    f"; the paired t-test on {pair.shared_repeats} repeats' means gives p "
                    f"{format_p(float(pair.p_paired))}"
                    if np.isfinite(float(pair.p_paired))
                    else ""
                )
                hover[i][j] = (
                    f"{row} and {column}: randomization test on d, p {format_p(float(pair.p))},"
                    f" adjusted p {format_p(q)} (Benjamini-Hochberg over the pairs){paired}"
                    + ("; imaged in different repeats, so the batch is in it" if apart else "")
                    if np.isfinite(q)
                    else f"{row} and {column}: not tested ({pair.status})"
                )
    panel_b = {"lower": d, "upper": p, "text": text, "hover": hover, "lower_scale": "d"}
    return {
        "names": [str(c) for c in names],
        "b": panel_b,
        "c": _paired_panel(lookup, names),
        "alpha": ALPHA,
        "cap": D_CAP,
    }


def _paired_panel(lookup: dict, names: list[Any]) -> dict[str, Any]:
    """Panel C: the paired t-test on the repeats' means, uncorrected p in the lower-left
    triangle, the Benjamini-Hochberg q above."""
    n = len(names)
    lower = [[None] * n for _ in range(n)]
    upper = [[None] * n for _ in range(n)]
    text = [[""] * n for _ in range(n)]
    hover = [[""] * n for _ in range(n)]
    for i, row in enumerate(names):
        for j, column in enumerate(names):
            pair = lookup.get((row, column)) or lookup.get((column, row))
            if i == j or pair is None:
                continue
            value = float(pair.p_paired if i > j else pair.q_paired)
            target = lower if i > j else upper
            target[i][j] = round(p_shade(value), 3) if np.isfinite(value) else 0.0
            text[i][j] = format_p(value) if np.isfinite(value) else "–"
            what = (
                "p, uncorrected"
                if i > j
                else "adjusted p, Benjamini-Hochberg over the pairs tested"
            )
            hover[i][j] = (
                f"{row} and {column}: paired t-test on {pair.shared_repeats} repeats' means "
                f"({pair.paired_values}), {what} {text[i][j]}"
                if np.isfinite(value)
                else f"{row} and {column}: not tested ({pair.shared_repeats} repeats imaged "
                f"both, fewer than {MIN_REPEATS})"
            )
    return {"lower": lower, "upper": upper, "text": text, "hover": hover, "lower_scale": "p"}


def _points(frame: pd.DataFrame, summaries: pd.DataFrame, seed: int) -> dict[str, Any]:
    """The observations of the figure's clusters, jittered within each, at most
    ``MAX_POINTS`` (a seeded sample of each cluster in proportion)."""
    pooled = summaries["repeat"].isna().all()
    rng = np.random.default_rng(seed)
    total = 0
    chunks = []
    for row in summaries.itertuples():
        mine = frame["condition"] == row.condition
        if not pooled:
            mine &= frame["repeat"] == row.repeat
        block = frame[mine]
        total += len(block)
        chunks.append((row, block))
    share = min(1.0, MAX_POINTS / total) if total else 1.0
    out: dict[str, list] = {"x": [], "y": [], "cluster": [], "label": [], "shape": []}
    for index, (row, block) in enumerate(chunks):
        if share < 1.0 and len(block):
            keep = max(1, round(len(block) * share))
            block = block.iloc[np.sort(rng.choice(len(block), keep, replace=False))]
        jitter = rng.uniform(-JITTER, JITTER, size=len(block)) * row.width
        out["x"] += [round(float(row.x + j), 3) for j in jitter]
        out["y"] += [_number(v) for v in block["value"]]
        out["cluster"] += [index] * len(block)
        # Each technical repeat's shape, by its rank within the biological repeat, as in the
        # static figure.
        ranks = {t: i for i, t in enumerate(dict.fromkeys(block["technical"]))}
        out["shape"] += [ranks[t] for t in block["technical"]]
        out["label"] += [
            " · ".join(
                str(v) for v in values if v is not None and pd.notna(v) and str(v) != "single"
            )
            for values in block[["repeat", "technical", "field"]].itertuples(index=False)
        ]
    out["shown"], out["total"] = len(out["x"]), total
    return out


def interactive_caption(figure: Any) -> str:
    """The static figure's title in sentences, and what hovering a point gives."""
    lines = [line.strip() for line in getattr(figure, "caption", "").split("\n") if line.strip()]
    if len(lines) > 1:
        lines[1] += "; hover for its repeat, well and field"
    # The first line starts with the feature's own name, kept as written.
    return ". ".join(lines[:1] + [line[:1].upper() + line[1:] for line in lines[1:]]) + "."


def feature_specs(
    drawn: dict[str, list[Any]],
    prepared: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Any,
    colours: dict[Any, str],
    names: dict[str, str],
) -> list[dict[str, Any]]:
    """One spec per static figure (see the module); ``names`` are the figures' file names,
    ``colours`` each condition's colour."""
    specs = []
    for feature, parts in drawn.items():
        frame = _observations(prepared, fovs, feature, setup)
        for index, figure in enumerate(parts, start=1):
            summaries = getattr(figure, "cluster_summaries", None)
            if summaries is None or summaries.empty:
                continue
            conditions = list(dict.fromkeys(summaries["condition"]))
            centres = summaries.groupby("condition", sort=False)["x"].agg(["min", "max"])
            name = names[f"{feature}#{index}"]
            own = getattr(figure, "colours", {})
            spec = {
                "id": name,
                "feature": str(feature),
                "caption": interactive_caption(figure),
                "ylabel": figure.axes[0].get_ylabel(),
                "conditions": [
                    {
                        "name": "all fields" if c is None else str(c),
                        "centre": round(float(centres.loc[c].mean()), 3),
                        # The static figure's own colours: each part colours its own.
                        "colour": own.get(c, colours.get(c, COLOURS[0])),
                    }
                    for c in conditions
                ],
                "clusters": [
                    {
                        "condition": conditions.index(row.condition),
                        "repeat": None
                        if row.repeat is None or pd.isna(row.repeat)
                        else str(row.repeat),
                        "x": round(float(row.x), 3),
                        "width": float(row.width),
                        "summary": _number(row.summary),
                        "low": _number(row.low),
                        "high": _number(row.high),
                    }
                    for row in summaries.itertuples()
                ],
                "points": _points(frame, summaries, setup.random_state + index),
                "pairwise": _pairwise(getattr(figure, "pairwise", None), conditions),
            }
            specs.append(spec)
    return specs


__all__ = ["MAX_POINTS", "feature_specs", "interactive_caption", "plotly_js"]
