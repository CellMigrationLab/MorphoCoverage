"""The guided session's other feature tables (private; ``guided``): asked for at load, each
with a path, a name and its tag, checked against the observation table (owner, 2026-10-05).

``ask_tables`` asks "Do you have other feature tables for the same observations?"; each table
is read, asked whether it is explainable (each column read and tested one by one, like area
or intensity) or not (an embedding such as DINO, read only as a whole; owner, 2026-10-06),
and checked with the ones before it (``attach_feature_tables``), and one that cannot be used
is said and asked again. ``read_tables`` reads tables given to ``guide`` (a DataFrame or a
CSV path by name, or either with its tag), and ``entries`` records them in the saved answers:
each name, path, tag and checksum, of the rows sorted by their identity so that the row order
does not count; answers are reused only for the same names, checksums and tags (a saved
table without a tag is non-explainable), and ``changed`` names the table that differs (owner,
2026-10-06).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .._wording import count
from ..feature_tables import (
    EXPLAINABLE_TAG,
    NON_EXPLAINABLE_TAG,
    attach_feature_tables,
    identity_text,
    read_tag,
)
from . import _guided_ask as q

QUESTION = (
    "Do you have other feature tables for the same observations, e.g. deep-learning "
    "embeddings such as DINO?"
)
TAG_QUESTION = (
    "Is {name} explainable, each column read and tested one by one, like area or intensity? "
    "No: an embedding such as DINO, read only as a whole."
)
# Each table: its DataFrame, the path it was read from (None for a DataFrame) and its tag.
Tables = dict[str, tuple[pd.DataFrame, str | None, str]]


def read_table(source: Any, name: str = "feature table") -> tuple[pd.DataFrame, str | None]:
    """A table given as a DataFrame or a CSV path, and its path; ``name`` says which table
    a missing file was meant to be."""
    if isinstance(source, pd.DataFrame):
        return source, None
    path = Path(str(source)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"the {name} {str(path)!r} does not exist")
    return pd.read_csv(path), str(path)


def read_tables(given: Mapping[str, Any] | None) -> Tables:
    """The tables given to ``guide`` by name, read: each a DataFrame or a CSV path,
    non-explainable, or either with its tag, ``(source, "explainable")``."""
    tables: Tables = {}
    for name, source in (given or {}).items():
        tag: str | None = NON_EXPLAINABLE_TAG
        if isinstance(source, tuple) and len(source) == 2:
            source, tag = source[0], read_tag(source[1])
        if tag is None:
            raise ValueError(f"tag the table {name!r} 'explainable' or 'non-explainable'")
        tables[name] = (*read_table(source), tag)
    return tables


def tagged(tables: Tables) -> dict[str, tuple[pd.DataFrame, str]]:
    """The tables with their tags, as ``attach_feature_tables`` takes them."""
    return {name: (table, tag) for name, (table, _, tag) in tables.items()}


def ask_tables(
    ask: Callable[[str], str],
    say: Callable[[str], None],
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
) -> Tables:
    """The other feature tables, asked for one at a time, each checked with the others."""
    tables: Tables = {}
    if not q.yes_no(ask, say, QUESTION, False):
        return tables
    while True:
        path = q.text(ask, "Path to the table (Enter when there are no more):")
        if not path:
            return tables
        try:
            table, read = read_table(path)
        except (OSError, ValueError) as error:  # fallback: asked again, with what was wrong
            say(f"It cannot be read: {error}.")
            continue
        name = q.text(
            ask, "Its name, as the report shows it (for example DINOv2):", Path(path).stem
        )
        explainable = q.yes_no(ask, say, TAG_QUESTION.format(name=name), False)
        tag = EXPLAINABLE_TAG if explainable else NON_EXPLAINABLE_TAG
        trial = {**tagged(tables), name: (table, tag)}
        try:
            attached = attach_feature_tables(objects, fovs, trial)
        except (ValueError, TypeError) as error:  # fallback: asked again, with what was wrong
            say(f"It cannot be used: {error}.")
            continue
        tables[name] = (table, read, tag)
        columns = count(len(attached.columns[name]), "feature column")
        say(f"{name}: {columns}, {tag}, checked.")


def _sorted(table: pd.DataFrame, objects: pd.DataFrame) -> pd.DataFrame:
    """``table`` with its rows sorted by its identity columns (those it shares with the
    observation table, matched as ``attach_feature_tables`` matches them), so a table saved
    in another row order has the same checksum (owner, 2026-10-06)."""
    key = [c for c in table.columns if c in objects.columns]
    if not key:
        return table
    text = pd.DataFrame({c: [identity_text(v) for v in table[c].tolist()] for c in key})
    return table.iloc[text.sort_values(key, kind="stable").index.to_numpy()]


def entries(tables: Tables, objects: pd.DataFrame) -> list[dict[str, Any]]:
    """The tables as the saved answers record them: name, path, tag and checksum, of the rows
    sorted by their identity."""
    from .guided import table_checksum  # guided imports this module

    return [
        {
            "name": name,
            "path": path,
            "tag": tag,
            "checksum": table_checksum(_sorted(table, objects)),
        }
        for name, (table, path, tag) in tables.items()
    ]


def tag_of(entry: Mapping[str, Any]) -> str:
    """A saved table's tag; non-explainable when saved without one (before 2026-10-06)."""
    return read_tag(entry.get("tag")) or NON_EXPLAINABLE_TAG


def identity(dataset: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """The tables' names, checksums and tags, which must match for saved answers to be
    reused."""
    return [(e["name"], e["checksum"], tag_of(e)) for e in dataset.get("feature_tables") or []]


def changed(saved: Mapping[str, Any], dataset: Mapping[str, Any]) -> list[str]:
    """What differs between the tables the answers were saved for and these: ``the
    observation table``, ``the FOV table``, ``the feature table 'DINOv2'`` (changed, added or
    no longer given), so a refusal names it (owner, 2026-10-06)."""
    found = []
    if saved.get("checksum") != dataset.get("checksum"):
        found.append("the observation table")
    if saved.get("fovs_checksum", "missing") != dataset.get("fovs_checksum"):
        found.append("the FOV table")
    before = {name: rest for name, *rest in identity(saved)}
    now = {name: rest for name, *rest in identity(dataset)}
    for name in dict.fromkeys([*before, *now]):
        if name not in now:
            found.append(f"the feature table {name!r}, not given this time")
        elif name not in before:
            found.append(f"the feature table {name!r}, new")
        elif before[name] != now[name]:
            found.append(f"the feature table {name!r}")
    if saved.get("description") != dataset.get("description"):
        found.append("the experiment description")
    return found


__all__ = [
    "QUESTION",
    "TAG_QUESTION",
    "Tables",
    "ask_tables",
    "changed",
    "entries",
    "identity",
    "read_table",
    "read_tables",
    "tag_of",
    "tagged",
]
