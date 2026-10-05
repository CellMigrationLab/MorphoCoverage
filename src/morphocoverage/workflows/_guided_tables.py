"""The guided session's other feature tables (private; ``guided``): asked for at load, each
with a path and a name, checked against the observation table (owner, 2026-10-05).

``ask_tables`` asks "Do you have other feature tables for the same observations?"; each table
is read and checked with the ones before it (``attach_feature_tables``), and one that cannot
be used is said and asked again. ``read_tables`` reads tables given to ``guide`` (a
DataFrame or a CSV path by name), and ``entries`` records them in the saved answers: each
name, path and checksum; answers are reused only for the same names and checksums.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .._wording import count
from ..feature_tables import attach_feature_tables
from . import _guided_ask as q

QUESTION = (
    "Do you have other feature tables for the same observations, e.g. deep-learning "
    "embeddings such as DINO?"
)
# Each table: its DataFrame and the path it was read from (None for a DataFrame).
Tables = dict[str, tuple[pd.DataFrame, str | None]]


def _read(source: Any) -> tuple[pd.DataFrame, str | None]:
    """A table given as a DataFrame or a CSV path, and its path."""
    if isinstance(source, pd.DataFrame):
        return source, None
    path = Path(str(source)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"the feature table {str(path)!r} does not exist")
    return pd.read_csv(path), str(path)


def read_tables(given: Mapping[str, Any] | None) -> Tables:
    """The tables given to ``guide`` by name, read."""
    return {name: _read(source) for name, source in (given or {}).items()}


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
            table, read = _read(path)
        except (OSError, ValueError) as error:
            say(f"It cannot be read: {error}.")
            continue
        name = q.text(
            ask, "Its name, as the report shows it (for example DINOv2):", Path(path).stem
        )
        trial = {**{n: t for n, (t, _) in tables.items()}, name: table}
        try:
            attached = attach_feature_tables(objects, fovs, trial)
        except (ValueError, TypeError) as error:
            say(f"It cannot be used: {error}.")
            continue
        tables[name] = (table, read)
        say(f"{name}: {count(len(attached.columns[name]), 'feature column')}, checked.")


def entries(tables: Tables) -> list[dict[str, Any]]:
    """The tables as the saved answers record them: name, path and checksum."""
    from .guided import table_checksum  # guided imports this module

    return [
        {"name": name, "path": path, "checksum": table_checksum(table)}
        for name, (table, path) in tables.items()
    ]


def identity(dataset: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The tables' names and checksums, which must match for saved answers to be reused."""
    return [(e["name"], e["checksum"]) for e in dataset.get("feature_tables") or []]


__all__ = ["QUESTION", "Tables", "ask_tables", "entries", "identity", "read_tables"]
