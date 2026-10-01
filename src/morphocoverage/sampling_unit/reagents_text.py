"""Whether independent reagents for one target agree (question 9, with a ``target`` column).

One table: per target with two or more reagents, the reagents and in how many
of the repeats holding them all every one is on the same side of the control;
one note: agreement between independent reagents speaks against an off-target
effect of one, and it is described, not tested. ``reagents_headline`` gives the
"At a glance" words.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .._caveats import assumption
from .._style import table, wrap
from .._wording import count


def _targets(contrast: Any) -> pd.DataFrame:
    return getattr(contrast, "targets", pd.DataFrame())


def _agree(row: dict[str, Any]) -> str:
    n, agree, direction = int(row["repeats"]), int(row["agree"]), row["direction"]
    if direction in ("lower", "higher"):
        return f"all {direction} in {agree} of {n}"
    if direction == "mixed":
        return f"on one side in {agree} of {n}, lower in some and higher in others"
    return f"on one side in none of {n}"


def reagents_parts(contrast: Any, indent: int = 2) -> tuple[list[str], list[str]]:
    """The agreement table under ``indent``, and its note (empty without targets)."""
    targets = _targets(contrast)
    if targets.empty:
        return [], []
    column = contrast.settings.get("target_col") or "target"
    rows = [
        [str(row["target"]), ", ".join(map(str, row["reagents"])), _agree(row)]
        for row in targets.to_dict("records")
    ]
    lines = [
        wrap(
            f"Independent reagents for one target ({column}), against the control in each repeat:",
            indent,
            hang=indent + 2,
        ),
        table(rows, ["target", "reagents", "repeats"], indent=indent + 2),
    ]
    note = assumption(
        "Reagents of one target that go the same way in every repeat speak against an "
        "off-target effect of one of them; this is described, not tested, since they share the "
        "control."
    )
    return lines, [note]


def reagents_headline(contrast: Any) -> str | None:
    """``reagents agree in every repeat for 5 of 6 targets``, or ``None`` without targets."""
    targets = _targets(contrast)
    if targets.empty:
        return None
    every = int(
        (
            (targets["agree"] == targets["repeats"])
            & targets["direction"].isin(["lower", "higher"])
        ).sum()
    )
    return f"reagents agree in every repeat for {every} of {count(len(targets), 'target')}"


__all__ = ["reagents_headline", "reagents_parts"]
