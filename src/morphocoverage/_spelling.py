"""Typing errors of a name (private; ``metadata.canonical_spelling``): ``biological_repat`` for
``biological_repeat``, ``conditon`` for ``condition``."""

from __future__ import annotations

from collections.abc import Mapping


def edits(first: str, second: str) -> int:
    """The edits between two names: insertions, deletions, substitutions and swaps of two
    neighbouring letters (the optimal string alignment distance)."""
    rows = [list(range(len(second) + 1))]
    for i, a in enumerate(first, 1):
        row = [i]
        for j, b in enumerate(second, 1):
            row.append(min(rows[-1][j] + 1, row[j - 1] + 1, rows[-1][j - 1] + (a != b)))
            if i > 1 and j > 1 and a == second[j - 2] and first[i - 2] == b:
                row[j] = min(row[j], rows[-2][j - 2] + 1)
        rows.append(row)
    return rows[-1][-1]


def typing_error_of(name: str, targets: Mapping[str, str]) -> str | None:
    """The target ``name`` is a typing error of, from ``{written form: name meant}``: one edit
    for a name of up to 11 letters, two for a longer one when the lengths differ by at most
    one, so a longer column (``technicalrepeatid``) is not taken for the shorter name."""
    for target, meant in targets.items():
        allowed = 2 if len(target) >= 12 and abs(len(name) - len(target)) <= 1 else 1
        if 0 < edits(name, target) <= allowed:
            return meant
    return None


__all__ = ["edits", "typing_error_of"]
