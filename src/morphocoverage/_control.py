"""The declared control and the declared comparisons, resolved in one place (private).

Every analysis that compares conditions takes ``control``, a mapping of FOV-table columns to
the control's value, ``{"condition": "siCTRL"}``: each condition is then compared with it,
biological repeat by biological repeat. It may instead take ``comparisons`` (owner,
2026-10-05): a sequence of ``(condition, reference)`` pairs, each one test of that condition
against that reference, per biological repeat, in the repeats where both are imaged. A
condition may be in several (``THBS5_SiCol6a1`` against ``THBS5_SiNC`` and against
``VEC_SiCol6a1``) and a reference may be tested itself; each comparison's rows are labelled
``THBS5_SiCol6a1 vs THBS5_SiNC`` (``comparison_label``) in the column that holds the
condition with a single control, so every result keyed on it reads per comparison. Where an
analysis needs one reference population (unusual objects, the object rules, clustering, the
positive control), the references are pooled (``reference_population``): a control value may
then be a tuple of accepted values.

Shared-private: used by the analyses that select the control or the comparisons' sides and
the texts that name them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._wording import listing

# What joins a comparison's condition and reference in its label.
VS = " vs "


def _accepted(spec: Any) -> list[Any]:
    """The values a control column accepts: one, or each of a tuple, list or set."""
    return list(spec) if isinstance(spec, (tuple, list, set, frozenset)) else [spec]


def control_mask(table: pd.DataFrame, control: Mapping[str, Any]) -> np.ndarray:
    """Which rows of ``table`` match ``control`` (every column; a tuple accepts any of it)."""
    mask = np.ones(len(table), dtype=bool)
    for column, spec in control.items():
        if isinstance(spec, (tuple, list, set, frozenset)):
            mask &= table[column].isin(list(spec)).to_numpy(dtype=bool)
        else:
            mask &= table[column].eq(spec).to_numpy(dtype=bool)
    return mask


def group_matches(fovs: pd.DataFrame, control: Mapping[str, Any]) -> bool:
    """Whether one analysis group (its FOV rows, not empty) holds the control's values only,
    one combination of them."""
    if not len(fovs):
        return False
    values = fovs[list(control)].drop_duplicates()
    return len(values) == 1 and bool(control_mask(values, control).all())


def control_name(control: Mapping[str, Any] | None, column: str | None = None) -> str:
    """The control in words: its values joined by `` · `` (``siCTRL``), a pooled column's
    accepted values by ``/``; ``column`` names one column's only."""
    control = control or {}
    specs = [control[column]] if column is not None else list(control.values())
    return " · ".join("/".join(str(v) for v in _accepted(spec)) for spec in specs)


def control_text(control: Mapping[str, Any] | None) -> str:
    """The control as messages print it: ``{'condition': 'siCTRL'}``; references pooled read
    ``{'condition': 'VEC/VEC_SiNC'}``."""
    control = dict(control or {})
    if not any(isinstance(v, (tuple, list, set, frozenset)) for v in control.values()):
        return str(control)
    return str({column: control_name(control, column) for column in control})


def check_comparisons(comparisons: Any, name: str = "comparisons") -> tuple[tuple[Any, Any], ...]:
    """``comparisons`` as a tuple of ``(condition, reference)`` pairs; ``TypeError`` or
    ``ValueError`` for anything else, a pair of one condition with itself or a pair twice."""
    if isinstance(comparisons, (str, Mapping)) or not isinstance(comparisons, Sequence):
        raise TypeError(f"{name} must be a sequence of (condition, reference) pairs")
    pairs = []
    for pair in comparisons:
        if isinstance(pair, str) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise TypeError(f"{name} must hold (condition, reference) pairs, got {pair!r}")
        condition, reference = pair
        if not pd.api.types.is_scalar(condition) or not pd.api.types.is_scalar(reference):
            raise TypeError(f"{name} must name each condition by one label, got {pair!r}")
        if condition == reference:
            raise ValueError(f"{name} compares {condition!r} with itself")
        pairs.append((condition, reference))
    if not pairs:
        raise ValueError(f"{name} must hold at least one (condition, reference) pair")
    twice = [p for i, p in enumerate(pairs) if p in pairs[:i]]
    if twice:
        raise ValueError(f"{name} names {twice[0]!r} twice")
    return tuple(pairs)


def comparison_label(condition: Any, reference: Any) -> str:
    """``THBS5_SiCol6a1 vs THBS5_SiNC``: one comparison, as its rows are labelled."""
    return f"{condition}{VS}{reference}"


def comparison_pairs(comparisons: Sequence[tuple[Any, Any]]) -> dict[str, tuple[Any, Any]]:
    """Each comparison's label with its ``(condition, reference)``, in the declared order."""
    return {comparison_label(c, r): (c, r) for c, r in comparisons}


def references(comparisons: Sequence[tuple[Any, Any]] | None) -> list[Any]:
    """The references of ``comparisons``, in the declared order (empty without)."""
    return list(dict.fromkeys(r for _, r in comparisons or ()))


def reference_population(
    control: Mapping[str, Any] | None,
    comparisons: Sequence[tuple[Any, Any]] | None,
    condition_col: str,
) -> Mapping[str, Any] | None:
    """The reference population of the analyses that need one: the control when declared,
    else every comparison's reference pooled (``{condition_col: ("VEC", "VEC_SiNC")}``)."""
    if control is not None:
        return control
    if not comparisons:
        return None
    return {condition_col: tuple(references(comparisons))}


def versus(condition: Any, control: Any) -> str:
    """``cond vs ctrl`` for a row of ``condition`` against ``control``; a comparison's label
    already says it, so it is returned as is."""
    text = str(condition)
    if text.endswith(f"{VS}{control}"):
        return text
    return f"{text}{VS}{control}"


def compared_name(settings: Mapping[str, Any]) -> str:
    """What a result's conditions were compared with, for its sentences: the control's
    values (``siCTRL``), or with comparisons ``their references`` (named in each row)."""
    if settings.get("comparisons"):
        return "their references"
    return control_name(settings.get("control"))


def not_found(comparisons: Sequence[tuple[Any, Any]], conditions: Sequence[Any]) -> str | None:
    """Why ``comparisons`` cannot run on these ``conditions``: the labels absent, or
    ``None``."""
    present = set(conditions)
    missing = [c for pair in comparisons for c in pair if c not in present]
    if not missing:
        return None
    return (
        f"the comparisons name conditions absent from the tables: "
        f"{listing([repr(c) for c in dict.fromkeys(missing)], 6)} (values are matched "
        "exactly, case and spaces included)"
    )


__all__ = [
    "VS",
    "check_comparisons",
    "compared_name",
    "comparison_label",
    "comparison_pairs",
    "control_mask",
    "control_name",
    "control_text",
    "group_matches",
    "not_found",
    "reference_population",
    "references",
    "versus",
]
