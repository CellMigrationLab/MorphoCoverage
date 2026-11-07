"""The analysis groups and the two sides of each comparison (private).

``describe_groups`` gives each analysis group its condition, its biological repeat and
whether it is the control. ``contrasts_of`` lists what is compared, repeat by repeat, and
``sides`` the treated and control groups of one of them. With a single control (``control``)
each condition other than the control is compared with the control groups of its repeat.
With declared comparisons (``_control``, owner 2026-10-05) each ``(condition, reference)``
pair is compared, labelled ``condition vs reference`` (``_control.comparison_label``): the
treated groups are the condition's in the repeat, the control groups the reference's, and a
condition may be on either side of several comparisons.

Shared-private: re-exported by ``_units`` for the contrast, the ranking, the discovery, the
distribution and ``retrospective.effect_preservation``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .._control import comparison_pairs, not_found
from .._identity import identity_key
from ..schema import ValidatedInput
from ._common import no_match_message


class Described(list):
    """The described groups (a list of dicts), with the declared comparisons by label
    (``None`` with a single control)."""

    comparisons: dict[str, tuple[Any, Any]] | None = None


def describe_groups(
    all_groups: list[tuple[dict[str, Any], ValidatedInput]],
    condition_col: str,
    repeat_col: str | None,
    control: Mapping[str, Any] | None,
    comparisons: Sequence[tuple[Any, Any]] | None = None,
) -> Described:
    """Each analysis group with its condition, repeat and whether it is the control.

    Raises when a group spans several conditions or repeats (the column is
    then missing from ``group_cols``), when no group matches ``control``, or when a
    comparison names a condition the tables do not hold. With ``comparisons`` no group is
    the control: each comparison names its own sides (``sides``).
    """

    def value_of(group_data: ValidatedInput, column: str) -> Any:
        values = group_data.fovs[column].drop_duplicates()
        if len(values) != 1:
            raise ValueError(
                f"an analysis group spans several values of {column!r}; add it to group_cols"
            )
        return values.iloc[0]

    if comparisons is None and not control:
        raise ValueError("declare the control (a mapping of FOV-table column to value)")
    described = Described()
    for group, group_data in all_groups:
        described.append(
            {
                "group": group,
                "condition": value_of(group_data, condition_col),
                "repeat": value_of(group_data, repeat_col) if repeat_col else None,
                "is_control": comparisons is None
                and all(
                    value_of(group_data, column) == value
                    for column, value in (control or {}).items()
                ),
            }
        )
    if comparisons is not None:
        problem = not_found(comparisons, [d["condition"] for d in described])
        if problem:
            raise ValueError(problem)
        described.comparisons = comparison_pairs(comparisons)
        refuse_unshared_repeat_labels(described)
        return described
    if not any(d["is_control"] for d in described):
        raise ValueError(no_match_message(all_groups, control or {}, "control"))
    refuse_unshared_repeat_labels(described)
    return described


def refuse_unshared_repeat_labels(described: Described) -> None:
    """Refuse a comparison whose two sides have repeats but no repeat label in common (#617).

    The repeats of two compared conditions are paired by their folder label (the
    ``biological_repeat`` value). A pairing confirmed under different folder names never reaches
    this brick, so labels that share nothing would read as "not compared", or be paired with
    whatever label happens to match. Until the Spec v1 reader carries the confirmed repeat group
    the comparison is refused; nothing is paired here. A repeat that only one side holds is not a
    mismatch (an uneven design keeps its shared labels).

    Raises
    ------
    ValueError
        Naming the repeat labels of both sides, for the first comparison that shares none.
    """
    pairs = getattr(described, "comparisons", None)
    controls = sorted({str(d["condition"]) for d in described if d["is_control"]})
    if pairs is not None:
        compared = list(pairs.values())
    else:
        conditions = dict.fromkeys(d["condition"] for d in described if not d["is_control"])
        compared = [(condition, None) for condition in conditions]
    for treated, reference in compared:
        treated_labels = _repeat_labels(described, lambda d, c=treated: d["condition"] == c)
        reference_labels = _repeat_labels(
            described,
            lambda d, r=reference: d["condition"] == r if r is not None else d["is_control"],
        )
        if treated_labels and reference_labels and not set(treated_labels) & set(reference_labels):
            reference_name = str(reference) if reference is not None else "/".join(controls)
            raise ValueError(
                f"the repeat labels of {str(treated)!r} ({_listed(treated_labels)}) and of "
                f"{reference_name!r} ({_listed(reference_labels)}) differ and share none: "
                "repeats are paired by their folder label, and a pairing confirmed under "
                "different folder names is not read until the Spec v1 reader lands. Refused, "
                "not guessed."
            )


def _repeat_labels(described: list[dict[str, Any]], keep: Any) -> list[Any]:
    """The repeat labels of the groups ``keep`` selects, in a stable order (none unlabelled)."""
    found = {d["repeat"] for d in described if keep(d) and d["repeat"] is not None}
    return sorted(found, key=identity_key)


def _listed(labels: Sequence[Any]) -> str:
    """``'R1', 'R2'``."""
    return ", ".join(repr(label) for label in labels)


def contrasts_of(described: list[dict[str, Any]]) -> list[tuple[Any, Any]]:
    """The ``(repeat, condition)`` pairs to compare with the control, in a stable order; with
    comparisons, ``(repeat, label)`` for each repeat the comparison's condition is in."""
    pairs = getattr(described, "comparisons", None)
    if pairs is not None:
        found = {
            (d["repeat"], label)
            for label, (condition, _) in pairs.items()
            for d in described
            if d["condition"] == condition
        }
    else:
        found = {(d["repeat"], d["condition"]) for d in described if not d["is_control"]}
    return sorted(found, key=lambda pair: (identity_key(pair[0]), identity_key(pair[1])))


def sides(
    described: list[dict[str, Any]], repeat: Any, condition: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The treated groups of ``condition`` and the control groups, within ``repeat``; with
    comparisons, ``condition`` is a comparison's label and the control groups its
    reference's."""
    pairs = getattr(described, "comparisons", None)
    if pairs is not None:
        treated_condition, reference = pairs[condition]
        in_repeat = [d for d in described if d["repeat"] == repeat]
        treated = [d["group"] for d in in_repeat if d["condition"] == treated_condition]
        controls = [d["group"] for d in in_repeat if d["condition"] == reference]
        return treated, controls
    controls = [d["group"] for d in described if d["is_control"] and d["repeat"] == repeat]
    treated = [
        d["group"]
        for d in described
        if not d["is_control"] and d["repeat"] == repeat and d["condition"] == condition
    ]
    return treated, controls


def control_label(described: list[dict[str, Any]], controls: list[dict[str, Any]]) -> str:
    """The control conditions of ``controls``, joined by ``/`` (empty without controls)."""
    return "/".join(sorted({str(d["condition"]) for d in described if d["group"] in controls}))


def reference_of(
    described: list[dict[str, Any]],
    condition: Any,
    control: Mapping[str, Any] | None,
    condition_col: str,
) -> str:
    """The control a row of ``condition`` names when its repeat holds none: the declared
    control's condition, or the comparison's reference."""
    pairs = getattr(described, "comparisons", None)
    if pairs is not None:
        return str(pairs[condition][1])
    return str((control or {})[condition_col])


__all__ = [
    "Described",
    "contrasts_of",
    "control_label",
    "describe_groups",
    "reference_of",
    "sides",
]
