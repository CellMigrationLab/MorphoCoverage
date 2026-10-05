"""The contrast table of a ``Setup``: one control, or several with their conditions (private;
owner, 2026-10-05).

``Setup.control`` names one control, ``{"condition": "siCTRL"}``, and every condition is
compared with it within each biological repeat, as always. It may instead name several
controls, each mapping a treatment key to its condition (a plain list of conditions gives no
keys)::

    control={"condition": {
        "VEC_SiNC": {"SiCol6a1": "VEC_SiCol6a1", "SiFBN2": "VEC_SiFBN2"},
        "THBS5_SiNC": {"SiCol6a1": "THBS5_SiCol6a1", "SiFBN2": "THBS5_SiFBN2"},
        "VEC": {"THBS5": "THBS5"},
    }}
    control_parents={"VEC_SiNC": ("VEC", "transfection"), "THBS5_SiNC": ("THBS5", "transfection")}
    master_control="VEC_SiNC"

Each link, a condition to its control or a control to its parent (any condition, a member of
another control included; a cycle is refused), is one main verdict (``own_control``). Two
links with the same treatment key under different references give one ``across_normalised``
row: the two conditions each normalised to its own reference, the master's (else the first
declared) side as the reference. With a master, each other control against it on raw values
(``vs_master_raw``), unless a link already joins them. These rows make the contrast table
(``_control.CONTRAST_COLUMNS``), shown in the report and saved as ``contrasts.csv``;
``Setup.contrasts`` takes a table of the same columns in its place.

The normalised figures show each condition relative to its own control, a control relative
to its parent (else itself, so 1); a condition without its own control in a repeat is left
out there (``left_out_sentence``). Unusual observations and the population enrichment use
each condition's own control; the object rules, clustering and a positive control pool the
references (``reference_control``).

Shared-private: used by ``workflows.setup`` and the workflows that route the contrasts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._control import (
    ACROSS,
    CONTRAST_COLUMNS,
    MASTER_RAW,
    OWN,
    across_label,
    check_contrast_table,
    comparison_label,
    own_pairs,
    raw_label,
    table_conditions,
)
from .._wording import listing
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..normalize import control_mean_problem, control_means, negative_values_sentence
from ..retrospective.overview import normalized_sentence

if TYPE_CHECKING:
    from .setup import Setup

Link = tuple[Any, Any, Any]  # (condition, its reference, treatment key or None)


def _several(setup: Setup) -> Mapping[Any, Any] | None:
    spec = (setup.control or {}).get(CONDITION_COLUMN)
    return spec if isinstance(spec, Mapping) else None


def _members(value: Any, members: Any) -> list[tuple[Any, Any]]:
    """``(key, condition)`` of one control's declared conditions (key ``None`` for a list)."""
    if isinstance(members, Mapping):
        found = list(members.items())
    elif isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
        raise TypeError(
            f"control {value!r} must map treatment keys to its conditions, or list them; "
            f"got {members!r}"
        )
    else:
        found = [(None, m) for m in members]
    for key, member in found:
        if not pd.api.types.is_scalar(member) or not pd.api.types.is_scalar(key):
            raise TypeError(f"control {value!r} names {member!r}, not one condition label")
    return found


def links(setup: Setup) -> list[Link]:
    """Every declared link, members first (in the declared order), then the parents."""
    spec = _several(setup)
    if spec is None:
        return []
    found = [
        (m, value, key) for value, members in spec.items() for key, m in _members(value, members)
    ]
    for value, parent in (setup.control_parents or {}).items():
        target, key = parent if isinstance(parent, tuple) else (parent, None)
        found.append((value, target, key))
    return found


def check_controls(setup: Setup) -> None:
    """Refuse several controls, parents or a master the workflows could not use
    (``TypeError``/``ValueError``)."""
    if setup.contrasts is not None:
        check_contrast_table(setup.contrasts)
    spec = _several(setup)
    parents = setup.control_parents
    if spec is None:
        if parents:
            raise ValueError(
                "control_parents link several controls: declare them as "
                "control={'condition': {control: {treatment: condition, ...}, ...}}"
            )
        if (
            setup.master_control is not None
            and (setup.control or {}).get(CONDITION_COLUMN) != setup.master_control
        ):
            raise ValueError("master_control names one of several declared controls")
        return
    if len(setup.control) > 1 or not spec:
        raise ValueError(
            "several controls are named by the condition column alone: "
            "control={'condition': {control: {treatment: condition, ...}, ...}}"
        )
    if parents is not None and not isinstance(parents, Mapping):
        raise TypeError("control_parents must map a control to its parent condition")
    for value, parent in (parents or {}).items():
        if value not in spec:
            raise ValueError(f"control_parents names {value!r}, which is not a declared control")
        ok = isinstance(parent, tuple) and len(parent) == 2 or pd.api.types.is_scalar(parent)
        if not ok:
            raise TypeError(f"the parent of {value!r} is a condition, or (condition, treatment)")
    found = links(setup)
    reference: dict[Any, Any] = {}
    for child, target, _ in found:
        if child == target:
            raise ValueError(f"{child!r} is declared as its own control")
        if child in reference:
            raise ValueError(
                f"{child!r} belongs to two controls, {reference[child]!r} and {target!r}: a "
                "condition belongs to one control (link controls with control_parents)"
            )
        reference[child] = target
    keyed = [(target, key) for _, target, key in found if key is not None]
    twice = [k for i, k in enumerate(keyed) if k in keyed[:i]]
    if twice:
        raise ValueError(f"treatment {twice[0][1]!r} names two conditions of {twice[0][0]!r}")
    for start, first in reference.items():
        seen, node = {start}, first
        while node in reference:
            if node in seen:
                raise ValueError(f"the controls form a cycle through {node!r}")
            seen.add(node)
            node = reference[node]
    master = setup.master_control
    if master is not None and master not in spec:
        raise ValueError(f"master_control {master!r} is not one of the declared controls")


def generated(setup: Setup) -> pd.DataFrame | None:
    """The contrast table the declaration gives (see the module), ``None`` for one control."""
    spec = _several(setup)
    if spec is None:
        return None
    found = links(setup)
    rows = [
        _row(OWN, child, target, None, None, OWN, comparison_label(child, target))
        for child, target, _ in found
    ]
    master = setup.master_control
    rank = {c: i for i, c in enumerate(dict.fromkeys([master, *spec, *(t for _, t, _ in found)]))}
    by_key: dict[Any, list[Link]] = {}
    for link in found:
        if link[2] is not None:
            by_key.setdefault(link[2], []).append(link)
    for key, same in by_key.items():
        same = sorted(same, key=lambda link: rank[link[1]])
        for i, (b, ref_b, _) in enumerate(same):
            for a, ref_a, _ in same[i + 1 :]:
                if ref_a != ref_b:
                    label = across_label(a, ref_a, b, ref_b, key)
                    rows.append(_row(ACROSS, a, b, ref_a, ref_b, ACROSS, label))
    if master is not None:
        joined = {frozenset((child, target)) for child, target, _ in found}
        for value in spec:
            if value != master and frozenset((value, master)) not in joined:
                rows.append(
                    _row(
                        MASTER_RAW, value, master, None, None, MASTER_RAW, raw_label(value, master)
                    )
                )
    return pd.DataFrame(rows, columns=CONTRAST_COLUMNS)


def _row(*values: Any) -> dict[str, Any]:
    return dict(zip(CONTRAST_COLUMNS, values, strict=True))


def contrast_table(setup: Setup) -> pd.DataFrame | None:
    """``Setup.contrasts`` (checked), else the generated table, else ``None`` (one control)."""
    if setup.contrasts is not None:
        return check_contrast_table(setup.contrasts)
    table = generated(setup)
    return None if table is None else check_contrast_table(table)


def comparisons(setup: Setup) -> tuple[tuple[Any, Any], ...] | None:
    """The main verdicts as ``(condition, control)`` pairs; ``None`` with one control."""
    table = contrast_table(setup)
    return None if table is None else own_pairs(table)


def own_references(setup: Setup) -> dict[Any, Any] | None:
    """Each condition's own control (a control's parent, else itself); ``None`` with one
    control."""
    table = contrast_table(setup)
    if table is None:
        return None
    own = dict(own_pairs(table))
    for reference in list(own.values()):
        own.setdefault(reference, reference)
    return own


def experiments(setup: Setup) -> dict[Any, tuple[Any, ...]] | None:
    """Each declared control's experiment, ``{control: (control, its conditions..., its
    parent)}``: the control, the conditions declared under it and its parent when one is
    linked (a control linked to it as a child is its own experiment). With a contrast table
    given in place of a declaration, each reference and the conditions compared with it.
    An experiment held within another is left out; ``None`` with fewer than two (owner,
    2026-10-05)."""
    table = contrast_table(setup)
    if table is None:
        return None
    spec = _several(setup)
    found: dict[Any, list[Any]] = {}
    if spec is not None:
        for control, members in spec.items():
            found[control] = [control, *(m for _, m in _members(control, members))]
        for control, parent in (setup.control_parents or {}).items():
            target = parent[0] if isinstance(parent, tuple) else parent
            found.setdefault(control, [control]).append(target)
    else:
        for condition, control in own_pairs(table):
            found.setdefault(control, [control]).append(condition)
    sets = {control: set(members) for control, members in found.items()}
    kept = {
        control: tuple(dict.fromkeys(members))
        for control, members in found.items()
        if not any(sets[control] < sets[other] for other in sets)
    }
    return kept if len(kept) >= 2 else None


def compares(setup: Setup) -> bool:
    """Whether conditions are compared: a control on the condition column, or contrasts."""
    return setup.contrasts is not None or (
        setup.control is not None and CONDITION_COLUMN in setup.control
    )


def contrast_arguments(setup: Setup) -> dict[str, Any]:
    """The ``control``, or the contrast table, as the comparing analyses take them."""
    table = contrast_table(setup)
    if table is None:
        return {"control": setup.control}
    return {"control": None, "comparisons": own_pairs(table)}


def reference_control(setup: Setup) -> Mapping[str, Any] | None:
    """The reference population of the object rules, clustering and a positive control:
    the control, or every own control pooled."""
    own = own_references(setup)
    if own is None:
        return setup.control
    return {CONDITION_COLUMN: tuple(dict.fromkeys(own.values()))}


def figure_control(setup: Setup) -> Mapping[str, Any] | None:
    """What figures draw first: the control, or the master (else the first declared)."""
    spec = _several(setup)
    if spec is not None:
        return {CONDITION_COLUMN: setup.master_control or next(iter(spec))}
    if setup.contrasts is not None and setup.control is None:
        table = contrast_table(setup)
        return {CONDITION_COLUMN: table.loc[table["kind"] == OWN, "reference"].iloc[0]}
    return setup.control


def single_control(setup: Setup) -> Mapping[str, Any] | None:
    """The control when there is one and no contrast table (``None`` otherwise)."""
    return setup.control if contrast_table(setup) is None else None


def declared_conditions(setup: Setup) -> list[Any]:
    """Every condition the contrast table names (empty with one control)."""
    table = contrast_table(setup)
    return [] if table is None else table_conditions(table)


def left_out(setup: Setup, fovs: pd.DataFrame) -> dict[Any, list[Any]]:
    """Per condition, the repeats where it is held without its own control, which the
    normalised views leave out (empty with one control)."""
    own = own_references(setup)
    if own is None or REPEAT_COLUMN not in fovs or CONDITION_COLUMN not in fovs:
        return {}
    held = fovs.groupby(CONDITION_COLUMN, sort=False)[REPEAT_COLUMN].unique()
    found = {}
    for condition, repeats in held.items():
        mine = set(held.get(own.get(condition), [])) if condition in own else set()
        missing = [r for r in repeats if r not in mine]
        if missing:
            found[condition] = missing
    return found


def left_out_sentence(setup: Setup, fovs: pd.DataFrame) -> str:
    """``left out of the normalised values where their own control is not imaged: VEC_SiNC
    in R4, R5 and R6; ...`` (empty without)."""
    found = left_out(setup, fovs)
    if not found:
        return ""
    parts = [f"{c} in {listing([str(r) for r in repeats], 6)}" for c, repeats in found.items()]
    shown = "; ".join(parts[:6]) + (f"; and {len(parts) - 6} more" if len(parts) > 6 else "")
    return f"left out of the normalised values where their own control is not imaged: {shown}"


def controls_sentence(setup: Setup) -> str | None:
    """The several controls in a sentence (``None`` with one): each with its conditions,
    the parents, the master."""
    spec = _several(setup)
    if spec is None:
        return (
            None
            if setup.contrasts is None
            else ("Contrasts: as declared in the contrast table (contrasts.csv)")
        )
    each = "; ".join(
        f"{value} for {listing([str(m) for _, m in _members(value, members)], 8)}"
        for value, members in spec.items()
    )
    text = f"Controls: {each}"
    parents = [
        f"{value} under {parent[0] if isinstance(parent, tuple) else parent}"
        for value, parent in (setup.control_parents or {}).items()
    ]
    if parents:
        text += f"; {listing(parents, 6)}"
    master = setup.master_control
    text += (
        ". Each condition is tested against its own control within each biological repeat"
        + (f"; the master control is {master}" if master is not None else "")
        + "; the contrasts are listed in contrasts.csv"
    )
    return text


def normalisation_problem(setup: Setup, objects: pd.DataFrame, fovs: pd.DataFrame) -> str | None:
    """Why the values cannot be normalised (``normalize.control_mean_problem``): to the one
    control, or to any own control (a repeat without it is left out, not a problem)."""
    own = setup.own_references
    if own is None:
        controls, leave = [setup.control], False
    else:
        controls, leave = [{CONDITION_COLUMN: r} for r in dict.fromkeys(own.values())], True
    for control in controls:
        problem = control_mean_problem(
            objects,
            fovs,
            feature_cols=list(setup.columns),
            control=control,
            repeat_col=setup.repeat_col(fovs),
            leave_out_missing=leave,
            **setup.common,
        )
        if problem is not None:
            return problem
    return None


def normalized_text(
    setup: Setup, fovs: pd.DataFrame, objects: pd.DataFrame | None = None
) -> str | None:
    """``Setup.normalized_sentence``: how the values are normalised and what reads them, or
    ``None``; with several controls, the conditions left out where their own control is not
    imaged; with ``objects``, the caution when a control holds negative values."""
    if setup.normalize is None:
        return None
    if setup.own_references is None:
        assert setup.control is not None
        text = (
            f"{normalized_sentence(setup.control, setup.repeat_col(fovs))}, for the outcomes "
            "judged on held-back fields, the condition contrast, the feature ranking and the "
            "plots; the stopping rules and the other analyses read the raw values"
        )
        if objects is not None:
            means = control_means(
                objects,
                fovs,
                feature_cols=setup.columns,
                control=setup.control,
                repeat_col=setup.repeat_col(fovs),
                **setup.common,
            )
            note = negative_values_sentence(means)
            text += f". Caution: {note}" if note else ""
        return text
    left = left_out_sentence(setup, fovs)
    return (
        "every feature divided by the mean of each condition's own control in the same "
        "biological repeat (a control by its parent's, else by its own), for the plots and the "
        "populations; the raw values are where levels across experiments are read, and the "
        "tests read them too, each normalising to its own control" + (f"; {left}" if left else "")
    )


__all__ = [
    "check_controls",
    "compares",
    "comparisons",
    "contrast_arguments",
    "contrast_table",
    "controls_sentence",
    "declared_conditions",
    "experiments",
    "figure_control",
    "generated",
    "left_out",
    "left_out_sentence",
    "links",
    "normalisation_problem",
    "normalized_text",
    "own_references",
    "reference_control",
    "single_control",
]
