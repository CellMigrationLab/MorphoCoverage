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
    """Refuse several controls, parents or a master the workflows could not use, and a
    contrast table that gives a declared condition another own control
    (``TypeError``/``ValueError``)."""
    if setup.contrasts is not None:
        check_contrast_table(setup.contrasts)
    spec = _several(setup)
    if spec is None:
        _check_one_control(setup)
        return
    _check_several(setup, spec)
    found = links(setup)
    reference = _references(found, setup.control_parents or {})
    if not reference and len(spec) > 1 and setup.contrasts is None:
        raise ValueError(_nothing_under(spec))
    keyed = [(target, key) for _, target, key in found if key is not None]
    twice = [k for i, k in enumerate(keyed) if k in keyed[:i]]
    if twice:
        raise ValueError(f"treatment {twice[0][1]!r} names two conditions of {twice[0][0]!r}")
    _check_cycles(reference)
    master = setup.master_control
    if master is not None and master not in spec:
        raise ValueError(f"master_control {master!r} is not one of the declared controls")
    if setup.contrasts is not None:
        _check_table_controls(setup.contrasts, reference)


def _check_one_control(setup: Setup) -> None:
    """Refuse parents, or a master other than the control, beside one control."""
    if setup.control_parents:
        raise ValueError(
            "control_parents link several controls: declare them as "
            "control={'condition': {control: {treatment: condition, ...}, ...}}"
        )
    if (
        setup.master_control is not None
        and (setup.control or {}).get(CONDITION_COLUMN) != setup.master_control
    ):
        raise ValueError("master_control names one of several declared controls")


def _check_several(setup: Setup, spec: Mapping[Any, Any]) -> None:
    """Refuse several controls named beside other columns or none, and parents that are not
    a mapping of declared controls to a condition (or a condition and a treatment)."""
    parents = setup.control_parents
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


def _references(found: list[Link], parents: Mapping[Any, Any]) -> dict[Any, Any]:
    """Each linked condition's one reference; a condition linked to itself or twice is
    refused."""
    reference: dict[Any, Any] = {}
    for child, target, _ in found:
        if child == target:
            raise ValueError(f"{child!r} is declared as its own control")
        if child in reference:
            raise ValueError(_twice(child, reference[child], target, parents))
        reference[child] = target
    return reference


def _check_cycles(reference: Mapping[Any, Any]) -> None:
    """Refuse references that lead back to a condition already met."""
    for start, first in reference.items():
        seen, node = {start}, first
        while node in reference:
            if node in seen:
                raise ValueError(f"the controls form a cycle through {node!r}")
            seen.add(node)
            node = reference[node]


def _check_table_controls(contrasts: pd.DataFrame, reference: Mapping[Any, Any]) -> None:
    """Refuse a contrast table that gives a declared condition another own control."""
    for condition, given in own_pairs(check_contrast_table(contrasts)):
        if reference.get(condition, given) != given:
            raise ValueError(
                f"the contrast table compares {condition!r} with {given!r} as its own "
                f"control, but the declared controls give {reference[condition]!r}: a "
                "condition's own control comes from the declaration"
            )


def _nothing_under(spec: Mapping[Any, Any]) -> str:
    """Why several controls with no condition under any of them, and no parent, are refused:
    no condition would be compared with its own control (audit 8B)."""
    names = listing([repr(c) for c in spec], 6)
    return (
        f"the controls {names} have no condition under them and no parent, so no condition "
        "would be compared with its own control: declare each control's conditions, "
        "control={'condition': {control: [condition, ...], ...}}, or link a control to its "
        "parent with control_parents={control: parent}"
    )


def _twice(child: Any, first: Any, second: Any, parents: Mapping[Any, Any]) -> str:
    """Why ``child`` cannot have both references: listed twice, or under two controls (a
    member and a parent, or two members)."""
    if first == second:
        return f"{child!r} is listed twice under control {first!r}"
    if child in parents:
        return (
            f"{child!r} is listed under control {first!r} and linked to the parent "
            f"{second!r}: a control is compared with one reference, so keep one of the two"
        )
    return (
        f"{child!r} belongs to two controls, {first!r} and {second!r}: a condition belongs "
        "to one control (link controls with control_parents)"
    )


def one_control(setup: Setup) -> dict[str, Any] | None:
    """A single control declared in the several-controls forms (a list, a keyed dict, a
    master naming it) as ``{"condition": control}``, so it gives exactly what one control
    gives (owner, 2026-10-05); ``None`` otherwise, and with parents or a contrast table."""
    spec = _several(setup)
    if spec is None or len(spec) != 1 or setup.control_parents or setup.contrasts is not None:
        return None
    return {CONDITION_COLUMN: next(iter(spec))}


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
    """Each condition's own control (a control's parent, else itself), from the declared
    controls when there are, else from the contrast table; ``None`` with one control."""
    table = contrast_table(setup)
    if table is None:
        return None
    # The declaration, when there is one, says each condition's own control (owner,
    # 2026-10-05); a contrast table adds the conditions it does not name.
    own = {**dict(own_pairs(table)), **{child: target for child, target, _ in links(setup)}}
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
            repeat_col=setup.pairing_col(fovs),
            leave_out_missing=leave,
            **setup.common,
        )
        if problem is not None:
            return problem
    return None


# Biological unusualness and the definition of new where it judges the positive control and
# the fields to check read the normalised values (owner, 2026-10-06, plan v2).
_BIOLOGY = "unusual observations, new states for the positive control and the fields to check"


def normalised_readers(several: bool = False) -> str:
    """What reads the normalised values and what reads the raw ones, the one list the overview,
    the declaration, ``docs/common/setup.md`` and the docstrings refer to (audit 8B): with one
    control, or with ``several``, each condition then normalised to its own. The fingerprint
    and the held-back outcomes read the normalised values in both (``Setup.compared_values``;
    owner, 2026-10-06, audit 8, seed 8)."""
    if not several:
        return (
            f"the figures, the normalised maps, the populations, the separation, {_BIOLOGY}, "
            "the fingerprint and the outcomes judged on held-back fields read the normalised "
            "values; the condition contrast, the feature ranking and the discovery read the raw "
            "values and normalise them themselves; the stopping rules, the acquisition "
            "anomalies and every other analysis read the raw values"
        )
    return (
        f"the figures, the normalised maps, the populations, the separation, {_BIOLOGY}, the "
        "fingerprint and the outcomes judged on held-back fields read the normalised values; "
        "the condition contrast, the feature ranking and the discovery read the raw values, "
        "each normalising to its own control; the stopping rules, the acquisition anomalies "
        "and every other analysis read the raw values, where levels across experiments are read"
    )


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
        words = setup.unit_words if setup.observation_unit is not None else "objects"
        text = (
            f"{normalized_sentence(setup.control, setup.repeat_col(fovs), words)}; "
            f"{normalised_readers()}"
        )
        if objects is not None:
            means = control_means(
                objects,
                fovs,
                feature_cols=setup.columns,
                control=setup.control,
                repeat_col=setup.pairing_col(fovs),
                **setup.common,
            )
            note = negative_values_sentence(means)
            text += f". Caution: {note}" if note else ""
        return text
    left = left_out_sentence(setup, fovs)
    return (
        "every feature divided by the mean of each condition's own control in the same "
        f"biological repeat (a control by its parent's, else by its own); "
        f"{normalised_readers(several=True)}" + (f"; {left}" if left else "")
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
    "normalised_readers",
    "normalized_text",
    "one_control",
    "own_references",
    "reference_control",
    "single_control",
]
