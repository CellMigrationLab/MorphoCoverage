"""The workflows read a MorphoFeature export's experiment description (MC-2, private).

``described_setup(experiment, readout=..., **fields)`` is a ``Setup`` whose controls default to
the description's references (``morphocoverage.described.DescribedExperiment``): one control,
or one ``own_control`` row per condition and its reference, so chains are compared as chains.
Controls the user declares (``control``, ``contrasts``, ``control_parents``,
``master_control``) are used instead; when they differ from the references the Setup records
the override (``Setup.described``), and the declaration, printed first by the report, names the
description's reference beside the user's choice (owner answer 1). An export of a described
run without the block defaults nothing (``MISSING``) and accepts controls declared by ID (owner
decision, 12 Nov 2026).

``with_targets`` gives the condition contrast the reagent grouping the user confirmed
(``Setup.targets``) as the FOV table's ``target`` column of a working copy; ``check_described``
checks the description again against the FOV table a workflow reads.

Shared-private: used by ``workflows.scan``, ``_pairing``, ``_shared`` and the package's
``workflows`` namespace (``described_setup``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from .._control import control_name, own_pairs
from .._wording import listing
from ..described import ControlSource, DescribedExperiment
from ..metadata import CONDITION_COLUMN, TARGET_COLUMN
from .setup import Setup

# The Setup fields that declare controls: any one of them replaces the default.
CONTROL_FIELDS = ("control", "contrasts", "control_parents", "master_control")


def _chosen_pairs(setup: Setup, experiment: DescribedExperiment) -> set[tuple[Any, Any]] | None:
    """The ``(condition, control)`` comparisons the Setup declares, on the described
    conditions; ``None`` for a control that is not one condition."""
    table = setup.contrast_table
    if table is not None:
        return set(own_pairs(table))
    control = dict(setup.control or {})
    value = control.get(CONDITION_COLUMN)
    if set(control) != {CONDITION_COLUMN} or not isinstance(value, str):
        return None
    return {(c, value) for c in experiment.present if c != value}


def _chosen_text(setup: Setup) -> str:
    """The user's controls in words."""
    table = setup.contrast_table
    if table is not None:
        return listing([f"{c} vs {r}" for c, r in own_pairs(table)], 8)
    return control_name(setup.control)


def described_setup(experiment: DescribedExperiment, **fields: Any) -> Setup:
    """A ``Setup`` (``fields`` as for ``Setup``) whose controls default to ``experiment``'s
    references (see the module); refused (``DescriptionRefused``, ``MISSING``) when the export
    of a described run has no experiment block and no control is declared."""
    if not isinstance(experiment, DescribedExperiment):
        raise TypeError("experiment must be a DescribedExperiment (read_experiment)")
    declared = any(fields.get(name) is not None for name in CONTROL_FIELDS)
    if not experiment.described:  # not a described run: the Setup as given
        return Setup(**fields)
    if not declared:
        fields.update(experiment.default_control())  # MISSING without a block
    setup = Setup(**fields)
    chosen = None
    if declared:
        pairs = _chosen_pairs(setup, experiment)
        if experiment.block is None or pairs != set(experiment.pairs):
            chosen = _chosen_text(setup)
    return replace(setup, described=ControlSource(experiment, chosen))


def described_design(design: dict[str, Any], description: Any, control_declared: bool) -> None:
    """The scan's design read from a described export (``description``, in place): the
    control suggested is the description's default, never a name read as a control, and the
    reagent groups are the description's target suggestions, never names read as reagents;
    an undescribed export, or none, leaves the design as scanned."""
    if description is None:
        return
    if not isinstance(description, DescribedExperiment):
        raise TypeError("description must be a DescribedExperiment (read_experiment)")
    if not description.described:
        return
    default = {} if description.missing is not None else description.default_control()
    control = default.get("control", {}).get(CONDITION_COLUMN)
    design["control_guesses"] = [control] if control is not None else []
    if not control_declared:
        design["control_guess"] = control
    design["reagent_groups"] = {}
    design["described"] = description
    suggestions = description.target_suggestions
    design["target_suggestions"] = suggestions.lines() if suggestions else []


def check_described(setup: Setup, fovs: pd.DataFrame) -> None:
    """Refuse a FOV table that disagrees with the Setup's description (``DISAGREES``)."""
    source = setup.described
    if source is None or source.experiment.block is None:
        return
    source.experiment.block.check_fovs(fovs, listing(list(source.experiment.sources), 3))


def with_targets(fovs: pd.DataFrame, setup: Setup) -> pd.DataFrame:
    """A working copy of the FOV table with the confirmed targets as its ``target`` column
    (the conditions named as the working tables name them); ``fovs`` without targets."""
    if not setup.targets:
        return fovs
    if TARGET_COLUMN in fovs.columns:
        raise ValueError(
            f"the FOV table has a {TARGET_COLUMN!r} column and the Setup declares targets: "
            "give the reagent grouping once, in one of them"
        )
    return fovs.assign(**{TARGET_COLUMN: fovs[CONDITION_COLUMN].map(dict(setup.targets))})


__all__ = [
    "CONTROL_FIELDS",
    "check_described",
    "described_design",
    "described_setup",
    "with_targets",
]
