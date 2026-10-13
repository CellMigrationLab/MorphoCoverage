"""The checks of a ``Setup`` as it is built (private; split out of ``Setup.__post_init__``):
each refuses a declaration the workflows could not use, with the message the user reads,
in the order ``__post_init__`` runs them.

Shared-private: used by ``workflows.setup``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from ..metadata import MORPHOBRICKS_GROUP_COLUMNS
from ..normalize import METHODS as NORMALIZATIONS
from ..objectives import RareTarget
from ..retrospective.criteria import OutcomeCriterion
from ._feature_space import table_columns

if TYPE_CHECKING:
    from .setup import Setup


def normalise_sequences(setup: Setup) -> None:
    """Refuse a readout that is not one column name, and store the sequence fields as
    tuples (a string refused)."""
    if setup.readout is not None and (not isinstance(setup.readout, str) or not setup.readout):
        raise TypeError("readout must be one column name (None: no explainable features)")
    for name in ("features", "tolerances", "effort_strata", "intensity_features"):
        value = getattr(setup, name)
        if value is None and name == "effort_strata":
            continue  # one stratum per condition (resolved_effort_strata)
        if isinstance(value, str):
            raise TypeError(f"{name} must be a sequence, not a string")
        object.__setattr__(setup, name, tuple(value))
    if setup.group_cols is not None:
        if isinstance(setup.group_cols, str):
            raise TypeError("group_cols must be a sequence of column names")
        object.__setattr__(setup, "group_cols", tuple(setup.group_cols))


def check_declared_types(setup: Setup) -> None:
    """Store the rules and the fingerprint as tuples, and refuse ``fields_independent``,
    ``fingerprint``, ``rare`` and ``rare_label`` of another type (a label needs its target)."""
    if setup.rules is not None:
        object.__setattr__(setup, "rules", tuple(setup.rules))
    if not isinstance(setup.fields_independent, (bool, np.bool_)):
        raise TypeError("fields_independent must be True or False")
    object.__setattr__(setup, "fields_independent", bool(setup.fields_independent))
    if isinstance(setup.fingerprint, OutcomeCriterion):
        raise TypeError("fingerprint must be a sequence of OutcomeCriterion, not one")
    object.__setattr__(setup, "fingerprint", tuple(setup.fingerprint))
    if any(not isinstance(c, OutcomeCriterion) for c in setup.fingerprint):
        raise TypeError("fingerprint must be a sequence of OutcomeCriterion")
    if setup.rare is not None and not isinstance(setup.rare, RareTarget):
        raise TypeError("rare must be a RareTarget")
    if setup.rare_label is not None:
        if not isinstance(setup.rare_label, str) or not setup.rare_label:
            raise TypeError("rare_label must be one object column name")
        if setup.rare is None:
            raise ValueError("rare_label labels a declared rare target: declare rare too")


def check_tolerances_and_control(setup: Setup) -> None:
    """Refuse tolerances that are not fractions, none without another target, and a control
    that does not map columns to values."""
    if any(not 0 < t < 1 for t in setup.tolerances):
        raise ValueError("tolerances must be fractions between 0 and 1, such as 0.05")
    if not setup.tolerances and not (setup.fingerprint or setup.rare is not None):
        raise ValueError("tolerances=() declares no mean target: declare the fingerprint")
    if setup.control is not None and (not isinstance(setup.control, Mapping) or not setup.control):
        raise TypeError("control must map FOV-table columns to values")


def check_normalize(setup: Setup) -> None:
    """Refuse an unknown normalisation, and one without a control naming the condition."""
    if setup.normalize is None:
        return
    if setup.normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {list(NORMALIZATIONS)} or None")
    chosen = setup.figure_control
    if chosen is None or MORPHOBRICKS_GROUP_COLUMNS[0] not in chosen:
        raise ValueError(
            "normalize needs control to name the condition column, for example "
            "control={'condition': 'MOCK'}"
        )


def check_declared_columns(setup: Setup) -> None:
    """Refuse intensity features and fingerprint columns that are not declared, and two
    criteria alike."""
    if set(setup.intensity_features) - set(setup.columns):
        raise ValueError("intensity_features must be declared columns (readout or features)")
    undeclared = set(setup.criterion_columns) - {*setup.columns, *table_columns(setup)}
    if undeclared:
        raise ValueError(
            f"fingerprint criteria read feature columns {sorted(undeclared)[:3]} (of "
            f"{len(undeclared)}) that are not declared: add them to features, or declare "
            "them as another feature table, feature_tables={name: columns}"
        )
    names = [c.name for c in setup.criteria()]
    if len(set(names)) != len(names):
        raise ValueError("the declared criteria must be distinct (same metric and features)")


__all__ = [
    "check_declared_columns",
    "check_declared_types",
    "check_normalize",
    "check_tolerances_and_control",
    "normalise_sequences",
]
