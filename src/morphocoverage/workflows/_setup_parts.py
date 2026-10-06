"""Parts of a ``Setup`` declared on their own (private): the positive control and its
checks, the default definition of new, the objective of the declared targets.

Shared-private: re-exported by ``workflows.setup``, where they are imported from.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._checks import check_seed, is_integer, is_number
from .._control import comparison_label
from .._former_repeat import read_fields
from .._parallel import check_n_jobs
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..objectives import ObjectiveConfig, SamplingObjective
from ..statistics import SCALES
from ..strategies import SupportNoveltyEstimator
from ..transforms import RobustStandardize, Standardize

if TYPE_CHECKING:
    from .setup import Setup

# From this many features on, a radius of sqrt(d) links too few objects of a state.
MANY_NOVELTY_FEATURES = 8


def default_novelty(features: Sequence[str]) -> SupportNoveltyEstimator:
    """The definition of "new" used in ``docs/preliminary_findings.md`` (entries 2, 3, 6).

    Robustly standardized features (``RobustStandardize``: median and MAD, so
    a few extreme objects such as clumps or debris cannot widen the scale and
    hide a new state elsewhere), a radius of ``sqrt(len(features))``, support
    from one other FOV, clusters of at least 3 unsupported objects, 5 FOVs
    before any estimate. It was chosen on idr0028 (development data) with the
    mean and SD, robust since audit 4: call it explicitly, and check it with
    ``representation_control`` on your data.

    A state's observations must be linked by steps of at most ``sqrt(d)``,
    though two of a state varying by one SD per feature sit ``sqrt(2 d)``
    apart: with 8 or more features ``Setup`` warns (reduce the dimension).
    """
    return SupportNoveltyEstimator(float(np.sqrt(len(features))), 1, RobustStandardize(), 5, 3)


@dataclass(frozen=True)
class PositiveControl:
    """A declared positive control, checked against ``Setup.control`` (report question R1).

    ``selector`` picks the positive-control groups by FOV-table values, for
    example ``{"condition": "YAP"}``. ``fovs_seen`` reference FOVs are seen
    before a positive FOV is judged; ``pair_cols`` pairs positive and
    reference FOVs (for example by plate). By default (``None``) they are
    paired within each repeat when the FOV table has a repeat column, so a
    shift between repeats is never read as a detected phenotype;
    ``pair_cols=()`` pools every repeat, as explicitly asked, and the
    declaration and the report say so. ``planted_objects`` also plants that
    many positive objects into reference FOVs.
    """

    selector: Mapping[str, Any]
    fovs_seen: int
    pair_cols: tuple[str, ...] | None = None
    planted_objects: int | None = None
    draws: int = 20

    def __post_init__(self) -> None:
        if self.pair_cols is not None:
            if isinstance(self.pair_cols, str):
                raise TypeError("pair_cols must be a sequence of column names, not a string")
            object.__setattr__(self, "pair_cols", tuple(self.pair_cols))
        read_fields(self, "selector", "pair_cols")  # the former name ``repeat``

    def resolved_pair_cols(self, fovs: pd.DataFrame) -> tuple[str, ...]:
        """``pair_cols`` as declared; by default the repeat column when the FOV table has one."""
        if self.pair_cols is not None:
            return self.pair_cols
        return (REPEAT_COLUMN,) if REPEAT_COLUMN in fovs.columns else ()


def check_positive(setup: Setup) -> None:
    """A positive control is a ``PositiveControl``, checked against the control with the
    definition of new: refused otherwise, with what to declare."""
    if setup.positive is None:
        return
    if not isinstance(setup.positive, PositiveControl):
        raise TypeError("positive must be a PositiveControl")
    if setup.reference_control is None:
        raise ValueError("a positive control is checked against the control: declare both")
    if setup.novelty is None:
        raise ValueError(
            "a positive control is judged with the definition of new: declare novelty "
            "(and features), or leave the positive control out"
        )
    # Fewer seen fields than the definition of new needs: no field is ever judged (F10).
    novelty, seen = setup.novelty, setup.positive.fovs_seen
    needed = max(novelty.min_fovs, novelty.min_other_fovs)
    if is_integer(seen) and seen < needed:
        name = "min_fovs" if novelty.min_fovs >= novelty.min_other_fovs else "min_other_fovs"
        raise ValueError(
            f"the positive control is judged after fovs_seen={seen} reference fields, but the "
            f"definition of new judges a field only after {needed} (its {name}): declare "
            f"fovs_seen={needed} or more"
        )


def check_scales(setup: Setup) -> None:
    """``scales`` (plan v2, owner, 2026-10-06): a mapping of explainable features to
    ``"ratio"`` (the paired test across repeats on the log ratio to the control) or
    ``"difference"``, stored as a dict; a feature not declared keeps the automatic rule
    (``statistics.log_scale``). A name that is no declared explainable column is refused once
    the columns are known (``features`` declared, or the explainable tables attached);
    a ratio on a value that is not positive is refused where the values are read."""
    if setup.scales is None:
        return
    if not isinstance(setup.scales, Mapping):
        raise TypeError("scales must map feature names to 'ratio' or 'difference'")
    object.__setattr__(setup, "scales", dict(setup.scales))
    for feature, scale in setup.scales.items():
        if not isinstance(feature, str) or scale not in SCALES:
            raise ValueError(
                f"scales must map feature names to 'ratio' or 'difference', not "
                f"{feature!r}: {scale!r}"
            )
    tables = setup.explainable_tables
    if setup.features or tables is not None:
        known = {*setup.columns, *(c for columns in (tables or {}).values() for c in columns)}
        unknown = sorted(set(setup.scales) - known)
        if unknown:
            raise ValueError(
                f"scales names {unknown[:3]}, not declared explainable features: a scale is "
                "declared for the features tested one by one (readout, features)"
            )


def check_presets(setup: Setup) -> None:
    """The novelty target, the detection preset and the computational settings."""
    target = setup.novelty_target
    if target is not None and (not is_number(target) or not 0 < float(target) < 1):
        raise ValueError("novelty_target must be a probability between 0 and 1, or None")
    if target is not None:
        object.__setattr__(setup, "novelty_target", float(target))
    detection = setup.minimum_detection
    if not is_number(detection) or not 0 < float(detection) <= 1:
        raise ValueError("minimum_detection must be a share in (0, 1], such as 0.9")
    object.__setattr__(setup, "minimum_detection", float(detection))
    object.__setattr__(setup, "random_state", check_seed(setup.random_state))
    check_n_jobs(setup.n_jobs)
    object.__setattr__(setup, "n_jobs", int(setup.n_jobs))


def warn_novelty_dimension(setup: Setup) -> None:
    """Warn when a standardization alone defines "new" on many features (P6-3)."""
    d = len(setup.features)
    novelty = setup.novelty
    if novelty is None or d < MANY_NOVELTY_FEATURES:
        return
    if isinstance(novelty.transform, (Standardize, RobustStandardize)):
        warnings.warn(
            f"novelty is judged on {d} standardized features at distance "
            f"{novelty.epsilon:g}, but two observations of a state varying by one SD per "
            f"feature sit about {np.sqrt(2 * d):.1f} apart, so small, ordinarily variable "
            "states are often missed: reduce the dimension (Chain([RobustStandardize(), "
            "PCA(k)])) or declare fewer features",
            UserWarning,
            stacklevel=3,
        )


def check_embedding_only(setup: Setup) -> None:
    """An embedding-only Setup (``readout=None``, owner 2026-10-06) declares no explainable
    feature, nothing to normalise and no intensity: refused otherwise, with what to do."""
    if setup.readout is not None:
        return
    if setup.features:
        raise ValueError(
            "features are explainable features, read one by one: declare the readout among "
            "them (readout=None declares none, for a report on the other feature tables alone)"
        )
    if setup.intensity_features:
        raise ValueError("intensity_features are explainable features: readout=None has none")


def check_few_fovs(setup: Setup) -> None:
    """``few_fovs``: an integer above the definition of new's ``min_other_fovs``."""
    if setup.few_fovs is None:
        return
    if setup.novelty is None:
        raise ValueError("few_fovs counts states under the definition of new: declare novelty")
    if not is_integer(setup.few_fovs):
        raise TypeError("few_fovs must be an integer number of fields")
    object.__setattr__(setup, "few_fovs", int(setup.few_fovs))
    if setup.few_fovs <= setup.novelty.min_other_fovs:
        raise ValueError(
            f"few_fovs ({setup.few_fovs}) must exceed the definition of new's min_other_fovs "
            f"({setup.novelty.min_other_fovs}): a state seen in that many other fields is "
            "already not new"
        )


def check_experiment(setup: Setup) -> None:
    """``experiment`` is ``None``, a column name, or a mapping condition -> experiment (kept
    as a dict); anything else is refused (audit 10, owner, 2026-10-06)."""
    value = setup.experiment
    if value is None or (isinstance(value, str) and value):
        return
    if not isinstance(value, Mapping) or not value:
        raise TypeError(
            "experiment must be a FOV-table column naming each condition's experiment, or a "
            "mapping condition -> experiment, such as {'THBS5': 'untransfected'}"
        )
    object.__setattr__(setup, "experiment", dict(value))


def experiment_families(setup: Setup, fovs: pd.DataFrame) -> dict[Any, Any] | None:
    """Each condition's declared experiment (``Setup.experiment``), ``None`` without one: the
    mapping as declared, or the FOV table's column, one experiment per condition (a control
    serving several experiments is left out: a test belongs to its condition's)."""
    value = setup.experiment
    if value is None:
        return None
    if isinstance(value, Mapping):
        return _with_labels(dict(value), setup)
    if value not in fovs.columns:
        raise ValueError(
            f"experiment names {value!r}, which is not a column of the FOV table: name the "
            "column holding each condition's experiment, or pass a mapping condition -> "
            "experiment"
        )
    pairs = fovs[[CONDITION_COLUMN, value]].dropna().drop_duplicates()
    counts = pairs[CONDITION_COLUMN].value_counts()
    controls = set((setup.own_references or {}).values())
    controls |= {(setup.single_control or {}).get(CONDITION_COLUMN)}
    several = [c for c in counts.index[counts > 1] if c not in controls]
    if several:
        raise ValueError(
            f"experiment: {several[:3]} hold several values of {value!r}; a condition other "
            "than a control belongs to one experiment"
        )
    multiple = set(counts.index[counts > 1])
    found = dict(zip(pairs[CONDITION_COLUMN], pairs[value], strict=True))
    return _with_labels({c: e for c, e in found.items() if c not in multiple}, setup)


def _with_labels(families: dict[Any, Any], setup: Setup) -> dict[Any, Any]:
    """``families`` with each comparison's label (``THBS5_SiFBN2 vs THBS5_SiNC``) mapped to
    its condition's experiment, as the tests name the rows of several controls."""
    for condition, reference in setup.comparisons or ():
        if condition in families:
            families.setdefault(comparison_label(condition, reference), families[condition])
    return families


def setup_objective(setup: Setup) -> ObjectiveConfig:
    """The objective whose components are the declared targets (``Setup.objective``).

    The population summary whenever the readout's mean is a target; a fingerprint adds
    the distribution component, a rare target the rare component; with none of them, no
    component (``no_target``: nothing decides a stop). The live decision and every replay
    run under it, so a rule that has no evidence about a declared target cannot alone say
    stop, and a target no rule decides stays ``unsupported`` in the decision view rather
    than being assumed adequate.
    """
    if not setup.mean_target:  # no mean: the fingerprint and the rare phenotype only
        if setup.rare is None and not setup.fingerprint:
            # No target at all: no component, nothing decides a stop (audit 10, owner,
            # 2026-10-06; before, a fingerprint was required that nobody declared).
            return ObjectiveConfig(SamplingObjective.NO_TARGET)
        if setup.rare is None:
            return ObjectiveConfig(SamplingObjective.DISTRIBUTION)
        if setup.fingerprint:  # owner, 2026-10-06 (audit 8): no population component
            return ObjectiveConfig(SamplingObjective.DISTRIBUTION_RARE, rare=setup.rare)
        return ObjectiveConfig(SamplingObjective.RARE_DISCOVERY, rare=setup.rare)
    if setup.rare is not None and setup.fingerprint:
        return ObjectiveConfig(SamplingObjective.ALL, rare=setup.rare)
    if setup.rare is not None:
        return ObjectiveConfig(SamplingObjective.BALANCED, rare=setup.rare)
    if setup.fingerprint:
        return ObjectiveConfig(SamplingObjective.PHENOTYPE)
    return ObjectiveConfig(SamplingObjective.POPULATION)


__all__ = [
    "MANY_NOVELTY_FEATURES",
    "PositiveControl",
    "check_embedding_only",
    "check_experiment",
    "check_few_fovs",
    "check_positive",
    "check_presets",
    "check_scales",
    "default_novelty",
    "experiment_families",
    "setup_objective",
    "warn_novelty_dimension",
]
