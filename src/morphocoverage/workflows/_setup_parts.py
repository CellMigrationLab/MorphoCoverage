"""Parts of a ``Setup`` declared on their own (private): the positive control and the
default definition of new.

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
from .._former_repeat import read_fields
from .._parallel import check_n_jobs
from ..metadata import REPEAT_COLUMN
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


__all__ = [
    "MANY_NOVELTY_FEATURES",
    "PositiveControl",
    "check_few_fovs",
    "check_presets",
    "default_novelty",
    "warn_novelty_dimension",
]
