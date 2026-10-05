"""Parts of a ``Setup`` declared on their own (private): the positive control and the
default definition of new.

Shared-private: re-exported by ``workflows.setup``, where they are imported from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._former_repeat import read_fields
from ..metadata import REPEAT_COLUMN
from ..strategies import SupportNoveltyEstimator
from ..transforms import RobustStandardize

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


__all__ = ["MANY_NOVELTY_FEATURES", "PositiveControl", "default_novelty"]
