"""What a dataset is, declared once for every workflow (``report``, ``optimise``, ``live``,
``benchmark``).

Every scientific choice the workflows make is a field here, visible and
recorded; the workflows only add the computational settings (draws, splits,
orders) and run the package's analyses in the order of ``docs/questions.md``.
A field left at ``None`` switches off the analyses that need it, and the
report then says "Not assessed" with what it would need.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..retrospective.hierarchy import SamplingHierarchy
from ..strategies import ScalarPrecisionStrategy, SupportNoveltyEstimator
from ..strategies.base import SamplingStrategy
from ..transforms import Standardize


def default_novelty(features: Sequence[str]) -> SupportNoveltyEstimator:
    """The definition of "new" used in ``docs/preliminary_findings.md`` (entries 2, 3, 6).

    Standardized features, a radius of ``sqrt(len(features))``, support from
    one other FOV, clusters of at least 3 unsupported objects, 5 FOVs before
    any estimate. It was chosen on idr0028 (development data): call it
    explicitly, and check it with ``representation_control`` on your data.
    """
    return SupportNoveltyEstimator(float(np.sqrt(len(features))), 1, Standardize(), 5, 3)


@dataclass(frozen=True)
class PositiveControl:
    """A declared positive control, checked against ``Setup.control`` (report question R1).

    ``selector`` picks the positive-control groups by FOV-table values, for
    example ``{"condition": "YAP"}``. ``fovs_seen`` reference FOVs are seen
    before a positive FOV is judged; ``pair_cols`` pairs positive and
    reference FOVs (for example by plate); ``planted_objects`` also plants that
    many positive objects into reference FOVs.
    """

    selector: Mapping[str, Any]
    fovs_seen: int
    pair_cols: tuple[str, ...] = ()
    planted_objects: int | None = None
    draws: int = 20


@dataclass(frozen=True)
class Setup:
    """The declaration of a dataset shared by the four workflows.

    - ``readout``: the scalar the microscopist reports (one feature column).
    - ``features``: the representation for unusual objects and novelty
      (explicit columns; never inferred). Empty means the readout only.
    - ``control``: the negative-control groups, for example
      ``{"condition": "MOCK"}``; it normalizes the JCB rule and is the
      reference for unusual objects.
    - ``positive``: a ``PositiveControl``, for the detection check.
    - ``tolerances``: the readout's mean within these fractions, the outcomes
      judged on held-back FOVs and the JCB targets; default the paper's 5% and
      2.5%.
    - ``novelty``: the definition of "new" (``default_novelty(features)`` or
      your own); ``None`` leaves the novelty questions not assessed.
    - ``hierarchy`` and ``effort_strata``: FOV, sample and repeat columns for
      "more fields or more samples?" and protocol reliability.
    - ``acquisition_order_col``, ``stage_cols``: acquisition order and stage
      coordinates, when recorded.
    - ``fov_col``, ``group_cols``, ``observation_unit``: as for
      ``validate_input_tables`` (MorphoBricks defaults when ``None``).
    - ``rules``: the stopping rules to assess, calibrate and run live; by
      default the JCB rule (upper-95 SD, first tolerance) per object and per
      FOV.
    """

    readout: str
    features: tuple[str, ...] = ()
    control: Mapping[str, Any] | None = None
    positive: PositiveControl | None = None
    tolerances: tuple[float, ...] = (0.05, 0.025)
    novelty: SupportNoveltyEstimator | None = None
    novelty_target: float = 0.05
    hierarchy: SamplingHierarchy | None = None
    effort_strata: tuple[str, ...] = ()
    acquisition_order_col: str | None = None
    stage_cols: tuple[str, str] | None = None
    fov_col: str | None = None
    group_cols: tuple[str, ...] | None = None
    observation_unit: str | None = None
    rules: tuple[SamplingStrategy, ...] | None = None
    random_state: int = 0
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.readout, str) or not self.readout:
            raise TypeError("readout must be one column name")
        for name in ("features", "tolerances", "effort_strata"):
            value = getattr(self, name)
            if isinstance(value, str):
                raise TypeError(f"{name} must be a sequence, not a string")
            object.__setattr__(self, name, tuple(value))
        if self.group_cols is not None:
            if isinstance(self.group_cols, str):
                raise TypeError("group_cols must be a sequence of column names")
            object.__setattr__(self, "group_cols", tuple(self.group_cols))
        if self.rules is not None:
            object.__setattr__(self, "rules", tuple(self.rules))
        if not self.tolerances or any(not 0 < t < 1 for t in self.tolerances):
            raise ValueError("tolerances must be fractions between 0 and 1, such as 0.05")
        if self.control is not None and (not isinstance(self.control, Mapping) or not self.control):
            raise TypeError("control must map FOV-table columns to values")
        if self.positive is not None and not isinstance(self.positive, PositiveControl):
            raise TypeError("positive must be a PositiveControl")
        if self.positive is not None and self.control is None:
            raise ValueError("a positive control is checked against the control: declare both")
        # A declaration the workflows could not use is refused, never silently skipped.
        if self.novelty is not None and not self.features:
            raise ValueError(
                "novelty is computed on the feature columns: declare features (the readout "
                "alone is not the representation the definition of new was made for)"
            )
        if self.positive is not None and self.novelty is None:
            raise ValueError(
                "a positive control is judged with the definition of new: declare novelty "
                "(and features), or leave the positive control out"
            )

    @property
    def columns(self) -> list[str]:
        """Every feature column the workflows read: the readout, then the features."""
        return [self.readout, *(f for f in self.features if f != self.readout)]

    @property
    def common(self) -> dict[str, Any]:
        """The validation arguments shared by every analysis."""
        return {
            "fov_col": self.fov_col,
            "group_cols": None if self.group_cols is None else list(self.group_cols),
            "observation_unit": self.observation_unit,
        }

    def stopping_rules(self) -> tuple[SamplingStrategy, ...]:
        """``rules``, or the JCB rule per object and per FOV at the first tolerance."""
        if self.rules is not None:
            return self.rules
        target = self.tolerances[0]
        return (
            ScalarPrecisionStrategy(self.readout, target, "object", "upper_95"),
            ScalarPrecisionStrategy(self.readout, target, "fov", "upper_95"),
        )


__all__ = ["PositiveControl", "Setup", "default_novelty"]
