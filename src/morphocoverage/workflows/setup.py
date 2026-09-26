"""What a dataset is, declared once for every workflow (``report``, ``optimise``, ``live``,
``benchmark``).

Every scientific choice the workflows make is a field here, visible and
recorded; the workflows only add the computational settings (draws, splits,
orders) and run the package's analyses in the order of ``docs/questions.md``.
A field left at ``None`` switches off the analyses that need it, and the
report then says "Not assessed" with what it would need.

The declaration also says what the acquisition must preserve (the front door
of ``docs/questions.md``): a population summary (the mean of ``readout``
within ``tolerances``), a phenotypic fingerprint (``fingerprint``: outcome
criteria on the declared features), and a rare phenotype (``rare``: a
``RareTarget``, with ``rare_label`` naming the objects that carry it when a
label exists). ``Setup.criteria()`` turns the declaration into the outcome
criteria every planning analysis judges on held-back fields,
``Setup.objective()`` into the objective the live decision runs under (its
components are the declared targets), and ``Setup.declaration()`` states it
in sentences, which the report prints first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..evidence import Component
from ..metadata import (
    CONDITION_COLUMN,
    MORPHOBRICKS_GROUP_COLUMNS,
    REPEAT_COLUMN,
    TECHNICAL_REPEAT_COLUMN,
    column_has_values,
    drop_empty_optional_columns,
    resolve_fov_col,
)
from ..normalize import METHODS as NORMALIZATIONS
from ..normalize import normalize_to_control
from ..objectives import ObjectiveConfig, RareTarget, SamplingObjective
from ..outcomes import RareStateDetected, RelativeMeanError
from ..results import COMPONENT_WORDS
from ..retrospective.criteria import OutcomeCriterion
from ..retrospective.hierarchy import SamplingHierarchy
from ..retrospective.overview import normalized_sentence
from ..statistics import fovs_to_detect
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
    - ``normalize``: ``"control_mean"`` divides every feature (the readout
      included) by the mean of the control condition's objects of the same
      repeat before any retrospective analysis (``normalize_to_control``);
      needs ``control`` to name the condition column. The live session runs
      on raw values, since it sees one group at a time. ``None`` keeps the
      values as given.
    - ``positive``: a ``PositiveControl``, for the detection check.
    - ``tolerances``: the readout's mean within these fractions, the outcomes
      judged on held-back FOVs and the JCB targets; default the paper's 5% and
      2.5%. This is the population-summary target.
    - ``fingerprint``: the phenotypic-fingerprint target: outcome criteria on
      the declared features (for example an ``OutcomeCriterion`` on
      ``CovarianceError`` or ``QuantileError`` within a tolerance), judged on
      held-back FOVs beside the mean. Their features must be declared in
      ``features`` (or be the readout). Empty means the fingerprint is not
      declared and the planning answers speak of the mean only.
    - ``rare`` and ``rare_label``: the rare-phenotype target, a ``RareTarget``
      (its definition, prevalence and/or field incidence, detection
      probability, fields it must be seen in), and the object column that
      labels it (booleans or 0/1) when an independent label exists. With the
      label, held-back fields judge whether ``k`` fields showed the state
      (``RareStateDetected``); without it the target is stated and planned
      for, not judged.
    - ``novelty``: the definition of "new" (``default_novelty(features)`` or
      your own); ``None`` leaves the novelty questions not assessed.
    - ``hierarchy`` and ``effort_strata``: FOV, sample and repeat columns for
      "more fields or more samples?" and protocol reliability. When the FOV
      table has the MorphoBricks ``technical_repeat`` column (a well, a dish)
      and no hierarchy is declared, one is derived: the field within the
      technical repeat within the repeat, one stratum per condition
      (``resolved_hierarchy``, ``resolved_effort_strata``).
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
    normalize: str | None = None
    positive: PositiveControl | None = None
    tolerances: tuple[float, ...] = (0.05, 0.025)
    fingerprint: tuple[OutcomeCriterion, ...] = ()
    rare: RareTarget | None = None
    rare_label: str | None = None
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
        if isinstance(self.fingerprint, OutcomeCriterion):
            raise TypeError("fingerprint must be a sequence of OutcomeCriterion, not one")
        object.__setattr__(self, "fingerprint", tuple(self.fingerprint))
        if any(not isinstance(c, OutcomeCriterion) for c in self.fingerprint):
            raise TypeError("fingerprint must be a sequence of OutcomeCriterion")
        if self.rare is not None and not isinstance(self.rare, RareTarget):
            raise TypeError("rare must be a RareTarget")
        if self.rare_label is not None:
            if not isinstance(self.rare_label, str) or not self.rare_label:
                raise TypeError("rare_label must be one object column name")
            if self.rare is None:
                raise ValueError("rare_label labels a declared rare target: declare rare too")
        if not self.tolerances or any(not 0 < t < 1 for t in self.tolerances):
            raise ValueError("tolerances must be fractions between 0 and 1, such as 0.05")
        if self.control is not None and (not isinstance(self.control, Mapping) or not self.control):
            raise TypeError("control must map FOV-table columns to values")
        if self.positive is not None and not isinstance(self.positive, PositiveControl):
            raise TypeError("positive must be a PositiveControl")
        if self.positive is not None and self.control is None:
            raise ValueError("a positive control is checked against the control: declare both")
        if self.normalize is not None:
            if self.normalize not in NORMALIZATIONS:
                raise ValueError(f"normalize must be one of {list(NORMALIZATIONS)} or None")
            if self.control is None or MORPHOBRICKS_GROUP_COLUMNS[0] not in self.control:
                raise ValueError(
                    "normalize needs control to name the condition column, for example "
                    "control={'condition': 'MOCK'}"
                )
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
        undeclared = sorted(set(self.criterion_columns) - set(self.columns))
        if undeclared:
            raise ValueError(
                f"fingerprint criteria read feature columns {undeclared} that are not declared: "
                "add them to features (or use the readout)"
            )
        names = [c.name for c in self.criteria()]
        if len(set(names)) != len(names):
            raise ValueError("the declared criteria must be distinct (same metric and features)")

    @property
    def columns(self) -> list[str]:
        """Every feature column the workflows read: the readout, then the features.

        The stopping rules see exactly these columns in ``report``, ``optimise``,
        ``benchmark`` and ``live_session``, so one rule gives one verdict on the
        same data whichever workflow runs it.
        """
        return [self.readout, *(f for f in self.features if f != self.readout)]

    @property
    def criterion_columns(self) -> list[str]:
        """The feature columns the fingerprint criteria read, from each metric's provenance."""
        columns: list[str] = []
        for criterion in self.fingerprint:
            parameters = criterion.metric.parameters
            for name in [*parameters.get("features", []), parameters.get("feature")]:
                if isinstance(name, str) and name not in columns:
                    columns.append(name)
        return columns

    @property
    def judged_columns(self) -> list[str]:
        """The readout plus every column a declared criterion reads: what
        ``sampling_reliability`` judges (the rules read ``columns``)."""
        return [self.readout, *(c for c in self.criterion_columns if c != self.readout)]

    def prepared(self, objects: pd.DataFrame, fovs: pd.DataFrame) -> pd.DataFrame:
        """The object table every retrospective analysis reads: normalised when asked.

        With ``normalize="control_mean"`` every declared column is divided by
        its repeat's control mean; otherwise ``objects`` is returned unchanged.
        A repeat without a control raises (``normalize_to_control``).
        """
        if self.normalize is None:
            return objects
        assert self.control is not None
        return normalize_to_control(
            objects,
            fovs,
            feature_cols=self.columns,
            control=self.control,
            repeat_col=self.repeat_col(fovs),
            **self.common,
        )

    def repeat_col(self, fovs: pd.DataFrame) -> str | None:
        """The repeat column when the FOV table has one."""
        return REPEAT_COLUMN if REPEAT_COLUMN in fovs.columns else None

    def technical_col(self, fovs: pd.DataFrame) -> str | None:
        """The technical-repeat column when the FOV table has one with values in it."""
        return TECHNICAL_REPEAT_COLUMN if column_has_values(fovs, TECHNICAL_REPEAT_COLUMN) else None

    def resolved_hierarchy(
        self, objects: pd.DataFrame, fovs: pd.DataFrame
    ) -> SamplingHierarchy | None:
        """``hierarchy`` as declared, else derived from the MorphoBricks columns.

        With a ``technical_repeat`` column the field (``fov_col``, or the
        MorphoBricks identifier present) sits within the technical repeat,
        within the repeat when there is one. ``None`` without either, and when
        ``group_cols`` leave the technical repeat out: the analyses then pool
        the technical repeats, so there are no per-technical-repeat results to
        compare (declare ``hierarchy`` to ask for it anyway).
        """
        if self.hierarchy is not None:
            return self.hierarchy
        technical = self.technical_col(fovs)
        if technical is None:
            return None
        if self.group_cols is not None and technical not in self.group_cols:
            return None
        explicit = [self.fov_col] if self.fov_col is not None else []
        objects, fovs = drop_empty_optional_columns(objects, fovs, explicit)
        fov_col = resolve_fov_col(objects, fovs, self.fov_col)
        return SamplingHierarchy(fov_col, technical, self.repeat_col(fovs))

    def resolved_effort_strata(self, fovs: pd.DataFrame) -> list[str]:
        """``effort_strata`` as declared; with a derived hierarchy, one stratum per condition.

        A declared hierarchy with no strata pools every sample, as asked; a
        derived one never pools conditions silently.
        """
        if self.effort_strata or self.hierarchy is not None:
            return list(self.effort_strata)
        return [CONDITION_COLUMN] if CONDITION_COLUMN in fovs.columns else []

    def normalized_sentence(self, fovs: pd.DataFrame) -> str | None:
        """How the values are normalised, or ``None`` when they are not."""
        if self.normalize is None:
            return None
        assert self.control is not None
        return normalized_sentence(self.control, self.repeat_col(fovs))

    def objective(self) -> ObjectiveConfig:
        """The objective whose components are the declared targets.

        The population summary is always declared; a fingerprint adds the
        distribution component, a rare target the rare component. The live
        decision and every replay run under it, so a rule that has no evidence
        about a declared target cannot alone say stop, and a target no rule
        decides stays ``unsupported`` in the decision view rather than being
        assumed adequate.
        """
        if self.rare is not None and self.fingerprint:
            return ObjectiveConfig(SamplingObjective.ALL, rare=self.rare)
        if self.rare is not None:
            return ObjectiveConfig(SamplingObjective.BALANCED, rare=self.rare)
        if self.fingerprint:
            return ObjectiveConfig(SamplingObjective.PHENOTYPE)
        return ObjectiveConfig(SamplingObjective.POPULATION)

    def criteria(self) -> list[OutcomeCriterion]:
        """The declaration as outcome criteria, judged on held-back fields.

        The mean of the readout within each tolerance (population summary), the
        fingerprint criteria as declared, and, when ``rare_label`` is set, the
        rare state seen in at least ``rare.min_fov_occurrences`` fields.
        """
        criteria = [
            OutcomeCriterion(
                RelativeMeanError(self.readout),
                tolerance,
                "lower_is_better",
                f"the mean of {self.readout} within {tolerance:.1%}",
            )
            for tolerance in self.tolerances
        ]
        criteria += list(self.fingerprint)
        if self.rare is not None and self.rare_label is not None:
            fields = self.rare.min_fov_occurrences
            criteria.append(
                OutcomeCriterion(
                    RareStateDetected(self.rare_label, fields),
                    1.0,
                    "higher_is_better",
                    f"the rare phenotype ({self.rare.phenotype}) seen in at least "
                    f"{fields} field{'s' if fields != 1 else ''}",
                )
            )
        return criteria

    def declaration(self) -> tuple[str, ...]:
        """What this acquisition was asked to preserve, in sentences, one per target.

        Printed first by the report. Every answer that follows is read against
        it; a target that is not declared says so, so that "enough" is never
        read for a target the microscopist did not name.
        """
        tolerances = " and ".join(f"{t:.1%}" for t in self.tolerances)
        lines = [
            (
                f"Population summary: the mean of {self.readout} over all objects, within "
                f"{tolerances} of its value on held-back fields."
            )
        ]
        if self.fingerprint:
            named = "; ".join(c.description or c.name for c in self.fingerprint)
            lines.append(f"Phenotypic fingerprint: {named}; judged on held-back fields.")
        else:
            lines.append(
                "Phenotypic fingerprint: not declared (no distribution criterion in "
                "fingerprint), so the answers below speak of the mean, not of the "
                "distribution of phenotypes."
            )
        if self.rare is not None:
            lines.append(self._rare_declaration())
        else:
            lines.append(
                "Rare phenotype: not declared (no RareTarget in rare), so no answer below "
                "says whether a rare state was sampled enough."
            )
        new = "not declared" if self.novelty is None else self.novelty.name
        if self.features:
            representation = (
                f"{len(self.features)} declared feature columns for unusual objects and new "
                f"phenotypes; definition of new: {new}"
            )
        else:
            # Unusual objects and novelty read the declared features only (see report).
            representation = (
                "no feature columns declared, so unusual objects and new phenotypes are not "
                "assessed"
            )
        lines.append(f"Representation: readout {self.readout}; {representation}.")
        if self.normalize is not None:
            lines.append(
                "Values: every declared feature divided by the mean of the control "
                f"{dict(self.control or {})} objects of the same repeat (of the whole table "
                "without a repeat column) before every retrospective analysis; the live "
                "decision runs on raw values."
            )
        objective = self.objective()
        targets = " and ".join(
            COMPONENT_WORDS[c].lower() for c in Component if c in objective.required_components
        )
        lines.append(
            f"Live decision: the {objective.objective.value} objective ({targets}); a rule "
            "decides only the targets it has evidence about, and a target no rule decides "
            "stays unsupported rather than assumed adequate."
        )
        return tuple(lines)

    def _rare_declaration(self) -> str:
        target = self.rare
        assert target is not None
        parts = []
        if target.object_prevalence is not None:
            parts.append(f"{target.object_prevalence:.1%} of objects")
        if target.fov_incidence is not None:
            parts.append(f"{target.fov_incidence:.1%} of fields")
        fields = target.min_fov_occurrences
        line = (
            f"Rare phenotype: {target.phenotype}, present in {' and '.join(parts)}, to be seen "
            f"in at least {fields} field{'s' if fields != 1 else ''} with "
            f"{target.detection_probability:.0%} probability."
        )
        if target.fov_incidence is not None:
            needed = fovs_to_detect(target.fov_incidence, target.detection_probability, fields)
            line += (
                f" At that field incidence the binomial plan asks for {needed} independent "
                "representative fields; fields that cluster the state need more."
            )
        else:
            line += (
                " No field incidence was declared, so no field count is planned: object "
                "prevalence is never converted to field incidence."
            )
        if self.rare_label is not None:
            line += f" Judged on held-back fields with the label column {self.rare_label!r}."
        else:
            line += " No label column: stated and planned for, not judged on held-back fields."
        return line

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
