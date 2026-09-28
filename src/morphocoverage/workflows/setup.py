"""What a dataset is, declared once for every workflow (``report``, ``optimise``, ``live``,
``benchmark``).

Every scientific choice the workflows make is a field here or an explicit
argument of a workflow, visible and recorded; the workflows add the
computational settings (draws, splits, orders) and two provisional presets that
their summaries print (``optimise``'s ``protocol_threshold``, ``benchmark``'s
``alphas``), and run the package's analyses in the order of
``docs/questions.md``.
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
from ..planes import PLANE_AXES, planes_as_features
from ..results import COMPONENT_WORDS
from ..retrospective.criteria import OutcomeCriterion
from ..retrospective.hierarchy import SamplingHierarchy
from ..retrospective.overview import normalized_sentence
from ..schema import resolve_observation_unit, unit_plural
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
    - ``features``: the representation for unusual observations and novelty
      (explicit columns; never inferred). Empty means the readout only.
    - ``control``: the negative-control groups, for example
      ``{"condition": "MOCK"}``; it normalizes the JCB rule and is the
      reference for unusual observations (without it they are not assessed).
    - ``normalize``: ``"control_mean"`` divides every feature (the readout
      included) by the mean of the control condition's observations of the
      same repeat (``normalize_to_control``); needs ``control`` to name the
      condition column. The normalised values are used only where values are
      compared across repeats: by the outcome criteria judged on held-back
      fields, the condition contrast, the feature ranking and the plots. Every
      stopping rule runs on the raw values, in the report, ``optimise`` and
      ``benchmark`` exactly as in the live session (which sees one field at a
      time and no control), so one rule gives one verdict whichever workflow
      runs it; so do the other analyses (variability, controls, inspection,
      effort projection). ``None`` keeps the values as given.
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
      probability, fields it must be seen in), and the observation column
      that labels it (booleans or 0/1) when an independent label exists. With
      the label, held-back fields judge whether ``k`` fields showed the state
      (``RareStateDetected``); without it the target is stated and planned
      for, not judged.
    - ``novelty``: the definition of "new" (``default_novelty(features)`` or
      your own); ``None`` leaves the novelty questions not assessed.
    - ``novelty_target``: with ``novelty``, the chance that one more field
      shows something new that the report asks to rule out (default 5%). The
      report then also judges a report-only rule, "the upper bound of that
      chance is at most ``novelty_target``", beside ``rules``: it is not one of
      ``rules``, so the live session does not run it and ``optimise`` and
      ``benchmark`` do not calibrate it; the declaration says so.
    - ``few_fovs``: with ``novelty``, the report also counts the fields holding
      a state that fewer than ``few_fovs`` other fields show (seen, but not
      established). It must exceed the definition's ``min_other_fovs``;
      ``None`` means two more than it (3 for ``default_novelty``).
    - ``hierarchy`` and ``effort_strata``: FOV, sample and repeat columns for
      "more fields or more samples?" and protocol reliability. When the FOV
      table has the MorphoBricks ``technical_repeat`` column (a well, a dish)
      and no hierarchy is declared, one is derived: the field within the
      technical repeat within the repeat (``resolved_hierarchy``).
      ``effort_strata=None`` (the default) keeps one stratum per condition,
      declared or derived hierarchy alike, so conditions are never pooled
      silently; ``effort_strata=()`` pools every sample, as explicitly asked
      (``resolved_effort_strata``).
    - ``acquisition_order_col``, ``stage_cols``: acquisition order and stage
      coordinates, when recorded.
    - ``fov_col``, ``group_cols``, ``observation_unit``: as for
      ``validate_input_tables`` (MorphoBricks defaults when ``None``).
    - ``rules``: the stopping rules to assess, calibrate and run live; by
      default the JCB rule (upper-95 SD, first tolerance) per observation and
      per FOV.
    - ``random_state``: the seed of every random procedure the workflows run
      (subset draws, hold-outs, replay orders, bootstraps and randomization
      tests), so a run is reproducible; it is a computational setting, not a
      scientific one.
    - ``n_jobs``: worker processes for the analyses that can use them (``-1``:
      every CPU); results are identical for every value.
    - ``per_plane``: opt-in, the within-field axes (``("z",)``, or ``"c"``,
      ``"t"``) along which each declared column is split into one feature per
      plane (``area`` into ``area_z0``, ``area_z1``, ...;
      ``planes_as_features``), objects matched across planes by ``label_id``
      within their field. Empty (the default) keeps the columns as declared.
      With it, ``prepared`` returns the split table and ``plane_table`` the
      split table with its feature columns, which the analyses read in place
      of ``columns``; the field stays one FOV.
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
    few_fovs: int | None = None
    hierarchy: SamplingHierarchy | None = None
    effort_strata: tuple[str, ...] | None = None
    acquisition_order_col: str | None = None
    stage_cols: tuple[str, str] | None = None
    fov_col: str | None = None
    group_cols: tuple[str, ...] | None = None
    observation_unit: str | None = None
    rules: tuple[SamplingStrategy, ...] | None = None
    random_state: int = 0
    n_jobs: int = 1
    per_plane: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.readout, str) or not self.readout:
            raise TypeError("readout must be one column name")
        if isinstance(self.per_plane, str):
            raise TypeError(f"per_plane must be a sequence of axes, e.g. ({self.per_plane!r},)")
        object.__setattr__(self, "per_plane", tuple(self.per_plane))
        unknown = [a for a in self.per_plane if a not in PLANE_AXES]
        if unknown or len(set(self.per_plane)) != len(self.per_plane):
            raise ValueError(f"per_plane names distinct axes among {list(PLANE_AXES)}")
        for name in ("features", "tolerances", "effort_strata"):
            value = getattr(self, name)
            if value is None and name == "effort_strata":
                continue  # one stratum per condition (resolved_effort_strata)
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
        self._check_few_fovs()
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
        same data whichever workflow runs it. With ``per_plane`` these are the
        columns as measured, before the split: the split names depend on the
        planes in the data and come from ``plane_table``.
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

    def _check_few_fovs(self) -> None:
        if self.few_fovs is None:
            return
        if self.novelty is None:
            raise ValueError("few_fovs counts states under the definition of new: declare novelty")
        if isinstance(self.few_fovs, bool) or not isinstance(self.few_fovs, int):
            raise TypeError("few_fovs must be an integer number of fields")
        if self.few_fovs <= self.novelty.min_other_fovs:
            raise ValueError(
                f"few_fovs ({self.few_fovs}) must exceed the definition of new's min_other_fovs "
                f"({self.novelty.min_other_fovs}): a state seen in that many other fields is "
                "already not new"
            )

    @property
    def resolved_few_fovs(self) -> int | None:
        """``few_fovs``, or two more than the definition's ``min_other_fovs``; ``None``
        without a definition of new."""
        if self.novelty is None:
            return None
        return self.few_fovs if self.few_fovs is not None else self.novelty.min_other_fovs + 2

    def plane_table(
        self, objects: pd.DataFrame, fovs: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """With ``per_plane``: the observations with each declared column split per plane
        (``planes_as_features``, objects matched by ``label_id``) and the split feature
        columns; otherwise ``objects`` and ``columns`` unchanged."""
        if not self.per_plane:
            return objects, self.columns
        return planes_as_features(
            objects, fovs, features=self.columns, axes=self.per_plane, fov_col=self.fov_col
        )

    def prepared(self, objects: pd.DataFrame, fovs: pd.DataFrame) -> pd.DataFrame:
        """The observation table the outcome criteria, the contrast and the plots read.

        With ``per_plane`` the declared columns are first split per plane
        (``plane_table``). With ``normalize="control_mean"`` every declared
        (or split) column is then divided by its repeat's control mean;
        otherwise the table is returned as it is. A repeat without a control
        raises (``normalize_to_control``). The stopping rules and the other
        analyses read the raw values, as live.
        """
        table, columns = self.plane_table(objects, fovs)
        if self.normalize is None:
            return table
        assert self.control is not None
        return normalize_to_control(
            table,
            fovs,
            feature_cols=columns,
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
        """``effort_strata`` as declared; by default one stratum per condition.

        The default holds for a declared hierarchy as for a derived one, so
        conditions are never pooled silently; ``effort_strata=()`` pools every
        sample, as explicitly asked.
        """
        if self.effort_strata is not None:
            return list(self.effort_strata)
        return [CONDITION_COLUMN] if CONDITION_COLUMN in fovs.columns else []

    def normalized_sentence(self, fovs: pd.DataFrame) -> str | None:
        """How the values are normalised and what reads them, or ``None`` when they are not."""
        if self.normalize is None:
            return None
        assert self.control is not None
        return (
            f"{normalized_sentence(self.control, self.repeat_col(fovs))}, for the outcomes "
            "judged on held-back fields, the condition contrast, the feature ranking and the "
            "plots; the stopping rules and the other analyses read the raw values"
        )

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
        units = self.unit_words
        lines = [
            (
                f"Population summary: the mean of {self.readout} over all {units}, within "
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
        lines.append(f"Representation: readout {self.readout}; {self._representation()}.")
        if self.novelty is not None:
            lines.append(
                "Report-only rule: with this definition of new, 'Did I image enough?' also "
                "judges whether the chance that one more field shows something new is at most "
                f"{self.novelty_target:.1%} (its upper bound; novelty_target). It is not one of "
                "the stopping rules (rules): the live decision does not run it, and optimise "
                "and benchmark do not calibrate it. A state that fewer than "
                f"{self.resolved_few_fovs} other fields show is counted as seen in few fields "
                "(few_fovs)."
            )
        if self.normalize is not None:
            lines.append(
                "Values: every declared feature divided by the mean of the control "
                f"{dict(self.control or {})} {units} of the same repeat (of the whole table "
                "without a repeat column) for the outcomes judged on held-back fields, the "
                "condition contrast, the feature ranking and the plots; every stopping rule "
                "runs on the raw values, in the report, optimise and benchmark as in the live "
                "decision, and so do the other analyses."
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

    @property
    def unit_words(self) -> str:
        """The observation unit in the plural, for sentences (``objects`` when undeclared)."""
        return unit_plural(resolve_observation_unit(self.observation_unit)[0])

    def _representation(self) -> str:
        """What the declared features are used for; unusual observations need the control."""
        units = self.unit_words
        if not self.features:
            # Unusual observations and novelty read the declared features only (see report).
            return (
                f"no feature columns declared, so unusual {units} and new phenotypes are not "
                "assessed"
            )
        new = "not declared" if self.novelty is None else self.novelty.name
        if self.control is None:
            return (
                f"{len(self.features)} declared feature columns for new phenotypes (definition "
                f"of new: {new}); unusual {units} are not assessed without a control"
            )
        return (
            f"{len(self.features)} declared feature columns for unusual {units} (against the "
            f"control) and new phenotypes; definition of new: {new}"
        )

    def _rare_declaration(self) -> str:
        target = self.rare
        assert target is not None
        parts = []
        if target.object_prevalence is not None:
            parts.append(f"{target.object_prevalence:.1%} of {self.unit_words}")
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
        """``rules``, or the JCB rule per observation and per FOV at the first tolerance."""
        if self.rules is not None:
            return self.rules
        target = self.tolerances[0]
        return (
            ScalarPrecisionStrategy(self.readout, target, "object", "upper_95"),
            ScalarPrecisionStrategy(self.readout, target, "fov", "upper_95"),
        )


__all__ = ["PositiveControl", "Setup", "default_novelty"]
