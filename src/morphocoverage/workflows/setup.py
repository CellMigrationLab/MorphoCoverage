"""What a dataset is, declared once for every workflow (``report``, ``optimise``, ``live``,
``benchmark``).

Every scientific choice the workflows make is a field here or an explicit
argument of a workflow, visible and recorded; the workflows add the
computational settings (draws, splits, orders, resamples) and provisional
presets (``minimum_detection`` here, ``optimise``'s ``protocol_threshold``,
``benchmark``'s ``alphas``), which the declaration and the summaries print,
and run the package's analyses in the order of ``docs/questions.md``.
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

from .._checks import check_seed, is_integer, is_number
from .._parallel import check_n_jobs
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
from ..retrospective.criteria import OutcomeCriterion
from ..retrospective.hierarchy import SamplingHierarchy
from ..retrospective.overview import normalized_sentence
from ..schema import resolve_observation_unit, unit_plural
from ..strategies import ScalarPrecisionStrategy, SupportNoveltyEstimator, SupportNoveltyStrategy
from ..strategies.base import SamplingStrategy
from ..strategies.support_novelty import UPPER_QUANTITY
from ..transforms import Standardize
from ._declaration import declaration


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

    def resolved_pair_cols(self, fovs: pd.DataFrame) -> tuple[str, ...]:
        """``pair_cols`` as declared; by default the repeat column when the FOV table has one."""
        if self.pair_cols is not None:
            return self.pair_cols
        return (REPEAT_COLUMN,) if REPEAT_COLUMN in fovs.columns else ()


@dataclass(frozen=True)
class Setup:
    """The declaration of a dataset shared by the four workflows.

    - ``readout``: the scalar the microscopist reports (one feature column).
    - ``features``: the representation for unusual observations and novelty
      (explicit columns; never inferred). Empty means the readout only.
    - ``control``: the negative-control groups, for example
      ``{"condition": "MOCK"}``. Its mean is the reference of the Foundation's
      JCB object-rule check (``object_rule_check``; the stopping rules use each
      group's own mean), its observations the reference for unusual
      observations (without it they are not assessed), and, with ``normalize``,
      the divisor of the normalised values.
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
    - ``minimum_detection``: the share of positive-control checks the
      representation must recognize before the adequacy answers are read
      without a warning (default 0.9, a provisional preset, printed in the
      declaration).
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
      chance is at most ``novelty_target``" (``novelty_rule()``), beside
      ``rules``: unless it is one of ``rules`` (then it is judged once, as one
      of them), the live session does not run it and ``optimise`` and
      ``benchmark`` do not calibrate it; the declaration says so. ``None``
      answers the next-field question without a target and adds no rule.
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
      per FOV (the declaration names them and marks the default).
    - ``random_state``: the seed of every random procedure the workflows run
      (subset draws, hold-outs, replay orders, the report's random orders,
      the default benchmark families' resampling, bootstraps and randomization
      tests), so a run is reproducible; a non-negative integer, a
      computational setting, not a scientific one.
    - ``n_jobs``: worker processes for the analyses that can use them (``-1``:
      every CPU); results are identical for every value.
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
    novelty_target: float | None = 0.05
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
    minimum_detection: float = 0.9

    def __post_init__(self) -> None:
        if not isinstance(self.readout, str) or not self.readout:
            raise TypeError("readout must be one column name")
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
        self._check_presets()
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

    def _check_presets(self) -> None:
        """The novelty target, the detection preset and the computational settings."""
        target = self.novelty_target
        if target is not None and (not is_number(target) or not 0 < float(target) < 1):
            raise ValueError("novelty_target must be a probability between 0 and 1, or None")
        if target is not None:
            object.__setattr__(self, "novelty_target", float(target))
        detection = self.minimum_detection
        if not is_number(detection) or not 0 < float(detection) <= 1:
            raise ValueError("minimum_detection must be a share in (0, 1], such as 0.9")
        object.__setattr__(self, "minimum_detection", float(detection))
        object.__setattr__(self, "random_state", check_seed(self.random_state))
        check_n_jobs(self.n_jobs)
        object.__setattr__(self, "n_jobs", int(self.n_jobs))

    def _check_few_fovs(self) -> None:
        if self.few_fovs is None:
            return
        if self.novelty is None:
            raise ValueError("few_fovs counts states under the definition of new: declare novelty")
        if not is_integer(self.few_fovs):
            raise TypeError("few_fovs must be an integer number of fields")
        object.__setattr__(self, "few_fovs", int(self.few_fovs))
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

    def prepared(self, objects: pd.DataFrame, fovs: pd.DataFrame) -> pd.DataFrame:
        """The observation table the outcome criteria, the contrast and the plots read.

        With ``normalize="control_mean"`` every declared column is divided by
        its repeat's control mean; otherwise ``objects`` is returned unchanged.
        A repeat without a control raises (``normalize_to_control``). The
        stopping rules and the other analyses read the raw values, as live.
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
        read for a target the microscopist did not name. The presets the
        workflows apply without being asked (``minimum_detection``, the default
        stopping rules, the positive control's pairing) are printed too.
        """
        return declaration(self)

    @property
    def unit_words(self) -> str:
        """The observation unit in the plural, for sentences (``objects`` when undeclared)."""
        return unit_plural(resolve_observation_unit(self.observation_unit)[0])

    def novelty_rule(self) -> SupportNoveltyStrategy | None:
        """The novelty-bound rule ``novelty_target`` adds to the report, or ``None``.

        The rule ``sufficiency_report`` builds from ``novelty`` and
        ``novelty_target``: stop once the upper bound of the chance that one
        more field shows something new is at most the target.
        """
        if self.novelty is None or self.novelty_target is None:
            return None
        novelty = self.novelty
        return SupportNoveltyStrategy(
            novelty.epsilon,
            self.novelty_target,
            novelty.min_other_fovs,
            novelty.transform,
            novelty.min_fovs,
            novelty.min_novel_observations,
            UPPER_QUANTITY,
            novelty.confidence,
        )

    @property
    def novelty_rule_declared(self) -> bool:
        """Whether the novelty-bound rule of ``novelty_target`` is one of ``rules``."""
        rule = self.novelty_rule()
        return rule is not None and rule.config_id in {r.config_id for r in self.stopping_rules()}

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
