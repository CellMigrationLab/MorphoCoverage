"""What a dataset is, declared once for every workflow (``report``, ``optimise``, ``live``,
``benchmark``).

Every scientific choice the workflows make is a field here or an explicit
argument of a workflow, visible and recorded; the workflows add the
computational settings (draws, splits, orders, resamples) and provisional
presets (``minimum_detection`` here, ``optimise``'s ``protocol_threshold``,
``benchmark``'s ``alphas``), which the declaration and the summaries print,
and run the package's analyses in the order of ``docs/report/questions.md``.
A field left at ``None`` switches off the analyses that need it, and the
report then says "Not assessed" with what it would need.

The declaration also says what the acquisition must preserve (the front door
of ``docs/common/setup.md``): a population summary (the mean of ``readout``
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

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._former_repeat import read_fields
from .._wording import count, rate
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
from ..normalize import normalize_to_control, normalize_to_own
from ..objectives import ObjectiveConfig, RareTarget, SamplingObjective
from ..outcomes import RareStateDetected, RelativeMeanError
from ..retrospective.criteria import OutcomeCriterion
from ..retrospective.hierarchy import SamplingHierarchy
from ..schema import OBJECT_LIKE_UNITS, resolve_observation_unit, unit_words
from ..strategies import ScalarPrecisionStrategy, SupportNoveltyEstimator, SupportNoveltyStrategy
from ..strategies.base import SamplingStrategy
from ..strategies.support_novelty import UPPER_QUANTITY
from ._contrasts import (
    check_controls,
    compares,
    comparisons,
    contrast_arguments,
    contrast_table,
    figure_control,
    normalized_text,
    one_control,
    own_references,
    reference_control,
    single_control,
)
from ._declaration import declaration
from ._feature_space import check_feature_tables, table_columns
from ._setup_parts import (
    PositiveControl,
    check_embedding_only,
    check_few_fovs,
    check_presets,
    default_novelty,
    warn_novelty_dimension,
)


@dataclass(frozen=True)
class Setup:
    """The declaration of a dataset shared by the four workflows.

    - ``readout``: the scalar the microscopist reports (one feature column). ``None``
      with no ``features`` makes an embedding-only Setup (owner, 2026-10-06): the
      workflows then read the other feature tables alone (``feature_tables=``), and the
      targets are the fingerprint of each table and the labelled phenotype, never a mean
      (``docs/common/setup.md``).
    - ``features``: the representation for unusual observations and novelty
      (explicit columns; never inferred). Empty means the readout only.
    - ``control``: the negative-control groups, for example
      ``{"condition": "MOCK"}``. Its mean in each repeat is the reference of the
      Foundation's JCB object-rule check (``object_rule_check``; the stopping rules use each
      group's own mean), its observations the reference for unusual
      observations (without it they are not assessed), and, with ``normalize``,
      the divisor of the normalised values. Several controls, each mapping a treatment
      key to its condition (or listing them), owner 2026-10-05: ``{"condition":
      {"VEC_SiNC": {"SiCol6a1": "VEC_SiCol6a1"}, "VEC": {"THBS5": "THBS5"}}}``; each
      condition is tested against its own control, and the contrast table adds the
      comparisons across experiments (``_contrasts``). One control declared that way (a
      list, a keyed dict, a master), without parents, is ``{"condition": control}``: every
      condition is compared with it, as with one control (owner, 2026-10-05).
    - ``control_parents``, ``master_control``: with several controls, a control's parent
      condition (``"VEC"`` or ``("VEC", "transfection")``, a treatment key), tested as a
      main verdict; and the control the other controls are compared with on raw values.
    - ``contrasts``: a contrast table (``_control.CONTRAST_COLUMNS``) in its place.
    - ``intensity_features``: with several controls, declared columns compared between the
      controls for an acquisition gap, beside every declared column whose name contains
      "intensity" (``_acquisition_gap``, owner 2026-10-05).
    - ``feature_tables``: the non-explainable feature tables, each name with its columns,
      such as a DINO embedding (owner, 2026-10-05): never tested feature by feature, they
      join the maps, the populations, the copy checks and their own fingerprints, never
      novelty or unusual observations (owner, 2026-10-06; ``_feature_space``).
      ``explainable_tables``: the explainable ones (fiberfeatures, CellProfiler), each name
      with its final columns, explainable features to declare in ``features``; with none
      declared, every column of them is a feature (owner, 2026-10-06). Both are set by the
      workflows' ``feature_tables=``.
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
    - ``fields_independent``: declares that the fields are independent draws
      of their condition (default ``False``). The condition contrast and the
      feature ranking shuffle the technical repeats (``technical_repeat``),
      else a group column nested within condition x repeat (a well column in
      ``group_cols``); without either they give no p-value, since fields of
      one dish or well are not independent draws, unless this is ``True``.
    - ``fov_col``, ``group_cols``, ``observation_unit``: as for
      ``validate_input_tables`` (MorphoBricks defaults when ``None``).
    - ``rules``: the stopping rules to assess, calibrate and run live; by
      default the mean-precision rule (upper-95 SD, first tolerance) per observation and
      per FOV, per FOV only for observations other than objects such as patches or tracks
      (the declaration names them and marks the default).
    - ``random_state``: the seed of every random procedure the workflows run
      (subset draws, hold-outs, replay orders, the report's random orders,
      the default benchmark families' resampling, bootstraps and randomization
      tests), so a run is reproducible; a non-negative integer, a
      computational setting, not a scientific one.
    - ``n_jobs``: worker processes for the analyses that can use them (``-1``:
      every CPU); results are identical for every value.

    Per-plane features (a z-stack measured plane by plane) are split before
    the Setup: ``morphocoverage.planes_as_features`` turns ``area`` into
    ``area_z0``, ``area_z1``, ... (objects matched by ``label_id``), and those
    names are the ``readout`` and ``features`` declared here.
    """

    readout: str | None = None
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
    fields_independent: bool = False
    master_control: Any = None
    control_parents: Mapping[Any, Any] | None = None
    contrasts: pd.DataFrame | None = field(default=None, compare=False)
    feature_tables: Mapping[str, tuple[str, ...]] | None = None
    intensity_features: tuple[str, ...] = ()
    explainable_tables: Mapping[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if self.readout is not None and (not isinstance(self.readout, str) or not self.readout):
            raise TypeError("readout must be one column name (None: no explainable features)")
        for name in ("features", "tolerances", "effort_strata", "intensity_features"):
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
        read_fields(self, "control", "fov_col", "group_cols", "effort_strata")  # ``repeat``
        if self.rules is not None:
            object.__setattr__(self, "rules", tuple(self.rules))
        if not isinstance(self.fields_independent, (bool, np.bool_)):
            raise TypeError("fields_independent must be True or False")
        object.__setattr__(self, "fields_independent", bool(self.fields_independent))
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
        check_controls(self)
        single = one_control(self)
        if single is not None:  # one control in the new forms is one control
            object.__setattr__(self, "control", single)
            object.__setattr__(self, "master_control", None)
        if self.positive is not None and not isinstance(self.positive, PositiveControl):
            raise TypeError("positive must be a PositiveControl")
        if self.positive is not None and self.reference_control is None:
            raise ValueError("a positive control is checked against the control: declare both")
        if self.normalize is not None:
            if self.normalize not in NORMALIZATIONS:
                raise ValueError(f"normalize must be one of {list(NORMALIZATIONS)} or None")
            chosen = self.figure_control
            if chosen is None or MORPHOBRICKS_GROUP_COLUMNS[0] not in chosen:
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
        check_embedding_only(self)
        check_presets(self)
        check_few_fovs(self)
        warn_novelty_dimension(self)
        check_feature_tables(self)
        if set(self.intensity_features) - set(self.columns):
            raise ValueError("intensity_features must be declared columns (readout or features)")
        undeclared = set(self.criterion_columns) - {*self.columns, *table_columns(self)}
        if undeclared:
            raise ValueError(
                f"fingerprint criteria read feature columns {sorted(undeclared)[:3]} (of "
                f"{len(undeclared)}) that are not declared: add them to features, or declare "
                "them as another feature table, feature_tables={name: columns}"
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
        if self.readout is None:  # embedding-only: no explainable feature
            return []
        return [self.readout, *(f for f in self.features if f != self.readout)]

    @property
    def embedding_only(self) -> bool:
        """Whether no explainable feature is declared (no readout): the other feature tables
        are then all the report reads (owner, 2026-10-06)."""
        return self.readout is None

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
        first = [self.readout] if self.readout is not None else []
        return [*first, *(c for c in self.criterion_columns if c != self.readout)]

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
        stopping rules and the other analyses read the raw values, as live. With several
        controls, each condition over its own control (``normalize_to_own``), for figures.
        """
        if self.normalize is None or not self.columns:  # embedding-only: nothing divided
            return objects
        if (own := self.own_references) is not None:
            kept = {"repeat_col": self.repeat_col(fovs), "own": own, **self.common}
            return normalize_to_own(objects, fovs, feature_cols=self.columns, **kept)
        assert self.control is not None
        return normalize_to_control(
            objects,
            fovs,
            feature_cols=self.columns,
            control=self.control,
            repeat_col=self.repeat_col(fovs),
            **self.common,
        )

    def compared_values(self, objects: pd.DataFrame, fovs: pd.DataFrame) -> pd.DataFrame:
        """What the tests read: ``prepared``, or raw values with a contrast table."""
        return objects if self.comparisons is not None else self.prepared(objects, fovs)

    @property
    def contrast_table(self) -> pd.DataFrame | None:
        """``contrasts``, else the table several controls generate, else ``None``."""
        return contrast_table(self)

    @property
    def comparisons(self) -> tuple[tuple[Any, Any], ...] | None:
        """The main verdicts as ``(condition, control)`` pairs; ``None`` with one control."""
        return comparisons(self)

    @property
    def own_references(self) -> dict[Any, Any] | None:
        """Each condition's own control (a control's parent, else itself), or ``None``."""
        return own_references(self)

    @property
    def compares(self) -> bool:
        """Whether conditions are compared: a control on the condition, or contrasts."""
        return compares(self)

    @property
    def contrast_arguments(self) -> dict[str, Any]:
        """``control``, or the main verdicts' ``comparisons``, as the analyses take them."""
        return contrast_arguments(self)

    @property
    def reference_control(self) -> Mapping[str, Any] | None:
        """The control, or every own control pooled (``_contrasts``)."""
        return reference_control(self)

    @property
    def figure_control(self) -> Mapping[str, Any] | None:
        """What figures draw first: the control, or the master (else the first declared)."""
        return figure_control(self)

    @property
    def single_control(self) -> Mapping[str, Any] | None:
        """The control when there is one and no contrast table, else ``None``."""
        return single_control(self)

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

    def normalized_sentence(
        self, fovs: pd.DataFrame, objects: pd.DataFrame | None = None
    ) -> str | None:
        """How the values are normalised and what reads them, or ``None`` when they are not;
        with ``objects``, the caution when the control holds negative values."""
        return normalized_text(self, fovs, objects)

    def objective(self) -> ObjectiveConfig:
        """The objective whose components are the declared targets.

        The population summary is always declared; a fingerprint adds the
        distribution component, a rare target the rare component. The live
        decision and every replay run under it, so a rule that has no evidence
        about a declared target cannot alone say stop, and a target no rule
        decides stays ``unsupported`` in the decision view rather than being
        assumed adequate.
        """
        if self.readout is None:  # no mean: the fingerprint and the rare phenotype only
            if self.rare is not None and not self.fingerprint:
                return ObjectiveConfig(SamplingObjective.RARE_DISCOVERY, rare=self.rare)
            if self.rare is None:
                return ObjectiveConfig(SamplingObjective.DISTRIBUTION)
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
                f"the mean of {self.readout} within {rate(tolerance)}",
            )
            for tolerance in (self.tolerances if self.readout is not None else ())
        ]
        criteria += list(self.fingerprint)
        if self.rare is not None and self.rare_label is not None:
            fields = self.rare.min_fov_occurrences
            criteria.append(
                OutcomeCriterion(
                    RareStateDetected(self.rare_label, fields),
                    1.0,
                    "higher_is_better",
                    f"{self.rare_label} seen in at least {count(fields, 'field')}",
                )
            )
        return criteria

    def declaration(self) -> tuple[str, ...]:
        """What this acquisition was asked to preserve, in sentences, one per target.

        Printed first by the report, every answer read against it; a target not declared
        says so, so that "enough" is never read for a target the microscopist did not name.
        The presets applied without being asked (``minimum_detection``, the default
        stopping rules, the positive control's pairing) are printed too."""
        return declaration(self)

    @property
    def unit_words(self) -> str:
        """The observations in the plural, for sentences (``cells`` when undeclared)."""
        return unit_words(self.observation_unit)[1]

    @property
    def unit_word(self) -> str:  # the singular, ``cell`` when undeclared
        return unit_words(self.observation_unit)[0]

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
        """``rules``, or the mean-precision rule per observation and per FOV at the first
        tolerance; per FOV only for units other than objects (cells, nuclei), since patches
        or tracks of one image are far from independent draws (owner, 2026-10-02)."""
        if self.rules is not None:
            return self.rules
        if self.readout is None:  # the mean-precision rules read the readout
            return ()
        target = self.tolerances[0]
        per_fov = ScalarPrecisionStrategy(self.readout, target, "fov", "upper_95")
        if resolve_observation_unit(self.observation_unit)[0] not in OBJECT_LIKE_UNITS:
            return (per_fov,)
        return (ScalarPrecisionStrategy(self.readout, target, "object", "upper_95"), per_fov)


__all__ = ["PositiveControl", "Setup", "default_novelty"]
