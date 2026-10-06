"""Private helpers of the retrospective workflows: inputs, orders, replays and analyses.

- ``check_workflow_inputs`` refuses tables and setups the workflows cannot use, and
  runs ``check_tables`` so its warnings reach every workflow;
- ``orders`` builds the replay orders, every random one seeded from
  ``Setup.random_state``;
- ``replay_rules`` runs the stopping rules on the raw values (as the live
  session does) and judges the outcome criteria on the prepared values, so a
  rule gives the verdict in ``optimise`` and ``benchmark`` that it gives live;
- ``pooled_columns`` names the columns an analysis group spans several values
  of, so an analysis that needs them apart says "not assessed" instead of
  raising deep inside (the reasons and the assembled report are in
  ``workflows._document``);
- ``control_checks``, ``rare_clustering`` and ``conditions`` run the report's
  analyses against the control, of the rare label and between conditions.

Shared-private: every name in ``__all__`` is used by ``workflows.retrospective``
(``check_workflow_inputs`` also by ``workflows.figures``), not part of the public API.
"""

from __future__ import annotations

import warnings
import zlib
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pandas as pd

from .._control import not_found
from .._former_repeat import renamed_tables
from .._wording import count
from ..benchmark import AcquisitionOrder
from ..data_check import DataCheck, Problem, validate_input_tables
from ..data_check import check_tables as _check_tables
from ..metadata import (
    CONDITION_COLUMN,
    REPEAT_COLUMN,
    drop_empty_optional_columns,
    resolve_group_cols,
)
from ..outcomes import OutcomeMetric
from ..retrospective import replay_acquisitions
from ..retrospective.criteria import OutcomeCriterion
from ..retrospective.replay import ReplayResult
from ..sampling_unit import (
    FOVClustering,
    condition_contrast,
    feature_discovery,
    feature_ranking,
    fov_clustering,
    object_rule_check,
    unusual_objects,
)
from ..schema import ValidatedInput
from ..statistics import log_scale
from ..strategies import FixedFOVStrategy
from ..strategies.base import SamplingStrategy
from ..sufficiency import SufficiencyReport
from ._contrasts import declared_conditions
from ._document import pooling_reason
from ._feature_space import attached, table_columns, unusual_space, with_table
from ._rare_share import compared_columns, share_column, with_share
from .setup import Setup, default_novelty


def order_seeds(random_state: int, n: int) -> list[int]:
    """``n`` order seeds of their own for ``random_state``: ``0 .. n-1`` for seed 0.

    Consecutive within the block (dispersed orders then start at distinct
    fields), and blocks of different seeds never overlap, so ``random_state=1``
    shares no order with ``random_state=0``.
    """
    return [random_state * n + i for i in range(n)]


def orders(
    setup: Setup, permuted: int, dispersed: int, *, seed_permuted: bool = False
) -> list[AcquisitionOrder]:
    """The recorded order when there is one, ``permuted`` random and ``dispersed`` orders.

    The dispersed orders' seeds come from ``Setup.random_state``
    (``order_seeds``), since the replay seeds a dispersed order from its own
    seed only. ``seed_permuted`` does the same for the permuted orders: for
    ``sufficiency_report``, whose permutations are seeded by the order's seed
    alone; ``replay_acquisitions`` already seeds each permutation from its
    ``random_state``, so its orders keep the seeds ``0 .. permuted-1``.
    """
    chosen = [AcquisitionOrder.recorded()] if setup.acquisition_order_col else []
    permuted_seeds = order_seeds(setup.random_state, permuted) if seed_permuted else range(permuted)
    chosen += [AcquisitionOrder.permuted(seed) for seed in permuted_seeds]
    if setup.stage_cols is not None:
        seeds = order_seeds(setup.random_state, dispersed)
        chosen += [AcquisitionOrder.dispersed(seed) for seed in seeds]
    return chosen


def check_workflow_inputs(
    objects: Any,
    fovs: Any,
    setup: Any,
    *,
    warn: bool = True,
    tables_given: bool = False,
    check: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, DataCheck | None]:
    """Refuse inputs the workflows cannot use, before any analysis runs, and check the rest.

    Returns the tables every analysis of the workflow reads, with the former
    name ``repeat`` read as ``biological_repeat`` (one ``DeprecationWarning``;
    the caller's tables are not modified), and ``check_tables`` on the setup's
    columns and control, whose warnings
    (copied rows, labels that print alike, repeats without a control, ...) a
    workflow would otherwise never mention. With ``warn`` they are also issued
    as one ``UserWarning``; ``report`` prints them in its overview instead. Its
    errors are left to the analyses, which raise with the same report. With ``check=False``
    the tables are not checked (``None``): explainable tables hold declared features, so
    ``report`` checks the tables once they are attached.
    """
    if fovs is None:
        raise TypeError(
            "the workflows need the FOV table of every acquired field (fovs=None was given); "
            "morphocoverage.schema.fov_table_from_observations rebuilds one, without the "
            "fields that held no observation"
        )
    if not isinstance(objects, pd.DataFrame) or not isinstance(fovs, pd.DataFrame):
        raise TypeError("objects and fovs must be pandas DataFrames (every acquired FOV in fovs)")
    if not isinstance(setup, Setup):
        raise TypeError("setup must be a morphocoverage.workflows.Setup")
    objects, fovs = renamed_tables(objects, fovs)
    if not tables_given:  # the other feature tables a Setup names must be attached already
        attached(objects, fovs, setup, None)
    if setup.rare_label is not None and setup.rare_label not in objects.columns:
        raise ValueError(
            f"rare_label {setup.rare_label!r} is not a column of the object table: the rare "
            "phenotype cannot be judged on held-back fields without its labels"
        )
    if setup.rare_label is not None:
        # The analyses read the label as True/False: any other value ("no", 2) would be
        # True for some of them and another state for others.
        labels = objects[setup.rare_label].dropna()
        if not (pd.api.types.is_bool_dtype(labels) or set(labels.unique()) <= {0, 1}):
            raise ValueError(
                f"rare_label {setup.rare_label!r} must hold booleans or 0/1, got "
                f"{sorted(map(str, labels.unique()))[:5]}"
            )
    for feature, scale in (setup.scales or {}).items():  # a ratio needs positive values
        if feature in objects.columns:
            log_scale(objects[feature].to_numpy(dtype=float), scale, repr(feature))
    columns = setup.columns
    if not check:
        return objects, fovs, None
    if setup.embedding_only:  # the tables' columns, once attached (owner, 2026-10-06)
        columns = [c for c in table_columns(setup) if c in objects.columns]
        if not columns:
            return objects, fovs, None  # report checks them after attaching the tables
    check = _check_tables(
        objects, fovs, feature_cols=columns, control=setup.reference_control, **setup.common
    )
    declared = declared_conditions(setup)  # several controls and their conditions
    if declared:
        present = list(fovs.get(CONDITION_COLUMN, pd.Series()).unique())
        problem = not_found([tuple(declared)], present)
        if problem:
            raise ValueError(problem.replace("the comparisons name", "the controls name"))
    # The declared design the summaries state anyway (an assumed unit, empty fields) is not
    # repeated as a warning at every call.
    likely_mistakes = [p for p in check.warnings if not _states_the_design(p)]
    if warn and likely_mistakes:
        warnings.warn(
            f"check_tables found {count(len(likely_mistakes), 'warning')} in these tables: "
            + "; ".join(problem.title for problem in likely_mistakes)
            + " Print morphocoverage.check_tables(objects, fovs, ...).summary(detail='full') for "
            "where and how to fix each.",
            UserWarning,
            stacklevel=3,
        )
    return objects, fovs, check


def _states_the_design(problem: Problem) -> bool:
    """Whether a data-check warning restates the design (an assumed unit, empty fields)."""
    empty_fields = " field of the FOV table holds no observation." in problem.title or (
        " fields of the FOV table hold no observation." in problem.title
    )
    return problem.where == "observation_unit=None." or empty_fields


class _RevealedFields(OutcomeMetric):
    """A fingerprint of which fields a prefix and its reservoir hold (never reported).

    ``replay_rules`` judges it in both of its replays: equal values at every
    prefix of every run mean both replays held back the same reservoir and
    revealed the same fields in the same order, whatever the values.
    """

    name = "revealed_fields"

    def evaluate(self, full: ValidatedInput, stopped: ValidatedInput) -> float | None:
        """A 32-bit hash of the reservoir's and the prefix's sorted FOV keys."""
        keys = [
            sorted(map(str, table[list(data.fov_key)].itertuples(index=False, name=None)))
            for table, data in ((full.fovs, full), (stopped.fovs, stopped))
        ]
        return float(zlib.crc32(repr(keys).encode()))


_REVEALED = OutcomeCriterion(_RevealedFields(), 0.0, "lower_is_better")


def replay_rules(
    raw: pd.DataFrame,
    prepared: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    *,
    strategies: Sequence[SamplingStrategy],
    criteria: Sequence[OutcomeCriterion],
    **settings: Any,
) -> ReplayResult:
    """``replay_acquisitions`` with the rules on ``raw`` and the criteria on ``prepared``.

    Without normalisation (``prepared is raw``) this is one replay. With it,
    the decisions come from a replay of the raw values and the outcomes from a
    replay of the prepared ones. Hold-outs and orders depend only on the FOV
    table, the groups and the seeds, never on the values, so both replays
    hold back the same fields and reveal the same fields in the same order:
    checked at every prefix of every run with a fingerprint of the reservoir
    and the revealed fields' keys (``_RevealedFields``), besides the stream
    sizes and stage travel. The first replay judges only that fingerprint and
    the second runs only a fixed-count rule, whose decisions are discarded. So it is when a
    criterion reads columns the rules do not (another table's fingerprint): the second replay
    reads them too, the rules never.
    """
    common = {"feature_cols": setup.columns, "objective": setup.objective(), **setup.common}
    judged = [*setup.columns, *(c for c in setup.judged_columns if c not in setup.columns)]
    if prepared is raw and judged == setup.columns:
        return replay_acquisitions(
            raw, fovs, strategies=strategies, criteria=criteria, **common, **settings
        )
    decided = replay_acquisitions(
        raw, fovs, strategies=strategies, criteria=[_REVEALED], **common, **settings
    )
    scored = replay_acquisitions(
        prepared,
        fovs,
        strategies=[FixedFOVStrategy(1)],
        criteria=[*criteria, _REVEALED],
        **{**common, "feature_cols": judged},
        **{**settings, "keep_results": False},
    )
    revealed = scored.outcomes["criterion"] == _REVEALED.name
    same_fields = (
        decided.outcomes.reset_index(drop=True)
        .drop(columns="status")
        .equals(scored.outcomes.loc[revealed].reset_index(drop=True).drop(columns="status"))
    )
    if (
        not same_fields
        or not decided.streams.equals(scored.streams)
        or not decided.paths.equals(scored.paths)
    ):
        raise RuntimeError(
            "the replays of the raw and the normalised values revealed different fields; "
            "please report this as a bug"
        )
    return replace(
        decided,
        outcomes=scored.outcomes.loc[~revealed].reset_index(drop=True),
        settings={
            **decided.settings,
            "criteria": [c.parameters for c in criteria],
            "labels": {c.name: c.label for c in criteria},
            **(
                {"values": "stopping rules on raw values; outcome criteria on normalised values"}
                if prepared is not raw
                else {}
            ),
        },
    )


def pooled_columns(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, columns: Sequence[str]
) -> list[str]:
    """The ``columns`` of the FOV table that some analysis group holds several values of."""
    explicit = [*(setup.group_cols or ()), *([setup.fov_col] if setup.fov_col else [])]
    objects, table = drop_empty_optional_columns(objects, fovs, explicit)
    assert table is not None
    # The groups as the FOV table holds them: the validator attaches its group columns to
    # lean observation tables, so their absence from ``objects`` is no error here.
    groups = list(resolve_group_cols(table, None, setup.group_cols))
    pooled = []
    for column in columns:
        if column not in table.columns or column in groups:
            continue
        if groups:
            spread = table.groupby(groups, dropna=False)[column].nunique(dropna=False).max()
        else:
            spread = table[column].nunique(dropna=False)
        if spread > 1:
            pooled.append(column)
    return pooled


def report_only(sufficiency: SufficiencyReport, setup: Setup) -> SufficiencyReport:
    """The sufficiency report with the rules it adds beyond ``Setup.rules`` labelled.

    With a definition of new and ``novelty_target``, ``sufficiency_report``
    also judges the novelty bound as a rule. It is not one of the setup's
    rules, so the live session does not run it and ``optimise`` does not
    calibrate it: its answer says so.
    """
    if sufficiency.enough.empty:
        return sufficiency
    declared = {rule.config_id for rule in setup.stopping_rules()}
    enough = sufficiency.enough.copy()
    extra = ~enough["config_id"].isin(declared)
    enough.loc[extra, "answer"] = (
        "report-only rule, not one of the setup's rules (not run live, not calibrated): "
        + enough.loc[extra, "answer"]
    )
    return replace(sufficiency, enough=enough)


def sufficiency_rules(setup: Setup) -> tuple[list[SamplingStrategy], float | None]:
    """The rules and the ``novelty_target`` to pass to ``sufficiency_report``.

    ``sufficiency_report`` adds the novelty-bound rule of ``novelty_target``
    itself and refuses it a second time. When the setup's rules hold it (to
    run it live), it is judged once: on the same columns, the added copy is
    the setup's rule (so it is dropped from the rules passed); when the rules
    also read the readout, which novelty does not, the setup's rule is kept,
    as live judges it, and no target is passed.
    """
    rules = list(setup.stopping_rules())
    target = setup.novelty_target if setup.novelty is not None else None
    added = setup.novelty_rule()
    if added is None or not setup.novelty_rule_declared:
        return rules, target
    if set(setup.features) == set(setup.columns):
        return [rule for rule in rules if rule.config_id != added.config_id], target
    return rules, None


def unusual_label(objects: pd.DataFrame, setup: Setup) -> str:
    """The column name of the unusual-observation labels: it is printed (``unusual cells``)."""
    label = f"unusual {setup.unit_words}"
    return label if label not in objects.columns else f"{label} (report)"


def one_per_field(objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup) -> bool:
    """Whether every field with observations holds exactly one (one cell per image)."""
    columns = setup.columns or table_columns(setup)
    data = validate_input_tables(objects, fovs=fovs, feature_cols=columns, **setup.common)
    sizes = data.objects.groupby(list(data.fov_key), dropna=False).size()
    return bool(len(sizes)) and int(sizes.max()) == 1


def one_per_field_reasons(setup: Setup) -> dict[str, str]:
    """Why the questions about fields do not apply when each field holds one observation,
    by question (``with_reasons``): fields and observations coincide, so their difference
    cannot be measured, and a new state needing several similar observations in one field
    cannot occur."""
    unit, words = setup.unit_word, setup.unit_words
    reasons = {
        "unit": (
            f"not applicable: each field holds one {unit}, so fields and {words} coincide and "
            f"every per-{unit} answer is also the per-field one"
        ),
        "profile": (
            f"not applicable: with one {unit} per field, how {words} spread over fields "
            "cannot be measured"
        ),
    }
    novelty = setup.novelty
    needed = int(getattr(novelty, "min_novel_observations", 1)) if novelty is not None else 1
    if needed > 1:
        reasons["enough"] = (
            f"whether unseen {unit} states remain is not asked: a new state needs {needed} "
            f"similar {words} in one field, and each field holds one"
        )
        if setup.positive is not None:  # judged by the definition of new
            reasons["controls"] = (
                f"not applicable: the positive control is judged by the definition of new, "
                f"and a new state needs {needed} similar {words} in one field, while each "
                "field holds one"
            )
    return reasons


def control_checks(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    reasons: dict[str, str],
    per_field: bool = True,
) -> dict[str, Any]:
    """The Foundation's rule check and R4's unusual observations against the control.

    Both compare each analysis group with the control's groups, so group
    columns that pool the control's columns make them "not assessed". With one
    observation per field (``per_field=False``) neither applies (their fields
    are their observations), so neither is run.
    """
    reference = setup.reference_control
    assert reference is not None
    if not per_field:
        return {}
    common = setup.common
    pooled = pooled_columns(objects, fovs, setup, list(reference))
    if pooled:
        reasons["unit"] = pooling_reason(
            "the object-rule check against the control is", setup, pooled
        )
        if setup.features:
            reasons["profile"] = pooling_reason(
                f"unusual {setup.unit_words} and their clustering are", setup, pooled
            )
        return {}
    results: dict[str, Any] = {}
    if setup.mean_target:  # the mean-precision rules at the readout's tolerances
        results["object_rules"] = object_rule_check(
            objects,
            fovs,
            feature=setup.readout,
            targets=setup.tolerances,
            control=reference,
            random_state=setup.random_state,
            **common,
        )
    # On the explainable features when there are any, whatever the other tables; without
    # them, on the tables' weighted combined space (owner, 2026-10-06).
    columns, weights, space = unusual_space(objects, fovs, setup)
    if columns:
        unusual = unusual_objects(
            objects,
            fovs,
            feature_cols=columns,
            reference=reference,
            own=setup.own_references,  # several controls: each against its own
            random_state=setup.random_state,
            weights=weights,
            **common,
        )
        if setup.feature_tables or weights:  # which space, said beside the tables
            unusual.settings["space"] = space
        results["unusual"] = unusual
        label = unusual_label(objects, setup)
        # The label flags a share of ordinary objects by construction, so a field holds
        # the state only with the novelty rule's m of them, and the control groups'
        # own incidence is printed beside it (P6-1).
        novelty = setup.novelty or default_novelty(columns)
        results["clustering"] = fov_clustering(
            objects.assign(**{label: unusual.labels}),
            fovs,
            label_col=label,
            states=[True],
            random_state=setup.random_state,
            min_objects_per_fov=novelty.min_novel_observations,
            reference=reference,
            **common,
        )
    return results


def rare_clustering(objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup) -> FOVClustering:
    """R4 for the declared rare phenotype: are its labelled objects clustered in few fields?

    The fields to see it are planned for the declared target: seen in at least
    ``rare.min_fov_occurrences`` fields with ``rare.detection_probability``. A missing
    label is unknown, as everywhere (owner, 2026-10-05): ``fov_clustering`` leaves the object
    out, and a field whose objects all lack one out of the fields assessed.
    """
    assert setup.rare is not None and setup.rare_label is not None
    return fov_clustering(
        objects,
        fovs,
        label_col=setup.rare_label,
        states=[True],
        detection_probability=setup.rare.detection_probability,
        random_state=setup.random_state,
        min_fov_occurrences=setup.rare.min_fov_occurrences,
        **setup.common,
    )


def conditions(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, reasons: dict[str, str]
) -> dict[str, Any]:
    """R7: each condition against the control (or each declared comparison), per repeat, on
    the raw values, each test normalising to its own control itself."""
    held = fovs[CONDITION_COLUMN].nunique() if CONDITION_COLUMN in fovs.columns else 0
    if setup.compares and held == 1:  # the real reason, not the generic need (F10)
        only = fovs[CONDITION_COLUMN].dropna().iloc[0]
        reasons["conditions"] = (
            f"not applicable: the FOV table holds one condition, {only}, so no other "
            "condition is compared with the control"
        )
    if not (setup.compares and held > 1):
        return {}
    pooled = pooled_columns(objects, fovs, setup, [CONDITION_COLUMN, REPEAT_COLUMN])
    if pooled:
        reasons["conditions"] = pooling_reason(
            "the condition contrast and the feature ranking are", setup, pooled
        )
        return {}
    results = {
        "contrast": condition_contrast(
            objects,
            fovs,
            feature=setup.readout,
            random_state=setup.random_state,
            fields_independent=setup.fields_independent,
            scale=setup.scale_of(setup.readout),
            # One control, or the contrast table: its main verdicts and its secondary rows.
            **(
                {"control": setup.control}
                if setup.contrast_table is None
                else {"contrasts": setup.contrast_table}
            ),
            **setup.common,
        )
    }
    compared = compared_columns(setup)
    if len(compared) > 1:
        shared = with_share(objects, setup)
        results["discovery"] = feature_discovery(
            shared,
            fovs,
            features=compared,
            random_state=setup.random_state,
            n_jobs=setup.n_jobs,
            scales=setup.scales,
            **setup.contrast_arguments,
            **setup.common,
        )
        # The headline counts the share apart from the features (report_status_conditions).
        results["discovery"].settings["share"] = share_column(setup)
        found = results["discovery"]  # each feature's table named (owner, 2026-10-06)
        results["discovery"] = replace(
            found, tests=with_table(found.tests, setup), ranking=with_table(found.ranking, setup)
        )
        # One correction family on the page: the readout's tests take the discovery's.
        from ._families import shared_across, shared_family

        results["contrast"] = shared_family(
            results["contrast"], results["discovery"], setup.readout
        )
        # The secondary rows for every feature, each family over all of them.
        results["contrast"] = shared_across(results["contrast"], shared, fovs, compared, setup)
        results["ranking"] = feature_ranking(
            shared,
            fovs,
            features=compared,
            random_state=setup.random_state,
            fields_independent=setup.fields_independent,
            **setup.contrast_arguments,
            **setup.common,
        )
    return results


def reservoir_sizes(replay: ReplayResult) -> tuple[int, int] | None:
    """The smallest and largest reservoir the replay held back, or ``None`` without one."""
    if replay.streams.empty:
        return None
    sizes = replay.streams["reservoir_fovs"]
    return int(sizes.min()), int(sizes.max())


__all__ = [
    "check_workflow_inputs",
    "conditions",
    "control_checks",
    "order_seeds",
    "orders",
    "pooled_columns",
    "rare_clustering",
    "replay_rules",
    "report_only",
    "reservoir_sizes",
    "sufficiency_rules",
    "unusual_label",
]
