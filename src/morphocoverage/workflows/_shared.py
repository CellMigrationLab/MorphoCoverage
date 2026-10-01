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
from ..schema import ValidatedInput, resolve_observation_unit
from ..strategies import FixedFOVStrategy
from ..strategies.base import SamplingStrategy
from ..sufficiency import SufficiencyReport
from ._document import pooling_reason
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
    objects: Any, fovs: Any, setup: Any, *, warn: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, DataCheck]:
    """Refuse inputs the workflows cannot use, before any analysis runs, and check the rest.

    Returns the tables every analysis of the workflow reads, with the former
    name ``repeat`` read as ``biological_repeat`` (one ``DeprecationWarning``;
    the caller's tables are not modified), and ``check_tables`` on the setup's
    columns and control, whose warnings
    (copied rows, labels that print alike, repeats without a control, ...) a
    workflow would otherwise never mention. With ``warn`` they are also issued
    as one ``UserWarning``; ``report`` prints them in its overview instead. Its
    errors are left to the analyses, which raise with the same report.
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
    if setup.rare_label is not None and setup.rare_label not in objects.columns:
        raise ValueError(
            f"rare_label {setup.rare_label!r} is not a column of the object table: the rare "
            "phenotype cannot be judged on held-back fields without its labels"
        )
    check = _check_tables(
        objects, fovs, feature_cols=setup.columns, control=setup.control, **setup.common
    )
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
    the second runs only a fixed-count rule, whose decisions are discarded.
    """
    common = {"feature_cols": setup.columns, "objective": setup.objective(), **setup.common}
    if prepared is raw:
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
        **common,
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
            "values": "stopping rules on raw values; outcome criteria on normalised values",
        },
    )


def pooled_columns(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, columns: Sequence[str]
) -> list[str]:
    """The ``columns`` of the FOV table that some analysis group holds several values of."""
    explicit = [*(setup.group_cols or ()), *([setup.fov_col] if setup.fov_col else [])]
    objects, table = drop_empty_optional_columns(objects, fovs, explicit)
    assert table is not None
    groups = list(resolve_group_cols(objects, table, setup.group_cols))
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
    data = validate_input_tables(objects, fovs=fovs, feature_cols=[setup.readout], **setup.common)
    sizes = data.objects.groupby(list(data.fov_key), dropna=False).size()
    return bool(len(sizes)) and int(sizes.max()) == 1


def one_per_field_reasons(setup: Setup) -> dict[str, str]:
    """Why the questions about fields do not apply when each field holds one observation,
    by question (``with_reasons``): fields and observations coincide, so their difference
    cannot be measured, and a new state needing several similar observations in one field
    cannot occur."""
    unit = resolve_observation_unit(setup.observation_unit)[0]
    words = setup.unit_words
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
    assert setup.control is not None
    if not per_field:
        return {}
    common = setup.common
    pooled = pooled_columns(objects, fovs, setup, list(setup.control))
    if pooled:
        reasons["unit"] = pooling_reason(
            "the object-rule check against the control is", setup, pooled
        )
        if setup.features:
            reasons["profile"] = pooling_reason(
                f"unusual {setup.unit_words} and their clustering are", setup, pooled
            )
        return {}
    results: dict[str, Any] = {
        "object_rules": object_rule_check(
            objects,
            fovs,
            feature=setup.readout,
            targets=setup.tolerances,
            control=setup.control,
            random_state=setup.random_state,
            **common,
        )
    }
    if setup.features:
        unusual = unusual_objects(
            objects,
            fovs,
            feature_cols=list(setup.features),
            reference=setup.control,
            random_state=setup.random_state,
            **common,
        )
        results["unusual"] = unusual
        label = unusual_label(objects, setup)
        # The label flags a share of ordinary objects by construction, so a field holds
        # the state only with the novelty rule's m of them, and the control groups'
        # own incidence is printed beside it (P6-1).
        novelty = setup.novelty or default_novelty(setup.features)
        results["clustering"] = fov_clustering(
            objects.assign(**{label: unusual.labels}),
            fovs,
            label_col=label,
            states=[True],
            random_state=setup.random_state,
            min_objects_per_fov=novelty.min_novel_observations,
            reference=setup.control,
            **common,
        )
    return results


def rare_clustering(objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup) -> FOVClustering:
    """R4 for the declared rare phenotype: are its labelled objects clustered in few fields?

    The fields to see it are planned for the declared target: seen in at least
    ``rare.min_fov_occurrences`` fields with ``rare.detection_probability``.
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
    """R7: each condition against the control, per repeat, on the prepared values."""
    contrasted = setup.control is not None and CONDITION_COLUMN in setup.control
    if not (
        contrasted and CONDITION_COLUMN in fovs.columns and fovs[CONDITION_COLUMN].nunique() > 1
    ):
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
            control=setup.control,
            random_state=setup.random_state,
            fields_independent=setup.fields_independent,
            **setup.common,
        )
    }
    if len(setup.columns) > 1:
        results["discovery"] = feature_discovery(
            objects,
            fovs,
            features=setup.columns,
            control=setup.control,
            random_state=setup.random_state,
            **setup.common,
        )
        results["ranking"] = feature_ranking(
            objects,
            fovs,
            features=setup.columns,
            control=setup.control,
            random_state=setup.random_state,
            fields_independent=setup.fields_independent,
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
