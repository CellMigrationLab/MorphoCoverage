"""Private helpers of the retrospective workflows: inputs, orders, replays, parts, reasons.

- ``replay_rules`` runs the stopping rules on the raw values (as the live
  session does) and judges the outcome criteria on the prepared values, so a
  rule gives the verdict in ``optimise`` and ``benchmark`` that it gives live;
- ``pooled_columns`` names the columns an analysis group spans several values
  of, so an analysis that needs them apart says "not assessed" instead of
  raising deep inside;
- ``with_reasons`` adds those reasons to an assembled report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import pandas as pd

from ..benchmark import AcquisitionOrder
from ..metadata import (
    CONDITION_COLUMN,
    REPEAT_COLUMN,
    drop_empty_optional_columns,
    resolve_group_cols,
)
from ..retrospective import MicroscopistReport, SamplingHierarchy, replay_acquisitions
from ..retrospective.criteria import OutcomeCriterion
from ..retrospective.replay import ReplayResult
from ..retrospective.report import QUESTIONS
from ..sampling_unit import (
    condition_contrast,
    feature_ranking,
    fov_clustering,
    object_rule_check,
    unusual_objects,
)
from ..strategies import FixedFOVStrategy
from ..strategies.base import SamplingStrategy
from ..sufficiency import SufficiencyReport
from .setup import Setup

_UNUSUAL = "morphocoverage_unusual"


def orders(setup: Setup, permuted: int, dispersed: int) -> list[AcquisitionOrder]:
    """The recorded order when there is one, ``permuted`` random and ``dispersed`` orders."""
    chosen = [AcquisitionOrder.recorded()] if setup.acquisition_order_col else []
    chosen += [AcquisitionOrder.permuted(seed) for seed in range(permuted)]
    if setup.stage_cols is not None:
        chosen += [AcquisitionOrder.dispersed(seed) for seed in range(dispersed)]
    return chosen


def check_tables(objects: Any, fovs: Any, setup: Any) -> None:
    """Refuse inputs the workflows cannot use, before any analysis runs."""
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
    if setup.rare_label is not None and setup.rare_label not in objects.columns:
        raise ValueError(
            f"rare_label {setup.rare_label!r} is not a column of the object table: the rare "
            "phenotype cannot be judged on held-back fields without its labels"
        )


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
    reveal the same fields in the same order (checked); the first judges only
    one cheap criterion and the second runs only a fixed-count rule, whose
    decisions are discarded.
    """
    common = {"feature_cols": setup.columns, "objective": setup.objective(), **setup.common}
    decided = replay_acquisitions(
        raw,
        fovs,
        strategies=strategies,
        criteria=criteria if prepared is raw else list(criteria)[:1],
        **common,
        **settings,
    )
    if prepared is raw:
        return decided
    scored = replay_acquisitions(
        prepared,
        fovs,
        strategies=[FixedFOVStrategy(1)],
        criteria=criteria,
        **common,
        **{**settings, "keep_results": False},
    )
    if not decided.streams.equals(scored.streams) or not decided.paths.equals(scored.paths):
        raise RuntimeError(
            "the replays of the raw and the normalised values revealed different fields; "
            "please report this as a bug"
        )
    return replace(
        decided,
        outcomes=scored.outcomes,
        settings={
            **decided.settings,
            "criteria": scored.settings["criteria"],
            "labels": scored.settings["labels"],
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


def pooling_reason(analyses: str, setup: Setup, pooled: Sequence[str]) -> str:
    """Why ``analyses`` were not run: the analysis groups pool the columns they need apart."""
    named = ", ".join(repr(c) for c in pooled)
    return (
        f"{analyses} not assessed: group_cols {tuple(setup.group_cols or ())} pool {named} "
        f"(an analysis group holds several of its values), and they compare within one "
        f"{' and one '.join(pooled)}; add {named} to group_cols to assess them"
    )


def with_reasons(document: MicroscopistReport, reasons: Mapping[str, str]) -> MicroscopistReport:
    """The report with each question's reason for what it could not assess.

    A question with nothing assessed shows the reason instead of the generic
    need; one assessed in part keeps its answers, adds the reason, and the
    closing section lists it among the notes.
    """
    if not reasons:
        return document
    titles = dict(QUESTIONS)
    sections, notes = dict(document.sections), list(document.notes)
    for key, reason in reasons.items():
        sentence = f"{reason[:1].upper()}{reason[1:]}."
        if document.inputs.get(key) == "not assessed":
            sections[key] = f"  {sentence}"
        else:
            sections[key] = f"{sections[key]}\n\n  {sentence}"
            notes.append(f"{titles[key]} {sentence}")
    return replace(document, sections=sections, notes=tuple(notes))


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


def control_checks(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, reasons: dict[str, str]
) -> dict[str, Any]:
    """The Foundation's rule check and R4's unusual observations against the control.

    Both compare each analysis group with the control's groups, so group
    columns that pool the control's columns make them "not assessed".
    """
    assert setup.control is not None
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
        results["clustering"] = fov_clustering(
            objects.assign(**{_UNUSUAL: unusual.labels}),
            fovs,
            label_col=_UNUSUAL,
            states=[True],
            random_state=setup.random_state,
            **common,
        )
    return results


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
            **setup.common,
        )
    }
    if len(setup.columns) > 1:
        results["ranking"] = feature_ranking(
            objects,
            fovs,
            features=setup.columns,
            control=setup.control,
            random_state=setup.random_state,
            **setup.common,
        )
    return results


def plan_settings(settings: dict[str, Any]) -> str:
    """The run's settings in one sentence, the provisional preset first."""
    return (
        f"Settings: a sample counts as adequately represented at k fields when "
        f"{settings['protocol_threshold']:.0%} of random acquisitions of k fields preserved "
        f"the criterion (protocol_threshold, a provisional preset); "
        f"{settings['reservoir_fraction']:.0%} of each group's fields held back, "
        f"{settings['n_splits']} hold-outs, {settings['draws']} draws per field count, "
        f"{settings['permuted_orders']} random and {settings['dispersed_orders']} dispersed "
        f"orders; random_state {settings['random_state']}."
    )


def protocol_reason(
    setup: Setup, hierarchy: SamplingHierarchy, strata: Sequence[str], groups: Sequence[str]
) -> str | None:
    """Why protocol reliability cannot run on these analysis groups, or ``None``."""
    levels = [c for c in hierarchy.unit_key(hierarchy.sample) if c not in groups]
    apart = [c for c in strata if c not in groups]
    if not levels and not apart:
        return None
    named = ", ".join(repr(c) for c in [*levels, *apart])
    reason = (
        f"protocol reliability across samples is not assessed: group_cols {tuple(groups)} pool "
        f"{named}, so the reliability has no result per sample ({hierarchy.sample}"
    )
    reason += f") and per stratum ({', '.join(strata)}); " if apart else "); "
    reason += f"add {named} to group_cols"
    if apart:
        reason += " (or pass effort_strata=() to pool the strata explicitly)"
    return reason


def no_hierarchy_reasons(setup: Setup, fovs: pd.DataFrame) -> dict[str, str]:
    """Why questions 4 (across samples) and 7 were not assessed although the FOV
    table has technical repeats: ``group_cols`` leave them out."""
    technical = setup.technical_col(fovs)
    if technical is None:
        return {}  # the generic need (a technical_repeat column or a hierarchy) holds
    pooled = (
        f"group_cols {tuple(setup.group_cols or ())} leave {technical!r} out, so the "
        "technical repeats are pooled"
    )
    return {
        "effort": (
            f"not assessed: {pooled} and no sampling hierarchy is derived; add {technical!r} "
            "to group_cols, or declare hierarchy to model them anyway"
        ),
        "next_time": (
            f"protocol reliability across samples is not assessed: {pooled}, and the "
            f"reliability then has no result per technical repeat; add {technical!r} to "
            "group_cols"
        ),
    }


__all__: list[str] = []
