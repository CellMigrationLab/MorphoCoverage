"""One observation per field (one cell per image): the field questions on the rows as fields
(private; owner, 2026-10-06, replacing the earlier "not applicable" decision).

When every field with observations holds exactly one, the fields and the observations
coincide, and the questions about fields are asked of the rows as fields:

- the Foundation's spread over fields (``fov_variability``: the SD over fields, a design
  effect of 1 by construction);
- unusual fields, worded "unusual images" (``field_unit``), against the control as any
  unusual observations;
- novelty across fields and the positive control judged by it: the definition of new needs
  ``m`` similar observations in one field, which a field of one cannot hold, so its field
  version (``support_novelty.per_field``, which the estimator itself applies on such data,
  so the report, ``optimise``, ``benchmark`` and a live session agree) calls a field's
  observation new when no other field holds one within the radius: a state seen once,
  exploratory (module P10 of the validation protocol);
- "Did I image enough?" per field, with that definition.

The acquisition anomalies then read single observations too, so a field they name is not
listed again as a new state in question 6 (``listed_once``; owner, 2026-10-06, audit 10: 8
of 9 such fields were also "new states", one list calling them never biological, the other
a new biological state).

Only the ideas within a field stay not applicable (``one_per_field_reasons``): unusual cells
or a labelled state clustering in a few fields (a field of one holds a state or not), and
the mean-precision count rule checked with whole fields (random cells and whole fields are
the same draws).

Shared-private: used by ``workflows.retrospective``, ``_shared`` and ``_embedding_report``.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING

import pandas as pd

from .._wording import count
from ..data_check import validate_input_tables
from ..schema import OBJECT_LIKE_UNITS, resolve_observation_unit
from ..strategies.support_novelty import (
    SupportNoveltyEstimator,
    SupportNoveltyStrategy,
    per_field,
)
from ._feature_space import table_columns

if TYPE_CHECKING:
    from ..retrospective.inspection import InspectionResult
    from ..sampling_unit.acquisition import AcquisitionAnomalies
    from .setup import Setup

# The word for an observation that is a whole field (one cell per image).
FIELD_UNIT = "image"


def one_per_field(objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup) -> bool:
    """Whether every field with observations holds exactly one (one cell per image)."""
    columns = setup.columns or table_columns(setup)
    data = validate_input_tables(objects, fovs=fovs, feature_cols=columns, **setup.common)
    sizes = data.objects.groupby(list(data.fov_key), dropna=False).size()
    return bool(len(sizes)) and int(sizes.max()) == 1


def field_unit(setup: Setup) -> str | None:
    """The unit unusual observations are worded in with one per field: ``image`` for cells,
    nuclei or undeclared objects (an unusual cell is an unusual image), else the declared
    unit (a patch, a field summary)."""
    unit, declared = resolve_observation_unit(setup.observation_unit)
    return FIELD_UNIT if not declared or unit in OBJECT_LIKE_UNITS else setup.observation_unit


def field_novelty(novelty: SupportNoveltyEstimator) -> SupportNoveltyEstimator:
    """The definition of new on fields of one observation (``support_novelty.per_field``: new
    when no other field holds one within the radius), the estimator's own, so the report,
    ``optimise``, ``benchmark`` and a live session give one verdict."""
    return per_field(novelty)


def field_setup(setup: Setup) -> Setup:
    """``setup`` with the field version of its definition of new and of any novelty rule it
    declares (``per_field``), so the report's words and its rules name what is judged;
    unchanged without a definition needing several observations in one field."""
    rules = setup.rules
    changed = rules is not None and any(isinstance(r, SupportNoveltyStrategy) for r in rules)
    novelty = setup.novelty
    if (novelty is None or novelty.min_novel_observations <= 1) and not changed:
        return setup
    fields = {"novelty": None if novelty is None else per_field(novelty)}
    if changed:
        fields["rules"] = tuple(
            per_field(r) if isinstance(r, SupportNoveltyStrategy) else r for r in rules
        )
    with warnings.catch_warnings():  # the Setup warned once, when it was made
        warnings.simplefilter("ignore")
        return replace(setup, **fields)


def one_per_field_reasons(setup: Setup) -> dict[str, str]:
    """By question, what stays not applicable with one observation per field: the ideas
    within a field (see the module), each with its reason (``with_reasons``)."""
    unit, words = setup.unit_word, setup.unit_words
    reasons: dict[str, str] = {}
    if setup.mean_target and setup.reference_control is not None:
        reasons["unit"] = (
            f"the mean-precision count rule is not checked with whole fields: each field holds "
            f"one {unit}, so random {words} and whole fields are the same draws"
        )
    clustered = [
        *(["the labelled state"] if setup.rare_label is not None else []),
        *([f"unusual {words}"] if setup.reference_control is not None else []),
    ]
    if clustered:
        verb = "clusters" if clustered[-1] == "the labelled state" else "cluster"  # C13
        reasons["profile"] = (
            f"whether {' or '.join(clustered)} {verb} in a few fields is not applicable: each "
            f"field holds one {unit}, so a field holds a state or not"
        )
    return reasons


def listed_once(
    inspection: InspectionResult | None, anomalies: AcquisitionAnomalies | None, setup: Setup
) -> InspectionResult | None:
    """Question 6 without the new states of the fields the acquisition anomalies name, said
    in one sentence (``settings["listed_elsewhere"]``): with one observation per field both
    read the same single observations (see the module). Unchanged when none is named."""
    if inspection is None or anomalies is None or not anomalies.found or inspection.fovs.empty:
        return inspection
    keys = [c for c in anomalies.settings["fov_key"] if c in inspection.fovs.columns]
    named = set(map(tuple, anomalies.fields[keys].astype(str).to_numpy()))
    own = pd.Series(list(map(tuple, inspection.fovs[keys].astype(str).to_numpy())))
    drop = (inspection.fovs["reason"] == "new state").to_numpy() & own.isin(named).to_numpy()
    if not drop.any():
        return inspection
    fields = len(set(own[drop]))
    unit, units = setup.unit_word, setup.unit_words
    sentence = (
        f"{count(fields, 'field')} named among the acquisition anomalies "
        f"{'is' if fields == 1 else 'are'} not listed again as new states: each field holds one "
        f"{unit}, so both read the same single {units}."
    )
    settings = {**inspection.settings, "listed_elsewhere": sentence}
    kept = inspection.fovs[~drop].reset_index(drop=True)
    return replace(inspection, fovs=kept, settings=settings)


__all__ = [
    "FIELD_UNIT",
    "field_novelty",
    "field_setup",
    "field_unit",
    "listed_once",
    "one_per_field",
    "one_per_field_reasons",
]
