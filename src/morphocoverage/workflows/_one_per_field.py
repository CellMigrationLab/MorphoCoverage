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
  version (``field_novelty``) counts the ``m`` similar observations across fields instead: a
  field's observation is new when fewer than ``m`` other fields hold one within the same
  radius (each field one observation), so a new state is one seen in at most ``m`` fields;
  scattered outliers are then new too, as each field shows one observation;
- "Did I image enough?" per field, with that definition.

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

from ..data_check import validate_input_tables
from ..schema import OBJECT_LIKE_UNITS, resolve_observation_unit
from ..strategies.support_novelty import SupportNoveltyEstimator
from ._feature_space import table_columns

if TYPE_CHECKING:
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
    """The definition of new on fields of one observation (see the module): the ``m`` similar
    observations of a new state counted across fields, so support needs ``m`` other fields
    and a field's one observation can be new; the radius, transform and confidence kept."""
    needed = novelty.min_novel_observations
    if needed <= 1:
        return novelty
    others = max(novelty.min_other_fovs, needed)
    return replace(
        novelty,
        min_other_fovs=others,
        min_fovs=max(novelty.min_fovs, others + 1),
        min_novel_observations=1,
    )


def field_setup(setup: Setup) -> Setup:
    """``setup`` with the field version of its definition of new (``field_novelty``); a
    declared count of states seen in few fields is raised past its ``min_other_fovs`` when
    needed, the default follows it; unchanged without a definition of new."""
    novelty = setup.novelty
    if novelty is None or novelty.min_novel_observations <= 1:
        return setup
    field = field_novelty(novelty)
    few = setup.few_fovs
    few = None if few is None else max(few, field.min_other_fovs + 1)
    with warnings.catch_warnings():  # the Setup warned once, when it was made
        warnings.simplefilter("ignore")
        return replace(setup, novelty=field, few_fovs=few)


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
        *([f"unusual {words}"] if setup.reference_control is not None else []),
        *(["the labelled state"] if setup.rare_label is not None else []),
    ]
    if clustered:
        reasons["profile"] = (
            f"whether {' or '.join(clustered)} cluster in a few fields is not applicable: each "
            f"field holds one {unit}, so a field holds a state or not"
        )
    return reasons


__all__ = [
    "FIELD_UNIT",
    "field_novelty",
    "field_setup",
    "field_unit",
    "one_per_field",
    "one_per_field_reasons",
]
