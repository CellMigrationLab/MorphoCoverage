"""Representation controls: could these features recognize the phenotype I care about?

Question R1 of ``docs/report/questions.md``. A "nothing new" answer is only as good as
the features it was computed on: a phenotype they cannot separate is never new.
A declared control tests this on the dataset itself:

- **positive**: FOVs of a declared phenotype (for example a knockdown well),
  judged as candidate next FOVs against ``fovs_seen`` random FOVs of the
  reference. The share judged new is the **detection rate** for whole fields.
  With ``planted_objects``, that many objects of a random positive FOV are also
  added to remaining reference FOVs: the detection rate for a small group of such
  objects in an ordinary field. They come from other fields, so this approximates a rare state.
- **reference**: the declared negative population (for example MOCK). Its remaining FOVs,
  judged the same way, give the **false-alarm rate**. Without a reference, the undeclared
  rest of the stratum stands in for it, and the false-alarm rate is "not assessed".
- With neither, nothing can be said: "not assessed".
**Leave one field out.** The rates above judge a field against ``fovs_seen`` reference fields,
as a stopping rule judges the next field. ``inspect_fovs`` judges each field against every
other field of its analysis group, and the false-alarm rate depends strongly on how many
fields were seen (41% after 3 fields, 0% after 29, on one plate). So the same definition is
also measured that way, within the inspection's analysis groups (``lofo_group_cols``; by
default the analysis groups of the data, as ``inspect_fovs`` resolves them)
inside each stratum: each reference field against every other reference
field of its group (``reference_lofo``, the false-alarm rate of the
inspection, which the report uses for its "expected by chance" clause), and
each positive field against the reference fields of each reference group
(``positive_lofo``, one check per positive field and group: the detection
rate after a whole group was acquired, which the report uses to adjust the
unseen-state bound). Both are measured in every stratum that has the fields
they need, also where the rates after ``fovs_seen`` fields cannot be.

**Planted, leave one field out.** With ``planted_objects``, the same
planting is also judged as the unseen-state bound judges a field: each
reference field of an analysis group, with that many objects of a random
positive field added, against the other reference fields of its group
(``planted_lofo``). It is the detection rate of a small state in an
ordinary field, and the report adjusts the unseen-state bound by it
(``report_profile``): the whole-field rate describes whole-field phenotypes
only (97% whole-field against 19% for 5 planted objects on one dataset).

**Seen fields hold observations.** The ``fovs_seen`` reference fields of a
draw are drawn among the reference fields holding observations: an empty
field adds nothing to the reference, and with it among the seen fields the
definition of new often could not be judged at all (with ``fovs_seen`` equal
to the estimator's ``min_fovs``, any empty seen field left the whole draw
unjudged). Empty reference fields remain candidates, and are never new.
**Positive fields are judged only when they hold observations**: an empty
positive field cannot show the phenotype, and counting it as a miss made
the detection rate the share of positive fields with observations (67% at
30% empty fields, although every field with cells was recognized). Their
number is reported apart (``settings["positive_empty"]``).

"New" is the ``SupportNoveltyEstimator`` definition the report uses
(``novel_fovs_against``, transform fitted on the seen FOVs only). Reference and
positive FOVs are paired within ``pair_cols`` (for example the plate), so a
plate effect is not mistaken for a phenotype; pooling across plates must be
asked for with ``pair_cols=[]``. Rates are pooled over draws and candidates,
which are not independent; they describe these controls in this dataset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._acquisition import ACQUISITION, acquisition_identity
from .._checks import check_seed, is_integer
from .._frozen import owned
from .._parallel import check_n_jobs
from .._repeat_alias import refuse_repeat_alias
from ..data_check import validate_input_tables
from ..evaluator import FovSubsets
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS, column_names, empty_values
from ..schema import ValidatedInput
from ..strategies.support_novelty import SupportNoveltyEstimator
from ..sufficiency_text import novelty_definition
from ._control_draws import CANDIDATES, _ControlRun, _measure_strata
from .controls_text import control_summary


def _selector(value: Mapping[str, Any] | None, label: str) -> dict[str, tuple] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{label} must map FOV-table columns to accepted values")
    refuse_repeat_alias(**{label: value})
    selector = {}
    for column, accepted in value.items():
        if isinstance(accepted, str) or not isinstance(accepted, Sequence):
            accepted = (accepted,)
        selector[str(column)] = tuple(accepted)
    return selector


@dataclass(frozen=True)
class RepresentationControl:
    """Declared control FOVs, selected by FOV-table values.

    ``positive`` and ``reference`` map FOV-table columns to accepted values,
    for example ``{"condition": ["YAP"]}`` and ``{"condition": ["MOCK"]}``.
    ``planted_objects`` (needs both) also plants that many positive objects into
    reference FOVs.
    """

    positive: Mapping[str, Any] | None = None
    reference: Mapping[str, Any] | None = None
    planted_objects: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "positive", _selector(self.positive, "positive"))
        object.__setattr__(self, "reference", _selector(self.reference, "reference"))
        if self.planted_objects is not None:
            if not is_integer(self.planted_objects):
                raise TypeError("planted_objects must be a positive integer")
            if self.planted_objects < 1:
                raise ValueError("planted_objects must be a positive integer")
            object.__setattr__(self, "planted_objects", int(self.planted_objects))
            if self.positive is None or self.reference is None:
                raise ValueError("planting cells needs a positive and a reference control")

    @property
    def parameters(self) -> dict[str, Any]:
        """The declaration, for provenance."""
        return {
            "positive": None
            if self.positive is None
            else {k: list(v) for k, v in self.positive.items()},
            "reference": None
            if self.reference is None
            else {k: list(v) for k, v in self.reference.items()},
            "planted_objects": self.planted_objects,
        }


@owned("table", "pooled", "settings")
@dataclass(frozen=True)
class ControlResult:
    """Rates per stratum and candidate kind; ``status`` says what could be assessed.

    ``table`` has one row per stratum (``pair_cols`` values) and candidate
    (``reference``: false alarms; ``positive``: whole positive FOVs;
    ``planted``: reference FOVs with planted positive objects;
    ``reference_lofo``, ``positive_lofo`` and ``planted_lofo``: judged leave
    one field out, see the module), with ``judged_new``, ``candidates`` and
    ``rate``; ``pooled``
    the same over strata. ``status`` is ``"assessed"``, ``"positive only"``
    or ``"not assessed"`` (from the ``positive`` and ``reference`` rows).
    ``settings["planted_skipped"]`` lists the strata where no objects could
    be planted, with the reason; ``settings["positive_empty"]`` counts the
    positive fields without observations, which are not judged;
    ``settings["lofo_group_cols"]`` the analysis groups the
    leave-one-field-out rates were measured within, and
    ``settings["novelty_definition"]`` the definition of new in words, so a
    report combines the rates only with answers judged the same way.
    """

    table: pd.DataFrame
    pooled: pd.DataFrame
    status: str
    pair_cols: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)

    def summary(self, detail: str = "short") -> str:
        """The control's answer in words: short (one table of the rates, each caveat once)
        or ``detail="full"`` (the counts behind every rate, stratum by stratum)."""
        return control_summary(self, detail)


def _select(fovs: pd.DataFrame, selector: dict[str, tuple]) -> pd.Series:
    mask = pd.Series(True, index=fovs.index)
    for column, accepted in selector.items():
        if column not in fovs.columns:
            raise ValueError(f"control column {column!r} is not in the FOV table")
        mask &= fovs[column].isin(accepted)
    return mask


def _lofo_columns(data: ValidatedInput, lofo_group_cols: Sequence[str] | None) -> list[str]:
    """The analysis-group columns the leave-one-field-out rates are measured within.

    By default the analysis groups ``inspect_fovs`` resolves for these tables:
    the MorphoBricks group columns that identify a FOV here.
    """
    if lofo_group_cols is None:
        return [c for c in MORPHOBRICKS_GROUP_COLUMNS if c in data.fov_key and c != data.fov_col]
    columns = list(column_names(lofo_group_cols, "lofo_group_cols"))
    missing = [c for c in columns if c not in data.fovs.columns]
    if missing:
        raise ValueError(f"lofo_group_cols missing from the FOV table: {missing}")
    return columns


def _check_declaration(
    novelty: Any, control: Any, fovs_seen: Any, draws: Any, random_state: Any
) -> None:
    """Raise when the estimator, the control, the counts or the seed are not usable."""
    if not isinstance(novelty, SupportNoveltyEstimator):
        raise TypeError("novelty must be a SupportNoveltyEstimator")
    if not isinstance(control, RepresentationControl):
        raise TypeError("control must be a RepresentationControl")
    for value, label in ((fovs_seen, "fovs_seen"), (draws, "draws")):
        if not is_integer(value) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if not is_integer(random_state):
        raise TypeError("random_state must be an integer seed")


def _check_pairing(data: ValidatedInput, pair_cols: list[str]) -> None:
    """Raise when a pairing column is missing from the FOV table or has missing values."""
    missing = [c for c in pair_cols if c not in data.fovs.columns]
    if missing:
        raise ValueError(f"pair_cols missing from the FOV table: {missing}")
    unpaired = [c for c in pair_cols if empty_values(data.fovs[c]).any()]
    if unpaired:  # a missing value equals nothing, so those fields would silently drop out
        raise ValueError(f"pair_cols {unpaired} have missing values: those fields cannot be paired")


def _control_run(
    data: ValidatedInput,
    novelty: SupportNoveltyEstimator,
    control: RepresentationControl,
    lofo_cols: list[str],
    counts: tuple[int, int, int, int],
) -> _ControlRun:
    """The control's fields and what every stratum is measured with; ``counts`` are
    ``fovs_seen``, ``draws``, ``random_state`` and ``n_jobs``."""
    positive_mask = _select(data.fovs, control.positive)
    if control.reference is not None:
        base_mask = _select(data.fovs, control.reference)
        if (base_mask & positive_mask).any():
            raise ValueError("a FOV cannot be both a positive and a reference control")
    else:
        base_mask = ~positive_mask
    subsets = FovSubsets(data)
    observed = np.bincount(subsets.positions[subsets.positions >= 0], minlength=len(data.fovs)) > 0
    fovs_seen, draws, random_state, n_jobs = counts
    return _ControlRun(
        novelty=novelty,
        data=data,
        subsets=subsets,
        observed=observed,
        positive_mask=positive_mask,
        base_mask=base_mask,
        lofo_cols=lofo_cols,
        planted_objects=control.planted_objects,
        with_reference=control.reference is not None,
        fovs_seen=fovs_seen,
        draws=draws,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def _pooled(table: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """The rates over strata, and the status: what was actually measured."""
    pooled = (
        table.groupby("candidate", sort=False)[["judged_new", "candidates", "not_judged"]]
        .sum()
        .reset_index()
    )
    pooled["rate"] = [
        new / total if total else None
        for new, total in zip(pooled["judged_new"], pooled["candidates"], strict=True)
    ]
    measured = set(pooled.loc[pooled["candidates"] > 0, "candidate"])
    # The status says what was actually measured, not what was declared.
    status = "assessed" if {"positive", "reference"} <= measured else "positive only"
    if "positive" not in measured:
        status = "not assessed"
    return pooled, status


def representation_control(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    novelty: SupportNoveltyEstimator,
    control: RepresentationControl,
    pair_cols: Sequence[str],
    fovs_seen: int,
    draws: int = 20,
    random_state: int = 0,
    fov_col: str | None = None,
    observation_unit: str | None = None,
    n_jobs: int = 1,
    identity_cols: Sequence[str] = (),
    lofo_group_cols: Sequence[str] | None = None,
) -> ControlResult:
    """Detection and false-alarm rates of ``novelty`` for the declared controls.

    ``fovs_seen`` is how many reference FOVs holding observations are
    "already imaged" in each draw, ``draws`` how many random draws per
    stratum. See the module docstring.
    ``identity_cols`` are columns that, with condition, repeat and ``fov_col``,
    identify one FOV (for example ``well`` when position names repeat in every
    well). They only name FOVs: controls are still compared across them, within
    ``pair_cols``.
    ``lofo_group_cols`` are the analysis groups within which the
    leave-one-field-out rates are measured: give the ``group_cols`` of the
    ``inspect_fovs`` (and ``sufficiency_report``) whose answers they are read
    with; ``None`` (the default) takes the analysis groups these tables
    resolve to by default, and ``[]`` measures them over each whole stratum.
    ``n_jobs`` runs the novelty checks of the draws in that many worker
    processes (``-1``: every CPU); results are identical for every value
    (bit for bit when BLAS is single-threaded; see ``morphocoverage._parallel``).
    """
    check_n_jobs(n_jobs)
    _check_declaration(novelty, control, fovs_seen, draws, random_state)
    fovs_seen, draws, random_state = int(fovs_seen), int(draws), check_seed(random_state)
    feature_cols = column_names(feature_cols, "feature_cols")  # read once: may be a generator
    pair_cols = list(column_names(pair_cols, "pair_cols"))
    identity_cols = column_names(identity_cols, "identity_cols")
    refuse_repeat_alias(
        pair_cols=pair_cols, identity_cols=identity_cols, lofo_group_cols=lofo_group_cols
    )
    settings = {
        "control": control.parameters,
        "novelty": dict(novelty.parameters),
        "novelty_definition": novelty_definition(novelty, list(feature_cols)),
        "pair_cols": pair_cols,
        "fovs_seen": fovs_seen,
        "draws": draws,
        "random_state": random_state,
    }
    columns = ["stratum", "candidate", "judged_new", "candidates", "not_judged", "rate"]
    if control.positive is None:
        empty = pd.DataFrame(columns=columns)
        return ControlResult(
            empty, empty.copy(), "not assessed", tuple(pair_cols), {**settings, "reason": "none"}
        )
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=feature_cols,
        fov_col=fov_col,
        # Identity only: the strata come from pair_cols, never from these columns.
        group_cols=list(identity_cols),
        observation_unit=observation_unit,
    )
    _check_pairing(data, pair_cols)
    lofo_cols = _lofo_columns(data, lofo_group_cols)
    run = _control_run(data, novelty, control, lofo_cols, (fovs_seen, draws, random_state, n_jobs))
    strata = data.fovs[pair_cols].drop_duplicates().to_dict("records") if pair_cols else [{}]
    measured = _measure_strata(run, strata)
    table = pd.DataFrame(measured.rows, columns=[*pair_cols, *columns])
    pooled, status = _pooled(table)
    settings = {
        **settings,
        ACQUISITION: acquisition_identity(data),
        "lofo_group_cols": lofo_cols,
        "skipped_strata": measured.skipped,
        "planted_skipped": measured.planted_skipped,
        "positive_empty": measured.positive_empty,
        "reason": "unusable" if measured.skipped else None,
        "observation_unit": data.observation_unit,
        "observation_unit_declared": data.observation_unit_declared,
    }
    return ControlResult(table, pooled, status, tuple(pair_cols), settings)


__all__ = [
    "CANDIDATES",
    "ControlResult",
    "RepresentationControl",
    "control_summary",
    "representation_control",
]
