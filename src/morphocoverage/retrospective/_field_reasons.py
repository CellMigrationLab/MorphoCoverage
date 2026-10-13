"""The count and new-state reasons of "Which fields should I look at?" (private), and what
the groups of one ``inspect_fovs`` call gather beside their fields.

Split from ``inspection`` to keep that module under 600 lines; the reasons and their
thresholds are described there. Shared-private: used by ``retrospective.inspection``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._rare_label import known_flags
from .._style import group_label
from .._wording import count
from ..objectives import ObjectiveConfig, SamplingObjective
from ..schema import ValidatedInput, fov_positions, observation_words
from ..strategies.novelty_null import not_informative
from ..strategies.support_novelty import FOV_QUANTITY, SupportNoveltyEstimator
from ._stage_neighbours import neighbour_rows, stage_skipped

_POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


@dataclass
class _Gathered:
    """Beside the fields found: per reason, the groups where it could not be judged and why,
    the groups too small for the influence reason, each judged group's fields and null rate,
    the reasons judged somewhere, the fields inspected and those judged for new states."""

    skipped: list[dict] = field(default_factory=list)
    neighbours_skipped: list[dict] = field(default_factory=list)
    novelty_skipped: list[dict] = field(default_factory=list)
    small: list[dict] = field(default_factory=list)
    uninformative: list[dict] = field(default_factory=list)
    null: list[dict] = field(default_factory=list)
    checked: set[str] = field(default_factory=set)
    n_fovs: int = 0
    n_judged: int = 0

    def label(self, group: dict[str, Any]) -> None:
        """Name ``group`` in every entry gathered so far that has no name yet."""
        gathered = (
            *self.skipped,
            *self.neighbours_skipped,
            *self.novelty_skipped,
            *self.uninformative,
            *self.small,
        )
        for entry in gathered:
            entry.setdefault("label", group_label(group))  # the group as the summaries name it


def _counts(data: ValidatedInput, ratio: float) -> list[dict]:
    positions = fov_positions(data, data.fovs)
    counts = np.bincount(positions, minlength=len(data.fovs))
    median = float(np.median(counts))
    unit, words = observation_words(data)
    rows = []
    for fov, held in enumerate(counts):
        if held == 0 and median > 0:
            detail = f"no {words}; the median is {median:g}"
        elif median > 0 and (held >= ratio * median or held <= median / ratio):
            detail = f"{count(int(held), unit, words)}; the median is {median:g}"
        elif median == 0 and held > 0:
            # Most fields are empty: the fields with observations are the unusual ones.
            detail = f"{count(int(held), unit, words)}; most fields have none (the median is 0)"
        else:
            continue
        rows.append({"position": fov, "reason": "count", "value": float(held), "detail": detail})
    return rows


def _novel(
    data: ValidatedInput,
    novelty: SupportNoveltyEstimator,
    group: str,
    skipped: list[dict],
    label_col: str | None,
    uninformative: list[dict],
    null: list[dict],
) -> list[dict]:
    """FOVs holding a new state, with how many of its observations carry ``label_col``; when
    novelty is not estimable, the reason goes in ``skipped``, and when the definition is not
    informative at this group's size (its own null rate above 20%, ``novelty_null``; owner,
    2026-10-06, audit 10) in ``uninformative``, no field listed. ``null`` gathers each
    judged group's fields and null rate (the count chance alone would give)."""
    sizes = np.bincount(fov_positions(data, data.fovs), minlength=len(data.fovs))
    reason, rate = not_informative(novelty, sizes, data.feature_cols)
    if reason is not None:
        fields, features = int((sizes > 0).sum()), len(data.feature_cols)
        entry = {"group": group, "reason": reason, "rate": rate, "fields": fields}
        uninformative.append({**entry, "features": features})
        return []
    flags = novelty.novel_observation_flags(data, new_states_only=True)
    if flags is None:
        quantity = novelty.estimate(data, objective=_POPULATION).quantities[FOV_QUANTITY]
        skipped.append({"group": group, "reason": quantity.reason or "novelty is not estimable"})
        return []
    null.append({"fields": int((sizes > 0).sum()), "rate": rate})
    if not flags.any():
        return []
    positions = fov_positions(data, data.fovs)
    counts = np.bincount(positions[flags], minlength=len(data.fovs))
    labelled = None
    if label_col is not None:  # present and valid: checked once, in inspect_fovs
        marked = flags & known_flags(data.objects[label_col], f"label_col {label_col!r}")[1]
        labelled = np.bincount(positions[marked], minlength=len(data.fovs))
    unit, words = observation_words(data)

    def detail(fov: int, held: int) -> str:
        others = novelty.min_other_fovs  # more than 1 with one observation per field
        shown = "no other field shows" if others == 1 else f"fewer than {others} other fields show"
        text = f"{count(held, unit, words)} of a state {shown}"
        if labelled is not None and labelled[fov]:
            # The new state is the declared label: say so, it is what the user looks for.
            text += f" ({'all' if labelled[fov] == held else labelled[fov]} labelled {label_col})"
        return text

    return [
        {
            "position": fov,
            "reason": "new state",
            "value": float(held),
            "detail": detail(fov, int(held)),
        }
        for fov, held in enumerate(counts)
        if held
    ]


def _new_state_found(
    both: ValidatedInput,
    novelty: SupportNoveltyEstimator,
    novelty_cols: list[str],
    label_col: str | None,
    group: str,
    gathered: _Gathered,
) -> list[dict]:
    """The new-state reason in one group (``group``: its name), judged on ``novelty_cols``."""
    judged = both.with_tables(feature_cols=tuple(novelty_cols))
    before = len(gathered.novelty_skipped) + len(gathered.uninformative)
    found = _novel(
        judged,
        novelty,
        group,
        gathered.novelty_skipped,
        label_col,
        gathered.uninformative,
        gathered.null,
    )
    if len(gathered.novelty_skipped) + len(gathered.uninformative) == before:
        gathered.checked.add("new state")
        gathered.n_judged += len(both.fovs)
    return found


def _neighbour_found(
    group_data: ValidatedInput,
    stage_cols: Sequence[str],
    n_neighbours: int,
    neighbour_threshold: float,
    stage_unit: str,
    group: str,
    gathered: _Gathered,
) -> list[dict]:
    """The stage-neighbour reason in one group (``group``: its name)."""
    why = stage_skipped(group_data.fovs, *stage_cols, n_neighbours)
    if why is not None:
        gathered.neighbours_skipped.append({"group": group, "reason": why})
    else:
        gathered.checked.add("neighbours")
    return neighbour_rows(
        group_data,
        *stage_cols,
        n_neighbours,
        float(neighbour_threshold),
        stage_unit,
    )


__all__ = ["_Gathered", "_counts", "_neighbour_found", "_new_state_found"]
