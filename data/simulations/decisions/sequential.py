"""Sequential acquisition: the outcome generator of module P4 ("Did I image enough?" for a
mean, run as the microscope runs it).

One sample (one condition in one biological repeat, with its wells) is acquired field by
field; a stopping rule looks at the fields so far at fixed looks (``schedule``) and stops at
the first look its criterion passes. The fields come in acquisition order: within a well in
the layout's order (along which ``Design.autocorrelation`` correlates the field effects),
the wells in turn (``interleave``, the default: one field of each well, then the next) or
one after the other. The values are ``scalar.generate``'s, so every distribution, informative
cluster size, unrecorded wells and missing fields apply. The target is the sample's own mean
with every field imaged (``Sequence.truth``: the mean over its wells of each well's
conditional cell-weighted mean, ``scalar.ScalarData.conditional``; ``truth_field`` the
field-balanced one): a stop is false when the estimate at the stop misses it by more than
the tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from morphocoverage.metadata import TECHNICAL_REPEAT_COLUMN

from .design import FIELD_COLUMN, ORDER_COLUMN, Design
from .scalar import Scalar, generate


@dataclass(frozen=True)
class Sequence:
    """One sample acquired in order: its tables, the fields in acquisition order, each
    field's well (always, recorded or not), and the targets."""

    objects: pd.DataFrame
    fovs: pd.DataFrame
    order: tuple[str, ...]
    wells: tuple[str, ...]
    truth: float
    truth_field: float
    feature: str = "x0"

    def field_values(self, k: int | None = None) -> list[np.ndarray]:
        """The values of the first ``k`` fields (all by default), in order; empty arrays for
        empty fields."""
        chosen = self.order if k is None else self.order[:k]
        grouped = self.objects.groupby(FIELD_COLUMN)[self.feature]
        found = {name: values.to_numpy(dtype=float) for name, values in grouped}
        return [found.get(name, np.array([])) for name in chosen]

    def recorded_wells(self, k: int | None = None) -> tuple[str, ...] | None:
        """The wells of the first ``k`` fields as the tables record them; ``None`` when the
        design does not record wells."""
        if TECHNICAL_REPEAT_COLUMN not in self.fovs:
            return None
        return self.wells if k is None else self.wells[:k]


def _ordered(fovs: pd.DataFrame, interleave: bool) -> pd.DataFrame:
    """The fields of ``fovs`` (with their hidden ``well``) in acquisition order."""
    if interleave:
        return fovs.sort_values([ORDER_COLUMN, "well"], kind="stable")
    return fovs.sort_values(["well", ORDER_COLUMN], kind="stable")


def sequence(
    design: Design, spec: Scalar, rng: np.random.Generator, interleave: bool = True
) -> Sequence:
    """One sample of ``design`` (its first condition in its first repeat; the design should
    hold one) acquired in order (see the module)."""
    data = generate(design, spec, rng)
    fields = data.layout.fields
    first = fields[
        (fields["condition"] == design.conditions[0]) & (fields["biological_repeat"] == "R1")
    ]
    ordered = _ordered(first, interleave)
    conditional = data.conditional
    mine = conditional[
        conditional["well"].isin(set(first["well"])) & (conditional["feature"] == "x0")
    ]
    return Sequence(
        objects=data.objects[data.objects[FIELD_COLUMN].isin(set(first[FIELD_COLUMN]))],
        fovs=data.fovs[data.fovs[FIELD_COLUMN].isin(set(first[FIELD_COLUMN]))],
        order=tuple(ordered[FIELD_COLUMN]),
        wells=tuple(ordered["well"]),
        truth=float(mine["cell"].mean()),
        truth_field=float(mine["field"].mean()),
    )


def schedule(n_fields: int, first: int = 5, every: int = 5) -> list[int]:
    """The looks: after ``first`` fields, then every ``every`` fields, and at the last."""
    looks = list(range(first, n_fields + 1, every))
    if n_fields >= 1 and (not looks or looks[-1] != n_fields):
        looks.append(n_fields)
    return looks


__all__ = ["Sequence", "schedule", "sequence"]
