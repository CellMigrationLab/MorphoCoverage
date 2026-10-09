"""Declared sampling hierarchy: which columns are the FOV, the sample and the independent unit.

Generic grouping columns carry no automatic biological meaning (AGENTS.md
section 2): a ``well`` or ``plate`` column means different things in different
experiments. Analyses that aggregate across samples, or that ask whether more
FOVs or more samples would help, therefore need the hierarchy declared:

    observation -> FOV -> sample -> (technical levels) -> biological repeat

``SamplingHierarchy`` names the columns; it never infers anything from column
names. As for FOV identity (AGENTS.md section 19), a unit is identified by its
own value together with every level outside it: well ``J23`` of plate ``1A``
and well ``J23`` of plate ``2A`` are two samples. Nesting therefore holds by
construction; validation checks that every level is present and complete (no
missing or blank value) and that no FOV is listed twice.

Uncertainty for statements across samples counts the declared independent
unit (``biological_repeat`` when given, otherwise ``sample``), never FOVs,
objects, subset draws or reservoir splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .._former_repeat import read_fields
from ..metadata import column_names, empty_values


@dataclass(frozen=True)
class SamplingHierarchy:
    """FOV-table columns of the FOV, the sample and (optionally) the biological repeat.

    ``technical_levels`` names further columns between the sample and the
    repeat (for example a plate within a repeat), innermost first.
    """

    fov: str
    sample: str
    biological_repeat: str | None = None
    technical_levels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        levels = column_names(self.technical_levels, "technical_levels")
        object.__setattr__(self, "technical_levels", () if levels is None else levels)
        # A level named ``repeat``, the former name of the biological repeat's column.
        read_fields(self, "fov", "sample", "biological_repeat", "technical_levels")
        names = [self.fov, self.sample, *self.technical_levels]
        if self.biological_repeat is not None:
            names.append(self.biological_repeat)
        if not all(isinstance(name, str) and name for name in names):
            raise TypeError("hierarchy levels must be column names")
        if len(set(names)) != len(names):
            raise ValueError("hierarchy levels must be distinct columns")

    @property
    def levels(self) -> tuple[str, ...]:
        """Columns from the FOV outwards."""
        outer = (self.biological_repeat,) if self.biological_repeat is not None else ()
        return (self.fov, self.sample, *self.technical_levels, *outer)

    @property
    def independent_unit(self) -> str:
        """The level whose members are independent replicates for across-sample claims."""
        return self.biological_repeat if self.biological_repeat is not None else self.sample

    @property
    def parameters(self) -> dict[str, Any]:
        """The declaration, for provenance."""
        return {
            "fov": self.fov,
            "sample": self.sample,
            "biological_repeat": self.biological_repeat,
            "technical_levels": list(self.technical_levels),
        }

    def unit_key(self, level: str) -> list[str]:
        """Columns identifying one unit of ``level``: the level and every level outside it."""
        levels = list(self.levels)
        if level not in levels:
            raise ValueError(f"{level!r} is not a level of this hierarchy")
        return levels[levels.index(level) :]

    def validate(self, fovs: pd.DataFrame) -> None:
        """Raise unless every level is a complete column and no FOV is listed twice."""
        levels = list(self.levels)
        missing = [c for c in levels if c not in fovs.columns]
        if missing:
            raise ValueError(f"hierarchy columns missing from the FOV table: {missing}")
        for column in levels:
            # Blank text is missing too, as for the schema's identity columns: several
            # blank wells would otherwise become one sample.
            if empty_values(fovs[column]).any():
                raise ValueError(
                    f"hierarchy column {column!r} contains missing values or blank text"
                )
        repeated = fovs.duplicated(subset=levels, keep=False)
        if repeated.any():
            row = fovs.loc[repeated, levels].iloc[0].to_dict()
            raise ValueError(f"FOV listed more than once in the FOV table: {row}")

    def count(self, fovs: pd.DataFrame, level: str) -> int:
        """Number of distinct units of ``level`` in ``fovs``."""
        return int(fovs.drop_duplicates(subset=self.unit_key(level)).shape[0])


__all__ = ["SamplingHierarchy"]
