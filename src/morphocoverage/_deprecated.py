"""Deprecated names kept working after the renaming of cells to objects (private).

Observation rows are generic (cells, nuclei, patches, slices, ...), so public
names say *objects*. The former cell names still resolve, through a module
``__getattr__``, to the same object with a ``DeprecationWarning``; they are not
listed in ``__all__`` and will be removed in a later release.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

RENAMED = {
    "plan_cells_to_average": "plan_objects_to_average",
    "CellCountPlan": "ObjectCountPlan",
    "PlannedCellCountEstimator": "PlannedObjectCountEstimator",
    "PlannedCellCountStrategy": "PlannedObjectCountStrategy",
    "RandomCellRelativeError": "RandomObjectRelativeError",
    "cell_averaging_cv": "object_averaging_cv",
}


def module_getattr(
    module_name: str, namespace: Mapping[str, Any], names: tuple[str, ...]
) -> Callable[[str], Any]:
    """A module ``__getattr__`` resolving the deprecated ``names`` to their new objects."""

    def __getattr__(name: str) -> Any:
        if name in names:
            new = RENAMED[name]
            warnings.warn(
                f"{module_name}.{name} is deprecated; use {module_name}.{new}",
                DeprecationWarning,
                stacklevel=2,
            )
            return namespace[new]
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return __getattr__


__all__ = ["RENAMED", "module_getattr"]
