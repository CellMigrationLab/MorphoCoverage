"""The former cell names still work, with a DeprecationWarning, and are no longer exported."""

import importlib

import pytest

import morphocoverage
from morphocoverage._deprecated import RENAMED
from morphocoverage.retrospective import RepresentationControl

MODULES = {
    "morphocoverage": (
        "plan_cells_to_average",
        "CellCountPlan",
        "PlannedCellCountEstimator",
        "PlannedCellCountStrategy",
    ),
    "morphocoverage.strategies": (
        "plan_cells_to_average",
        "CellCountPlan",
        "PlannedCellCountEstimator",
        "PlannedCellCountStrategy",
    ),
    "morphocoverage.strategies.jcb_planning": (
        "plan_cells_to_average",
        "CellCountPlan",
        "PlannedCellCountEstimator",
        "PlannedCellCountStrategy",
    ),
    "morphocoverage.outcomes": ("RandomCellRelativeError",),
    "morphocoverage.strategies.scalar_precision": ("cell_averaging_cv",),
}


@pytest.mark.parametrize(
    "module_name, old", [(m, name) for m, names in MODULES.items() for name in names]
)
def test_a_former_cell_name_resolves_to_its_object_name_with_a_warning(module_name, old):
    module = importlib.import_module(module_name)
    with pytest.warns(DeprecationWarning, match=RENAMED[old]):
        value = getattr(module, old)
    assert value is getattr(module, RENAMED[old])
    assert old not in getattr(module, "__all__", ())


def test_unknown_names_still_raise_attribute_error():
    with pytest.raises(AttributeError):
        morphocoverage.no_such_name  # noqa: B018


def test_planted_cells_is_the_deprecated_spelling_of_planted_objects():
    with pytest.warns(DeprecationWarning, match="planted_objects"):
        control = RepresentationControl({"c": ["a"]}, {"c": ["b"]}, planted_cells=3)
    assert control.planted_objects == 3
    assert control == RepresentationControl({"c": ["a"]}, {"c": ["b"]}, planted_objects=3)
    assert "planted_cells" not in control.parameters
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="planted_objects only"):
        RepresentationControl({"c": ["a"]}, {"c": ["b"]}, planted_objects=3, planted_cells=3)
