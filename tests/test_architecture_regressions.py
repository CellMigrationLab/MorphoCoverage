"""Regression tests for architecture defects found in adversarial review of PR #1.

Each test encodes a scientific/data contract from AGENTS.md that the current
framework violates. They are marked ``xfail(strict=True)``: they document the
defect now, and will fail loudly (XPASS) once it is fixed, at which point the
marker must be removed so the contract stays protected.

The tests constrain behaviour, not a particular fix: where either a clear error
or a correct result would satisfy the contract, both are accepted.
"""

import pandas as pd
import pytest

from morphocoverage import (
    FixedFOVStrategy,
    ObjectiveConfig,
    SamplingObjective,
    SamplingStrategy,
    StrategyResult,
    evaluate_acquisition_prefixes,
    run_strategies,
    validate_input_tables,
)

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _n_fovs_or_error(**kwargs) -> int | None:
    """Return the validated FOV count, or None if validation raised ValueError."""
    try:
        return len(validate_input_tables(**kwargs).fovs)
    except ValueError:
        return None


@pytest.mark.xfail(strict=True, reason="group columns absent from FOV table are silently dropped")
def test_condition_in_objects_but_not_fov_table_is_not_silently_pooled():
    # Common real case: the FOV/acquisition table only lists image_file, while the
    # object table carries condition. Default grouping then pools control and drug.
    objects = pd.DataFrame(
        {
            "condition": ["control", "control", "drug", "drug"],
            "image_file": ["a.tif", "b.tif", "c.tif", "d.tif"],
            "area": [1.0, 2.0, 30.0, 40.0],
        }
    )
    fovs = pd.DataFrame({"image_file": ["a.tif", "b.tif", "c.tif", "d.tif"]})

    try:
        results = run_strategies(
            objects,
            fovs=fovs,
            strategies=[FixedFOVStrategy(3)],
            feature_cols=["area"],
            objective=POPULATION,
        )
    except ValueError:
        return
    assert {r.group.get("condition") for r in results} == {"control", "drug"}


@pytest.mark.xfail(strict=True, reason="pooling drops group columns from the FOV identity key")
def test_explicit_pooling_does_not_merge_fovs_with_reused_ids():
    # Microscope software commonly restarts position names per well/repeat.
    # Pooling repeats must not turn R1/F1 and R2/F1 into one FOV.
    objects = pd.DataFrame(
        {
            "repeat": ["R1", "R1", "R2", "R2"],
            "fov_id": ["F1", "F2", "F1", "F2"],
            "area": [1.0, 2.0, 3.0, 4.0],
        }
    )
    n_fovs = _n_fovs_or_error(objects=objects, feature_cols=["area"], group_cols=[])
    assert n_fovs in (None, 4)


@pytest.mark.xfail(strict=True, reason="image_file is preferred even when it is not one-to-one with FOVs")
def test_multi_position_image_file_does_not_collapse_fovs():
    # One multi-position file (e.g. .nd2/.czi scenes) holding four FOVs.
    objects = pd.DataFrame(
        {
            "image_file": ["plate.nd2"] * 4,
            "fov_id": ["P1", "P2", "P3", "P4"],
            "area": [1.0, 2.0, 3.0, 4.0],
        }
    )
    n_fovs = _n_fovs_or_error(objects=objects, feature_cols=["area"])
    assert n_fovs in (None, 4)


@pytest.mark.xfail(strict=True, reason="FOV column resolution silently differs between tables")
def test_fov_identifier_resolution_is_not_silently_inconsistent():
    # The object table has image_file, the FOV table does not; resolution falls
    # back to fov_id without telling the caller that the preferred key was skipped.
    objects = pd.DataFrame(
        {"image_file": ["a.tif", "b.tif"], "fov_id": ["F1", "F2"], "area": [1.0, 2.0]}
    )
    fovs = pd.DataFrame({"fov_id": ["F1", "F2", "F3"]})
    with pytest.raises(ValueError):
        validate_input_tables(objects, fovs=fovs, feature_cols=["area"])


@pytest.mark.xfail(strict=True, reason="sequential runner never validates the full object table")
def test_sequential_runner_rejects_objects_from_unlisted_fovs():
    # run_strategies raises on this input; the sequential runner silently drops it.
    objects = pd.DataFrame({"fov_id": ["F1", "F2", "GHOST"], "area": [1.0, 2.0, 999.0]})
    fovs = pd.DataFrame({"fov_id": ["F1", "F2"], "order": [1, 2]})
    with pytest.raises(ValueError):
        evaluate_acquisition_prefixes(
            objects,
            fovs,
            strategies=[FixedFOVStrategy(1)],
            feature_cols=["area"],
            acquisition_order_col="order",
            objective=POPULATION,
        )


@pytest.mark.xfail(strict=True, reason="tied acquisition order is resolved by row order")
def test_sequential_runner_rejects_tied_acquisition_order():
    # With ties, which FOV is "first" depends on table row order, so the
    # simulated acquisition history is not defined by the data.
    objects = pd.DataFrame({"fov_id": ["F1", "F2"], "area": [1.0, 2.0]})
    fovs = pd.DataFrame({"fov_id": ["F2", "F1"], "order": [1, 1]})
    with pytest.raises(ValueError):
        evaluate_acquisition_prefixes(
            objects,
            fovs,
            strategies=[FixedFOVStrategy(1)],
            feature_cols=["area"],
            acquisition_order_col="order",
            objective=POPULATION,
        )


class _CallCounter(SamplingStrategy):
    """Stateful strategy: its score is the number of calls this instance has seen."""

    name = "call_counter"

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, data, *, objective):
        self.calls += 1
        return StrategyResult(self.name, objective, score=float(self.calls))


@pytest.mark.xfail(strict=True, reason="one strategy instance is shared across analysis groups")
def test_strategy_state_does_not_flow_between_analysis_groups():
    # Any fitted state (scaler, PCA, reference sample, running statistics) held
    # by the instance would carry control-group information into the drug group.
    objects = pd.DataFrame(
        {"condition": ["control", "drug"], "image_file": ["a.tif", "b.tif"], "area": [1.0, 99.0]}
    )
    results = run_strategies(
        objects,
        strategies=[_CallCounter()],
        feature_cols=["area"],
        objective=POPULATION,
    )
    assert [r.score for r in results] == [1.0, 1.0]


class _MisreportingStrategy(SamplingStrategy):
    """Returns counts computed from objects only, as a naive strategy might."""

    name = "misreporting"

    def evaluate(self, data, *, objective):
        return StrategyResult(
            self.name,
            objective,
            score=0.0,
            n_fovs=data.objects[data.fov_col].nunique(),  # drops zero-object FOVs
            n_objects=len(data.objects),
        )


@pytest.mark.xfail(strict=True, reason="acquisition counts are strategy-reported, not framework-owned")
def test_framework_owns_fov_counts_including_zero_object_fovs():
    objects = pd.DataFrame({"fov_id": ["F1"], "area": [1.0]})
    fovs = pd.DataFrame({"fov_id": ["F1", "F2", "F3"]})
    [result] = run_strategies(
        objects,
        fovs=fovs,
        strategies=[_MisreportingStrategy()],
        feature_cols=["area"],
        objective=POPULATION,
    )
    assert result.n_fovs == 3


class _VisibleFovIds(SamplingStrategy):
    """Reports every FOV identifier reachable from the data it was given."""

    name = "visible_fov_ids"

    def evaluate(self, data, *, objective):
        visible = set(data.fovs[data.fov_col]) | set(data.objects[data.fov_col])
        for column in (data.fovs[data.fov_col], data.objects[data.fov_col]):
            if isinstance(column.dtype, pd.CategoricalDtype):
                visible |= set(column.cat.categories)
        return StrategyResult(self.name, objective, score=None, diagnostics={"ids": visible})


@pytest.mark.xfail(strict=True, reason="categorical dtype carries future FOV ids into prefixes")
def test_prefix_does_not_expose_future_fov_ids_via_categorical_levels():
    ids = pd.Categorical(["F1", "F2", "F3"])
    objects = pd.DataFrame({"fov_id": ids, "area": [1.0, 2.0, 3.0]})
    fovs = pd.DataFrame({"fov_id": ids, "order": [1, 2, 3]})
    results = evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=[_VisibleFovIds()],
        feature_cols=["area"],
        acquisition_order_col="order",
        objective=POPULATION,
    )
    assert results[0].result.diagnostics["ids"] == {"F1"}


class _FreeFormDecision(SamplingStrategy):
    name = "free_form"

    def evaluate(self, data, *, objective):
        return StrategyResult(self.name, objective, score=1.0, decision="probably fine")


@pytest.mark.xfail(strict=True, reason="decision is an unvalidated free-form string")
def test_decision_vocabulary_is_validated():
    # Benchmarks aggregate stop/continue across strategies; an unknown label
    # would silently count as neither.
    with pytest.raises(ValueError):
        run_strategies(
            pd.DataFrame({"fov_id": ["F1"], "area": [1.0]}),
            strategies=[_FreeFormDecision()],
            feature_cols=["area"],
            objective=POPULATION,
        )
