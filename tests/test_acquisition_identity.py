"""The identity of the acquisition a result was computed on (``_acquisition``, ``_source``)."""

from __future__ import annotations

import dataclasses
import functools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _helpers_tables import _one_condition_tables, _variability

from morphocoverage import FixedFOVStrategy, ObjectiveConfig, SamplingObjective
from morphocoverage._acquisition import (
    acquisition_identity,
    check_one_acquisition,
    result_identity,
)
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    RedundancyResult,
    RepresentationControl,
    SamplingHierarchy,
    adequacy_points,
    dataset_overview,
    effort_projection,
    inspect_fovs,
    microscopist_report,
    order_efficiency,
    protocol_reliability,
    redundancy_distance,
    replay_acquisitions,
    representation_control,
    sampling_reliability,
    stopping_calibration,
)
from morphocoverage.retrospective.report import _RESULT_TYPES
from morphocoverage.sampling_unit import (
    condition_contrast,
    descriptor_spread_check,
    feature_discovery,
    feature_ranking,
    fov_clustering,
    fov_variability,
    object_rule_check,
    unusual_objects,
)
from morphocoverage.schema import validate_input_tables
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.workflows import (
    Setup,
    default_novelty,
    document,
    html_report,
    optimise,
    report,
)
from morphocoverage.workflows._source import source_identity


def test_the_identity_ignores_row_order_and_values_but_not_the_fields():
    objects, fovs = _one_condition_tables()
    one = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    shuffled = validate_input_tables(
        objects.assign(x=objects["x"] * 3), fovs=fovs.iloc[::-1], feature_cols=["x"]
    )
    other_objects, other_fovs = _one_condition_tables("g")
    other = validate_input_tables(other_objects, fovs=other_fovs, feature_cols=["x"])
    assert acquisition_identity(one) == acquisition_identity(shuffled)
    assert acquisition_identity(one) != acquisition_identity(other)


def _experiment(name: str, group_cols: list | None = None):
    objects = pd.DataFrame({"experiment": name, "fov_id": "f1", "x": [1.0, 2.0]})
    fovs = pd.DataFrame({"experiment": [name], "fov_id": ["f1"]})
    groups = ["experiment"] if group_cols is None else group_cols
    return validate_input_tables(
        objects, fovs=fovs, feature_cols=["x"], fov_col="fov_id", group_cols=groups
    )


def test_a_custom_identity_column_is_part_of_the_acquisition_identity():
    """Field f1 of experiment run_A and of run_B are two acquisitions (#416).

    Only the MorphoBricks group columns and the FOV column were hashed, so a custom
    identity column named in ``group_cols`` was invisible and the two collided.
    """
    run_a, run_b = _experiment("run_A"), _experiment("run_B")
    assert acquisition_identity(run_a) != acquisition_identity(run_b)
    with pytest.raises(ValueError, match="different acquisitions"):
        check_one_acquisition(
            {"a": _variability_settings(run_a), "b": _variability_settings(run_b)}
        )


def test_the_identity_ignores_column_order_and_morphobricks_group_selection():
    """Reordered FOV-table columns, and grouping by fewer MorphoBricks columns, keep it."""
    objects, fovs = _one_condition_tables()
    base = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    columns = list(reversed(fovs.columns))
    reordered = validate_input_tables(objects, fovs=fovs[columns], feature_cols=["x"])
    pooled = validate_input_tables(objects, fovs=fovs, feature_cols=["x"], group_cols=[])
    assert acquisition_identity(base) == acquisition_identity(reordered)
    assert acquisition_identity(base) == acquisition_identity(pooled)


def _variability_settings(data):
    from types import SimpleNamespace

    return SimpleNamespace(settings={"acquisition": acquisition_identity(data)})


def test_a_result_without_an_identity_is_refused_not_noted():
    """Q-MC-a: every result records its acquisition, so one that records none (rebuilt by
    hand, or edited) cannot be checked and is refused, alone or beside others; no note."""
    objects, fovs = _one_condition_tables()
    result = _variability(objects, fovs)

    class Bare:
        def __init__(self) -> None:
            self.settings: dict = {}

    with pytest.raises(ValueError, match="effort records no acquisition identity"):
        check_one_acquisition({"variability": result, "effort": Bare()})
    with pytest.raises(ValueError, match="effort records no acquisition identity"):
        check_one_acquisition({"effort": Bare()})
    check_one_acquisition({"variability": result, "effort": None})
    bare = RedundancyResult(pd.DataFrame(), {})
    with pytest.raises(ValueError, match="redundancy records no acquisition identity"):
        microscopist_report(redundancy=bare)
    stripped = dataclasses.replace(result, settings={})
    with pytest.raises(ValueError, match="variability records no acquisition identity"):
        microscopist_report(variability=stripped)


def _stage_tables(prefix: str = "f"):
    """Fields of one condition and repeat on a 4 x 3 stage grid, 100 µm apart, with a
    gradient along x, so neighbouring fields are alike."""
    objects, fovs = _one_condition_tables(prefix, n=12)
    fovs = fovs.assign(stage_x=[100.0 * (i % 4) for i in range(12)])
    fovs = fovs.assign(stage_y=[100.0 * (i // 4) for i in range(12)])
    column = objects["position_id"].map(dict(zip(fovs["position_id"], fovs["stage_x"])))
    return objects.assign(x=objects["x"] + column / 20), fovs


def _redundancy(objects, fovs):
    return redundancy_distance(
        objects,
        fovs,
        feature_cols=["x"],
        stage_cols=("stage_x", "stage_y"),
        bin_edges=[0, 150, 300, 500],
        n_permutations=9,
    )


def test_a_redundancy_result_records_its_acquisition_identity():
    objects, fovs = _stage_tables()
    result = _redundancy(objects, fovs)
    assert isinstance(result, RedundancyResult)
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    assert result.settings["acquisition"] == acquisition_identity(data)
    assert result.settings["acquisition"] == _variability(objects, fovs).settings["acquisition"]
    assert list(result.settings["stage_cols"]) == ["stage_x", "stage_y"]
    assert list(result.table["unit"]) == ["µm"]
    # Owned, like every frozen result: the caller's table cannot change it.
    with pytest.raises(TypeError, match="read-only"):
        result.settings["acquisition"] = "other"


def test_redundancy_from_another_acquisition_is_refused():
    objects, fovs = _stage_tables()
    other_objects, other_fovs = _stage_tables("g")
    variability = _variability(objects, fovs)
    with pytest.raises(ValueError, match="different acquisitions"):
        microscopist_report(
            variability=variability, redundancy=_redundancy(other_objects, other_fovs)
        )
    report = microscopist_report(variability=variability, redundancy=_redundancy(objects, fovs))
    assert "redundancy_distance" in set(report.profile["source"])


def _tables(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows, fovs = [], []
    for condition in ("ctrl", "drug"):
        for repeat in ("R1", "R2", "R3"):
            for f in range(6):
                pid = f"{condition}{repeat}{f}"
                fovs.append(
                    {
                        "position_id": pid,
                        "condition": condition,
                        "biological_repeat": repeat,
                        "repeat_group": repeat,
                    }
                )
                for _ in range(8):
                    rows.append({**fovs[-1], "x": rng.normal(), "y": rng.normal()})
    return pd.DataFrame(rows), pd.DataFrame(fovs)


SETUP = Setup(
    readout="x", features=("x", "y"), control={"condition": "ctrl"}, observation_unit="cell"
)


@pytest.fixture(scope="module")
def run():
    objects, fovs = _tables()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return objects, fovs, report(objects, fovs, SETUP)


def test_the_identity_ignores_row_and_column_order():
    objects, fovs = _tables()
    base = source_identity(objects, fovs)
    shuffled = objects.sample(frac=1, random_state=1)[list(reversed(objects.columns))]
    assert source_identity(shuffled, fovs.iloc[::-1]) == base


def test_the_identity_sees_a_value_a_row_a_column_and_a_type():
    objects, fovs = _tables()
    base = source_identity(objects, fovs)
    changed = objects.copy()
    changed.loc[3, "x"] += 1e-9
    assert source_identity(changed, fovs) != base
    assert source_identity(objects.iloc[:-1], fovs) != base  # a row
    assert source_identity(objects, fovs.iloc[:-1]) != base  # a zero-object FOV removed
    assert source_identity(objects.assign(z=0), fovs) != base  # a column
    typed = pd.DataFrame({"position_id": [1, "1"], "v": [1.0, 2.0]})
    texts = pd.DataFrame({"position_id": ["1", "1"], "v": [1.0, 2.0]})
    assert source_identity(typed, typed) != source_identity(texts, texts)


def test_the_report_records_its_source(run):
    objects, fovs, result = run
    assert result.source == source_identity(objects, fovs)


def test_the_page_refuses_other_tables_and_names_why(run, tmp_path):
    objects, fovs, result = run
    other = objects.assign(x=objects["x"] + 100)
    with pytest.raises(ValueError, match="not the tables the report was run on"):
        html_report(result, other, fovs, tmp_path)
    with pytest.raises(ValueError, match="not the tables the report was run on"):
        html_report(result, objects, fovs.iloc[:-1], tmp_path)  # a FOV missing
    assert not (tmp_path / "report.html").exists()


def test_a_report_without_a_source_is_refused(run, tmp_path):
    from dataclasses import replace

    objects, fovs, result = run
    unsourced = replace(result, source=None)
    with pytest.raises(ValueError, match="this report records no source tables"):
        html_report(unsourced, objects, fovs, tmp_path)
    assert not (tmp_path / "report.html").exists()


def test_document_takes_one_report_and_one_plan_of_one_acquisition(run):
    objects, fovs, result = run
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = optimise(objects, fovs, SETUP, fov_counts=[2, 3], reservoir_fraction=0.35)
        other = optimise(
            _tables(1)[0], _tables(1)[1], SETUP, fov_counts=[2, 3], reservoir_fraction=0.35
        )
    assert document(result, plan) is not None
    with pytest.raises(ValueError, match="one acquisition"):
        document(result, other)
    with pytest.raises(ValueError, match="one report and one plan"):
        document(result, result)


# --- Every result producer records the identity, on every path ---------------------------

SYNTHETIC = Path(__file__).resolve().parents[1] / "data" / "synthetic"
FEATURES = ["area", "intensity", "eccentricity"]
CONTROL = {"condition": "control"}
CRITERION = OutcomeCriterion(RelativeMeanError("area"), 0.05, "lower_is_better")
HIERARCHY = SamplingHierarchy("image_file", "technical_repeat", "biological_repeat")


@functools.cache
def _synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    """2 conditions x 2 biological repeats x 2 wells x 6 fields, with stage coordinates."""
    objects = pd.read_csv(SYNTHETIC / "synthetic_cells.csv")
    fovs = pd.read_csv(SYNTHETIC / "synthetic_fovs.csv").assign(
        repeat_group=lambda t: t["biological_repeat"]
    )
    repeats = sorted(fovs["biological_repeat"].unique())[:2]

    def part(table: pd.DataFrame) -> pd.DataFrame:
        chosen = table["technical_repeat"].isin({"B02", "B03", "D02", "D03"})
        chosen &= table["biological_repeat"].isin(repeats)
        return table.loc[chosen].reset_index(drop=True)

    return part(objects), part(fovs)


def _control_only() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The control fields alone: nothing to compare, the not-assessed paths."""
    objects, fovs = _synthetic()
    return objects[objects["condition"] == "control"], fovs[fovs["condition"] == "control"]


@functools.cache
def _replay():
    objects, fovs = _synthetic()
    return replay_acquisitions(
        objects,
        fovs,
        feature_cols=["area"],
        strategies=[FixedFOVStrategy(4)],
        criteria=[CRITERION],
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        reservoir_fraction=0.34,
        orders=[AcquisitionOrder.permuted(0)],
        stage_cols=("stage_x", "stage_y"),
        n_splits=2,
    )


@functools.cache
def _reliability():
    objects, fovs = _synthetic()
    return sampling_reliability(
        objects,
        fovs,
        feature_cols=["area"],
        criteria=[CRITERION],
        fov_counts=[2, 3],
        reservoir_fraction=0.34,
        n_splits=2,
        draws=5,
    )


def _novelty_control(positive):
    def run(tables):
        return representation_control(
            *tables,
            feature_cols=FEATURES,
            novelty=default_novelty(FEATURES),
            control=RepresentationControl(positive=positive, reference=CONTROL),
            pair_cols=["biological_repeat"],
            fovs_seen=2,
            draws=2,
        )

    return run


# Each producer as ``(objects, fovs) -> result``; "empty" and "not assessed" are the paths
# that return before measuring anything (no other condition, no positive control, no pair of
# fields within the bins).
PRODUCERS = {
    "variability": (lambda t: fov_variability(*t, feature="area"), _synthetic),
    "controls": (_novelty_control({"condition": "drug"}), _synthetic),
    "controls, not assessed": (_novelty_control(None), _synthetic),
    "contrast": (
        lambda t: condition_contrast(
            *t, feature="area", control=CONTROL, n_permutations=50, n_draws=20
        ),
        _synthetic,
    ),
    "contrast, control only": (
        lambda t: condition_contrast(*t, feature="area", control=CONTROL, n_draws=5),
        _control_only,
    ),
    "ranking": (
        lambda t: feature_ranking(*t, features=FEATURES, control=CONTROL, n_permutations=50),
        _synthetic,
    ),
    "ranking, control only": (
        lambda t: feature_ranking(*t, features=FEATURES, control=CONTROL, n_permutations=50),
        _control_only,
    ),
    "discovery": (
        lambda t: feature_discovery(*t, features=FEATURES, control=CONTROL, n_permutations=50),
        _synthetic,
    ),
    "discovery, control only": (
        lambda t: feature_discovery(*t, features=FEATURES, control=CONTROL, n_permutations=50),
        _control_only,
    ),
    "sufficiency": (
        lambda t: sufficiency_report(
            *t,
            feature_cols=["area"],
            strategies=[FixedFOVStrategy(4)],
            objective=ObjectiveConfig(SamplingObjective.POPULATION),
            orders=[AcquisitionOrder.permuted(0)],
        ),
        _synthetic,
    ),
    "redundancy": (
        lambda t: redundancy_distance(
            *t,
            feature_cols=["area"],
            stage_cols=("stage_x", "stage_y"),
            bin_edges=[0, 600, 1200],
            n_permutations=20,
        ),
        _synthetic,
    ),
    "redundancy, no pair within the bins": (
        lambda t: redundancy_distance(
            *t, feature_cols=["area"], stage_cols=("stage_x", "stage_y"), bin_edges=[1e6, 2e6]
        ),
        _synthetic,
    ),
    "object_rules": (
        lambda t: object_rule_check(*t, feature="area", control=CONTROL, n_draws=50),
        _synthetic,
    ),
    "descriptor_spread": (
        lambda t: descriptor_spread_check(*t, feature="area", threshold=10, n_draws=5),
        _synthetic,
    ),
    "unusual": (
        lambda t: unusual_objects(*t, feature_cols=FEATURES, reference=CONTROL, seed_checks=0),
        _synthetic,
    ),
    "clustering": (
        lambda t: fov_clustering(*t, label_col="rare", states=[True], n_permutations=50),
        _synthetic,
    ),
    "overview": (lambda t: dataset_overview(*t, feature_cols=FEATURES), _synthetic),
    "inspection": (lambda t: inspect_fovs(*t, feature_cols=FEATURES), _synthetic),
    "effort": (
        lambda t: effort_projection(
            *t, feature="area", hierarchy=HIERARCHY, strata=["condition"], n_boot=20
        ),
        _synthetic,
    ),
    "reliability": (lambda t: _reliability(), _synthetic),
    # Derived: they inherit the identity of the replay or reliability they were computed on.
    "calibration": (lambda t: stopping_calibration(_replay()), _synthetic),
    "adequacy": (lambda t: adequacy_points(_replay()), _synthetic),
    "efficiency": (lambda t: order_efficiency(_replay()), _synthetic),
    "protocol": (
        lambda t: protocol_reliability(
            _reliability(), HIERARCHY, threshold=0.9, strata=["condition"], n_boot=20
        ),
        _synthetic,
    ),
}


@pytest.mark.parametrize("name", sorted(PRODUCERS))
def test_every_result_producer_records_the_acquisition_identity(name):
    """Every result the report takes records its acquisition, on every path, so the report
    never refuses a valid result (an early "not assessed" return included)."""
    produce, tables = PRODUCERS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = produce(tables())
    objects, fovs = tables()
    expected = acquisition_identity(
        validate_input_tables(objects, fovs=fovs, feature_cols=["area"])
    )
    assert result_identity(result) == expected
    # The case name before any comma is the report keyword; the producer must return exactly
    # the type the report takes for it, so each case vouches for one result type.
    assert type(result) is _RESULT_TYPES[_keyword(name)], type(result)


def _keyword(name: str) -> str:
    return name.split(",", 1)[0]


def test_every_report_result_type_has_a_producer_under_test():
    """Each report keyword has at least one case, and each case names a report keyword; with
    the exact type check above, a keyword whose case returns another type fails there."""
    assert {_keyword(name) for name in PRODUCERS} == set(_RESULT_TYPES)


def test_a_control_without_positive_is_not_assessed_and_assembles():
    """A declared reference without a positive control returns "not assessed" before any
    measurement; it records the identity, so the report accepts it (it refused it)."""
    result = _novelty_control(None)(_synthetic())
    assert result.status == "not assessed"
    report = microscopist_report(
        controls=result, variability=fov_variability(*_synthetic(), feature="area")
    )
    assert report.inputs["controls"] == "given"


def test_a_dataframe_is_refused_as_redundancy():
    table = _redundancy(*_stage_tables()).table
    with pytest.raises(TypeError, match="redundancy must be a RedundancyResult"):
        microscopist_report(redundancy=table)
