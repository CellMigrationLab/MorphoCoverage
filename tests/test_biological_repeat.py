"""The biological repeat's column is ``biological_repeat``; its former name ``repeat`` still reads.

The MorphoBricks project owner renamed the shared column ``repeat`` to
``biological_repeat`` (``technical_repeat`` is unchanged). These tests check:

- the new name works end to end (``check_tables``, ``report``, ``optimise``,
  the live session, ``LiveTables`` and ``LiveFolder``);
- a table, a column argument or a control mapping that uses ``repeat`` is read
  as ``biological_repeat`` with exactly one ``DeprecationWarning`` per call,
  never modifies the caller's tables, and gives the results of the new name
  (the seeds included: the biological repeat seeds under its former name);
- a table holding both columns is used when they are identical (with the
  warning) and refused when they differ;
- every near miss of either repeat is refused with "did you mean ...?", so a
  pipeline using ``bio_repeat`` or ``tec_repeat`` never pools its repeats;
- ``single``, CellPatcher's value for a level with one member, is one member.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from morphocoverage import (
    FixedFOVStrategy,
    ObjectiveConfig,
    OnlineSession,
    SamplingObjective,
    ScalarPrecisionStrategy,
    check_tables,
    evaluate_acquisition_prefixes,
    integrity_checks,
    plan_objects_to_average,
    planes_as_features,
    run_strategies,
    validate_input_tables,
)
from morphocoverage.benchmark import (
    AcquisitionOrder,
    BenchmarkDataset,
    BenchmarkRegistry,
    run_benchmark,
)
from morphocoverage.calibration import calibrate_next_fov_novelty
from morphocoverage.metadata import (
    MORPHOBRICKS_GROUP_COLUMNS,
    MORPHOBRICKS_SHARED_COLUMNS,
    REPEAT_COLUMN,
    canonical_spelling,
)
from morphocoverage.normalize import normalize_to_control
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.plots import plots_of_data
from morphocoverage.retrospective import (
    RepresentationControl,
    SamplingHierarchy,
    dataset_overview,
    effort_projection,
    inspect_fovs,
    redundancy_distance,
    replay_acquisitions,
    representation_control,
    sampling_reliability,
)
from morphocoverage.retrospective.criteria import OutcomeCriterion
from morphocoverage.sampling_unit import (
    condition_contrast,
    descriptor_spread_check,
    feature_ranking,
    fov_clustering,
    fov_variability,
    object_rule_check,
    unusual_objects,
)
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.workflows import (
    LiveFolder,
    LiveTables,
    PositiveControl,
    Setup,
    default_novelty,
    live_session,
    optimise,
    report,
    write_fov,
)

SYNTHETIC = Path(__file__).resolve().parents[1] / "data" / "synthetic"
FEATURES = ("area", "intensity", "eccentricity")
FORMER = "repeat"
POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small part of the synthetic acquisition: 2 conditions x 2 repeats x 2 wells x 6 fields."""
    objects = pd.read_csv(SYNTHETIC / "synthetic_cells.csv")
    fovs = pd.read_csv(SYNTHETIC / "synthetic_fovs.csv")
    keep = {"B02", "B03", "D02", "D03"}
    repeats = sorted(fovs[REPEAT_COLUMN].unique())[:2]

    def part(table: pd.DataFrame) -> pd.DataFrame:
        chosen = table["technical_repeat"].isin(keep) & table[REPEAT_COLUMN].isin(repeats)
        return table.loc[chosen].reset_index(drop=True)

    return part(objects), part(fovs)


def _former(table: pd.DataFrame) -> pd.DataFrame:
    """``table`` with the biological repeat under its former name."""
    return table.rename(columns={REPEAT_COLUMN: FORMER})


def _setup(**changes) -> Setup:
    settings = {
        "readout": "area",
        "features": FEATURES,
        "control": {"condition": "control"},
        "observation_unit": "cell",
        "acquisition_order_col": "acquisition_order",
        "stage_cols": ("stage_x", "stage_y"),
        "rules": (FixedFOVStrategy(4), ScalarPrecisionStrategy("area", 0.05, "fov", "upper_95")),
    }
    return Setup(**{**settings, **changes})


def _deprecations(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]


def _call(function, *args, **kwargs):
    """``function(*args, **kwargs)`` and the DeprecationWarnings it gave."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(*args, **kwargs)
    return result, _deprecations(caught)


def _quiet(function, *args, **kwargs):
    """``function(*args, **kwargs)``, which must give no DeprecationWarning."""
    result, deprecations = _call(function, *args, **kwargs)
    assert deprecations == [], deprecations
    return result


def _once(function, *args, **kwargs):
    """``function(*args, **kwargs)``, which must give exactly one former-name warning."""
    result, deprecations = _call(function, *args, **kwargs)
    assert len(deprecations) == 1, deprecations
    message = deprecations[0]
    assert "'repeat' is the former name of 'biological_repeat'" in message
    assert "rename the column" in message and "will be removed in a later version" in message
    return result


def _comparable(value):
    """What two runs must agree on: tables, summaries and results without timings."""
    if isinstance(value, pd.DataFrame | pd.Series):
        return value.to_json(orient="split", default_handler=str)
    if isinstance(value, list | tuple):
        return [_comparable(v) for v in value]
    if isinstance(value, dict):
        return {k: _comparable(v) for k, v in value.items()}
    if hasattr(value, "without_timing"):
        return value.without_timing()
    if hasattr(value, "step") and hasattr(value, "result"):  # a SequentialResult
        return (value.step, value.acquired_fovs, value.result.without_timing())
    if hasattr(value, "objects") and hasattr(value, "fov_key"):  # a ValidatedInput
        return (
            _comparable(value.objects),
            _comparable(value.fovs),
            value.fov_key,
            value.group_cols,
        )
    if hasattr(value, "summary"):
        return value.summary(detail="full")
    if dataclasses.is_dataclass(value):
        fields = [f.name for f in dataclasses.fields(value) if "seconds" not in f.name]
        return {name: _comparable(getattr(value, name)) for name in fields}
    return value


# --- The new name, end to end ------------------------------------------------


def test_the_shared_columns_name_the_biological_repeat():
    assert REPEAT_COLUMN == "biological_repeat"
    assert MORPHOBRICKS_GROUP_COLUMNS == ("condition", "biological_repeat", "technical_repeat")
    assert "biological_repeat" in MORPHOBRICKS_SHARED_COLUMNS
    assert FORMER not in MORPHOBRICKS_SHARED_COLUMNS


def test_the_new_name_works_end_to_end(tmp_path):
    objects, fovs = _tables()
    check = _quiet(
        check_tables, objects, fovs, feature_cols=list(FEATURES), observation_unit="cell"
    )
    assert check.ok
    assert "Analysis groups: condition, biological_repeat, technical_repeat." in check.understood
    setup = _setup()
    reported = _quiet(report, objects, fovs, setup)
    contrast = reported.results["contrast"].table
    assert REPEAT_COLUMN in contrast.columns and FORMER not in contrast.columns
    plan = _quiet(
        optimise,
        objects,
        fovs,
        setup,
        fov_counts=[2, 4],
        reservoir_fraction=0.34,
        n_splits=2,
        draws=5,
        permuted_orders=2,
        dispersed_orders=1,
    )
    assert REPEAT_COLUMN in plan.results["reliability"].curve.columns
    session = _quiet(live_session, setup)
    updates = _quiet(LiveTables(session).update, objects, fovs)
    assert len(updates) == len(fovs)
    assert session.group_cols == ("condition", "biological_repeat", "technical_repeat")
    assert all(REPEAT_COLUMN in r.group for u in updates for r in u.results)
    folder = LiveFolder(tmp_path, live_session(setup))
    key = ["condition", REPEAT_COLUMN, "technical_repeat", "image_file"]
    for _, field in fovs.head(3).iterrows():
        mine = objects.merge(field[key].to_frame().T, on=key)
        write_fov(tmp_path, mine, field.to_dict())
    assert len(_quiet(folder.poll)) == 3


# --- The former name: one warning, the caller's tables untouched, the same results ------------


def _analyses():
    """Public analyses taking the linked tables, each as ``(objects, fovs) -> result``."""
    common = {"observation_unit": "cell"}
    control = {"condition": "control"}
    criterion = OutcomeCriterion(RelativeMeanError("area"), 0.05, "lower_is_better")
    rules = [FixedFOVStrategy(4)]
    return {
        "validate_input_tables": lambda o, f: validate_input_tables(
            o, fovs=f, feature_cols=list(FEATURES), **common
        ),
        "check_tables": lambda o, f: check_tables(
            o, f, feature_cols=list(FEATURES), control=control, **common
        ),
        "run_strategies": lambda o, f: run_strategies(
            o, fovs=f, strategies=rules, feature_cols=["area"], objective=POPULATION, **common
        ),
        "run_benchmark": lambda o, f: run_benchmark(
            BenchmarkRegistry(
                [
                    BenchmarkDataset(
                        name="part",
                        role="development",
                        load=lambda: (o, f),
                        feature_cols=("area",),
                        acquisition_order_col="acquisition_order",
                        observation_unit="cell",
                    )
                ]
            ),
            "part",
            strategies=rules,
            metrics=[RelativeMeanError("area")],
            objective=POPULATION,
        ),
        "replay_acquisitions": lambda o, f: replay_acquisitions(
            o,
            f,
            feature_cols=["area"],
            strategies=rules,
            criteria=[criterion],
            objective=POPULATION,
            reservoir_fraction=0.34,
            orders=[AcquisitionOrder.permuted(0)],
            n_splits=2,
            **common,
        ),
        "calibrate_next_fov_novelty": lambda o, f: calibrate_next_fov_novelty(
            o,
            f,
            feature_cols=list(FEATURES),
            novelty=default_novelty(FEATURES),
            fov_counts=[5],
            draws=3,
            random_state=0,
            **common,
        ),
        "redundancy_distance": lambda o, f: redundancy_distance(
            o,
            f,
            feature_cols=["area"],
            stage_cols=("stage_x", "stage_y"),
            bin_edges=[0, 600, 1200],
            n_permutations=20,
            **common,
        ),
        "descriptor_spread_check": lambda o, f: descriptor_spread_check(
            o, f, feature="area", threshold=10, n_draws=5, **common
        ),
        "evaluate_acquisition_prefixes": lambda o, f: evaluate_acquisition_prefixes(
            o,
            f,
            strategies=rules,
            feature_cols=["area"],
            acquisition_order_col="acquisition_order",
            objective=POPULATION,
            **common,
        ),
        "integrity_checks": lambda o, f: integrity_checks(
            o, f, feature_cols=list(FEATURES), **common
        ),
        "fov_variability": lambda o, f: fov_variability(o, f, feature="area", **common),
        "object_rule_check": lambda o, f: object_rule_check(
            o, f, feature="area", control=control, n_draws=50, **common
        ),
        "condition_contrast": lambda o, f: condition_contrast(
            o, f, feature="area", control=control, n_permutations=50, n_draws=20, **common
        ),
        "feature_ranking": lambda o, f: feature_ranking(
            o, f, features=list(FEATURES), control=control, n_permutations=50, **common
        ),
        "unusual_objects": lambda o, f: unusual_objects(
            o, f, feature_cols=list(FEATURES), reference=control, seed_checks=0, **common
        ),
        "fov_clustering": lambda o, f: fov_clustering(
            o, f, label_col="rare", states=[True], n_permutations=50, **common
        ),
        "normalize_to_control": lambda o, f: normalize_to_control(
            o, f, feature_cols=["area"], control=control, **common
        ),
        "dataset_overview": lambda o, f: dataset_overview(
            o, f, feature_cols=list(FEATURES), control=control, **common
        ),
        "inspect_fovs": lambda o, f: inspect_fovs(o, f, feature_cols=list(FEATURES), **common),
        "sampling_reliability": lambda o, f: sampling_reliability(
            o,
            f,
            feature_cols=["area"],
            criteria=[criterion],
            fov_counts=[2, 3],
            reservoir_fraction=0.34,
            n_splits=2,
            draws=5,
            **common,
        ),
        "effort_projection": lambda o, f: effort_projection(
            o,
            f,
            feature="area",
            hierarchy=SamplingHierarchy("image_file", "technical_repeat", "biological_repeat"),
            strata=["condition"],
            n_boot=20,
            **common,
        ),
        "sufficiency_report": lambda o, f: sufficiency_report(
            o,
            f,
            feature_cols=["area"],
            strategies=rules,
            objective=POPULATION,
            orders=[AcquisitionOrder.permuted(0)],
            **common,
        ),
        "representation_control": lambda o, f: representation_control(
            o,
            f,
            feature_cols=list(FEATURES),
            novelty=default_novelty(FEATURES),
            control=RepresentationControl(positive={"condition": "drug"}, reference=control),
            pair_cols=[REPEAT_COLUMN],
            fovs_seen=2,
            draws=2,
            **common,
        ),
        "plots_of_data": lambda o, f: [
            ax.get_title(loc="left")
            for figure in plots_of_data(o, f, features=["area"], n_boot=20)["area"]
            for ax in figure.axes
        ],
        "plan_objects_to_average": lambda o, f: plan_objects_to_average(
            o,
            fovs=f,
            feature="area",
            target_cv=0.05,
            sd_estimate="point",
            sample_cols=["condition", "biological_repeat", "technical_repeat"],
            control=control,
            **common,
        ),
    }


@pytest.mark.parametrize("name", sorted(_analyses()))
def test_the_former_name_gives_one_warning_and_the_results_of_the_new_name(name):
    analysis = _analyses()[name]
    objects, fovs = _tables()
    expected = _quiet(analysis, objects, fovs)
    old_objects, old_fovs = _former(objects), _former(fovs)
    copies = old_objects.copy(), old_fovs.copy()
    result = _once(analysis, old_objects, old_fovs)
    # The caller's tables are not modified.
    pd.testing.assert_frame_equal(old_objects, copies[0])
    pd.testing.assert_frame_equal(old_fovs, copies[1])
    assert _comparable(result) == _comparable(expected)


def test_a_table_the_analysis_returns_names_the_new_column():
    objects, fovs = _tables()
    normalized = _once(
        normalize_to_control,
        _former(objects),
        _former(fovs),
        feature_cols=["area"],
        control={"condition": "control"},
    )
    assert REPEAT_COLUMN in normalized.columns and FORMER not in normalized.columns
    stack = pd.concat([objects.assign(z=z) for z in (0, 1)], ignore_index=True)
    per_object, names = _once(planes_as_features, _former(stack), _former(fovs), features=["area"])
    assert names == ["area_z0", "area_z1"]
    assert REPEAT_COLUMN in per_object.columns and FORMER not in per_object.columns


def _report_text(objects, fovs, setup):
    return report(objects, fovs, setup).summary(detail="full")


def test_the_workflows_read_the_former_name_with_one_warning():
    objects, fovs = _tables()
    setup = _setup()
    expected = _quiet(_report_text, objects, fovs, setup)
    assert _once(_report_text, _former(objects), _former(fovs), setup) == expected
    settings = {
        "fov_counts": [2, 3],
        "reservoir_fraction": 0.34,
        "n_splits": 2,
        "draws": 5,
        "permuted_orders": 2,
        "dispersed_orders": 1,
    }
    plan = _quiet(optimise, objects, fovs, setup, **settings).summary(detail="full")
    old = _once(optimise, _former(objects), _former(fovs), setup, **settings)
    assert old.summary(detail="full") == plan


def _live_results(objects, fovs, setup):
    """Every result of a live session fed field by field, and the resolved groups."""
    session = live_session(setup)
    repeat = REPEAT_COLUMN if REPEAT_COLUMN in fovs.columns else FORMER
    key = ["condition", repeat, "technical_repeat", "image_file"]
    results = []
    for field in fovs.to_dict("records"):
        mine = objects.merge(pd.DataFrame([{k: field[k] for k in key}]), on=key)
        results.append(session.add_fov(mine, field))
    return results, session.group_cols


def test_the_live_session_reads_the_former_name_one_warning_per_field():
    objects, fovs = _tables()
    objects, fovs = objects.head(60), fovs.head(6)
    setup = _setup()
    expected, groups = _quiet(_live_results, objects, fovs, setup)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old, old_groups = _live_results(_former(objects), _former(fovs), setup)
    # One call to add_fov per field, one warning per call.
    assert len(_deprecations(caught)) == len(fovs)
    assert old_groups == groups == ("condition", "biological_repeat", "technical_repeat")
    assert _comparable(old) == _comparable(expected)


def test_live_tables_and_live_folders_read_the_former_name_once_per_call(tmp_path):
    objects, fovs = _tables()
    setup = _setup()
    expected = _quiet(LiveTables(live_session(setup)).update, objects, fovs)
    old = _once(LiveTables(live_session(setup)).update, _former(objects), _former(fovs))
    assert [_comparable(u.results) for u in old] == [_comparable(u.results) for u in expected]
    assert all(REPEAT_COLUMN in u.fov_metadata for u in old)
    key = ["condition", REPEAT_COLUMN, "technical_repeat", "image_file"]
    for _, field in fovs.head(4).iterrows():
        mine = objects.merge(field[key].to_frame().T, on=key)
        write_fov(tmp_path, _former(mine), _former(field.to_frame().T).iloc[0].to_dict())
    folder = LiveFolder(tmp_path, live_session(setup))
    updates = _once(folder.poll)
    assert len(updates) == 4
    assert all(REPEAT_COLUMN in u.fov_metadata and FORMER not in u.fov_metadata for u in updates)
    groups = [tuple(u.results[0].group) for u in updates]
    assert set(groups) == {("condition", "biological_repeat", "technical_repeat")}


def test_column_arguments_and_setup_fields_naming_the_former_name_are_translated():
    objects, fovs = _tables()
    groups = ["condition", FORMER, "technical_repeat"]
    data = _once(
        validate_input_tables, objects, fovs=fovs, feature_cols=["area"], group_cols=groups
    )
    assert data.group_cols == ("condition", "biological_repeat", "technical_repeat")
    check = _once(
        check_tables,
        objects,
        fovs,
        feature_cols=["area"],
        group_cols=groups,
        control={"condition": "control", FORMER: fovs[REPEAT_COLUMN].iloc[0]},
    )
    assert check.ok, check.summary()
    setup = _once(
        Setup,
        readout="area",
        control={"condition": "control"},
        group_cols=groups,
        effort_strata=("condition", FORMER),
    )
    assert setup.group_cols == ("condition", "biological_repeat", "technical_repeat")
    assert setup.effort_strata == ("condition", "biological_repeat")
    positive = _once(PositiveControl, {"condition": "drug"}, fovs_seen=2, pair_cols=(FORMER,))
    assert positive.pair_cols == ("biological_repeat",)
    hierarchy = _once(SamplingHierarchy, "image_file", "technical_repeat", FORMER)
    assert hierarchy.biological_repeat == "biological_repeat"
    session = _once(
        OnlineSession,
        strategies=[FixedFOVStrategy(2)],
        feature_cols=["area"],
        objective=POPULATION,
        group_cols=groups,
    )
    first = fovs.iloc[0].to_dict()
    mine = objects[objects["image_file"].eq(first["image_file"])]
    mine = mine[mine["technical_repeat"].eq(first["technical_repeat"])]
    mine = mine[mine[REPEAT_COLUMN].eq(first[REPEAT_COLUMN])]
    results = _quiet(session.add_fov, mine, first)
    assert set(results[0].group) == {"condition", "biological_repeat", "technical_repeat"}


# --- Both columns ---------------------------------------------------------------------


def test_both_columns_identical_warn_and_use_the_new_one():
    objects, fovs = _tables()
    both = objects.assign(**{FORMER: objects[REPEAT_COLUMN]})
    both_fovs = fovs.assign(**{FORMER: fovs[REPEAT_COLUMN]})
    expected = _quiet(validate_input_tables, objects, fovs=fovs, feature_cols=["area"])
    data = _once(validate_input_tables, both, fovs=both_fovs, feature_cols=["area"])
    assert FORMER not in data.objects.columns and FORMER not in data.fovs.columns
    assert _comparable(data) == _comparable(expected)
    check = _once(check_tables, both, both_fovs, feature_cols=["area"])
    assert check.ok


def test_both_columns_that_differ_are_refused():
    objects, fovs = _tables()
    other = objects.assign(**{FORMER: "plate 1"})
    with pytest.raises(ValueError, match="both 'repeat' .* and 'biological_repeat'"):
        validate_input_tables(other, fovs=fovs, feature_cols=["area"])
    check = check_tables(other, fovs, feature_cols=["area"])
    assert not check.ok
    assert any("different values" in p.title for p in check.errors)
    session = OnlineSession(
        strategies=[FixedFOVStrategy(2)], feature_cols=["area"], objective=POPULATION
    )
    first = fovs.iloc[0].to_dict()
    with pytest.raises(ValueError, match="with different values"):
        session.add_fov(objects.head(0), {**first, FORMER: "another"})
    assert session.fov_col is None  # the refused field left the session as it was
    with pytest.raises(ValueError, match="names both 'repeat'"):
        check_tables(
            objects,
            fovs,
            feature_cols=["area"],
            control={REPEAT_COLUMN: "a", FORMER: "b"},
        ).raise_if_errors()


# --- Near misses ------------------------------------------------------------------------

NEAR_MISSES = {
    "bio_repeat": "biological_repeat",
    "biological repeat": "biological_repeat",
    "biologicalrepeat": "biological_repeat",
    "Biological_Repeat": "biological_repeat",
    "Metadata_biological_repeat": "biological_repeat",
    "bio_rep": "biological_repeat",
    "biological_replicate": "biological_repeat",
    "Repeat": "biological_repeat",
    "repeats": "biological_repeat",
    "Metadata_repeat": "biological_repeat",
    "tec_repeat": "technical_repeat",
    "technical repeat": "technical_repeat",
    "tech_repeat": "technical_repeat",
    "tech_rep": "technical_repeat",
    "technical_replicate": "technical_repeat",
    "TechnicalRepeat": "technical_repeat",
}


@pytest.mark.parametrize("column", sorted(NEAR_MISSES))
def test_each_near_miss_is_refused_with_the_column_meant(column):
    meant = NEAR_MISSES[column]
    assert canonical_spelling(column) == meant
    objects, fovs = _tables()
    renamed = {meant: column}
    objects, fovs = objects.rename(columns=renamed), fovs.rename(columns=renamed)
    hint = f"did you mean '{meant}'"
    with pytest.raises(ValueError, match=f"(?i){hint}"):
        validate_input_tables(objects, fovs=fovs, feature_cols=["area"])
    check = check_tables(objects, fovs, feature_cols=["area"])
    assert not check.ok
    assert any(hint in p.fix.lower() for p in check.errors), check.summary()
    session = OnlineSession(
        strategies=[FixedFOVStrategy(2)], feature_cols=["area"], objective=POPULATION
    )
    with pytest.raises(ValueError, match=f"(?i){hint}"):
        session.add_fov(objects.head(0), fovs.iloc[0].to_dict())


def test_the_exact_names_are_not_near_misses():
    for column in (*MORPHOBRICKS_SHARED_COLUMNS, "replicate", "plate", "well", "rep"):
        assert canonical_spelling(column) is None, column


# --- A level with one member: ``single`` ----------------------------------------------


def test_single_is_one_member_of_its_level():
    objects, fovs = _tables()
    first = fovs[REPEAT_COLUMN].iloc[0]
    one, one_fovs = objects[objects[REPEAT_COLUMN] == first], fovs[fovs[REPEAT_COLUMN] == first]
    single = {REPEAT_COLUMN: "single"}
    labelled = one.assign(**single), one_fovs.assign(**single)
    data = validate_input_tables(*labelled[:1], fovs=labelled[1], feature_cols=["area"])
    assert data.group_cols == ("condition", "biological_repeat", "technical_repeat")
    assert set(data.fovs[REPEAT_COLUMN]) == {"single"}
    check = check_tables(*labelled, feature_cols=["area"], observation_unit="cell")
    assert check.ok and not [p for p in check.warnings if "single" in p.title]
    overview = dataset_overview(*labelled, feature_cols=["area"], observation_unit="cell")
    assert overview.counts["repeats"] == 1
    # The label is not data: the answers are those of the plate's own label, but for the label.
    rules = [FixedFOVStrategy(4)]
    settings = {"strategies": rules, "feature_cols": ["area"], "objective": POPULATION}
    as_single = run_strategies(labelled[0], fovs=labelled[1], **settings)
    as_given = run_strategies(one, fovs=one_fovs, **settings)
    assert [r.decision for r in as_single] == [r.decision for r in as_given]
    assert {r.group[REPEAT_COLUMN] for r in as_single} == {"single"}
    # A technical repeat of one member per biological repeat, too: one well per condition.
    well = {"technical_repeat": "single"}
    in_one = one["technical_repeat"].isin(["B02", "D02"])
    in_one_fovs = one_fovs["technical_repeat"].isin(["B02", "D02"])
    wells = one.loc[in_one].assign(**well), one_fovs.loc[in_one_fovs].assign(**well)
    data = validate_input_tables(wells[0], fovs=wells[1], feature_cols=["area"])
    assert "technical_repeat" in data.group_cols
    assert set(data.fovs["technical_repeat"]) == {"single"}
    overview = dataset_overview(*wells, feature_cols=["area"], observation_unit="cell")
    assert overview.counts["technical_repeats"] == 2  # one per condition


def test_an_empty_repeat_is_not_recorded_and_refused():
    objects, fovs = _tables()
    with pytest.raises(ValueError, match="missing group values"):
        validate_input_tables(
            objects.assign(**{REPEAT_COLUMN: np.nan}),
            fovs=fovs.assign(**{REPEAT_COLUMN: np.nan}),
            feature_cols=["area"],
        )


def test_the_biological_repeat_seeds_as_before_the_renaming():
    from morphocoverage._seeds import group_identity

    assert group_identity({"condition": "a", REPEAT_COLUMN: "1A"}) == group_identity(
        {"condition": "a", FORMER: "1A"}
    )


def test_the_warning_names_the_callers_line():
    objects, fovs = _tables()
    for call in (
        lambda: validate_input_tables(_former(objects), fovs=_former(fovs), feature_cols=["area"]),
        lambda: Setup(readout="area", group_cols=["condition", FORMER]),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        former = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert [w.filename for w in former] == [__file__]
