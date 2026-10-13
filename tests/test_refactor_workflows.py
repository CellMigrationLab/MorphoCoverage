"""Characterisation of the workflows' branches that neither the golden runs nor the other
tests execute, pinned on the code before the refactor of their long functions (2026-10-12):
the exact messages, in the order the checks fire, and the exact text."""

import warnings
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from morphocoverage._integrity_tables import integrity_by_table
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import OutcomeCriterion
from morphocoverage.workflows import Setup, benchmark, optimise, scan_design
from morphocoverage.workflows._document import assemble
from morphocoverage.workflows._enough_targets import fingerprint_precision
from morphocoverage.workflows._families import shared_family
from morphocoverage.workflows._shared import check_workflow_inputs
from morphocoverage.workflows.guided import Answers
from morphocoverage.workflows.live import LiveFolder, LiveTables, live_session, write_fov


def _tables(conditions=("siCTRL", "A"), fields=2, cells=3):
    rng = np.random.default_rng(0)
    rows, fovs = [], []
    for condition in conditions:
        for field in range(fields):
            image = f"{condition}_{field}.tif"
            key = {"condition": condition, "biological_repeat": "2026_01_01", "image_file": image}
            fovs.append(key)
            for label in range(cells):
                rows.append({**key, "label_id": label + 1, "area": float(rng.normal(100, 10))})
    return pd.DataFrame(rows), pd.DataFrame(fovs)


SETUP = Setup(readout="area", control={"condition": "siCTRL"})


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"interval": 0}, "interval must be positive"),
        ({"until_stop": "x"}, "until_stop must be False, True, 'all_groups' or 'latest_group'"),
        ({"stop_on": "x"}, "stop_on must be 'decision' or 'rule', not 'x'"),
    ],
)
def test_watch_refuses_its_arguments(tmp_path, arguments, message):
    with pytest.raises(ValueError) as raised:
        LiveFolder(tmp_path, live_session(SETUP)).watch(on_update=None, **arguments)
    assert str(raised.value) == message


def test_live_tables_refuse_tables_that_shrink_or_late_objects():
    objects, fovs = _tables()
    with pytest.raises(TypeError) as raised:
        LiveTables(live_session(SETUP)).update(objects.to_dict(), fovs)
    assert str(raised.value) == "objects and fovs must be pandas DataFrames"
    tables = LiveTables(live_session(SETUP))
    tables.update(objects.iloc[:6], fovs.iloc[:2])
    with pytest.raises(ValueError) as raised:
        tables.update(objects.iloc[:3], fovs.iloc[:2])
    assert str(raised.value) == "the tables only grow: rows already read were removed"
    with pytest.raises(ValueError) as raised:
        tables.update(pd.concat([objects.iloc[:6], objects.iloc[:1]]), fovs.iloc[:2])
    assert str(raised.value) == (
        "objects appended for field(s) already added [('siCTRL', '2026_01_01', "
        "'siCTRL_0.tif')]; append a field's objects before its FOV row"
    )


def test_live_tables_hold_objects_until_their_field():
    objects, fovs = _tables()
    tables = LiveTables(live_session(SETUP))
    assert tables.update(objects.iloc[:3], fovs.iloc[:0]) == []
    assert tables.pending_objects == 3
    updates = tables.update(objects.iloc[:6], fovs.iloc[:1])
    assert [u.fov_metadata["image_file"] for u in updates] == ["siCTRL_0.tif"]
    assert [(u.n_objects, u.pending_objects) for u in updates] == [(3, 3)]
    assert tables.pending_objects == 3 and len(tables.updates) == 1


@pytest.mark.parametrize(
    ("fovs", "setup", "kind", "message"),
    [
        (
            None,
            SETUP,
            TypeError,
            (
                "the workflows need the FOV table of every acquired field (fovs=None was given); "
                "morphocoverage.schema.fov_table_from_observations rebuilds one, without the "
                "fields that held no observation"
            ),
        ),
        (
            "fovs",
            SETUP,
            TypeError,
            "objects and fovs must be pandas DataFrames (every acquired FOV in fovs)",
        ),
        ("table", "setup", TypeError, "setup must be a morphocoverage.workflows.Setup"),
    ],
)
def test_workflow_inputs_are_refused_in_order(fovs, setup, kind, message):
    objects, table = _tables()
    with pytest.raises(kind) as raised:
        check_workflow_inputs(objects, table if fovs == "table" else fovs, setup)
    assert str(raised.value) == message


def test_workflow_inputs_refuse_controls_absent_from_the_tables():
    objects, fovs = _tables()
    setup = Setup(readout="area", control={"condition": {"siCTRL": ["A"], "Z": ["Q"]}})
    with pytest.raises(ValueError) as raised:
        check_workflow_inputs(objects, fovs, setup)
    assert str(raised.value) == (
        "the controls name conditions absent from the tables: 'Q' and 'Z' (values are "
        "matched exactly, case and spaces included)"
    )


def test_answers_name_several_controls_with_parents_and_master():
    answers = Answers(
        loaded_correctly=True,
        control={"condition": {"A": ["a", "b"], "B": {"t": "c"}, "C": []}},
        aim="population",
        features=["area", "dna."],
        readout="area",
        observation_unit="cell",
        control_parents={"A": "C", "B": ["C", "t"]},
        master_control="A",
    )
    assert answers.lines() == [
        "Control: A for a, b; B for c; C for -; A under C; B under C; master A.",
        "Aim: compare the population averages of the conditions (the readout's mean).",
        "Features (2): area, dna.",
        "Readout: area.",
        "One row is a cell.",
    ]


# Setup.__post_init__: each message, and the first check that fires when two would.
SETUP_ERRORS = [
    ({"readout": ""}, TypeError, "readout must be one column name (None: no explainable features)"),
    ({"features": "area"}, TypeError, "features must be a sequence, not a string"),
    ({"intensity_features": "x"}, TypeError, "intensity_features must be a sequence, not a string"),
    ({"group_cols": "x"}, TypeError, "group_cols must be a sequence of column names"),
    ({"fields_independent": 1}, TypeError, "fields_independent must be True or False"),
    ({"fingerprint": (1,)}, TypeError, "fingerprint must be a sequence of OutcomeCriterion"),
    ({"rare": 3}, TypeError, "rare must be a RareTarget"),
    ({"rare_label": 3}, TypeError, "rare_label must be one object column name"),
    (
        {"rare_label": "m"},
        ValueError,
        "rare_label labels a declared rare target: declare rare too",
    ),
    (
        {"tolerances": (1.5,)},
        ValueError,
        "tolerances must be fractions between 0 and 1, such as 0.05",
    ),
    ({"control": {}}, TypeError, "control must map FOV-table columns to values"),
    (
        {"normalize": "x", "control": {"condition": "c"}},
        ValueError,
        "normalize must be one of ['control_mean'] or None",
    ),
    (
        {"intensity_features": ("b",)},
        ValueError,
        "intensity_features must be declared columns (readout or features)",
    ),
    # Two problems: the earlier check fires.
    (
        {"features": "area", "group_cols": "x"},
        TypeError,
        "features must be a sequence, not a string",
    ),
    ({"group_cols": "x", "rare": 3}, TypeError, "group_cols must be a sequence of column names"),
    ({"rare": 3, "tolerances": (2,)}, TypeError, "rare must be a RareTarget"),
    (
        {"tolerances": (2,), "control": {}},
        ValueError,
        "tolerances must be fractions between 0 and 1, such as 0.05",
    ),
    (
        {"control": {}, "control_parents": {"A": "B"}},
        TypeError,
        "control must map FOV-table columns to values",
    ),
    (
        {"control_parents": {"A": "B"}, "normalize": "x"},
        ValueError,
        (
            "control_parents link several controls: declare them as "
            "control={'condition': {control: {treatment: condition, ...}, ...}}"
        ),
    ),
    (
        {"normalize": "x", "intensity_features": ("b",)},
        ValueError,
        "normalize must be one of ['control_mean'] or None",
    ),
]


@pytest.mark.parametrize(("changes", "kind", "message"), SETUP_ERRORS)
def test_setup_refuses_with_its_messages_in_order(changes, kind, message):
    with pytest.raises(kind) as raised:
        Setup(**{"readout": "a", **changes})
    assert str(raised.value) == message


CONTROL_ERRORS = [
    (
        {"control": {"condition": "A"}, "master_control": "B"},
        ValueError,
        "master_control names one of several declared controls",
    ),
    (
        {"control": {"condition": {"A": ["a"]}, "well": "x"}},
        ValueError,
        (
            "several controls are named by the condition column alone: "
            "control={'condition': {control: {treatment: condition, ...}, ...}}"
        ),
    ),
    (
        {"control": {"condition": {"A": ["a"]}}, "control_parents": [("A", "B")]},
        TypeError,
        "control_parents must map a control to its parent condition",
    ),
    (
        {"control": {"condition": {"A": ["a"]}}, "control_parents": {"A": ("B", "t", "u")}},
        TypeError,
        "the parent of 'A' is a condition, or (condition, treatment)",
    ),
    (
        {
            "control": {"condition": {"A": {"t": "x"}, "B": {"t": "y"}}},
            "control_parents": {"B": ("A", "t")},
        },
        ValueError,
        "treatment 't' names two conditions of 'A'",
    ),
    (
        {
            "control": {"condition": {"A": ["a"], "B": ["b"]}},
            "control_parents": {"A": "B", "B": "A"},
        },
        ValueError,
        "the controls form a cycle through 'A'",
    ),
    (
        {"control": {"condition": {"A": ["a"], "B": ["A"]}}, "control_parents": {"A": "Z"}},
        ValueError,
        (
            "'A' is listed under control 'B' and linked to the parent 'Z': a control is compared "
            "with one reference, so keep one of the two"
        ),
    ),
]


@pytest.mark.parametrize(("changes", "kind", "message"), CONTROL_ERRORS)
def test_several_controls_are_refused_with_their_messages(changes, kind, message):
    with pytest.raises(kind) as raised:
        Setup(**{"readout": "a", **changes})
    assert str(raised.value) == message


def test_a_contrast_table_cannot_give_another_own_control():
    table = pd.DataFrame(
        {
            "condition": ["a"],
            "reference": ["A"],
            "kind": ["own_control"],
            "key": [None],
            "label": ["a vs A"],
        }
    )
    with pytest.raises(ValueError) as raised:
        Setup(readout="a", control={"condition": {"B": ["a"]}}, contrasts=table)
    assert str(raised.value) == (
        "the contrast table compares 'a' with 'A' as its own control, but the declared "
        "controls give 'B': a condition's own control comes from the declaration"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Setup(readout="a", control={"condition": {"A": ["a"]}}, contrasts=table)


def test_answers_name_every_optional_answer():
    answers = Answers(
        loaded_correctly=True,
        control={"condition": {"A": ["a"]}},
        aim="population",
        features=["area", "dna."],
        readout="area",
        observation_unit="cell",
        positive={"condition": "YAP"},
        rare_label="m",
        fingerprint_tolerance=0.1,
        experiment={"a": "e1", "A": "e2"},
        scales={"area": "log"},
        dataset={
            "feature_tables": [
                {"name": "D", "path": None},
                {"name": "E", "path": "x.csv", "tag": "explainable"},
            ]
        },
    )
    assert answers.lines() == [
        "Control: A for a.",
        "Aim: compare the population averages of the conditions (the readout's mean).",
        "Features (2): area, dna.",
        "Readout: area.",
        "One row is a cell.",
        "Other feature tables: D (non-explainable, a DataFrame), E (explainable, x.csv).",
        "Positive control: YAP.",
        "Phenotype column: m (its share as the rare target, seen in 3 fields with 95% chance).",
        "Fingerprint tolerance: 10%.",
        "Experiments, each corrected on its own: e1 and e2.",
    ]


def _live_folder(folder):
    objects, fovs = _tables()
    for _, field in fovs.iterrows():
        mine = objects.loc[objects["image_file"] == field["image_file"], ["label_id", "area"]]
        write_fov(folder, mine, dict(field))
    return LiveFolder(folder, live_session(SETUP))


def test_watch_hands_each_update_over_and_stops_on_ctrl_c(tmp_path):
    seen = []
    updates = _live_folder(tmp_path / "a").watch(interval=0.01, timeout=0.05, on_update=seen.append)
    names = ["siCTRL_0.tif", "siCTRL_1.tif", "A_0.tif", "A_1.tif"]
    assert [u.fov_metadata["image_file"] for u in seen] == names
    assert [u.fov_metadata["image_file"] for u in updates] == names

    def interrupt(update):
        raise KeyboardInterrupt

    kept = _live_folder(tmp_path / "b").watch(interval=0.01, on_update=interrupt)
    assert [u.fov_metadata["image_file"] for u in kept] == ["siCTRL_0.tif"]


def test_setup_without_any_target_and_benchmark_without_families():
    with pytest.raises(ValueError) as raised:
        Setup(readout="a", tolerances=())
    assert str(raised.value) == "tolerances=() declares no mean target: declare the fingerprint"
    objects, fovs = _tables()
    with pytest.raises(TypeError) as raised:
        benchmark(objects, fovs, SETUP, reservoir_fraction=0.3, families=[])
    assert str(raised.value) == "families must be a non-empty sequence of Family"


def test_assemble_puts_the_integrity_checks_in_the_overview_without_one():
    objects, fovs = _tables(fields=4, cells=5)
    objects.loc[5:9, "area"] = objects.loc[0:4, "area"].to_numpy()  # a copied field
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        integrity = integrity_by_table(objects, fovs, tables={"features": ["area"]})
    document = assemble({"integrity": integrity}, SETUP)
    assert document.overview == integrity.summary(detail="full")
    assert document.overview.startswith(
        "❌ Data integrity — 1 finding in 2 fields (structure to inspect, not a judgment)\n"
    )
    assert (
        "⚠️ 6. Which fields should I look at? — not assessed; needs inspect_fovs; integrity "
        "names 2 fields"
    ) in document.summary()


def test_fingerprint_precision_without_an_error_to_lower_is_none():
    objects, fovs = _tables()
    criterion = OutcomeCriterion(RelativeMeanError("area"), 0.9, "higher_is_better")
    assert (
        fingerprint_precision(objects, fovs, Setup(readout="area", fingerprint=(criterion,)))
        is None
    )


@dataclass(frozen=True)
class _Contrast:
    repeats: pd.DataFrame
    cells: pd.DataFrame
    settings: dict


class _Discovery:
    tests = pd.DataFrame(
        {
            "feature": ["area"],
            "condition": ["A"],
            "p_paired": [0.01],
            "q_paired": [0.02],
            "paired_values": ["raw"],
            "p_cells": [0.03],
            "q_cells": [0.04],
            "d": [0.5],
            "normalised": [False],
            "balanced": [False],
        }
    )
    settings: ClassVar[dict] = {}


def test_shared_family_leaves_conditions_the_discovery_did_not_test():
    contrast = _Contrast(
        pd.DataFrame({"condition": ["A", "B"], "p_paired_t": [0.5, 0.6], "q_paired_t": [0.5, 0.6]}),
        pd.DataFrame(
            {
                "condition": ["A", "B"],
                "d_cells": [0.0, 0.0],
                "p_cells": [1.0, 1.0],
                "q_cells": [1.0, 1.0],
            }
        ),
        {"x": 1},
    )
    shared = shared_family(contrast, _Discovery(), "area")
    assert shared.repeats.to_csv(index=False) == (
        "condition,p_paired_t,q_paired_t\nA,0.01,0.02\nB,0.6,0.6\n"
    )
    assert shared.cells.to_csv(index=False) == (
        "condition,d_cells,p_cells,q_cells\nA,0.5,0.03,0.04\nB,0.0,1.0,1.0\n"
    )
    assert shared.settings == {"x": 1, "family_features": 1, "family_share": None}


def test_scan_counts_planes_and_conditions_without_repeats():
    with pytest.raises(TypeError) as raised:
        scan_design([1, 2])
    assert str(raised.value) == "objects must be a pandas DataFrame, not list"
    objects, _ = _tables()
    planes = objects.assign(z=objects["label_id"] % 2).drop(columns=["biological_repeat"])
    scan = scan_design(planes)
    assert scan.design["planes"] == {"z": (2, 2)}
    assert scan.design["repeats"] == []
    assert scan.layout.to_csv() == "condition,observations\nsiCTRL,6\nA,6\n"


def test_optimise_measures_the_redundancy_distance_with_stage_coordinates():
    rng = np.random.default_rng(0)
    rows, fovs = [], []
    for condition in ("siCTRL", "A"):
        for repeat in ("2026_01_01", "2026_01_02"):
            for field in range(8):
                image = f"{condition}_{repeat}_{field}.tif"
                key = {"condition": condition, "biological_repeat": repeat, "image_file": image}
                stage = {"stage_x": float(field % 4) * 100, "stage_y": float(field // 4) * 100}
                fovs.append({**key, **stage, "acquisition_order": field})
                for label in range(6):
                    area = float(rng.normal(100, 10))
                    rows.append({**key, "label_id": label + 1, "area": area})
    setup = Setup(
        readout="area", control={"condition": "siCTRL"}, stage_cols=("stage_x", "stage_y")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = optimise(
            pd.DataFrame(rows),
            pd.DataFrame(fovs),
            setup,
            fov_counts=[2, 4],
            reservoir_fraction=0.3,
            n_splits=2,
            draws=5,
            permuted_orders=2,
            dispersed_orders=1,
            redundancy_bins=[0, 150, 400],
        )
    assert plan.results["redundancy"].to_csv(index=False) == (
        "condition,biological_repeat,redundancy_distance,pairs_closer,smallest_lag,largest_lag,"
        "fov_mean_variance,null_share,beyond_chance,permutations,share,unit,reason\n"
        "siCTRL,2026_01_01,115.53300858899107,10,115.53300858899107,243.90689358613267,"
        "26.906492434623566,1.0,False,199,0.9,µm,\n"
        "siCTRL,2026_01_02,115.53300858899107,10,115.53300858899107,243.90689358613267,"
        "6.191046651809311,1.0,False,199,0.9,µm,\n"
        "A,2026_01_01,115.53300858899107,10,115.53300858899107,243.90689358613267,"
        "27.05491116345264,1.0,False,199,0.9,µm,\n"
        "A,2026_01_02,243.90689358613267,24,115.53300858899107,243.90689358613267,"
        "19.445996867611395,0.425,False,199,0.9,µm,\n"
    )
