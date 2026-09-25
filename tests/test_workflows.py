"""The workflows (report, optimise, benchmark, live): they run end to end and add nothing.

Constructed data only check that each mode runs, is deterministic and keeps
its contracts; the real-data runs are in ``docs/workflows.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, _fov_table, _objects_for

from morphocoverage import validate_input_tables
from morphocoverage.results import Decision
from morphocoverage.strategies import FixedFOVStrategy
from morphocoverage.workflows import (
    LiveFolder,
    LiveTables,
    PositiveControl,
    Setup,
    benchmark,
    default_families,
    default_novelty,
    document,
    live_session,
    optimise,
    report,
    write_fov,
)
from morphocoverage.workflows.live import MANIFEST, OBJECTS_FILE

READOUT = FEATURE_COLS[0]


def _acquisition(n_fovs: int = 16, n_objects: int = 12, seed: int = 7):
    rng = np.random.default_rng(seed)
    fov_tables, object_tables = [], []
    for condition, shift in (("control", 0.0), ("drug", 3.0)):
        for repeat in ("R1", "R2"):
            fovs = _fov_table(n_fovs, condition=condition, repeat=repeat, grid_width=4)
            objects = _objects_for(fovs, np.full(n_fovs, n_objects), rng)
            objects.loc[:, list(FEATURE_COLS)] += shift
            fov_tables.append(fovs)
            object_tables.append(objects)
    return pd.concat(object_tables, ignore_index=True), pd.concat(fov_tables, ignore_index=True)


def _setup(**changes) -> Setup:
    settings = {
        "readout": READOUT,
        "features": FEATURE_COLS,
        "control": {"condition": "control"},
        "acquisition_order_col": "acquisition_order",
        "stage_cols": ("stage_x", "stage_y"),
    }
    return Setup(**{**settings, **changes})


def test_setup_rejects_ambiguous_declarations():
    with pytest.raises(TypeError):
        Setup(readout=READOUT, features="feature_b")
    with pytest.raises(ValueError):
        Setup(readout=READOUT, tolerances=(5,))
    with pytest.raises(ValueError, match="declare both"):
        Setup(readout=READOUT, positive=PositiveControl({"condition": "drug"}, fovs_seen=4))
    assert Setup(readout=READOUT, features=FEATURE_COLS).columns == list(FEATURE_COLS)


def test_report_answers_the_report_questions_and_leaves_the_rest_out():
    objects, fovs = _acquisition()
    before = (objects.copy(), fovs.copy())
    setup = _setup(
        novelty=default_novelty(FEATURE_COLS),
        positive=PositiveControl({"condition": "drug"}, fovs_seen=4, pair_cols=("repeat",)),
    )
    result = report(objects, fovs, setup)
    text = result.summary()
    assert "1. Can I count objects" in text and "4. What is limiting" in text
    assert "5. How many fields" not in text  # a planning answer never enters the report
    pd.testing.assert_frame_equal(objects, before[0])
    pd.testing.assert_frame_equal(fovs, before[1])
    assert report(objects, fovs, setup).summary() == text  # deterministic


def test_report_without_optional_declarations_says_what_is_missing():
    objects, fovs = _acquisition()
    text = report(objects, fovs, Setup(readout=READOUT)).summary()
    assert "Not assessed" in text


def test_optimise_and_document_merge_report_and_plan():
    objects, fovs = _acquisition()
    setup = _setup()
    plan = optimise(
        objects,
        fovs,
        setup,
        fov_counts=[2, 4],
        reservoir_fraction=0.25,
        n_splits=2,
        draws=5,
        permuted_orders=2,
        dispersed_orders=1,
    )
    text = plan.summary()
    assert "5. How many fields" in text and "1. Can I count objects" not in text
    merged = document(report(objects, fovs, setup), plan).summary()
    assert "1. Can I count objects" in merged and "5. How many fields" in merged


def test_benchmark_runs_on_the_given_families():
    objects, fovs = _acquisition()
    families = default_families(READOUT, n_resamples=20)[:2]
    result = benchmark(
        objects,
        fovs,
        _setup(),
        reservoir_fraction=0.25,
        families=families,
        n_splits=2,
        permuted_orders=2,
        n_boot=50,
    )
    assert set(result.frontiers) and result.summary()
    assert result.families == ("JCB per object", "JCB per field")


def _fields(objects, fovs):
    for _, row in fovs.sort_values(["condition", "repeat", "acquisition_order"]).iterrows():
        mine = objects[
            objects["condition"].eq(row["condition"])
            & objects["repeat"].eq(row["repeat"])
            & objects["image_file"].eq(row["image_file"])
        ]
        yield mine, row[["condition", "repeat", "image_file", "acquisition_order"]].to_dict()


def _decisions(updates):
    return [[(r.decision, r.n_fovs, r.n_observations) for r in u.results] for u in updates]


def test_live_folder_matches_live_tables_and_add_fov(tmp_path):
    objects, fovs = _acquisition(n_fovs=6)
    setup = _setup()
    folder = LiveFolder(tmp_path, live_session(setup))
    tables = LiveTables(live_session(setup))
    direct = live_session(setup)
    seen_objects, seen_fovs = objects.iloc[:0], fovs.iloc[:0]
    by_folder, by_tables, by_session = [], [], []
    for mine, metadata in _fields(objects, fovs):
        write_fov(tmp_path, mine[setup.columns], metadata)
        by_folder += folder.poll()
        seen_objects = pd.concat([seen_objects, mine])
        seen_fovs = pd.concat([seen_fovs, pd.DataFrame([metadata])])
        by_tables += tables.update(seen_objects, seen_fovs)
        by_session.append(direct.add_fov(mine, metadata))
    assert len(by_folder) == len(by_tables) == len(fovs)
    expected = [[(r.decision, r.n_fovs, r.n_observations) for r in u] for u in by_session]
    assert _decisions(by_folder) == _decisions(by_tables) == expected
    assert folder.poll() == [] and tables.update(seen_objects, seen_fovs) == []


def test_live_folder_waits_for_an_unterminated_manifest_line(tmp_path):
    objects, fovs = _acquisition(n_fovs=3)
    fields = list(_fields(objects, fovs))
    live = LiveFolder(tmp_path, live_session(_setup()))
    write_fov(tmp_path, fields[0][0][list(FEATURE_COLS)], fields[0][1])
    with open(tmp_path / MANIFEST, "a") as handle:
        handle.write("control,R1,img_001")  # a line still being written
    assert len(live.poll()) == 1
    with open(tmp_path / MANIFEST, "a") as handle:
        handle.write(".tif,2,\n")
    update = live.poll()
    assert len(update) == 1 and update[0].n_objects == 0  # no object file: zero objects


def test_live_folder_counts_a_field_without_objects_and_keeps_text_ids(tmp_path):
    session = live_session(Setup(readout=READOUT, fov_col="position_id", group_cols=()))
    live = LiveFolder(tmp_path, session)
    empty = pd.DataFrame(columns=[READOUT])
    write_fov(tmp_path, empty, {"position_id": "001"})
    write_fov(tmp_path, pd.DataFrame({READOUT: [1.0, 2.0]}), {"position_id": "002"})
    updates = live.poll()
    assert [u.n_objects for u in updates] == [0, 2]
    assert updates[-1].results[0].n_fovs == 2
    assert updates[0].fov_metadata == {"position_id": "001"}


def test_write_fov_refuses_changed_manifest_columns_and_reserved_names(tmp_path):
    write_fov(tmp_path, pd.DataFrame({READOUT: [1.0]}), {"position_id": "a"})
    with pytest.raises(ValueError, match="manifest columns"):
        write_fov(tmp_path, pd.DataFrame({READOUT: [1.0]}), {"position_id": "b", "well": "B2"})
    with pytest.raises(ValueError, match=OBJECTS_FILE):
        write_fov(tmp_path, pd.DataFrame({READOUT: [1.0]}), {OBJECTS_FILE: "x"})
    assert not list((tmp_path / "objects").glob(".*.tmp"))


def test_live_tables_buffers_objects_until_their_fov_row():
    objects, fovs = _acquisition(n_fovs=3)
    fields = list(_fields(objects, fovs))
    live = LiveTables(live_session(_setup()))
    first, second = fields[0][0], fields[1][0]
    both = pd.concat([first, second])
    assert live.update(both, pd.DataFrame([fields[0][1]])) != []
    assert live.pending_objects == len(second)  # waiting for its FOV row
    updates = live.update(both, pd.DataFrame([fields[0][1], fields[1][1]]))
    assert [u.n_objects for u in updates] == [len(second)] and live.pending_objects == 0


def test_live_tables_rejects_late_objects_shrinking_tables_and_repeated_fields():
    objects, fovs = _acquisition(n_fovs=3)
    (first, meta), (second, _) = list(_fields(objects, fovs))[:2]
    live = LiveTables(live_session(_setup()))
    live.update(first.iloc[:5], pd.DataFrame([meta]))
    with pytest.raises(ValueError, match="already added"):
        live.update(first, pd.DataFrame([meta]))
    live = LiveTables(live_session(_setup()))
    live.update(first, pd.DataFrame([meta]))
    with pytest.raises(ValueError, match="only grow"):
        live.update(first.iloc[:0], pd.DataFrame([meta]))
    with pytest.raises(ValueError, match="twice"):
        live.update(pd.concat([first, second]), pd.DataFrame([meta, meta]))


def test_live_until_stop_ends_the_watch(tmp_path):
    rule = FixedFOVStrategy(2)
    live = LiveFolder(tmp_path, live_session(_setup(), rules=(rule,)))
    objects, fovs = _acquisition(n_fovs=4)
    for mine, metadata in list(_fields(objects, fovs))[:4]:
        write_fov(tmp_path, mine[list(FEATURE_COLS)], metadata)
    seen = live.watch(interval=0.01, timeout=1, until_stop=True, on_update=None)
    assert seen[-1].results[0].decision == Decision.STOP and len(seen) == 2


def test_summary_questions_select_and_reject():
    objects, fovs = _acquisition()
    setup = _setup()
    validate_input_tables(objects, fovs=fovs, feature_cols=setup.columns)
    result = report(objects, fovs, setup)
    only = result.document.summary(["unit"])
    assert "1. Can I count objects" in only and "4. What is limiting" not in only
    with pytest.raises((KeyError, ValueError)):
        result.document.summary(["not_a_question"])
