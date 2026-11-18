"""The workflows (report, optimise, benchmark, live): they run end to end and add nothing.

Constructed data only check that each mode runs, is deterministic and keeps
its contracts; the real-data run times are in ``docs/report/running_a_report.md``.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from _helpers_embedding import SEVERAL, _cells, _embedding_only, _one_per_field, _several
from _helpers_feature_tables import _fingerprint_data, _fingerprint_setup
from _helpers_workflows import (
    KEY,
    QUICK,
    READOUT,
    _acquisition,
    _fingerprint,
    _flat,
    _rare,
    _report,
    _scenario_report,
    _setup,
    _well_acquisition,
    _wells,
)
from scenarios import FEATURE_COLS

import morphocoverage.workflows._document as document_module
import morphocoverage.workflows.benchmarks as benchmarks_module
import morphocoverage.workflows.retrospective as workflow_module
from morphocoverage import RareTarget, validate_input_tables
from morphocoverage._style import Status
from morphocoverage.multivariate_outcomes import CovarianceError
from morphocoverage.results import Decision
from morphocoverage.retrospective import SamplingHierarchy
from morphocoverage.retrospective.replay import replay_acquisitions
from morphocoverage.retrospective.report import DECLARATION_TITLE, LIMITS_TITLE
from morphocoverage.retrospective.report_status import calibration_status
from morphocoverage.sampling_unit import condition_contrast, fov_variability
from morphocoverage.strategies import DescriptorSpreadStrategy, FixedFOVStrategy
from morphocoverage.strategies.rare_planning import fovs_to_detect
from morphocoverage.workflows import (
    Family,
    LiveFolder,
    LiveTables,
    Plan,
    PositiveControl,
    Report,
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
from morphocoverage.workflows._shared import orders, replay_rules
from morphocoverage.workflows.live import MANIFEST, OBJECTS_FILE
from morphocoverage.workflows.retrospective import CALIBRATION_SHORT, CALIBRATION_TITLE


def test_setup_rejects_ambiguous_declarations():
    with pytest.raises(TypeError):
        Setup(readout=READOUT, features="feature_b")
    with pytest.raises(ValueError):
        Setup(readout=READOUT, tolerances=(5,))
    with pytest.raises(ValueError, match="declare both"):
        Setup(readout=READOUT, positive=PositiveControl({"condition": "drug"}, fovs_seen=4))
    assert Setup(readout=READOUT, features=FEATURE_COLS).columns == list(FEATURE_COLS)


def test_setup_declares_what_to_preserve_and_refuses_undeclared_targets():
    plain = Setup(readout=READOUT)
    declared = plain.declaration()
    assert declared[0].startswith(f"Population summary: the mean of {READOUT}")
    assert "5% and 2.5%" in declared[0]
    assert declared[1].startswith("Phenotypic fingerprint: not declared")
    assert declared[2].startswith("Rare phenotype: not declared")
    assert [c.name for c in plain.criteria()] == [
        f"relative_mean_error:{READOUT} <= 0.05",
        f"relative_mean_error:{READOUT} <= 0.025",
    ]
    assert plain.judged_columns == [READOUT]

    full = _setup(fingerprint=(_fingerprint(),), rare=_rare(), rare_label="state")
    declared = full.declaration()
    assert "the covariance of the features within 50%" in declared[1]
    needed = fovs_to_detect(0.2, 0.95, 1)
    assert f"asks for {needed} independent representative fields" in declared[2]
    assert "label column 'state'" in declared[2]
    names = [c.name for c in full.criteria()]
    assert names[2].startswith("covariance_error:")
    assert names[3] == "rare_state_detected:state:1 >= 1"
    assert full.judged_columns == [READOUT, *(f for f in FEATURE_COLS if f != READOUT)]

    # A prevalence-only target is stated but never converted to a field count.
    prevalence_only = RareTarget(
        object_prevalence=0.01, detection_probability=0.95, min_fov_occurrences=1, phenotype="x"
    )
    line = _setup(rare=prevalence_only).declaration()[2]
    assert "Without a field incidence or a label, no field count is planned" in line
    assert "not judged" in line
    # With a label, the fields to see it are counted from the data, not warned about.
    labelled = _setup(rare=prevalence_only, rare_label="state").declaration()[2]
    assert "counted from the share of fields holding it in the data" in labelled

    assert plain.objective().objective.value == "population"
    assert _setup(fingerprint=(_fingerprint(),)).objective().objective.value == "phenotype"
    assert _setup(rare=_rare()).objective().objective.value == "balanced"
    assert full.objective().objective.value == "all"
    assert declared[-1].startswith(
        "Live decision: a stop needs every declared target (population summary and"
    )
    with pytest.raises(ValueError, match="normalize must be"):
        Setup(readout=READOUT, control={"condition": "control"}, normalize="z-score")
    with pytest.raises(ValueError, match="name the condition column"):
        Setup(readout=READOUT, normalize="control_mean")
    normalised = _setup(normalize="control_mean")
    assert "divided by the mean of the control" in normalised.declaration()[-2]
    with pytest.raises(TypeError, match="fingerprint"):
        Setup(readout=READOUT, fingerprint=_fingerprint())  # one criterion, not a sequence
    with pytest.raises(TypeError, match="fingerprint"):
        Setup(readout=READOUT, fingerprint=(CovarianceError(FEATURE_COLS, True),))
    with pytest.raises(ValueError, match="not declared"):
        Setup(readout=READOUT, fingerprint=(_fingerprint(),))  # features not declared
    with pytest.raises(ValueError, match="distinct"):
        _setup(fingerprint=(_fingerprint(0.5), _fingerprint(0.5)))
    with pytest.raises(TypeError, match="RareTarget"):
        Setup(readout=READOUT, rare={"fov_incidence": 0.2})
    with pytest.raises(ValueError, match="declare rare"):
        Setup(readout=READOUT, rare_label="state")


def test_the_documents_open_with_the_declaration_and_judge_every_declared_criterion():
    objects, fovs = _acquisition()
    # A labelled state in two fields of every group: the rare criterion has something to judge.
    objects["state"] = False
    for _, group in objects.groupby(["condition", "biological_repeat"]):
        first_two = group["image_file"].drop_duplicates().iloc[:2]
        objects.loc[group.index[group["image_file"].isin(first_two)], "state"] = True
    setup = _setup(fingerprint=(_fingerprint(),), rare=_rare(), rare_label="state")
    result = report(objects, fovs, setup)
    assert result.summary().startswith("Microscopist's report — ")
    text = result.summary(detail="full")
    opening = f"{result.document.statuses['overview'].marker} What is in the dataset"
    assert text.startswith(opening) and DECLARATION_TITLE in text
    fingerprint_line = text.index("Covariance fingerprint: the covariance")  # C22
    assert fingerprint_line < text.index("Foundation: why count")
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
    curve = plan.results["reliability"].curve
    assert set(curve["criterion"]) == {c.name for c in setup.criteria()}
    text = plan.summary(detail="full")
    assert text.startswith(DECLARATION_TITLE)
    assert "how many fields do you need for the covariance of the features within 50%?" in text
    assert "state seen in at least 1 field" in text
    merged = document(result, plan).summary(detail="full")
    assert merged.startswith(opening) and DECLARATION_TITLE in merged
    with pytest.raises(ValueError, match="rare_label"):
        report(objects.drop(columns="state"), fovs, setup)
    # Regression (#344): a label coded 0/2 was True where it was cast to bool and another
    # state where it was compared with True; text labels failed converting to float.
    for coded in (objects["state"].astype(int) * 2, np.where(objects["state"], "yes", "no")):
        with pytest.raises(ValueError, match="rare_label 'state' must hold booleans or 0/1"):
            report(objects.assign(state=coded), fovs, setup)


def test_report_answers_the_report_questions_and_leaves_the_rest_out():
    objects, fovs = _acquisition()
    before = (objects.copy(), fovs.copy())
    setup = _setup(
        novelty=default_novelty(FEATURE_COLS),
        positive=PositiveControl(
            {"condition": "drug"}, fovs_seen=5, pair_cols=("biological_repeat",)
        ),
    )
    result = report(objects, fovs, setup)
    text = result.summary(detail="full")
    assert "Foundation: why count fields" in text and "3. What is limiting" in text
    assert "4. How many fields" not in text  # a planning answer never enters the report
    # A control condition and another condition: the contrast runs, per repeat.
    assert "9. Do the conditions differ" in text and "drug vs control" in text
    assert list(result.results["contrast"].table["biological_repeat"]) == ["R1", "R2"]
    opening = f"{result.document.statuses['overview'].marker} What is in the dataset"
    assert text.startswith(opening) and "What this dataset cannot tell me" in text
    assert result.results["overview"].counts["fields"] == len(fovs)
    # Normalised to the control: the contrast reads the normalised values (the control
    # mean of the readout is 1 in every repeat); the other analyses the raw values.
    normalised_setup = _setup(normalize="control_mean")
    normalised = report(objects, fovs, normalised_setup)
    assert "Values: every feature divided by the mean of the control" in normalised.summary(
        detail="full"
    )
    assert "values: every feature divided by the mean of the control" in normalised.summary()
    prepared = normalised_setup.prepared(objects, fovs)
    control = prepared[prepared["condition"] == "control"]
    assert control.groupby("biological_repeat")[READOUT].mean().tolist() == pytest.approx(
        [1.0, 1.0]
    )
    # The contrast reads the raw values and normalises them itself, so its raw columns are
    # raw (owner, 2026-10-05).
    expected = condition_contrast(
        objects, fovs, feature=READOUT, control={"condition": "control"}, random_state=0
    )
    pd.testing.assert_frame_equal(normalised.results["contrast"].table, expected.table)
    pd.testing.assert_frame_equal(
        normalised.results["variability"].groups,
        fov_variability(objects, fovs, feature=READOUT).groups,
    )
    pd.testing.assert_frame_equal(objects, before[0])
    pd.testing.assert_frame_equal(fovs, before[1])
    assert report(objects, fovs, setup).summary(detail="full") == text  # deterministic


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
    assert "4. How many fields" in text and "Foundation: why count fields" not in text
    merged = document(report(objects, fovs, setup), plan).summary()
    assert "Foundation: why count fields" in merged and "4. How many fields" in merged


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
    for _, row in fovs.sort_values(
        ["condition", "biological_repeat", "acquisition_order"]
    ).iterrows():
        mine = objects[
            objects["condition"].eq(row["condition"])
            & objects["biological_repeat"].eq(row["biological_repeat"])
            & objects["image_file"].eq(row["image_file"])
        ]
        yield (
            mine,
            row[["condition", "biological_repeat", "image_file", "acquisition_order"]].to_dict(),
        )


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


def test_live_updates_end_with_one_view_per_target():
    objects, fovs = _acquisition(n_fovs=4)
    # A fingerprint is declared, so the live objective is phenotype; the default JCB
    # rules decide the population summary only, and the view says so.
    setup = _setup(fingerprint=(_fingerprint(),))
    tables = LiveTables(live_session(setup))
    updates = tables.update(objects, fovs)
    assert updates
    text = " ".join(updates[-1].summary(detail="full").split())
    assert "Population summary:" in text
    assert "Phenotypic fingerprint: no rule decides it" in text
    assert (
        "Overall (phenotype objective): unsupported (no rule decides the phenotypic fingerprint)"
        in text
    )
    # The short block says the same: not decided, and which target no rule decides.
    short = " ".join(updates[-1].summary().split())
    assert short.startswith("➖ field 4 ")
    assert "not decided: no rule decides the phenotypic fingerprint" in short
    assert "phenotypic fingerprint ➖ no rule decides it" in short
    assert all(r.decision.value == "unsupported" for r in updates[-1].results)
    assert all(r.rule_decision is not None for r in updates[-1].results)
    # Without a fingerprint the same rules decide the population objective.
    plain = LiveTables(live_session(_setup())).update(objects, fovs)
    assert "Overall (population objective):" in plain[-1].summary(detail="full")


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
    # The refused field left no object file, so the corrected retry gets the next name.
    assert [p.name for p in (tmp_path / "objects").iterdir()] == ["fov_000001.csv"]
    retried = write_fov(tmp_path, pd.DataFrame({READOUT: [1.0]}), {"position_id": "b"})
    assert retried.name == "fov_000002.csv"


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
    assert "Foundation: why count fields" in only and "3. What is limiting" not in only
    with pytest.raises((KeyError, ValueError)):
        result.document.summary(["not_a_question"])


# --- Regressions from the adversarial review (docs/live/live_session.md) ---


def _simple_session():
    return live_session(Setup(readout=READOUT, fov_col="position_id", group_cols=()))


def test_live_tables_refuses_key_columns_of_different_types():
    # position_id 1 (number) in the objects and "1" (text) in the FOV table never match:
    # the field would be counted as empty and its objects never used.
    live = LiveTables(_simple_session())
    fovs = pd.DataFrame({"position_id": ["1"]})
    objects = pd.DataFrame({"position_id": [1, 1, 1], READOUT: [1.0, 2.0, 3.0]})
    with pytest.raises(TypeError, match="position_id"):
        live.update(objects, fovs)


def test_live_tables_keeps_later_fields_when_one_field_fails():
    live = LiveTables(_simple_session())
    objects = pd.DataFrame(
        {"position_id": ["P1", "P1", "P2", "P3", "P3"], READOUT: [1.0, 2.0, np.inf, 5.0, 6.0]}
    )
    fovs = pd.DataFrame({"position_id": ["P1", "P2", "P3"]})
    with pytest.raises(ValueError):
        live.update(objects, fovs)  # P2 holds an infinite value
    assert [u.fov_metadata["position_id"] for u in live.updates] == ["P1"]
    assert live.pending_objects == 3  # P2 and P3 wait; nothing is lost
    with pytest.raises(ValueError):
        live.update(objects, fovs)  # the broken field is retried, never skipped
    assert len(live.updates) == 1  # P3 was not added past P2


def test_live_folder_keeps_later_fields_when_one_field_fails(tmp_path):
    live = LiveFolder(tmp_path, _simple_session())
    write_fov(tmp_path, pd.DataFrame({READOUT: [1.0, 2.0]}), {"position_id": "P1"})
    write_fov(tmp_path, pd.DataFrame({READOUT: [np.inf]}), {"position_id": "P2"})
    write_fov(tmp_path, pd.DataFrame({READOUT: [5.0]}), {"position_id": "P3"})
    with pytest.raises(ValueError):
        live.poll()
    assert [u.fov_metadata["position_id"] for u in live.updates] == ["P1"]
    with pytest.raises(ValueError):
        live.poll()  # the broken field is retried, never skipped


def test_write_fov_of_a_frame_without_columns_is_a_zero_object_field(tmp_path):
    live = LiveFolder(tmp_path, _simple_session())
    write_fov(tmp_path, pd.DataFrame(), {"position_id": "P1"})
    write_fov(tmp_path, pd.DataFrame({READOUT: [1.0]}), {"position_id": "P2"})
    assert [u.n_objects for u in live.poll()] == [0, 1]


def test_live_folder_refuses_a_missing_object_file(tmp_path):
    live = LiveFolder(tmp_path, _simple_session())
    write_fov(tmp_path, pd.DataFrame({READOUT: [1.0]}), {"position_id": "P1"}, name="f1")
    manifest = tmp_path / MANIFEST
    manifest.write_text(manifest.read_text().replace("objects/f1.csv", "objects/F1.csv"))
    with pytest.raises(FileNotFoundError, match="F1.csv"):
        live.poll()


def test_live_update_shows_objects_still_waiting_for_their_field():
    live = LiveTables(_simple_session())
    objects = pd.DataFrame({"position_id": ["P1", "P9"], READOUT: [1.0, 2.0]})
    (update,) = live.update(objects, pd.DataFrame({"position_id": ["P1"]}))
    assert update.pending_objects == 1 and "1 object row waits" in update.summary()


def test_setup_refuses_declarations_the_workflows_would_skip():
    novelty = default_novelty(FEATURE_COLS)
    positive = PositiveControl({"condition": "drug"}, fovs_seen=4)
    control = {"condition": "control"}
    with pytest.raises(ValueError, match="novelty"):
        Setup(readout=READOUT, features=FEATURE_COLS, control=control, positive=positive)
    with pytest.raises(ValueError, match="features"):
        Setup(readout=READOUT, novelty=novelty)


def test_positive_control_keeps_the_declared_fov_identity():
    # Two wells per condition and repeat, each naming its positions s0..s3: the well is
    # part of the FOV identity, as the setup's group columns say.
    objects, fovs = _acquisition(n_fovs=8)
    index = fovs.groupby(["condition", "biological_repeat"]).cumcount()
    fovs = fovs.assign(well=np.where(index < 4, "w1", "w2"), pos=[f"s{i % 4}" for i in index])
    names = fovs.set_index(["condition", "biological_repeat", "image_file"])[["well", "pos"]]
    objects = objects.join(names, on=["condition", "biological_repeat", "image_file"])
    setup = _setup(
        novelty=default_novelty(FEATURE_COLS),
        positive=PositiveControl(
            {"condition": "drug"}, fovs_seen=5, pair_cols=("biological_repeat",)
        ),
        fov_col="pos",
        group_cols=("condition", "biological_repeat", "well"),
        acquisition_order_col=None,
        stage_cols=None,
    )
    assert "controls" in report(objects, fovs, setup).results


def _with_technical_repeats(objects, fovs):
    """Two wells per condition and repeat, fields alternating between them."""
    fovs = fovs.copy()
    fovs["technical_repeat"] = [f"w{i % 2}" for i in range(len(fovs))]
    objects = objects.merge(
        fovs[["condition", "biological_repeat", "image_file", "technical_repeat"]],
        on=["condition", "biological_repeat", "image_file"],
    )
    return objects, fovs


def test_a_technical_repeat_column_derives_the_hierarchy_and_ranks_the_features():
    from morphocoverage.retrospective import SamplingHierarchy
    from morphocoverage.workflows import plots

    objects, fovs = _with_technical_repeats(*_acquisition())
    setup = _setup()
    hierarchy = setup.resolved_hierarchy(objects, fovs)
    assert hierarchy == SamplingHierarchy("image_file", "technical_repeat", "biological_repeat")
    assert setup.resolved_effort_strata(fovs) == ["condition"]
    declared = _setup(hierarchy=SamplingHierarchy("image_file", "technical_repeat"))
    assert declared.resolved_hierarchy(objects, fovs) == declared.hierarchy
    # A declared hierarchy keeps the conditions apart too (second audit, I-2), unless
    # pooling is asked for explicitly.
    assert declared.resolved_effort_strata(fovs) == ["condition"]
    pooled = _setup(hierarchy=declared.hierarchy, effort_strata=())
    assert pooled.resolved_effort_strata(fovs) == []
    assert _setup().resolved_hierarchy(*_acquisition()) is None
    result = report(objects, fovs, setup)
    assert result.ranking is not None and set(result.ranking.tests["feature"]) == set(FEATURE_COLS)
    assert (result.ranking.tests["permutation_unit"] == "technical_repeat").all()
    text = result.summary(detail="full")
    assert "technical repeats (technical_repeat, a well or a dish" in text
    assert "Which features separate the conditions?" in text
    assert "technical repeat" in result.summary().split("What is in the dataset")[1]
    figures = plots(objects, fovs, setup, ranking=result.ranking, n_boot=10)
    expected = [READOUT, *[f for f in result.ranking.top() if f != READOUT]]
    assert list(figures) == expected
    with pytest.raises(TypeError, match="ranking"):
        plots(objects, fovs, setup, ranking="x", n_boot=10)
    plan = optimise(
        objects,
        fovs,
        setup,
        fov_counts=[2, 4],
        reservoir_fraction=0.25,
        n_splits=2,
        draws=5,
        permuted_orders=1,
        dispersed_orders=1,
    )
    assert "effort" in plan.results and "protocol" in plan.results
    effort = plan.results["effort"]
    assert set(effort.components["stratum"]) == {"condition=control", "condition=drug"}
    assert "more fields or more technical repeats, for the mean" in plan.summary(detail="full")
    assert "more fields or technical repeats?" in plan.summary()
    assert "7. More fields or more samples?" in plan.summary()


def test_an_empty_technical_repeat_column_derives_no_hierarchy_and_reports_none():
    objects, fovs = _acquisition()
    objects, fovs = objects.assign(technical_repeat=None), fovs.assign(technical_repeat=None)
    setup = _setup()
    assert setup.technical_col(fovs) is None
    assert setup.resolved_hierarchy(objects, fovs) is None
    result = report(objects, fovs, setup)
    assert "technical repeat" not in result.results["overview"].summary()
    assert (result.results["contrast"].table["permutation_unit"] == "field").all()
    session = live_session(setup)
    tables = LiveTables(session)
    first = fovs.iloc[[0]]
    update = tables.update(objects[objects["image_file"] == first["image_file"].iloc[0]], first)
    assert "technical_repeat" not in tables.key
    assert update is not None


def test_saved_figures_stay_in_the_folder_and_never_share_a_file(tmp_path):
    from morphocoverage.workflows import plots
    from morphocoverage.workflows.figures import _file_name

    objects, fovs = _acquisition(n_fovs=4, n_objects=5)
    objects = objects.rename(columns={"feature_a": "a/b", "feature_b": "../outside"})
    setup = Setup(readout="a/b", features=("a/b", "../outside"))
    folder = tmp_path / "figures"
    plots(objects, fovs, setup, features=["a/b", "../outside"], folder=folder, n_boot=5)
    assert sorted(p.name for p in tmp_path.rglob("*.png")) == [".._outside.png", "a_b.png"]
    assert all(p.parent == folder for p in tmp_path.rglob("*.png"))
    taken: set[str] = set()
    names = [_file_name(name, taken) for name in ("x_1", "x_2", "x_1", "X_2", "")]
    assert names == ["x_1", "x_2", "x_1_2", "X_2_2", "feature"]


def test_beyond_the_mean_says_not_significant_and_no_trend_from_two_repeats() -> None:
    # C16: "the same way in every repeat, not tested (fewer than 3 repeats)" read as a trend
    # from 2 repeats (a25); C2: "no difference" for "not significant" (a24).
    a25 = " ".join(_scenario_report("a25_mix_several_norm_cells_tech_fp_expl").summary().split())
    beyond = a25[a25.index("Beyond the mean") :]
    assert "differs beyond the mean" not in beyond
    assert "the same way in every repeat, not tested" not in beyond
    assert "in both repeats" in beyond
    a24 = _scenario_report("a24_expl_one_norm_image_none_fp_expl")
    text = " ".join(a24.summary().split())
    assert "no difference beyond the mean" not in text
    assert "no difference in the distributions" not in a24.document.headlines["conditions"]
    assert "beyond the mean (exploratory), not significant" in a24.document.headlines["conditions"]


def _limits(text: str) -> str:
    return " ".join(text[text.index("What this dataset cannot tell me") :].split())


def test_the_closing_section_points_to_each_section_in_both_forms() -> None:
    """One line per limit pointing to its section, the reason only in the section (owner,
    2026-10-06; G3 had repeated the reasons in the short form)."""
    _, result = _report("a01_tagged_several_image_none_rare")
    reason = "each field holds one image, so a field holds a state or not"
    for detail in ("short", "full"):
        text = result.summary(detail=detail)
        limits = _limits(text)
        assert "3. What is limiting my sampling? Answered in part: see question 3." in limits
        assert "Answered in part: see the Foundation." in limits
        assert reason not in limits
        assert reason in " ".join(text.split())  # said once, in its section
    assert "no conclusion possible here, see question 10" in _limits(result.summary())
    # The page leaves out what it cannot answer, so its closing card keeps the reasons.
    page = " ".join(result.document.limits(None, "short", pointers=False).split())
    assert reason in page


def test_the_design_sentence_follows_declared_experiments() -> None:
    """With ``Setup(experiment=...)`` an experiment may hold several controls (the ECM siRNA
    set): the design says the declared experiments, never "each with its own control"."""
    import numpy as np
    import pandas as pd

    layout = {
        "VEC": ["R1", "R2", "R3"],
        "THBS5": ["R1", "R2", "R3"],
        "V_SiNC": ["R4", "R5", "R6"],
        "V_SiA": ["R4", "R5", "R6"],
        "T_SiNC": ["R7", "R8", "R9"],
        "T_SiA": ["R7", "R8", "R9"],
    }
    rng = np.random.default_rng(5)
    rows = [
        {
            "condition": condition,
            "biological_repeat": repeat,
            "repeat_group": repeat,
            "image_file": f"{condition}_{repeat}_{field}.tif",
            "label_id": label,
            "area": 10 * np.exp(rng.normal(0, 0.3)),
        }
        for condition, repeats in layout.items()
        for repeat in repeats
        for field in range(3)
        for label in range(10)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    control = {"condition": {"VEC": ["THBS5"], "V_SiNC": ["V_SiA"], "T_SiNC": ["T_SiA"]}}
    sets = {c: "untransfected" if c in ("VEC", "THBS5") else "siRNA" for c in layout}
    texts = {}
    for name, extra in (("derived", {}), ("declared", {"experiment": sets})):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setup = Setup(readout="area", control=control, observation_unit="cell", **extra)
            texts[name] = " ".join(report(objects, fovs, setup).summary("full").split())
    assert "Design: 3 sub-experiments, each with its own control in its own 3" in texts["derived"]
    assert "Design: 2 experiments as declared, in 3 biological repeats" in texts["declared"]
    assert "each with its own control" not in texts["declared"]
    # The q-values are said corrected within the declared experiment (owner, 2026-10-06).
    assert "Benjamini-Hochberg within the condition's experiment" in texts["declared"]
    assert "within the condition's sub-experiment" not in texts["declared"]
    assert "Benjamini-Hochberg within the condition's sub-experiment" in texts["derived"]


def _tables():
    rng = np.random.default_rng(0)
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
                for _ in range(6):
                    rows.append({**fovs[-1], "x": rng.normal()})
    return pd.DataFrame(rows), pd.DataFrame(fovs)


@pytest.mark.parametrize(
    ("argument", "bad"),
    [
        ("reservoir_fraction", "0.35"),
        ("reservoir_fraction", True),
        ("reservoir_fraction", float("nan")),
        ("protocol_threshold", "0.9"),
        ("protocol_threshold", True),
        ("protocol_threshold", float("inf")),
    ],
)
def test_optimise_refuses_string_bool_and_non_finite_shares(argument, bad):
    objects, fovs = _tables()
    setup = Setup(readout="x", control={"condition": "ctrl"}, observation_unit="cell")
    arguments = {"fov_counts": [2, 3], "reservoir_fraction": 0.35, argument: bad}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="between 0 and 1|share in"):
            optimise(objects, fovs, setup, **arguments)


# Absolute spread of the readout's mean over resamples: about 0.3 on the raw
# values (near 10) and 0.03 once divided by the control mean (near 1), so the
# rule stops at other fields on the two scales.
SPREAD = DescriptorSpreadStrategy(READOUT, 0.1, "fov", 20, 0)


RULES = (FixedFOVStrategy(3), SPREAD)


def _pooled(**changes) -> Setup:
    """A setup for pooled groups: the recorded order ties across the repeats it pools."""
    return _setup(acquisition_order_col=None, stage_cols=None, **changes)


def _live_verdicts(objects, fovs, setup):
    """Each rule's own verdict after the last field of every group, fed field by field."""
    ordered = fovs.sort_values(["condition", "biological_repeat", "acquisition_order"]).reset_index(
        drop=True
    )
    tables = LiveTables(live_session(setup))
    parts = [objects.merge(ordered.iloc[[i]][KEY], on=KEY) for i in range(len(ordered))]
    updates = []
    for i in range(len(ordered)):
        updates += tables.update(pd.concat(parts[: i + 1], ignore_index=True), ordered[: i + 1])
    last = {}
    for update in updates:
        for result in update.results:
            key = (
                *(result.group[c] for c in ("condition", "biological_repeat", "technical_repeat")),
            )
            last[(*key, result.config_id)] = result.rule_decision.value
    return last


def test_i1_every_rule_replay_reads_the_raw_values_as_live():
    # 8 fields per well: the per-field spread needs 5 fields with objects (audit 4, P4-6).
    objects, fovs = _well_acquisition(n_fovs=16)
    raw_setup = _setup(rules=RULES, stage_cols=None)
    setup = _setup(rules=RULES, stage_cols=None, normalize="control_mean")
    prepared = setup.prepared(objects, fovs)
    settings = {
        "reservoir_fraction": 0.25,
        "orders": orders(setup, 2, 0),
        "acquisition_order_col": "acquisition_order",
        "n_splits": 2,
        "random_state": 0,
        "incremental": True,
    }
    arguments = {
        "feature_cols": setup.columns,
        "strategies": list(RULES),
        "criteria": setup.criteria(),
        "objective": setup.objective(),
        **settings,
    }
    raw = replay_acquisitions(objects, fovs, **arguments)
    normalised = replay_acquisitions(prepared, fovs, **arguments)
    # The two scales give the rule different verdicts, so the test can tell them apart.
    assert not raw.decisions.equals(normalised.decisions)
    workflow = replay_rules(
        objects,
        prepared,
        fovs,
        setup,
        strategies=list(RULES),
        criteria=setup.criteria(),
        **settings,
    )
    pd.testing.assert_frame_equal(workflow.decisions, raw.decisions)
    pd.testing.assert_frame_equal(workflow.outcomes, normalised.outcomes)
    # The report judges each rule on the raw values: the verdicts of the raw setup and
    # of the live session, which never sees normalised values.
    final = report(objects, fovs, setup).results["sufficiency"].final
    expected = report(objects, fovs, raw_setup).results["sufficiency"].final
    columns = ["condition", "biological_repeat", "technical_repeat", "config_id", "verdict"]
    pd.testing.assert_frame_equal(final[columns], expected[columns])
    live = _live_verdicts(objects, fovs, setup)
    for row in final.itertuples(index=False):
        assert (
            row.verdict
            == live[(row.condition, row.biological_repeat, row.technical_repeat, row.config_id)]
        )
    # optimise and benchmark use the same replay: their decisions are the raw ones.
    plan = optimise(objects, fovs, setup, **QUICK)
    plan_raw = optimise(objects, fovs, raw_setup, **QUICK)
    pd.testing.assert_frame_equal(
        plan.results["calibration"].table, plan_raw.results["calibration"].table
    )
    family = [Family("spread", SPREAD, "max_descriptor_spread", (0.05, 0.1, 0.4))]
    bench = {
        name: benchmark(
            objects,
            fovs,
            chosen,
            reservoir_fraction=0.25,
            families=family,
            n_splits=1,
            permuted_orders=1,
            n_boot=10,
        )
        for name, chosen in (("normalised", setup), ("raw", raw_setup))
    }
    first = next(iter(bench["raw"].frontiers))
    pd.testing.assert_frame_equal(
        bench["normalised"].frontiers[first].scores, bench["raw"].frontiers[first].scores
    )
    assert "every stopping rule runs on the raw values" in " ".join(setup.declaration())


def test_i3_optimise_says_why_protocol_or_effort_is_not_assessed():
    objects, fovs = _well_acquisition()
    # A hierarchy level (the repeat) pooled by group_cols: no crash, a reason.
    plan = optimise(objects, fovs, _pooled(group_cols=("condition", "technical_repeat")), **QUICK)
    assert "protocol" not in plan.results and "effort" in plan.results
    text = plan.summary()
    assert (
        "Protocol reliability across samples is not assessed" in text
        and "'biological_repeat'" in text
    )
    # A condition stratum pooled: the reason offers explicit pooling.
    plan = optimise(
        objects, fovs, _pooled(group_cols=("biological_repeat", "technical_repeat")), **QUICK
    )
    assert "protocol" not in plan.results
    assert "effort_strata=()" in plan.document.sections["next_time"]
    # The technical repeat left out: the effort section names the column that exists.
    plan = optimise(objects, fovs, _pooled(group_cols=("condition", "biological_repeat")), **QUICK)
    effort = plan.document.sections["effort"]
    assert "leave 'technical_repeat' out" in effort
    # The generic need no longer tells the user to add a column that is there.
    assert "technical_repeat column (a well, a dish) or declare" not in plan.summary()
    # A declared hierarchy on those groups: the effort projection runs, the protocol says why.
    hierarchy = SamplingHierarchy("image_file", "technical_repeat", "biological_repeat")
    declared = _pooled(group_cols=("condition", "biological_repeat"), hierarchy=hierarchy)
    plan = optimise(objects, fovs, declared, **QUICK)
    assert "effort" in plan.results and "protocol" not in plan.results
    assert "pool 'technical_repeat'" in plan.document.sections["next_time"]


def test_i6_control_analyses_on_pooled_groups_are_not_assessed_with_the_reason():
    objects, fovs = _well_acquisition()
    pooled = report(objects, fovs, _pooled(group_cols=("condition",)))
    assert "contrast" not in pooled.results and "object_rules" in pooled.results
    assert "pool 'biological_repeat'" in pooled.document.sections["conditions"]
    everything = report(objects, fovs, _pooled(group_cols=()))
    assert "object_rules" not in everything.results and "unusual" not in everything.results
    assert "object-rule check against the control is not assessed" in everything.summary()
    assert "pool 'condition'" in everything.document.sections["profile"]


def test_i12_presets_are_printed_and_the_docstrings_say_what_is_chosen():
    objects, fovs = _well_acquisition()
    plan = optimise(objects, fovs, _setup(stage_cols=None), **QUICK)
    assert "90% of random acquisitions" in plan.summary() and "protocol_threshold" in plan.summary()
    family = [Family("spread", SPREAD, "max_descriptor_spread", (0.05, 0.4))]
    bench = benchmark(
        objects,
        fovs,
        _setup(stage_cols=None),
        reservoir_fraction=0.25,
        families=family,
        n_splits=1,
        permuted_orders=1,
        n_boot=10,
    )
    assert "risk levels 5% and 10% of stopping too early (alphas" in bench.summary()
    assert "one criterion per tolerance" not in inspect.getdoc(optimise)
    from morphocoverage import workflows

    for doc in (workflows.__doc__, workflow_module.__doc__):
        assert "no scientific choice" not in doc


def _rare_target() -> RareTarget:
    return RareTarget(
        fov_incidence=0.2, detection_probability=0.95, min_fov_occurrences=1, phenotype="big"
    )


def test_i5_every_preset_is_printed_and_minimum_detection_reaches_the_report(monkeypatch):
    seen = {}
    assemble = document_module.microscopist_report

    def spy(**kwargs):
        seen.update(kwargs)
        return assemble(**kwargs)

    monkeypatch.setattr(document_module, "microscopist_report", spy)
    objects, fovs = _acquisition(n_fovs=6, n_objects=4)
    setup = _setup(stage_cols=None, minimum_detection=0.5)
    first = report(objects, fovs, setup)
    assert seen["minimum_detection"] == 0.5
    with pytest.raises(ValueError, match="minimum_detection"):
        _setup(minimum_detection=1.5)
    positive = _setup(
        novelty=default_novelty(FEATURE_COLS),
        positive=PositiveControl({"condition": "drug"}, fovs_seen=5),
        minimum_detection=0.8,
    )
    assert "fewer than 80% of its checks are recognized, a provisional threshold" in " ".join(
        positive.declaration()
    )
    rules = [line for line in setup.declaration() if line.startswith("Stopping rules:")]
    assert len(rules) == 1 and "(the default, rules=None" in rules[0]
    chosen = _setup(rules=(FixedFOVStrategy(4),)).declaration()
    assert any("fixed 4 fields [fixed_fov (target_fovs=4)] (rules)" in line for line in chosen)
    assert "5 random orders" in first.summary()
    # document() keeps the plan's settings sentence (the protocol_threshold preset) and the
    # report's.
    plan = optimise(objects, fovs, setup, **{**QUICK, "fov_counts": [2]}, protocol_threshold=0.8)
    merged = document(first, plan).summary()
    assert "(protocol_threshold, a provisional preset)" in merged
    assert "'Did I image enough?' replays each rule" in merged
    # benchmark: n_boot, min_fovs, the grids and the families' seed.
    captured = {}
    families = benchmarks_module.default_families

    def quick_families(readout, n_resamples=200, random_state=0):
        captured["random_state"] = random_state
        return families(readout, 10, random_state)[:1]

    monkeypatch.setattr(benchmarks_module, "default_families", quick_families)
    text = benchmark(
        objects,
        fovs,
        _setup(stage_cols=None, random_state=3),
        reservoir_fraction=0.25,
        n_splits=1,
        permuted_orders=1,
        n_boot=20,
        min_fovs=2,
    ).summary()
    text = " ".join(text.split())  # the settings are wrapped
    assert "from 20 bootstrap resamples (n_boot)" in text
    assert "rules evaluated from 2 fields on (min_fovs)" in text
    assert "JCB per object (averaging_cv 0.0025 to 0.4, 9 settings)" in text
    assert "resampling rules seeded with random_state 3" in text


def test_i7_the_report_clusters_the_declared_rare_label():
    objects, fovs = _acquisition(n_fovs=6, n_objects=4)
    objects = objects.assign(state=objects[READOUT] > objects[READOUT].quantile(0.9))
    setup = _setup(control=None, rare=_rare_target(), rare_label="state", stage_cols=None)
    result = report(objects, fovs, setup)
    clustering = result.results["rare_clustering"]
    assert clustering.settings["label_col"] == "state"
    assert clustering.settings["states"] == [True]
    profile = result.document.sections["profile"]
    assert "Are the state" in profile and result.document.inputs["profile"] == "given"
    assert "- State cells: clustered in a few fields" in result.document.short["profile"]
    assert "fov_clustering" in set(result.document.profile["source"])


def test_i8_document_refuses_parts_of_different_setups():
    first, other = _setup(), _setup(tolerances=(0.1,))
    with pytest.raises(ValueError, match="different Setups"):
        document(Report({}, None, first, source="s"), Plan({}, None, other, source="s"))
    # n_jobs changes no result, so parts differing only there merge (of one acquisition).
    merged = document(
        Report({}, None, first, source="s"), Plan({}, None, _setup(n_jobs=2), source="s")
    )
    assert merged.declaration == first.declaration()


def test_i13_the_unusual_labels_are_named_for_the_reader():
    objects, fovs = _acquisition(n_fovs=6, n_objects=4)
    result = report(objects, fovs, _setup(stage_cols=None, observation_unit="cell"))
    assert result.results["clustering"].settings["label_col"] == "unusual cells"
    assert "morphocoverage_unusual" not in result.summary()
    assert "Are the unusual cells spread over the fields?" in result.document.sections["profile"]


def test_i18_the_unusual_values_are_relative_to_the_control_of_their_repeat():
    # Plan v2 (owner, 2026-10-06): biological unusualness reads each value relative to the
    # control of its own repeat, so a shift of one repeat is no longer read as unusual and
    # the caution that the reference pools repeats has no object.
    objects, fovs = _acquisition(n_fovs=6, n_objects=4)
    result = report(objects, fovs, _setup(stage_cols=None))
    profile = result.document.sections["profile"]
    assert "pools the control cells of 2 repeats" not in profile
    declaration = " ".join(result.setup.declaration())
    assert "every value relative to the control of its repeat; exploratory" in declaration


def test_seed3_seed18_b12_embedding_only_gives_the_real_reason():
    objects, fovs, table = _one_per_field()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alone = report(objects, fovs, Setup(observation_unit="image"), feature_tables={"T": table})
        single = report(
            objects,
            fovs,
            Setup(control={"condition": "ctrl"}, observation_unit="image"),
            feature_tables={"T": table},
        )
    short = " ".join(alone.summary().split())
    # Plan v2 (owner, 2026-10-06): requested contrasts only, not yet supported by the report.
    assert "9. Do the conditions differ? — not applicable: no control is declared" in short
    assert "2. Did I image enough? — not applicable: no target declared; declare the" in short
    assert "fingerprint) or a labelled phenotype (rare with rare_label)" in short
    assert "unusual images not assessed without a control" in short
    # One image per field: the unusual images run (plan v2), only their clustering within
    # fields is said not to apply; the Foundation needs explainable features.
    short = " ".join(single.summary().split())
    assert "3. What limits my sampling? — unusual images" in short
    assert "cluster in a few fields is not applicable: each field holds one image" in short
    assert "Foundation. Why count fields? — not applicable: no explainable features" in short
    assert "unusual images on the 6 columns of T (each relative to its own control's mean" in short


def test_b8_b9_embedding_only_values_and_space_are_said_as_they_are():
    *_, result = _embedding_only(normalize="control_mean")
    full = " ".join(result.summary(detail="full").split())
    assert "Values: as given (not normalised)" not in full
    assert "Values: the other feature tables centred per biological repeat" in full
    # One table: its columns, not a weighted combined space.
    # Centred tables, against each own control: never "as normalised" (audit 10, B7, C11).
    space = "unusual cells on the 12 columns of DINO (centred per biological repeat, against "
    assert space in full
    assert "exploratory), no definition of new" in full
    assert "weighted combined space" not in full


def test_seed6_the_batch_line_says_unusual_observations_read_the_explainable_features():
    objects, fovs, table = _cells()
    objects["area"] = np.random.default_rng(1).normal(10, 1, len(objects))
    setup = Setup(
        readout="area",
        features=("area", "mitotic"),
        control={"condition": "ctrl"},
        normalize="control_mean",
        observation_unit="cell",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup, feature_tables={"DINO": table})
    full = " ".join(result.summary(detail="full").split())
    assert (
        "the centred map, the populations and the separation of the conditions read them, "
        "never a test" in full
    )


def test_b10_embedding_only_says_the_rows_across_experiments_are_not_tested():
    objects, fovs, tables = _several()
    setup = Setup(control=SEVERAL, master_control="cA", observation_unit="cell")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup, feature_tables={"DINO": tables["DINO"]})
    full = " ".join(result.summary(detail="full").split())
    assert "tested feature by feature only" not in full
    assert "there are no explainable features to test them on" in full


def test_b5_without_a_technical_repeat_the_fields_alone_are_resampled():
    objects, fovs, _ = _several()
    setup = Setup(readout="area", control={"condition": "cA"}, observation_unit="cell")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    full = " ".join(result.summary(detail="full").split())
    assert "from resampling the fields as acquired" in full
    assert "resampling the technical repeats" not in full and "within wells" not in full


def test_optimise_does_not_judge_a_group_too_small_for_the_common_k_and_names_it():
    objects, fovs = _fingerprint_data(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = optimise(
            objects, fovs, _fingerprint_setup(), fov_counts=(2,), reservoir_fraction=0.35
        )
    curve = plan.results["reliability"].curve
    pilot = curve[curve["condition"] == "pilot"]
    assert len(pilot) and (pilot["defined_draws"] == 0).all() and pilot["preserved"].isna().all()
    others = curve[curve["condition"] != "pilot"]
    assert (others["defined_draws"] > 0).all()
    assert (
        "The covariance fingerprint, DINO covariance within 15%, is not assessable in "
        "pilot · repeat R1: "
        "fewer than "
        "5 k, 150 cells." in plan.document.notes
    )


def test_item10_one_image_per_field_leaves_the_rare_clustering_out_of_question_3():
    import sys
    import warnings
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from _helpers_embedding import FEATURES, MITOTIC, _one_per_field

    from morphocoverage.workflows import Setup, report

    objects, fovs, table = _one_per_field()
    objects["mitotic"] = (np.random.default_rng(2).random(len(objects)) < 0.3).astype(int)
    common = {"control": {"condition": "ctrl"}, "observation_unit": "image"}
    rare = {"rare": MITOTIC, "rare_label": "mitotic"}
    explainable = Setup(readout="f1", features=FEATURES, **common, **rare)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for setup, tables in ((explainable, None), (Setup(**common, **rare), {"T": table})):
            short = " ".join(report(objects, fovs, setup, feature_tables=tables).summary().split())
            # Plan v2: question 3 runs on the images; only the clustering within fields of
            # the labelled state (and of the unusual images) is said not to apply.
            assert "or unusual images cluster in a few fields is not applicable" in short
            assert "spread like random images" not in short


def test_df1_optimise_runs_when_group_cols_leave_the_technical_repeat_out():
    objects, fovs = _wells(*_acquisition(n_fovs=8, n_objects=6))
    setup = _setup(group_cols=("condition", "biological_repeat"), stage_cols=None)
    assert setup.resolved_hierarchy(objects, fovs) is None
    plan = optimise(
        objects,
        fovs,
        setup,
        fov_counts=[2, 4],
        reservoir_fraction=0.25,
        n_splits=2,
        draws=5,
        permuted_orders=1,
        dispersed_orders=0,
    )
    assert "protocol" not in plan.results and "effort" not in plan.results


def test_df3_question_7_reads_the_raw_values_with_their_variance_between_repeats():
    # Second audit, I-1: normalised values only for outcomes, contrast, ranking and
    # plots; the effort projection keeps the variance between repeats of the raw values.
    objects, fovs = _wells(*_acquisition(n_fovs=8, n_objects=6))
    settings = {
        "fov_counts": [2],
        "reservoir_fraction": 0.25,
        "n_splits": 1,
        "draws": 3,
        "permuted_orders": 1,
        "dispersed_orders": 0,
    }
    plan = optimise(objects, fovs, _setup(normalize="control_mean", stage_cols=None), **settings)
    raw = optimise(objects, fovs, _setup(stage_cols=None), **settings)
    assert "Values are normalised" not in plan.results["effort"].summary()
    pd.testing.assert_frame_equal(
        plan.results["effort"].components, raw.results["effort"].components
    )


def test_optimise_measures_the_redundancy_distance_with_stage_coordinates():
    rng = np.random.default_rng(0)
    rows, fovs = [], []
    for condition in ("siCTRL", "A"):
        for repeat in ("2026_01_01", "2026_01_02"):
            for field in range(8):
                image = f"{condition}_{repeat}_{field}.tif"
                key = {
                    "condition": condition,
                    "biological_repeat": repeat,
                    "repeat_group": repeat,
                    "image_file": image,
                }
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
    redundancy = plan.results["redundancy"]
    assert redundancy.settings["acquisition"] == plan.results["efficiency"].settings["acquisition"]
    assert redundancy.table.to_csv(index=False) == (
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


@pytest.fixture(scope="module")
def plan():
    objects, fovs = _acquisition()
    return optimise(
        objects,
        fovs,
        _setup(),
        fov_counts=[2, 4],
        reservoir_fraction=0.25,
        n_splits=2,
        draws=5,
        permuted_orders=2,
        dispersed_orders=1,
    )


def test_plan_has_its_own_title_and_the_rule_check_at_a_glance(plan):
    short = plan.summary()
    assert short.startswith("Plan the next experiment — 768 cells in 64 fields")
    status, answer = calibration_status(plan.results["calibration"])
    glance = short.split("\n\n")[1]
    assert f"{status.marker} {CALIBRATION_SHORT}" in glance
    assert _flat(answer) in _flat(glance)
    # The rule check is a section before the limits, and the settings close the plan.
    section = short.index(f"{status.marker} {CALIBRATION_TITLE}")
    assert section < short.index(f"\n{LIMITS_TITLE}\n") < short.index("Settings: a sample")
    full = plan.summary(detail="full")
    assert full.index(CALIBRATION_TITLE) < full.index(f"\n{LIMITS_TITLE}\n")
    assert "90% of random acquisitions" in _flat(full) and "90% of random" in _flat(short)
    with pytest.raises(ValueError, match="short"):
        plan.summary(detail="everything")


def test_calibration_status_follows_its_rule(plan):
    calibration = plan.results["calibration"]
    shares = calibration.table["not_preserved_at_stop"].dropna()
    status = calibration_status(calibration)[0]
    assert status is (Status.CAUTION if shares.max() > 0.1 else Status.OK)
    safe = calibration.table.assign(not_preserved_at_stop=0.0)
    ok = type(calibration)(calibration.runs, safe, calibration.group_cols, calibration.settings)
    assert calibration_status(ok)[0] is Status.OK
    empty = type(calibration)(calibration.runs, safe.iloc[0:0], calibration.group_cols, {})
    assert calibration_status(empty)[0] is Status.NOT_ASSESSED
    assert calibration_status(None)[0] is Status.NOT_ASSESSED
