"""The workflows (report, optimise, benchmark, live): they run end to end and add nothing.

Constructed data only check that each mode runs, is deterministic and keeps
its contracts; the real-data runs are in ``docs/workflows.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, _fov_table, _objects_for

from morphocoverage import RareTarget, validate_input_tables
from morphocoverage.multivariate_outcomes import CovarianceError
from morphocoverage.results import Decision
from morphocoverage.retrospective import OutcomeCriterion
from morphocoverage.retrospective.report import DECLARATION_TITLE
from morphocoverage.sampling_unit import condition_contrast, fov_variability
from morphocoverage.strategies import FixedFOVStrategy
from morphocoverage.strategies.rare_planning import fovs_to_detect
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


def _fingerprint(tolerance: float = 0.5) -> OutcomeCriterion:
    return OutcomeCriterion(
        CovarianceError(FEATURE_COLS, True),
        tolerance,
        "lower_is_better",
        f"the covariance of the features within {tolerance:.0%}",
    )


def _rare() -> RareTarget:
    return RareTarget(
        fov_incidence=0.2, detection_probability=0.95, min_fov_occurrences=1, phenotype="planted"
    )


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
    assert "No field incidence was declared" in line and "not judged" in line

    assert plain.objective().objective.value == "population"
    assert _setup(fingerprint=(_fingerprint(),)).objective().objective.value == "phenotype"
    assert _setup(rare=_rare()).objective().objective.value == "balanced"
    assert full.objective().objective.value == "all"
    assert declared[-1].startswith("Live decision: the all objective (population summary and")
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
    for _, group in objects.groupby(["condition", "repeat"]):
        first_two = group["image_file"].drop_duplicates().iloc[:2]
        objects.loc[group.index[group["image_file"].isin(first_two)], "state"] = True
    setup = _setup(fingerprint=(_fingerprint(),), rare=_rare(), rare_label="state")
    result = report(objects, fovs, setup)
    assert result.summary().startswith("Microscopist's report — ")
    text = result.summary(detail="full")
    opening = f"{result.document.statuses['overview'].marker} What is in the dataset"
    assert text.startswith(opening) and DECLARATION_TITLE in text
    fingerprint_line = text.index("Phenotypic fingerprint: the covariance")
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
    assert "the rare phenotype (planted) seen in at least 1 field" in text
    merged = document(result, plan).summary(detail="full")
    assert merged.startswith(opening) and DECLARATION_TITLE in merged
    with pytest.raises(ValueError, match="rare_label"):
        report(objects.drop(columns="state"), fovs, setup)


def test_report_answers_the_report_questions_and_leaves_the_rest_out():
    objects, fovs = _acquisition()
    before = (objects.copy(), fovs.copy())
    setup = _setup(
        novelty=default_novelty(FEATURE_COLS),
        positive=PositiveControl({"condition": "drug"}, fovs_seen=4, pair_cols=("repeat",)),
    )
    result = report(objects, fovs, setup)
    text = result.summary(detail="full")
    assert "Foundation: why count fields" in text and "3. What is limiting" in text
    assert "4. How many fields" not in text  # a planning answer never enters the report
    # A control condition and another condition: the contrast runs, per repeat.
    assert "9. Do the conditions differ" in text and "drug vs control" in text
    assert list(result.results["contrast"].table["repeat"]) == ["R1", "R2"]
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
    assert control.groupby("repeat")[READOUT].mean().tolist() == pytest.approx([1.0, 1.0])
    expected = condition_contrast(
        prepared, fovs, feature=READOUT, control={"condition": "control"}, random_state=0
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


# --- Regressions from the adversarial review (docs/workflows.md, "Live sources") ---


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
    index = fovs.groupby(["condition", "repeat"]).cumcount()
    fovs = fovs.assign(well=np.where(index < 4, "w1", "w2"), pos=[f"s{i % 4}" for i in index])
    names = fovs.set_index(["condition", "repeat", "image_file"])[["well", "pos"]]
    objects = objects.join(names, on=["condition", "repeat", "image_file"])
    setup = _setup(
        novelty=default_novelty(FEATURE_COLS),
        positive=PositiveControl({"condition": "drug"}, fovs_seen=2, pair_cols=("repeat",)),
        fov_col="pos",
        group_cols=("condition", "repeat", "well"),
        acquisition_order_col=None,
        stage_cols=None,
    )
    assert "controls" in report(objects, fovs, setup).results


def _with_technical_repeats(objects, fovs):
    """Two wells per condition and repeat, fields alternating between them."""
    fovs = fovs.copy()
    fovs["technical_repeat"] = [f"w{i % 2}" for i in range(len(fovs))]
    objects = objects.merge(
        fovs[["condition", "repeat", "image_file", "technical_repeat"]],
        on=["condition", "repeat", "image_file"],
    )
    return objects, fovs


def test_a_technical_repeat_column_derives_the_hierarchy_and_ranks_the_features():
    from morphocoverage.retrospective import SamplingHierarchy
    from morphocoverage.workflows import plots

    objects, fovs = _with_technical_repeats(*_acquisition())
    setup = _setup()
    hierarchy = setup.resolved_hierarchy(objects, fovs)
    assert hierarchy == SamplingHierarchy("image_file", "technical_repeat", "repeat")
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
