"""Regressions of the 2026-09-26 function audit: workflows, plots, live and data flow (H, DF).

Each test names the finding it pins (the first function audit, docs/audit/README.md)
and fails on the code before the fix.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS
from test_workflows import READOUT, _acquisition, _setup

import morphocoverage.plots as plots_module
import morphocoverage.workflows._shared as shared_module
import morphocoverage.workflows.figures as figures_module
import morphocoverage.workflows.retrospective as workflow_module
from morphocoverage import ObjectiveConfig, SamplingObjective
from morphocoverage.plots import PALETTE, plots_of_data
from morphocoverage.sampling_unit import FeatureRanking
from morphocoverage.schema import validate_input_tables
from morphocoverage.strategies import FOVBootstrapEstimator
from morphocoverage.workflows import (
    LiveFolder,
    LiveTables,
    Setup,
    benchmark,
    default_families,
    live_session,
    optimise,
    plots,
    report,
    write_fov,
)
from morphocoverage.workflows.live import MANIFEST

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _pooled_wells():
    """9 biological repeats (more than are drawn side by side), 2 wells each named alike."""
    rng = np.random.default_rng(0)
    rows, fields = [], []
    for repeat in range(9):
        for well in range(2):
            shift = rng.normal(0, 3)
            for field in range(3):
                key = {
                    "condition": "c",
                    "biological_repeat": f"r{repeat}",
                    "technical_repeat": f"w{well}",
                    "image_file": f"f{field}.tif",
                }
                fields.append(key)
                rows += [{**key, "x": shift + rng.normal()} for _ in range(10)]
    return pd.DataFrame(rows), pd.DataFrame(fields)


def test_h1_pooled_repeats_keep_each_well_within_its_repeat(monkeypatch):
    captured = []
    original = plots_module.bootstrap_nested

    def spy(units, *args, **kwargs):
        captured.append([len(unit) for unit in units])
        return original(units, *args, **kwargs)

    monkeypatch.setattr(plots_module, "bootstrap_nested", spy)
    plots_of_data(*_pooled_wells(), features=["x"], n_boot=50)
    # 9 repeats, each with the 6 fields of its 2 wells (the repeats alone resampled: audit
    # 3, H-2), never 2 wells of 27 fields
    assert captured == [[6] * 9]


def test_h5_the_palette_note_and_limit_say_what_is_drawn():
    assert not any(colour in ("#8C8C8C", "#999999") for colour in PALETTE)
    with pytest.raises(ValueError, match="max_technical can be at most 8"):
        plots_of_data(*_pooled_wells(), features=["x"], n_boot=10, max_technical=9)
    figure = plots_of_data(*_pooled_wells(), features=["x"], n_boot=10, max_technical=1)["x"][0]
    texts = [t.get_text() for t in figure.axes[0].texts]
    assert any("the first 1 shaped in order, the others small dots" in t for t in texts)


def test_h2_every_workflow_gives_the_rules_the_same_columns(monkeypatch):
    seen = {}
    replay, sufficiency = shared_module.replay_acquisitions, workflow_module.sufficiency_report

    def replay_spy(*args, feature_cols, **kwargs):
        seen.setdefault("replay", []).append(list(feature_cols))
        return replay(*args, feature_cols=feature_cols, **kwargs)

    def sufficiency_spy(*args, feature_cols, **kwargs):
        seen["sufficiency"] = list(feature_cols)
        return sufficiency(*args, feature_cols=feature_cols, **kwargs)

    monkeypatch.setattr(shared_module, "replay_acquisitions", replay_spy)
    monkeypatch.setattr(workflow_module, "sufficiency_report", sufficiency_spy)
    objects, fovs = _acquisition(n_fovs=8, n_objects=6)
    setup = _setup(stage_cols=None)
    report(objects, fovs, setup)
    families = default_families(READOUT, n_resamples=10)[:1]
    benchmark(
        objects,
        fovs,
        setup,
        reservoir_fraction=0.35,
        families=families,
        n_splits=1,
        permuted_orders=1,
        n_boot=10,
    )
    assert seen["sufficiency"] == setup.columns
    assert seen["replay"] == [setup.columns]
    assert list(live_session(setup).feature_cols) == setup.columns


def test_h3_the_benchmark_bootstrap_of_the_mean_reads_the_readout_only():
    family = default_families(READOUT, n_resamples=20)[-1]
    assert family.strategy.estimators[0].features == (READOUT,)
    objects, fovs = _acquisition(n_fovs=8, n_objects=6)
    everything = validate_input_tables(objects, fovs=fovs, feature_cols=list(FEATURE_COLS))
    alone = validate_input_tables(objects, fovs=fovs, feature_cols=[READOUT])
    selected = FOVBootstrapEstimator("centroid", None, 50, 0, 2, features=(READOUT,))
    plain = FOVBootstrapEstimator("centroid", None, 50, 0, 2)
    name = "fov_bootstrap_centroid_max_se"
    assert selected.estimate(everything, objective=POPULATION).value(name) == pytest.approx(
        plain.estimate(alone, objective=POPULATION).value(name)
    )
    # Without features the configuration is the one it always was.
    assert "features" not in plain.parameters


def test_h4_the_report_inspection_keeps_every_reason_with_novelty():
    from morphocoverage.workflows import default_novelty

    objects, fovs = _acquisition(n_fovs=8, n_objects=6)
    result = report(objects, fovs, _setup(novelty=default_novelty(FEATURE_COLS)))
    settings = result.results["inspection"].settings
    assert settings["novelty"] is not None and settings["novelty_cols"] == list(FEATURE_COLS)
    # With a contrast, a field's influence is on the result, not on a declared summary.
    on_result = "contrast" in result.results
    summaries = [] if on_result else [{"feature": READOUT, "statistic": "mean"}]
    assert settings["summaries"] == summaries
    assert ("influence" in settings["checked"]) and (
        settings.get("influence_on") == ("result" if on_result else None)
    )
    assert settings["count_ratio"] == 3.0 and settings["stage_cols"] == ["stage_x", "stage_y"]


def test_h6_live_identifiers_keep_their_leading_zeros(tmp_path):
    objects = pd.DataFrame({"position_id": ["001"] * 5, "x": np.arange(5.0) + 1})
    write_fov(
        tmp_path, objects, {"condition": "A", "biological_repeat": "01", "position_id": "001"}
    )
    update = LiveFolder(tmp_path, live_session(Setup(readout="x"))).poll()[0]
    assert update.n_objects == 5 and update.fov_metadata["position_id"] == "001"


def test_h11_a_field_written_to_an_empty_manifest_gets_its_header(tmp_path):
    (tmp_path / MANIFEST).write_text("")
    write_fov(tmp_path, pd.DataFrame({"x": [1.0]}), {"image_file": "a.tif"})
    assert (tmp_path / MANIFEST).read_text().splitlines()[0] == "image_file,objects_file"


def test_h14_h16_live_tables_keep_integers_and_say_the_unit():
    fovs = pd.DataFrame({"plate": [1, 2], "image_file": [10, 11]})
    objects = pd.DataFrame({"plate": [1, 1], "image_file": [10, 10], "x": [1.0, 2.0]})
    session = live_session(Setup(readout="x", observation_unit="nucleus", group_cols=("plate",)))
    update = LiveTables(session).update(objects, fovs)[0]
    assert isinstance(update.fov_metadata["plate"], int)
    assert "2 nuclei" in update.summary()


def test_h9_criteria_with_the_same_label_each_keep_a_frontier():
    objects, fovs = _acquisition(n_fovs=8, n_objects=6)
    setup = _setup(tolerances=(0.05, 0.0502), stage_cols=None)
    labels = [c.label for c in setup.criteria()]
    # Declared rates are written exactly (audit 3, _wording.rate), so the two labels
    # no longer read the same; each criterion still keeps its own frontier.
    assert labels == ["the mean of feature_a within 5%", "the mean of feature_a within 5.02%"]
    result = benchmark(
        objects,
        fovs,
        setup,
        reservoir_fraction=0.35,
        families=default_families(READOUT, n_resamples=10)[:1],
        n_splits=1,
        permuted_orders=1,
        n_boot=10,
    )
    assert len(result.frontiers) == 2


def test_contrast_summary_without_any_object_count_that_fits():
    from morphocoverage.sampling_unit import condition_contrast

    objects, fovs = _acquisition(n_fovs=8, n_objects=6)
    result = condition_contrast(
        objects, fovs, feature=READOUT, control={"condition": "control"}, n_permutations=20
    )
    # The effect size from every object is still given (audit 3, H-19), with no draws.
    assert result.effects.empty and "from 50 objects" not in result.summary()
    # Across repeats the short form leaves it to the full one's table.
    assert "over control SD, every object" in result.summary(detail="full")


def test_h10_df2_plots_give_the_top_places_to_the_best_ranked(monkeypatch):
    """The readout is plotted only when it ranks (owner, 2026-10-02: the report is neutral)."""
    seen = {}

    def spy(objects, fovs, *, features, value_note=None, **kwargs):
        seen["features"], seen["note"] = list(features), value_note
        return {}

    monkeypatch.setattr(figures_module, "plots_of_data", spy)
    ranking = FeatureRanking(
        pd.DataFrame(),
        pd.DataFrame({"feature": [READOUT, FEATURE_COLS[1]], "score": [3.0, 2.0]}),
        {},
    )
    objects, fovs = _acquisition(n_fovs=8, n_objects=6)
    plots(objects, fovs, _setup(), ranking=ranking, top=1)
    assert seen["features"] == [READOUT] and seen["note"] is None
    other_first = FeatureRanking(
        pd.DataFrame(),
        pd.DataFrame({"feature": [FEATURE_COLS[1], READOUT], "score": [3.0, 2.0]}),
        {},
    )
    plots(objects, fovs, _setup(), ranking=other_first, top=1)
    assert seen["features"] == [FEATURE_COLS[1]]
    plots(objects, fovs, _setup(normalize="control_mean"), ranking=ranking, top=1)
    assert seen["note"] == "relative to control"


def test_h12_h13_the_declaration_says_what_is_assessed():
    text = "\n".join(Setup(readout="x").declaration())
    assert "no feature columns declared, so unusual cells and new phenotypes are not" in text
    normalised = Setup(readout="x", control={"condition": "MOCK"}, normalize="control_mean")
    assert "(of the whole table without a repeat column)" in "\n".join(normalised.declaration())


def _wells(objects, fovs):
    fovs = fovs.assign(technical_repeat=[f"w{i % 2}" for i in range(len(fovs))])
    keys = ["condition", "biological_repeat", "image_file", "technical_repeat"]
    return objects.merge(fovs[keys]), fovs


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
