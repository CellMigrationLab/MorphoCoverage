"""The whole pipeline on a synthetic acquisition whose answers are known (``data/synthetic``).

``data/synthetic/make_synthetic.py`` builds the tables from declared variance
components, a declared drug effect, planted rare cells and empty fields;
``expected.json`` holds the truth. Each test runs one part of the pipeline the
way a user would and checks its answer against that truth: exact counts where
the construction fixes them, and tolerances from the design where the answer
is an estimate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from morphocoverage import FixedFOVStrategy, ScalarPrecisionStrategy, check_tables
from morphocoverage.sampling_unit import fov_clustering
from morphocoverage.workflows import (
    LiveFolder,
    Setup,
    live_session,
    optimise,
    plots,
    report,
    write_fov,
)

SYNTHETIC = Path(__file__).resolve().parents[1] / "data" / "synthetic"
FEATURES = ("area", "intensity", "eccentricity")
FIELD_KEY = ["condition", "biological_repeat", "technical_repeat", "image_file"]


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The committed tables, as a user reads them, and the truth."""
    objects = pd.read_csv(SYNTHETIC / "synthetic_cells.csv")
    fovs = pd.read_csv(SYNTHETIC / "synthetic_fovs.csv")
    return objects, fovs, json.loads((SYNTHETIC / "expected.json").read_text())


def synthetic_setup(**changes) -> Setup:
    """How a user would declare this acquisition."""
    settings = {
        "readout": "area",
        "features": FEATURES,
        "control": {"condition": "control"},
        "observation_unit": "cell",
        "acquisition_order_col": "acquisition_order",
        "stage_cols": ("stage_x", "stage_y"),
        "rules": (FixedFOVStrategy(8), ScalarPrecisionStrategy("area", 0.05, "fov", "upper_95")),
    }
    return Setup(**{**settings, **changes})


@pytest.fixture(scope="module")
def tables():
    return load()


@pytest.fixture(scope="module")
def reported(tables):
    objects, fovs, _ = tables
    return report(objects, fovs, synthetic_setup())


def test_the_committed_tables_are_what_the_generator_builds(tables):
    sys.path.insert(0, str(SYNTHETIC))
    try:
        from make_synthetic import build, expected
    finally:
        sys.path.remove(str(SYNTHETIC))
    objects, fovs, truth = tables
    built_objects, built_fovs = build()
    pd.testing.assert_frame_equal(objects, built_objects, check_dtype=False)
    pd.testing.assert_frame_equal(fovs, built_fovs, check_dtype=False)
    assert json.loads(json.dumps(expected(built_objects, built_fovs))) == truth


def test_the_checker_understands_the_tables(tables):
    objects, fovs, truth = tables
    check = check_tables(
        objects,
        fovs,
        feature_cols=list(FEATURES),
        observation_unit="cell",
        control={"condition": "control"},
    )
    assert check.ok and not check.errors
    assert [w.title for w in check.warnings] == [
        f"{truth['counts']['empty_fields']} fields of the FOV table hold no observation."
    ]
    assert (
        "A field is identified by 'image_file' within condition, biological_repeat, "
        "technical_repeat." in check.understood
    )


def test_the_overview_counts_what_was_built(reported, tables):
    counts = tables[2]["counts"]
    text = reported.document.overview
    assert (
        f"{counts['cells']} cells in {counts['fields']} acquired fields, "
        f"{counts['empty_fields']} of them without cells" in text
    )
    assert f"{counts['technical_repeats']} technical repeats" in text


def test_field_to_field_variation_is_recovered(reported, tables):
    truth = tables[2]["truth"]
    groups = reported.results["variability"].groups
    assert len(groups) == tables[2]["counts"]["analysis_groups"]
    assert (groups["status"] == "ok").all()
    # 30 wells of 5 fields: the median ICC is within 0.08 of the truth.
    assert groups["icc"].median() == pytest.approx(truth["icc_within_well"], abs=0.08)
    assert groups["design_effect"].median() == pytest.approx(
        truth["design_effect_within_well"], rel=0.4
    )
    # 150 fields with cells: the share of unusual fields within 0.1 of its prediction.
    assert groups["unusual_fov_share"].mean() == pytest.approx(truth["unusual_fov_share"], abs=0.1)


def test_a_fixed_count_above_the_acquired_fields_is_never_met(reported):
    enough = reported.results["sufficiency"].enough
    fixed = enough[enough["criterion"] == "fixed_fov"]
    assert len(fixed) == 30 and (fixed["sampled_enough"] == "no").all()
    assert fixed["answer"].str.contains("not met by the 6 acquired FOVs").all()


def test_the_drug_effect_is_found_in_every_repeat_with_fields_as_the_unit(reported, tables):
    truth = tables[2]["truth"]
    contrast = reported.results["contrast"].table
    assert list(contrast["biological_repeat"]) == tables[2]["design"]["repeats"]
    assert (contrast["status"] == "ok").all()
    # Within a repeat the effect over the cell SD; 5 wells a side move it by about 0.2.
    assert (
        contrast["d_objects"]
        .between(truth["cohen_d_cells"] - 0.5, truth["cohen_d_cells"] + 0.5)
        .all()
    )
    assert (contrast["d_fields"] > contrast["d_objects"]).all()
    assert (contrast["permutation_unit"] == "technical_repeat").all()
    assert (contrast["p_bonferroni"] < 0.05).all()


def test_only_the_affected_feature_is_ranked(reported, tables):
    ranking = reported.results["ranking"]
    assert ranking.top() == [tables[2]["answers"]["ranking_top"]]
    unranked = ranking.ranking[ranking.ranking["score"].isna()]["feature"]
    assert sorted(unranked) == sorted(tables[2]["answers"]["ranking_unranked"])


def test_every_empty_field_is_listed_for_inspection(reported, tables):
    fields = reported.results["inspection"].fovs
    empty = fields[fields["detail"].str.startswith("no cells")]
    assert len(empty) == tables[2]["counts"]["empty_fields"]
    assert set(empty["image_file"]) == {"field_04.tif"}


def test_the_planted_rare_cells_are_unusual_and_clustered(reported, tables):
    rare = tables[2]["truth"]["rare"]
    unusual = reported.results["unusual"]
    objects = tables[0]
    planted = objects["rare"].to_numpy()
    assert unusual.labels[planted].all()  # every planted cell is far from the control
    clustering = fov_clustering(objects, tables[1], label_col="rare", states=[True])
    table = clustering.table
    host = table[
        (table["biological_repeat"] == rare["repeat"]) & (table["technical_repeat"] == rare["well"])
    ]
    assert int(host["state_objects"].iloc[0]) == tables[2]["counts"]["rare_cells"]
    assert int(host["fovs_with_state"].iloc[0]) == tables[2]["counts"]["rare_fields"]
    assert float(host["p_clustered"].iloc[0]) <= 0.05
    assert (table.drop(host.index)["state_objects"] == 0).all()


def test_the_effort_projection_recovers_the_control_components(tables):
    objects, fovs, truth = tables
    plan = optimise(
        objects,
        fovs,
        synthetic_setup(rules=(FixedFOVStrategy(4),)),
        fov_counts=[2, 3],
        reservoir_fraction=0.34,
        n_splits=2,
        draws=10,
        permuted_orders=1,
        dispersed_orders=0,
    )
    assert {"protocol", "effort"} <= set(plan.results)
    components = plan.results["effort"].components
    control = components[components["condition"] == "control"].set_index("level")["estimate"]
    variance = truth["truth"]["variance"]
    # 1 128 control cells: the cell variance within 15%; 75 fields and 15 wells, wider.
    assert control["observation"] == pytest.approx(variance["cell"], rel=0.15)
    assert control["image_file"] == pytest.approx(variance["field"], rel=0.6)
    assert control["technical_repeat"] == pytest.approx(variance["well"], rel=1.0)


def test_a_live_session_stops_every_well_at_the_declared_count(tables, tmp_path):
    objects, fovs, truth = tables
    stop_at = truth["answers"]["fixed_4_fields_live_stop"]
    session = live_session(synthetic_setup(rules=(FixedFOVStrategy(stop_at),)))
    folder = LiveFolder(tmp_path, session)
    stops: dict[tuple, int] = {}
    for _, field in fovs.sort_values("acquisition_order").iterrows():
        mine = objects.merge(field[FIELD_KEY].to_frame().T, on=FIELD_KEY)
        write_fov(tmp_path, mine, field[FIELD_KEY].to_dict())
        for update in folder.poll():
            for result in update.results:
                if result.decision.value == "stop":
                    stops.setdefault(tuple(result.group.values()), result.n_fovs)
    assert len(stops) == truth["counts"]["analysis_groups"]
    assert set(stops.values()) == {stop_at}


def test_plots_draw_the_readout_with_its_ranking(reported, tables):
    objects, fovs, _ = tables
    figures = plots(
        objects, fovs, synthetic_setup(), ranking=reported.results["ranking"], n_boot=50
    )
    assert list(figures) == ["area"]  # the readout is the best-ranked feature
    assert figures["area"][0].axes[0].get_ylabel() == "area"


def test_normalising_to_the_control_puts_every_control_mean_at_one(tables):
    objects, fovs, _ = tables
    setup = synthetic_setup(normalize="control_mean")
    prepared = setup.prepared(objects, fovs)
    control = prepared[prepared["condition"] == "control"]
    assert np.allclose(control.groupby("biological_repeat")["area"].mean(), 1.0)
