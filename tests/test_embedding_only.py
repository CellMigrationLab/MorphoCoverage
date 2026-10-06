"""Embedding-only reports (owner, 2026-10-06): a Setup with no explainable feature and no
readout, the other feature tables alone, through report, html_report, scan_design, guide,
optimise and live; what needs explainable features said once each with ➖, the whole-table
test the only test of "Do the conditions differ?", unusual observations on the tables."""

import re
import warnings

import numpy as np
import pandas as pd
import pytest

from morphocoverage.multivariate_outcomes import CovarianceError
from morphocoverage.objectives import RareTarget
from morphocoverage.retrospective.criteria import OutcomeCriterion
from morphocoverage.strategies import FOVBootstrapStrategy
from morphocoverage.transforms import Standardize
from morphocoverage.workflows import (
    Setup,
    guide,
    html_report,
    live_session,
    optimise,
    report,
    scan_design,
)

KEY = ["condition", "biological_repeat", "image_file", "label_id"]


def _cells(seed: int = 0, columns: int = 12):
    """3 conditions in 3 repeats, 4 fields of 6 labelled cells; an embedding of ``columns``
    columns with condition ``a`` shifted; a 0/1 label in the observation table."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "condition": c,
            "biological_repeat": f"R{r}",
            "image_file": f"{c}_{r}_{f}.tif",
            "label_id": k,
        }
        for c in ("ctrl", "a", "b")
        for r in range(3)
        for f in range(4)
        for k in range(1, 7)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects[KEY[:3]].drop_duplicates().reset_index(drop=True)
    values = rng.normal(size=(len(objects), columns))
    values[objects["condition"].eq("a").to_numpy(), :2] += 1.0
    table = pd.concat(
        [objects[KEY], pd.DataFrame(values, columns=[f"e{j}" for j in range(columns)])], axis=1
    )
    objects["mitotic"] = (rng.random(len(objects)) < 0.2).astype(int)
    return objects, fovs, table


@pytest.fixture(scope="module")
def run():
    objects, fovs, table = _cells()
    setup = Setup(control={"condition": "ctrl"}, observation_unit="cell")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup, feature_tables={"DINO": table})
    return objects, fovs, table, result


def test_a_setup_without_readout_declares_no_explainable_feature():
    setup = Setup(control={"condition": "ctrl"})
    assert setup.embedding_only and setup.columns == [] and setup.stopping_rules() == ()
    with pytest.raises(ValueError, match="declare the readout among them"):
        Setup(features=("area",))
    # normalize centres the tables per repeat, never divides them (owner, 2026-10-06).
    centred = Setup(control={"condition": "ctrl"}, normalize="control_mean")
    assert centred.prepared(pd.DataFrame({"x": [1.0]}), pd.DataFrame()) is not None


def test_without_tables_an_embedding_only_report_is_refused():
    objects, fovs, _ = _cells()
    with pytest.raises(ValueError, match="pass them, feature_tables=") as error:
        report(objects, fovs, Setup(control={"condition": "ctrl"}, observation_unit="cell"))
    assert "readout" in str(error.value)


def test_what_needs_explainable_features_is_said_once_each_with_the_marker(run):
    *_, result = run
    full = result.summary(detail="full")
    short = result.summary()
    assert full.count("mean-precision rules are not applicable") == 1
    assert (
        short.count("➖ Per feature (the discovery, the heatmap, SuperPlots): not applicable") == 1
    )
    assert "➖ Foundation. Why count fields? — not applicable: no explainable features" in short
    assert "➖ 6. Which fields should I look at? — not applicable: no explainable features" in short
    assert result.discovery is None and "contrast" not in result.results


def test_the_whole_table_test_is_the_answer_to_do_the_conditions_differ(run):
    *_, result = run
    section = result.document.sections["conditions"]
    first = section.strip().splitlines()[0]
    assert first.startswith("The whole tables, each condition against its own control")
    assert "- a vs ctrl: shifted the same way in 3 of 3 repeats" in " ".join(section.split())
    assert result.results["whole_table"].tests["condition"].tolist() == ["a", "b"]


def test_unusual_observations_read_the_tables_and_say_so(run):
    *_, result = run
    unusual = result.results["unusual"]
    assert unusual.settings["feature_cols"][:2] == ["e0", "e1"]
    assert "the 12 columns of DINO (there are no explainable features)" in unusual.settings["space"]
    assert "on the 12 columns of DINO" in " ".join(unusual.summary().split())


def test_the_page_is_built_without_the_per_feature_parts(run, tmp_path):
    objects, fovs, table, result = run
    page = html_report(result, objects, fovs, tmp_path, feature_tables={"DINO": table})
    text = re.sub(r"<script.*?</script>", "", page.read_text(), flags=re.DOTALL)
    visible = re.sub(r"<[^>]+>", " ", text)
    assert "The whole tables" in visible and "None" not in visible
    assert (tmp_path / "figures" / "tests_whole_table.csv").exists()
    assert "Every feature against the control" not in visible


def test_the_scan_checks_the_tables_when_there_is_no_numeric_feature():
    objects, fovs, table = _cells()
    scan = scan_design(
        objects.drop(columns="mitotic"),
        fovs,
        observation_unit="cell",
        feature_tables={"DINO": table},
    )
    assert scan.data_check.ok and scan.settings["feature_cols"] == []


def test_the_guided_session_skips_features_and_readout_and_offers_the_aim(tmp_path):
    objects, fovs, table = _cells()
    said: list[str] = []
    # Loaded, a cell, one control (ctrl), no positive control, mitotic, both, 50%, stop.
    answers = iter(["y", "1", "", "", "", "1", "3", "0.5", "2"])
    result = guide(
        objects,
        fovs,
        folder=tmp_path,
        ask=lambda prompt: next(answers),
        say=said.append,
        feature_tables={"DINO": table},
    )
    assert result.answers is not None and result.answers.readout is None
    assert result.answers.aim == "fingerprint_label" and result.answers.rare_label == "mitotic"
    text = "\n".join(said)
    assert "Which features do you want to assess?" not in text
    assert "With no explainable features, what should the imaging get right?" in text
    setup = result.setup
    assert setup.embedding_only and setup.rare_label == "mitotic"
    assert [c.description for c in setup.fingerprint] == [
        "the covariance of the standardised DINO columns within 50%"
    ]


def test_optimise_judges_the_tables_fingerprint_and_refuses_without_a_target():
    objects, fovs, table = _cells()
    names = tuple(f"e{j}" for j in range(12))
    merged = objects.merge(table, on=KEY)
    criterion = OutcomeCriterion(CovarianceError(names, True), 0.5, "lower_is_better", "DINO")
    setup = Setup(
        control={"condition": "ctrl"},
        observation_unit="cell",
        feature_tables={"DINO": names},
        fingerprint=(criterion,),
        rare=RareTarget(
            detection_probability=0.95,
            min_fov_occurrences=2,
            object_prevalence=0.2,
            phenotype="mitotic",
        ),
        rare_label="mitotic",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = optimise(merged, fovs, setup, fov_counts=[2], reservoir_fraction=0.5)
    criteria = set(plan.results["reliability"].curve["criterion"])
    assert len(criteria) == 2  # DINO on its components, and the label
    assert "not applicable: no stopping rule, as there is no readout" in plan.summary()
    bare = Setup(
        control={"condition": "ctrl"}, observation_unit="cell", feature_tables={"DINO": names}
    )
    with pytest.raises(ValueError, match="has none it can judge"):
        optimise(merged, fovs, bare, fov_counts=[2], reservoir_fraction=0.5)


def test_live_accepts_rules_on_the_tables_and_refuses_without_them():
    names = tuple(f"e{j}" for j in range(12))
    setup = Setup(observation_unit="cell", feature_tables={"DINO": names})
    with pytest.raises(ValueError, match="live_session needs stopping rules"):
        live_session(setup)
    rule = FOVBootstrapStrategy("covariance", 0.5, Standardize(), 20, 0, 3)
    session = live_session(setup, rules=(rule,))
    assert list(session.feature_cols) == list(names)
