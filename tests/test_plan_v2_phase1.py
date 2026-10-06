"""Plan v2, Phase 1 corrections (owner, 2026-10-06): no held-back field in a fit, the wording
of the reviews, the three-repeat verdict, the declared scale, the exploratory labels and the
report without a control."""

import html
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_discovery import _tables as _discovery_tables
from test_feature_tables_wide import _wide
from test_guided import Script as GuidedScript
from test_guided import _cells as guided_cells

from morphocoverage._control import NO_CONTROL
from morphocoverage._wording import WITHIN_FAMILY
from morphocoverage.embedding import robust_components
from morphocoverage.multivariate_outcomes import ComponentsError, CovarianceError, QuantileError
from morphocoverage.retrospective.criteria import OutcomeCriterion
from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.sampling_unit.discovery import feature_discovery
from morphocoverage.sampling_unit.reading_text import LIMITED_REPLICATION
from morphocoverage.statistics import repeat_effect
from morphocoverage.workflows import Setup, guide, load_answers, optimise, report
from morphocoverage.workflows._acquisition_gap import gap_caution
from morphocoverage.workflows._exploratory import (
    VALIDATION,
    exploratory_html,
    exploratory_line,
    exploratory_names,
)
from morphocoverage.workflows._fingerprint_space import fingerprint_space, noun
from morphocoverage.workflows._guided_ask import ask_scales, ask_tolerance
from morphocoverage.workflows._guided_controls import contrast_lines
from morphocoverage.workflows._html_across import _across, across_frame

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT / "data" / "lab"))
sys.path.insert(0, str(ROOT / "data" / "scenarios"))
import catalog
import make_scenarios


@pytest.fixture(scope="module")
def screen():
    """The lab's filopodia screen: 3 biological repeats, one coverslip a side and repeat."""
    data = catalog.load("cct8_filopodia_screen")
    return condition_contrast(
        *data.tables, feature="Filopodia number", control=data.control, observation_unit="cell"
    )


# --- 1. The fingerprint's components are fitted on the fields not held back ---------------


def test_no_held_back_field_enters_the_fit_of_the_fingerprints_components(monkeypatch):
    merged, fovs, names = _wide(fields=8, cells=5)
    criterion = OutcomeCriterion(CovarianceError(names, True), 0.5, "lower_is_better", "DINO")
    setup = Setup(
        readout="area",
        features=("area", "shape"),
        fingerprint=(criterion,),
        feature_tables={"DINO": names},
        observation_unit="cell",
    )
    pooled = fingerprint_space(merged, fovs, setup).setup.fingerprint[0].metric
    assert isinstance(pooled, ComponentsError)
    calls = []
    evaluate = ComponentsError.evaluate

    def spy(self, full, stopped):
        calls.append((self, full.objects.copy()))
        return evaluate(self, full, stopped)

    monkeypatch.setattr(ComponentsError, "evaluate", spy)
    optimise(
        merged,
        fovs,
        setup,
        fov_counts=(2, 4),
        reservoir_fraction=0.35,
        n_splits=2,
        draws=2,
        permuted_orders=1,
        dispersed_orders=0,
    )
    assert calls
    k = len(pooled.loadings)
    for metric, reference in calls:
        # The held-back fields that judge the stop, and the group they belong to.
        held = set(reference["image_file"])
        same = merged[["condition", "biological_repeat"]].eq(
            reference[["condition", "biological_repeat"]].iloc[0]
        )
        group = merged[same.all(axis=1)]
        training = group[~group["image_file"].isin(held)]
        assert held and not held & set(training["image_file"])
        centre, _scale, loadings = robust_components(training[list(names)].to_numpy(float), k)
        assert np.allclose(metric.loadings, loadings) and np.allclose(metric.centre, centre)
        assert not np.allclose(metric.loadings, pooled.loadings)


def test_a_metric_without_fitted_state_is_its_own_trained_copy():
    metric = CovarianceError(("area", "shape"), True)
    criterion = OutcomeCriterion(metric, 0.5, "lower_is_better")
    assert criterion.trained(None) is criterion  # type: ignore[arg-type]


# --- 2. Wording ----------------------------------------------------------------------------


def test_the_two_fold_gap_is_said_to_be_an_engineering_threshold_for_quality_control():
    caution = gap_caution([{"controls": ("a", "b"), "feature": "intensity", "fold": 13.0}])
    assert caution.endswith("2-fold is an engineering threshold for quality control, not a test.")
    docs = (DOCS / "report" / "conditions.md").read_text()
    assert "engineering threshold for quality" in docs


def test_a_covariance_criterion_is_called_the_covariance_fingerprint():
    covariance = OutcomeCriterion(CovarianceError(("a", "b"), True), 0.1, "lower_is_better")
    quantiles = OutcomeCriterion(QuantileError(("a",), True, (0.1, 0.9)), 0.1, "lower_is_better")
    assert noun(covariance) == "covariance fingerprint"
    assert noun(quantiles) == "fingerprint"
    said = []
    ask_tolerance(lambda _: "", said.append, ["a", "b"])
    assert said[0].startswith("The fingerprint judged is a covariance fingerprint: the covariance")
    assert "not the whole distribution" in said[0]


def test_the_report_says_once_that_the_false_discovery_rate_is_within_each_family(screen):
    joined = " ".join(screen.summary().split())
    assert joined.count(WITHIN_FAMILY) == 1
    docs = (DOCS / "common" / "what_happens.md").read_text()
    assert "controls the false discovery rate within each family" in " ".join(docs.split())


def test_one_well_per_condition_is_no_within_repeat_technical_replication(screen):
    joined = " ".join(screen.summary().split())
    said = "its control have one well in each repeat, no within-repeat technical replication"
    assert said in joined


def test_the_docs_give_the_exact_chance_of_one_direction_in_every_repeat():
    docs = " ".join((DOCS / "report" / "conditions.md").read_text().split())
    assert "the exact chance that all `n` repeats go the same way is `2^(1-n)`: 25% at 3" in docs
    assert "12.5% at 4" in docs and "3.1% at 6" in docs


def test_the_whole_table_correction_is_described_as_asymptotic_not_exact():
    docs = " ".join((DOCS / "report" / "conditions.md").read_text().split())
    assert "asymptotic high-dimensional, low-sample-size (HDLSS) correlation correction" in docs
    assert "not an exact one" in docs


# --- 3. Three repeats: the repeat contrasts, the effect and its t interval ----------------


@pytest.mark.parametrize("values", ["normalised", "raw"])
def test_the_t_interval_of_three_repeats_covers_its_level(values):
    rng = np.random.default_rng(11)
    truth, covered, reps = 0.3, 0, 4000
    for _ in range(reps):
        effect = repeat_effect(values, rng.normal(truth, 0.4, size=3))
        target = np.exp(truth) if values == "normalised" else truth
        covered += effect["ci_low"] <= target <= effect["ci_high"]
    # 95% nominal; the Monte Carlo SE of the share is about 0.0035.
    assert 0.935 <= covered / reps <= 0.965


def test_three_repeats_show_each_repeats_change_and_the_interval_once_said_limited(screen):
    row = screen.repeats.set_index("condition").loc["siCCT8#5"]
    assert row["repeats"] == 3 and np.isfinite(row["ci_low"]) and row["ci_low"] < row["ci_high"]
    assert row["repeat_contrasts"].count(": ") == 3
    joined = " ".join(screen.summary().split())
    assert "in the 3 repeats, ×0.56 overall, 95% t interval ×0.35 to ×0.917" in joined
    assert joined.count("very limited biological replication") == 1
    assert LIMITED_REPLICATION in joined


def test_more_repeats_show_no_limited_replication_line():
    objects, fovs = _discovery_tables(n_repeats=4)
    contrast = condition_contrast(
        objects, fovs, feature="area", control={"condition": "ctrl"}, n_permutations=99
    )
    joined = " ".join(contrast.summary().split())
    assert "very limited biological replication" not in joined
    assert "in the 3 repeats" not in joined
    assert np.isfinite(contrast.repeats["ci_low"]).all()  # the CSV carries it at any n


def test_the_discovery_and_its_csv_carry_each_repeats_change_and_the_interval():
    objects, fovs = _discovery_tables(n_repeats=3)
    result = feature_discovery(
        objects,
        fovs,
        features=["area", "noise"],
        control={"condition": "ctrl"},
        n_permutations=999,
        observation_unit="cell",
    )
    tests = result.tests.set_index("feature")
    assert {"repeat_contrasts", "ci_low", "ci_high"} <= set(result.tests.columns)
    area = tests.loc["area"]
    assert area["ci_low"] < area["change"] < area["ci_high"]
    assert area["repeat_contrasts"].startswith("r0: ")
    text = result.summary()
    assert "change in each repeat" in text and "95% t interval" in text


def test_the_page_table_and_its_csv_carry_each_repeats_change_and_the_interval(tmp_path):
    objects, fovs = _discovery_tables(n_repeats=3)
    discovery = feature_discovery(
        objects,
        fovs,
        features=["area", "noise"],
        control={"condition": "ctrl"},
        n_permutations=999,
        observation_unit="cell",
    )
    result = SimpleNamespace(
        discovery=discovery,
        results={"contrast": None},
        setup=SimpleNamespace(readout="area", unit_word="cell"),
    )
    frame = across_frame(result)
    assert {"repeat_contrasts", "ci_low", "ci_high"} <= set(frame.columns)
    markup, _ = _across(result, tmp_path)
    assert "<th>change in each repeat</th><th>95% t interval</th>" in markup
    assert html.escape(LIMITED_REPLICATION) in markup
    saved = pd.read_csv(tmp_path / "tests_across_repeats.csv")
    assert {"repeat_contrasts", "ci_low", "ci_high"} <= set(saved.columns)


# --- 4. The scale declared per feature ------------------------------------------------------


def test_the_setup_declares_a_scale_per_feature_and_refuses_what_it_cannot_use():
    setup = Setup(
        readout="area",
        features=("area", "velocity"),
        scales={"area": "ratio", "velocity": "difference"},
    )
    assert setup.scale_of("area") == "ratio" and setup.scale_of("velocity") == "difference"
    assert setup.scale_of("shape") is None  # undeclared: the automatic rule
    with pytest.raises(ValueError, match="'ratio' or 'difference'"):
        Setup(readout="area", features=("area",), scales={"area": "log"})
    with pytest.raises(ValueError, match="not declared explainable features"):
        Setup(readout="area", features=("area",), scales={"speed": "ratio"})


def test_a_declared_ratio_on_a_value_that_is_not_positive_is_refused():
    objects, fovs = _discovery_tables(n_repeats=3)
    objects.loc[0, "area"] = -1.0
    with pytest.raises(ValueError, match="declared on the ratio scale.*not positive"):
        condition_contrast(
            objects, fovs, feature="area", control={"condition": "ctrl"}, scale="ratio"
        )
    setup = Setup(readout="area", control={"condition": "ctrl"}, scales={"area": "ratio"})
    with pytest.raises(ValueError, match="'area' is declared on the ratio scale"):
        report(objects, fovs, setup)


def test_a_declared_difference_replaces_the_automatic_ratio():
    objects, fovs = _discovery_tables(n_repeats=3)
    automatic = condition_contrast(objects, fovs, feature="area", control={"condition": "ctrl"})
    declared = condition_contrast(
        objects, fovs, feature="area", control={"condition": "ctrl"}, scale="difference"
    )
    assert automatic.repeats["fold_change"].notna().all()
    assert declared.repeats["fold_change"].isna().all()
    assert declared.repeats["q_paired_t_normalised"].isna().all()
    joined = " ".join(declared.summary().split())
    assert "the differences from each repeat's control, as declared in the Setup" in joined
    assert "the raw q: the feature is declared on the difference scale" in joined
    discovery = feature_discovery(
        objects,
        fovs,
        features=["area", "noise"],
        control={"condition": "ctrl"},
        n_permutations=999,
        observation_unit="cell",
        scales={"area": "difference"},
    )
    tests = discovery.tests.set_index("feature")
    assert tests.loc["area", "paired_values"] == "raw"
    assert tests.loc["noise", "paired_values"] == "normalised"  # the automatic rule
    assert "declared in the Setup: area, the difference" in " ".join(discovery.summary().split())


def test_the_guided_session_proposes_ratios_for_intensities_and_areas_and_saves_them(tmp_path):
    objects = pd.DataFrame(
        {"area": [1.0, 2.0], "Mean intensity": [3.0, 4.0], "shift": [-1.0, 2.0], "len": [1, 2]}
    )
    said = []
    yes = ask_scales(lambda _: "", said.append, objects, ["area", "Mean intensity", "shift"])
    assert yes == {"area": "ratio", "Mean intensity": "ratio"}  # shift can be negative
    no = ask_scales(lambda _: "n", said.append, objects, ["area", "len"])
    assert no == {"area": "difference"}
    assert ask_scales(lambda _: "", said.append, objects, ["len"]) == {}
    script = GuidedScript(["y", "", "", "", "", "", "1,3", "area"], scales=False)
    script.answers.insert(8, "n")  # the scale question, after the readout
    result = guide(
        guided_cells(), folder=tmp_path, ask=script.ask, say=script.say, run_report=False
    )
    assert any(p.startswith("Test the changes of area") for p in script.prompts)
    assert result.answers.scales == {"area": "difference"}
    assert result.setup.scales == {"area": "difference"}
    assert load_answers(tmp_path).scales == {"area": "difference"}
    assert "Tested as differences to the control: area." in result.answers.lines()


# --- 5. Exploratory labels -------------------------------------------------------------------


def _scenario_report(name):
    loaded = make_scenarios.load(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        setup = Setup(**loaded.setup)
        return report(loaded.objects, loaded.fovs, setup, feature_tables=loaded.feature_tables)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "a24_expl_one_norm_image_none_fp_expl",
            {
                "enough": "The covariance fingerprint is exploratory",
                "conditions": "Beyond the mean is exploratory",
                "populations": "Population enrichment is exploratory",
            },
        ),
        ("a02_expl_several_norm_cells_tech_positive", {"enough": "Novelty is exploratory"}),
        ("a01_tagged_several_image_none_rare", {"enough": "The rare target is exploratory"}),
    ],
)
def test_each_exploratory_analysis_is_said_once_in_its_section(name, expected):
    result = _scenario_report(name)
    document = result.document
    for key, start in expected.items():
        for form in (document.sections[key], document.short[key]):
            joined = " ".join(form.split())
            assert joined.count("exploratory: not yet validated") == 1, key
            assert f"{start}: not yet validated by the benchmark of {VALIDATION}." in joined
    said = " ".join(result.summary().split())
    assert said.count("exploratory: not yet validated") == len(
        exploratory_names(result.results, result.setup)
    )


def test_the_page_says_it_where_the_section_text_is_not_printed():
    result = SimpleNamespace(
        results={"distribution": object(), "populations": None},
        setup=Setup(readout="area"),
    )
    assert "Beyond the mean is exploratory" in exploratory_html(result, "conditions")
    assert exploratory_html(result, "populations") == ""
    assert exploratory_line(["the rare target", "novelty"]).startswith(
        "The rare target and novelty are exploratory"
    )


# --- 6. Without a control -------------------------------------------------------------------


def test_without_a_control_requested_contrasts_are_said_not_yet_supported():
    objects, fovs = _discovery_tables(n_repeats=3)
    result = report(objects, fovs, Setup(readout="area", observation_unit="cell"))
    said = " ".join(result.summary().split())
    sentence = (
        "not applicable: no control is declared: comparisons are available only for explicitly "
        "requested contrasts estimable from the repeat design, which the report does not "
        "support yet"
    )
    assert sentence in said
    section = " ".join(result.document.sections["conditions"].split())
    assert section.count("explicitly requested contrasts") == 1  # one sentence, its section
    assert "descriptive only" not in said and "described, not tested" not in said
    assert contrast_lines(None, None, None) == [f"{NO_CONTROL[:1].upper()}{NO_CONTROL[1:]}."]
