"""Tests of ``workflows._exploratory``: the analyses not yet validated, marked exploratory once."""

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from morphocoverage.workflows import Setup, report
from morphocoverage.workflows._exploratory import (
    VALIDATION,
    exploratory_html,
    exploratory_line,
    exploratory_names,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "scenarios"))
import make_scenarios


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
