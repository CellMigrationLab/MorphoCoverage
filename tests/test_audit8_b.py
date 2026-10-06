"""Regression tests of audit 8B (2026-10-06): the paths the matrix of tables, controls,
normalisation, units and targets found crashing or printing what did not happen."""

import re
import warnings

import pytest
from test_embedding_only import _cells

from morphocoverage.objectives import RareTarget
from morphocoverage.workflows import Setup, html_report, report

MITOTIC = RareTarget(
    detection_probability=0.95, min_fov_occurrences=2, object_prevalence=0.2, phenotype="mitotic"
)


def _visible(page: str) -> str:
    text = re.sub(r"<script.*?</script>", "", page, flags=re.DOTALL)
    return " ".join(re.sub(r"<[^>]+>", " ", text).split())


def _embedding_only(**fields):
    objects, fovs, table = _cells()
    setup = Setup(control={"condition": "ctrl"}, observation_unit="cell", **fields)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup, feature_tables={"DINO": table})
    return objects, fovs, table, result


@pytest.mark.parametrize(
    "fields",
    [{"normalize": "control_mean"}, {"rare": MITOTIC, "rare_label": "mitotic"}],
    ids=["normalize", "rare_label"],
)
def test_b1_b2_an_embedding_only_page_is_built_with_normalize_or_a_rare_label(fields, tmp_path):
    objects, fovs, table, result = _embedding_only(**fields)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        page = html_report(result, objects, fovs, tmp_path, feature_tables={"DINO": table})
    visible = _visible(page.read_text())
    assert "The whole tables" in visible and "Plan the next experiment" in visible
    if "normalize" in fields:  # the populations counted the centred values: said on them
        assert (tmp_path / "figures" / "separation.csv").exists()
