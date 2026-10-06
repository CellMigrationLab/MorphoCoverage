"""Plan v2, Phase 1 corrections (owner, 2026-10-06): no held-back field in a fit, the wording
of the reviews, the three-repeat verdict, the declared scale, the exploratory labels and the
report without a control."""

import sys
from pathlib import Path

import numpy as np
import pytest
from test_feature_tables_wide import _wide

from morphocoverage._wording import WITHIN_FAMILY
from morphocoverage.embedding import robust_components
from morphocoverage.multivariate_outcomes import ComponentsError, CovarianceError, QuantileError
from morphocoverage.retrospective.criteria import OutcomeCriterion
from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.workflows import Setup, optimise
from morphocoverage.workflows._acquisition_gap import gap_caution
from morphocoverage.workflows._fingerprint_space import fingerprint_space, noun
from morphocoverage.workflows._guided_ask import ask_tolerance

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT / "data" / "lab"))
import catalog


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
    assert "each condition has one well there, no within-repeat technical replication" in joined


def test_the_docs_give_the_exact_chance_of_one_direction_in_every_repeat():
    docs = " ".join((DOCS / "report" / "conditions.md").read_text().split())
    assert "the exact chance that all `n` repeats go the same way is `2^(1-n)`: 25% at 3" in docs
    assert "12.5% at 4" in docs and "3.1% at 6" in docs


def test_the_whole_table_correction_is_described_as_asymptotic_not_exact():
    docs = " ".join((DOCS / "report" / "conditions.md").read_text().split())
    assert "asymptotic high-dimensional, low-sample-size (HDLSS) correlation correction" in docs
    assert "not an exact one" in docs
