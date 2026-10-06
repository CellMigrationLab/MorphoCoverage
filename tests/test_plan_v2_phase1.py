"""Plan v2, Phase 1 corrections (owner, 2026-10-06): no held-back field in a fit, the wording
of the reviews, the three-repeat verdict, the declared scale, the exploratory labels and the
report without a control."""

import numpy as np
from test_feature_tables_wide import _wide

from morphocoverage.embedding import robust_components
from morphocoverage.multivariate_outcomes import ComponentsError, CovarianceError
from morphocoverage.retrospective.criteria import OutcomeCriterion
from morphocoverage.workflows import Setup, optimise
from morphocoverage.workflows._fingerprint_space import fingerprint_space

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
