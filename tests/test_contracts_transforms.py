"""Feature-transform contracts: no future-FOV leakage and JSON provenance.
"""


from dataclasses import replace

import pandas as pd
import pytest
from contract_registry import (
    SCENARIO_IDS,
    TRANSFORM_IDS,
    TRANSFORMS_UNDER_CONTRACT,
    replay,
)
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    Estimator,
    Evidence,
    SamplingStrategy,
    transforms,
)

# --- 5. Feature transform contracts --------------------------------------------


def test_every_package_transform_is_under_contract():
    exported = {
        obj
        for obj in vars(transforms).values()
        if isinstance(obj, type)
        and issubclass(obj, transforms.FeatureTransform)
        and obj is not transforms.FeatureTransform
    }
    covered = {type(transform) for transform in TRANSFORMS_UNDER_CONTRACT}
    for transform in TRANSFORMS_UNDER_CONTRACT:
        if isinstance(transform, transforms.Chain):
            covered |= {type(step) for step in transform.transforms}
    missing = sorted(cls.__name__ for cls in exported - covered)
    assert not missing, f"transforms missing from TRANSFORMS_UNDER_CONTRACT: {missing}"


class _TransformProbe(Estimator):
    """Fits a transform on what it is given and reports the fit and the output."""

    name = "transform_probe"

    def __init__(self, transform):
        self.transform = transform

    def estimate(self, data, *, objective):
        fitted = self.transform.fit(data)
        output = fitted.apply(data).tolist() if fitted.fitted else None
        return Evidence(diagnostics={"transform": fitted.provenance(), "output": output})


@pytest.mark.parametrize("transform", TRANSFORMS_UNDER_CONTRACT, ids=TRANSFORM_IDS)
@pytest.mark.parametrize("name", SCENARIO_IDS)
def test_transforms_are_unaffected_by_future_fovs(name, transform):
    # AGENTS.md section 3: preprocessing for prospective evaluation must not
    # use future FOVs. Altering every FOV after step k must leave the fitted
    # state and the transformed features at steps <= k unchanged.
    scenario = SCENARIOS[name]()
    probe = [SamplingStrategy([_TransformProbe(transform)])]
    original = replay(scenario, probe)
    order = scenario.fovs.set_index(["condition", "repeat", "image_file"])[ORDER_COL]
    cutoff = int(order.median())
    objects = scenario.objects.copy()
    key = pd.MultiIndex.from_frame(objects[["condition", "repeat", "image_file"]])
    future = order.reindex(key).to_numpy() > cutoff
    objects.loc[future, list(FEATURE_COLS)] = objects.loc[future, list(FEATURE_COLS)] * 7.0 + 50.0
    altered = replay(replace(scenario, objects=objects), probe)
    assert len(original) == len(altered)
    compared = 0
    for before, after in zip(original, altered, strict=True):
        assert before.acquired_fovs == after.acquired_fovs
        latest = order[before.acquired_fovs[-1]]
        if latest <= cutoff:
            assert before.result.diagnostics == after.result.diagnostics
            compared += 1
    assert compared > 0
    if future.any():
        # The alteration is real: the full-acquisition fits differ.
        assert original[-1].result.diagnostics != altered[-1].result.diagnostics


@pytest.mark.parametrize("transform", TRANSFORMS_UNDER_CONTRACT, ids=TRANSFORM_IDS)
def test_fitted_transforms_are_json_provenance(transform):
    import json

    scenario = SCENARIOS["multimodal"]()
    for item in replay(scenario, [SamplingStrategy([_TransformProbe(transform)])]):
        record = item.result.diagnostics["transform"]
        assert json.loads(json.dumps(record)) == record
        assert record["config_id"] == transform.config_id
        assert record["n_fit_fovs"] == item.step


