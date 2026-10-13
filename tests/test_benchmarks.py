"""Tests of ``workflows.benchmarks``: the benchmark workflow, its settings and grid warnings."""

from __future__ import annotations

import pandas as pd
import pytest
from _helpers_workflows import (
    AREA_SETUP,
    QUICK,
    READOUT,
    _acquisition,
    _base,
    _family_score,
    _flat,
    _frontier_rows,
    _setup,
    _sictrl_tables,
    _well_acquisition,
)

import morphocoverage.workflows._shared as shared_module
import morphocoverage.workflows.retrospective as workflow_module
from morphocoverage._style import CAUTION, OK, Status
from morphocoverage.retrospective.frontier import FrontierResult
from morphocoverage.retrospective.frontier_text import frontier_status
from morphocoverage.workflows import (
    Setup,
    benchmark,
    default_families,
    live_session,
    optimise,
    report,
)
from morphocoverage.workflows.benchmarks import Benchmark, grid_warnings


def test_i4_settings_print_the_fields_held_back_and_the_recorded_order():
    objects, fovs = _well_acquisition()
    plan = optimise(objects, fovs, _setup(stage_cols=None), **QUICK)
    low, high = plan.settings["reservoir_fovs"]
    # 25% of the three fields of each well rounds up to one field.
    assert (low, high) == (1, 1)
    text = " ".join(plan.summary().split())  # the settings are wrapped
    assert "1 field of each group held back (reservoir_fraction 25%, rounded up" in text
    assert "the recorded order, 1 random and 0 dispersed orders" in text
    result = benchmark(
        objects,
        fovs,
        _setup(stage_cols=None),
        reservoir_fraction=0.25,
        families=default_families(READOUT, n_resamples=10)[:1],
        n_splits=1,
        permuted_orders=1,
        n_boot=20,
    )
    text = " ".join(result.summary().split())
    assert "1 field of each group held back" in text
    assert "the recorded order and 1 random orders" in text


def test_i9_setup_random_state_seeds_the_default_families():
    for family in default_families(READOUT, n_resamples=10, random_state=7):
        parameters = family.strategy.parameters
        assert parameters.get("random_state", 7) == 7
    assert default_families(READOUT)[2].strategy.parameters["random_state"] == 0


def test_p9_6_benchmark_warns_when_a_grid_does_not_bracket_the_data():
    rows = []
    for setting in (0.001, 0.01, 0.1):
        for run in range(10):
            rows.append(
                {
                    "family": "JCS spread per object",
                    "setting": setting,
                    "stopped": False,
                    "effort": 20,
                }
            )
            rows.append(
                {
                    "family": "JCB per object",
                    "setting": setting,
                    "stopped": run < 5,
                    "effort": 1 + run,
                }
            )
    runs = pd.DataFrame(rows)
    result = FrontierResult(pd.DataFrame(), pd.DataFrame(), runs, {"min_fovs": 1})
    grids = {
        "JCS spread per object": {
            "quantity": "max_descriptor_spread",
            "thresholds": [0.001, 0.01, 0.1],
        },
        "JCB per object": {"quantity": "averaging_cv", "thresholds": [0.001, 0.01, 0.1]},
    }
    bench = Benchmark(
        {"the mean": result},
        ("JCS spread per object", "JCB per object"),
        {"families": grids, "readout": "area"},
    )
    warnings = grid_warnings(bench)
    assert len(warnings) == 1
    assert "the grid of JCS spread per object (max_descriptor_spread 0.001 to 0.1)" in warnings[0]
    assert "does not bracket these data: it never stopped at any setting tried" in warnings[0]
    assert "units of area" in warnings[0] and "widen the grid" in warnings[0]


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


def test_setup_without_any_target_and_benchmark_without_families():
    with pytest.raises(ValueError) as raised:
        Setup(readout="a", tolerances=())
    assert str(raised.value) == "tolerances=() declares no mean target: declare the fingerprint"
    objects, fovs = _sictrl_tables()
    with pytest.raises(TypeError) as raised:
        benchmark(objects, fovs, AREA_SETUP, reservoir_fraction=0.3, families=[])
    assert str(raised.value) == "families must be a non-empty sequence of Family"


@pytest.fixture(scope="module")
def bench():
    objects, fovs = _acquisition()
    return benchmark(
        objects,
        fovs,
        _setup(),
        reservoir_fraction=0.25,
        families=default_families(READOUT, n_resamples=20),
        n_splits=2,
        permuted_orders=2,
        n_boot=50,
    )


def test_benchmark_opens_with_the_worst_status_and_groups_grid_warnings(bench):
    warnings = grid_warnings(bench)
    assert warnings  # the default grids do not bracket these constructed data
    short, full = bench.summary(), bench.summary(detail="full")
    statuses = [frontier_status(result)[0] for result in bench.frontiers.values()]
    assert short.startswith(f"{CAUTION} Benchmark — fields saved against a fixed count")
    assert Status.CAUTION in statuses or warnings
    # Short: the grid advice once, each family's problem on its own line.
    assert _flat(short).count("widen the grid") == 1
    for warning in warnings:
        name = warning.split("the grid of ", 1)[1].split(" (", 1)[0]
        assert f"- {name} (" in short
        assert _flat(warning.removeprefix("Warning: the")) in _flat(full)
    for text in (short, full):
        assert "risk levels 5% and 10% of stopping too early (alphas" in _flat(text)
    for criterion in bench.frontiers:
        assert f"For {criterion}" in short


def test_benchmark_without_a_grid_problem_is_not_cautioned_for_one():
    scored = _frontier_rows([_base(0.05), _family_score("saver", 0.05, 4.0, 2.0, 0.5, 3.0)])
    # The family stopped at its setting, after the first field: its grid brackets the data.
    runs = pd.DataFrame({"family": ["saver"], "setting": [0.1], "stopped": [True], "effort": [3]})
    result = FrontierResult(scored.curve, scored.scores, runs, scored.settings)
    bench = Benchmark({"the mean": result}, ("saver",), {})
    assert not grid_warnings(bench)
    assert bench.summary().startswith(f"{OK} Benchmark")
