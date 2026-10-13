"""Tests of ``workflows._shared``: the workflows' inputs, orders, seeds and replays."""

from __future__ import annotations

import re
import warnings

import pandas as pd
import pytest
from _helpers_sampling_unit import CONTROL, _wells
from _helpers_tables import _plate_tables, _warning_titles
from _helpers_workflows import AREA_SETUP, _acquisition, _setup, _sictrl_tables

import morphocoverage.workflows._shared as shared_module
import morphocoverage.workflows.retrospective as workflow_module
from morphocoverage.retrospective import dataset_overview
from morphocoverage.strategies import FixedFOVStrategy
from morphocoverage.workflows import Setup, report
from morphocoverage.workflows._shared import (
    check_workflow_inputs,
    conditions,
    order_seeds,
    orders,
    replay_rules,
)


def test_i1_setup_random_state_seeds_the_report_random_orders(monkeypatch):
    seen = []
    sufficiency = workflow_module.sufficiency_report

    def spy(*args, **kwargs):
        seen.append([o.seed for o in kwargs["orders"] if o.kind == "permuted"])
        return sufficiency(*args, **kwargs)

    monkeypatch.setattr(workflow_module, "sufficiency_report", spy)
    objects, fovs = _acquisition(n_fovs=6, n_objects=4)
    runs = {}
    for seed in (0, 7):
        runs[seed] = report(objects, fovs, _setup(random_state=seed, stage_cols=None))
    assert seen == [order_seeds(0, 5), order_seeds(7, 5)]
    assert seen[0] == [0, 1, 2, 3, 4] and not set(seen[0]) & set(seen[1])
    assert "seeded from random_state); random_state 7." in " ".join(runs[7].summary().split())


def test_i10_setup_random_state_seeds_the_dispersed_orders():
    seeds = {
        state: [o.seed for o in orders(_setup(random_state=state), 2, 3) if o.kind == "dispersed"]
        for state in (0, 4)
    }
    assert seeds == {0: [0, 1, 2], 4: [12, 13, 14]}
    # The permuted orders of the replays keep 0.. (replay_acquisitions mixes random_state in).
    assert [o.seed for o in orders(_setup(random_state=4), 2, 0) if o.kind == "permuted"] == [0, 1]


def test_i12_replay_rules_checks_that_both_replays_reveal_the_same_fields(monkeypatch):
    objects, fovs = _acquisition(n_fovs=6, n_objects=4)
    setup = _setup(normalize="control_mean", rules=(FixedFOVStrategy(2),), stage_cols=None)
    settings = {
        "reservoir_fraction": 0.34,
        "orders": orders(setup, 2, 0),
        "acquisition_order_col": "acquisition_order",
        "n_splits": 1,
        "random_state": 0,
    }
    prepared = setup.prepared(objects, fovs)
    criteria = setup.criteria()
    kept = replay_rules(
        objects, prepared, fovs, setup, strategies=setup.rules, criteria=criteria, **settings
    )
    assert "revealed_fields" not in set(kept.outcomes["criterion"])
    assert [c["metric"] for c in kept.settings["criteria"]] == [
        c.metric.name for c in setup.criteria()
    ]
    replay = shared_module.replay_acquisitions

    def other_seed_for_the_scores(*args, **kwargs):
        if len(kwargs["criteria"]) > 1:  # the replay of the prepared values
            kwargs = {**kwargs, "random_state": 1}
        return replay(*args, **kwargs)

    monkeypatch.setattr(shared_module, "replay_acquisitions", other_seed_for_the_scores)
    with pytest.raises(RuntimeError, match="revealed different fields"):
        replay_rules(
            objects,
            prepared,
            fovs,
            setup,
            strategies=setup.rules,
            criteria=setup.criteria(),
            **settings,
        )


def test_p5_5_check_warnings_reach_the_workflows_and_the_overview():
    cells, fovs = _plate_tables(n_repeats=1)
    copied = pd.concat([cells, cells], ignore_index=True)  # the cell file read twice
    # One drug well typed 'drug ' in both tables.
    for table in (copied, fovs):
        table.loc[table["technical_repeat"] == "C03", "condition"] = "drug "
    setup = Setup(readout="area", control={"condition": "control"}, observation_unit="cell")
    with pytest.warns(UserWarning, match="check_tables found"):
        _, _, check = check_workflow_inputs(copied, fovs, setup)
    titles = _warning_titles(check)
    assert any("exact copy" in t for t in titles)
    assert any("differ only in spaces" in t for t in titles)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, quiet = check_workflow_inputs(copied, fovs, setup, warn=False)
    result = dataset_overview(
        copied, fovs, feature_cols=["area"], observation_unit="cell", data_check=quiet
    )
    overview = result.summary(detail="full")
    assert "The data check found" in overview and "exact copy" in overview
    # Labels that print alike are quoted, in the sentence and in the (aligned) table.
    assert "'drug', 'drug '" in overview and re.search(r"'drug ' +2026_01_01", overview)
    short = " ".join(result.summary().split())
    assert "⚠️ Data checks: 2 warnings (" in short and "exact copy" in short
    assert "'drug', 'drug '" in short


def test_q1_setup_fields_independent_reaches_the_report_contrast():
    objects, fovs = _wells(1, shift=0.5)
    objects, fovs = objects.drop(columns="technical_repeat"), fovs.drop(columns="technical_repeat")
    for independent, tested in ((False, False), (True, True)):
        setup = Setup("x", features=("y",), control=CONTROL, fields_independent=independent)
        results = conditions(objects, fovs, setup, {})
        assert bool(results["contrast"].table["p_value"].notna().all()) is tested
        assert bool(results["ranking"].tests["p_value"].notna().all()) is tested
    with pytest.raises(TypeError, match="fields_independent"):
        Setup("x", fields_independent="yes")


@pytest.mark.parametrize(
    ("fovs", "setup", "kind", "message"),
    [
        (
            None,
            AREA_SETUP,
            TypeError,
            (
                "the workflows need the FOV table of every acquired field (fovs=None was given); "
                "morphocoverage.schema.fov_table_from_observations rebuilds one, without the "
                "fields that held no observation"
            ),
        ),
        (
            "fovs",
            AREA_SETUP,
            TypeError,
            "objects and fovs must be pandas DataFrames (every acquired FOV in fovs)",
        ),
        ("table", "setup", TypeError, "setup must be a morphocoverage.workflows.Setup"),
    ],
)
def test_workflow_inputs_are_refused_in_order(fovs, setup, kind, message):
    objects, table = _sictrl_tables()
    with pytest.raises(kind) as raised:
        check_workflow_inputs(objects, table if fovs == "table" else fovs, setup)
    assert str(raised.value) == message


def test_workflow_inputs_refuse_controls_absent_from_the_tables():
    objects, fovs = _sictrl_tables()
    setup = Setup(readout="area", control={"condition": {"siCTRL": ["A"], "Z": ["Q"]}})
    with pytest.raises(ValueError) as raised:
        check_workflow_inputs(objects, fovs, setup)
    assert str(raised.value) == (
        "the controls name conditions absent from the tables: 'Q' and 'Z' (values are "
        "matched exactly, case and spaces included)"
    )
