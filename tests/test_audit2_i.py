"""Regressions of the second audit, group I: workflows and data flow (findings I-1 to I-12).

Each test names the finding it pins and fails on the code before the fix.
Constructed data only: every test checks a contract of the workflows, not how
well a rule performs.
"""

import inspect

import pandas as pd
import pytest
from _helpers_workflows import READOUT, _acquisition, _setup, _wells

import morphocoverage.workflows.retrospective as workflow_module
from morphocoverage.multivariate_outcomes import CovarianceError
from morphocoverage.results import Decision
from morphocoverage.retrospective import OutcomeCriterion, SamplingHierarchy
from morphocoverage.retrospective.replay import replay_acquisitions
from morphocoverage.strategies import (
    DescriptorSpreadStrategy,
    FixedFOVStrategy,
    SupportNoveltyEstimator,
)
from morphocoverage.transforms import Standardize
from morphocoverage.workflows import (
    Family,
    LiveFolder,
    LiveTables,
    Setup,
    benchmark,
    default_novelty,
    live_session,
    optimise,
    report,
    write_fov,
)
from morphocoverage.workflows._shared import orders, replay_rules

# Absolute spread of the readout's mean over resamples: about 0.3 on the raw
# values (near 10) and 0.03 once divided by the control mean (near 1), so the
# rule stops at other fields on the two scales.
SPREAD = DescriptorSpreadStrategy(READOUT, 0.1, "fov", 20, 0)
RULES = (FixedFOVStrategy(3), SPREAD)
QUICK = {
    "fov_counts": [2],
    "reservoir_fraction": 0.25,
    "n_splits": 1,
    "draws": 2,
    "permuted_orders": 1,
    "dispersed_orders": 0,
}
KEY = ["condition", "biological_repeat", "technical_repeat", "image_file"]


def _tables(n_fovs: int = 6, n_objects: int = 5):
    return _wells(*_acquisition(n_fovs=n_fovs, n_objects=n_objects))


def _pooled(**changes) -> Setup:
    """A setup for pooled groups: the recorded order ties across the repeats it pools."""
    return _setup(acquisition_order_col=None, stage_cols=None, **changes)


def _live_verdicts(objects, fovs, setup):
    """Each rule's own verdict after the last field of every group, fed field by field."""
    ordered = fovs.sort_values(["condition", "biological_repeat", "acquisition_order"]).reset_index(
        drop=True
    )
    tables = LiveTables(live_session(setup))
    parts = [objects.merge(ordered.iloc[[i]][KEY], on=KEY) for i in range(len(ordered))]
    updates = []
    for i in range(len(ordered)):
        updates += tables.update(pd.concat(parts[: i + 1], ignore_index=True), ordered[: i + 1])
    last = {}
    for update in updates:
        for result in update.results:
            key = (
                *(result.group[c] for c in ("condition", "biological_repeat", "technical_repeat")),
            )
            last[(*key, result.config_id)] = result.rule_decision.value
    return last


def test_i1_every_rule_replay_reads_the_raw_values_as_live():
    # 8 fields per well: the per-field spread needs 5 fields with objects (audit 4, P4-6).
    objects, fovs = _tables(n_fovs=16)
    raw_setup = _setup(rules=RULES, stage_cols=None)
    setup = _setup(rules=RULES, stage_cols=None, normalize="control_mean")
    prepared = setup.prepared(objects, fovs)
    settings = {
        "reservoir_fraction": 0.25,
        "orders": orders(setup, 2, 0),
        "acquisition_order_col": "acquisition_order",
        "n_splits": 2,
        "random_state": 0,
        "incremental": True,
    }
    arguments = {
        "feature_cols": setup.columns,
        "strategies": list(RULES),
        "criteria": setup.criteria(),
        "objective": setup.objective(),
        **settings,
    }
    raw = replay_acquisitions(objects, fovs, **arguments)
    normalised = replay_acquisitions(prepared, fovs, **arguments)
    # The two scales give the rule different verdicts, so the test can tell them apart.
    assert not raw.decisions.equals(normalised.decisions)
    workflow = replay_rules(
        objects,
        prepared,
        fovs,
        setup,
        strategies=list(RULES),
        criteria=setup.criteria(),
        **settings,
    )
    pd.testing.assert_frame_equal(workflow.decisions, raw.decisions)
    pd.testing.assert_frame_equal(workflow.outcomes, normalised.outcomes)
    # The report judges each rule on the raw values: the verdicts of the raw setup and
    # of the live session, which never sees normalised values.
    final = report(objects, fovs, setup).results["sufficiency"].final
    expected = report(objects, fovs, raw_setup).results["sufficiency"].final
    columns = ["condition", "biological_repeat", "technical_repeat", "config_id", "verdict"]
    pd.testing.assert_frame_equal(final[columns], expected[columns])
    live = _live_verdicts(objects, fovs, setup)
    for row in final.itertuples(index=False):
        assert (
            row.verdict
            == live[(row.condition, row.biological_repeat, row.technical_repeat, row.config_id)]
        )
    # optimise and benchmark use the same replay: their decisions are the raw ones.
    plan = optimise(objects, fovs, setup, **QUICK)
    plan_raw = optimise(objects, fovs, raw_setup, **QUICK)
    pd.testing.assert_frame_equal(
        plan.results["calibration"].table, plan_raw.results["calibration"].table
    )
    family = [Family("spread", SPREAD, "max_descriptor_spread", (0.05, 0.1, 0.4))]
    bench = {
        name: benchmark(
            objects,
            fovs,
            chosen,
            reservoir_fraction=0.25,
            families=family,
            n_splits=1,
            permuted_orders=1,
            n_boot=10,
        )
        for name, chosen in (("normalised", setup), ("raw", raw_setup))
    }
    first = next(iter(bench["raw"].frontiers))
    pd.testing.assert_frame_equal(
        bench["normalised"].frontiers[first].scores, bench["raw"].frontiers[first].scores
    )
    assert "every stopping rule runs on the raw values" in " ".join(setup.declaration())


def test_i2_a_declared_hierarchy_keeps_one_stratum_per_condition():
    objects, fovs = _tables()
    hierarchy = SamplingHierarchy("image_file", "technical_repeat", "biological_repeat")
    declared = _setup(hierarchy=hierarchy, stage_cols=None)
    assert declared.resolved_effort_strata(fovs) == ["condition"]
    assert _setup(hierarchy=hierarchy, effort_strata=()).resolved_effort_strata(fovs) == []
    plan = optimise(objects, fovs, declared, **QUICK)
    strata = set(plan.results["effort"].components["stratum"])
    assert strata == {"condition=control", "condition=drug"}
    assert "condition" in plan.results["protocol"].strata


def test_i3_optimise_says_why_protocol_or_effort_is_not_assessed():
    objects, fovs = _tables()
    # A hierarchy level (the repeat) pooled by group_cols: no crash, a reason.
    plan = optimise(objects, fovs, _pooled(group_cols=("condition", "technical_repeat")), **QUICK)
    assert "protocol" not in plan.results and "effort" in plan.results
    text = plan.summary()
    assert (
        "Protocol reliability across samples is not assessed" in text
        and "'biological_repeat'" in text
    )
    # A condition stratum pooled: the reason offers explicit pooling.
    plan = optimise(
        objects, fovs, _pooled(group_cols=("biological_repeat", "technical_repeat")), **QUICK
    )
    assert "protocol" not in plan.results
    assert "effort_strata=()" in plan.document.sections["next_time"]
    # The technical repeat left out: the effort section names the column that exists.
    plan = optimise(objects, fovs, _pooled(group_cols=("condition", "biological_repeat")), **QUICK)
    effort = plan.document.sections["effort"]
    assert "leave 'technical_repeat' out" in effort
    # The generic need no longer tells the user to add a column that is there.
    assert "technical_repeat column (a well, a dish) or declare" not in plan.summary()
    # A declared hierarchy on those groups: the effort projection runs, the protocol says why.
    hierarchy = SamplingHierarchy("image_file", "technical_repeat", "biological_repeat")
    declared = _pooled(group_cols=("condition", "biological_repeat"), hierarchy=hierarchy)
    plan = optimise(objects, fovs, declared, **QUICK)
    assert "effort" in plan.results and "protocol" not in plan.results
    assert "pool 'technical_repeat'" in plan.document.sections["next_time"]


def test_i4_livetables_key_is_the_fov_identity_of_the_validator():
    objects, fovs = _tables()
    # The same image_file names recur in every condition, repeat and well.
    assert fovs.duplicated("image_file").any()
    setup = _setup(rules=(FixedFOVStrategy(4),), group_cols=("condition",))
    ordered = fovs.sort_values(["biological_repeat", "acquisition_order"]).reset_index(drop=True)
    tables = LiveTables(live_session(setup))
    parts = [objects.merge(ordered.iloc[[i]][KEY], on=KEY) for i in range(len(ordered))]
    for i in range(len(ordered)):
        tables.update(pd.concat(parts[: i + 1], ignore_index=True), ordered[: i + 1])
    assert tables.key == ("condition", "biological_repeat", "technical_repeat", "image_file")
    assert [u.n_objects for u in tables.updates] == [len(p) for p in parts]
    # All at once, too.
    at_once = LiveTables(live_session(setup))
    at_once.update(pd.concat(parts, ignore_index=True), ordered)
    assert len(at_once.updates) == len(ordered)


def test_i5_the_report_takes_few_fovs_from_the_setup():
    objects, fovs = _tables(n_fovs=10)  # 5 fields per well: enough for min_other_fovs=4
    novelty = SupportNoveltyEstimator(1.5, 3, Standardize(), 3, 2)
    with pytest.raises(ValueError, match="must exceed"):
        _setup(novelty=novelty, few_fovs=3)
    assert _setup(novelty=novelty).resolved_few_fovs == 5
    result = report(objects, fovs, _setup(novelty=novelty, few_fovs=4, stage_cols=None))
    assert set(result.results["sufficiency"].next_fov["few_fovs"].dropna()) <= {4}
    assert "fewer than 4 other fields" in " ".join(result.setup.declaration())


def test_i6_control_analyses_on_pooled_groups_are_not_assessed_with_the_reason():
    objects, fovs = _tables()
    pooled = report(objects, fovs, _pooled(group_cols=("condition",)))
    assert "contrast" not in pooled.results and "object_rules" in pooled.results
    assert "pool 'biological_repeat'" in pooled.document.sections["conditions"]
    everything = report(objects, fovs, _pooled(group_cols=()))
    assert "object_rules" not in everything.results and "unusual" not in everything.results
    assert "object-rule check against the control is not assessed" in everything.summary()
    assert "pool 'condition'" in everything.document.sections["profile"]


def test_i8_setup_random_state_reaches_the_protocol_bootstrap(monkeypatch):
    seen = {}
    protocol = workflow_module.protocol_reliability

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return protocol(*args, **kwargs)

    monkeypatch.setattr(workflow_module, "protocol_reliability", spy)
    objects, fovs = _tables()
    plan = optimise(objects, fovs, _setup(random_state=7, stage_cols=None), **QUICK)
    assert seen["random_state"] == 7
    assert plan.results["protocol"].settings["random_state_protocol"] == 7
    for name in ("random_state", "n_jobs", "novelty_target", "few_fovs"):
        assert f"``{name}``" in Setup.__doc__


def test_i9_the_report_only_novelty_rule_is_labelled_and_declared():
    objects, fovs = _tables()
    setup = _setup(novelty=default_novelty(("feature_a", "feature_b")), stage_cols=None)
    assert any(line.startswith("Report-only rule:") for line in setup.declaration())
    enough = report(objects, fovs, setup).results["sufficiency"].enough
    declared = {rule.config_id for rule in setup.stopping_rules()}
    extra = enough[~enough["config_id"].isin(declared)]
    assert len(extra) and extra["answer"].str.startswith("report-only rule").all()
    ours = enough[enough["config_id"].isin(declared)]
    assert not ours["answer"].str.contains("report-only").any()


def test_i10_sentences_use_the_observation_unit_and_need_a_control_for_unusual_ones():
    setup = _setup(observation_unit="nucleus")
    assert "over all nuclei" in setup.declaration()[0]
    without = " ".join(_setup(observation_unit="cell", control=None).declaration())
    assert "unusual cells are not assessed without a control" in without
    assert "for unusual" not in without
    objects, fovs = _tables()
    session = live_session(_setup(observation_unit="nucleus", rules=(FixedFOVStrategy(4),)))
    first = fovs.iloc[[0]]
    waiting = objects.merge(fovs.iloc[[0, 1]][KEY], on=KEY)
    update = LiveTables(session).update(waiting, first)[0]
    assert "nucleus rows wait for their FOV rows" in update.summary()


def test_i11_watch_until_stop_follows_the_objective_and_stop_on_rule_each_verdict(tmp_path):
    objects, fovs = _tables()
    fingerprint = (
        OutcomeCriterion(CovarianceError(["feature_a", "feature_b"], True), 0.5, "lower_is_better"),
    )
    setup = _setup(rules=(FixedFOVStrategy(2),), fingerprint=fingerprint)
    group = fovs[(fovs["condition"] == "control") & (fovs["biological_repeat"] == "R1")]
    for _, row in group.sort_values("acquisition_order").iterrows():
        write_fov(tmp_path, objects.merge(row.to_frame().T[KEY], on=KEY), row.to_dict())
    seen = LiveFolder(tmp_path, live_session(setup)).watch(
        interval=0.01, timeout=1, until_stop=True, on_update=None
    )
    # The objective stays unsupported (no rule decides the fingerprint), so the watch
    # does not stop (owner, 2026-10-05, #85) and reads every field.
    assert len(seen) == len(group)
    seen = LiveFolder(tmp_path, live_session(setup)).watch(
        interval=0.01, timeout=5, until_stop=True, on_update=None, stop_on="rule"
    )
    last = seen[-1].results[0]
    # With stop_on="rule", the rule stops at the second field of each well (wells
    # alternate: w0, w1, w0, w1), and watch returns once both wells stop (audit 4, P2-5).
    assert last.decision != Decision.STOP and last.rule_decision == Decision.STOP
    assert len(seen) == 4 < len(group)


def test_i12_presets_are_printed_and_the_docstrings_say_what_is_chosen():
    objects, fovs = _tables()
    plan = optimise(objects, fovs, _setup(stage_cols=None), **QUICK)
    assert "90% of random acquisitions" in plan.summary() and "protocol_threshold" in plan.summary()
    family = [Family("spread", SPREAD, "max_descriptor_spread", (0.05, 0.4))]
    bench = benchmark(
        objects,
        fovs,
        _setup(stage_cols=None),
        reservoir_fraction=0.25,
        families=family,
        n_splits=1,
        permuted_orders=1,
        n_boot=10,
    )
    assert "risk levels 5% and 10% of stopping too early (alphas" in bench.summary()
    assert "one criterion per tolerance" not in inspect.getdoc(optimise)
    from morphocoverage import workflows

    for doc in (workflows.__doc__, workflow_module.__doc__):
        assert "no scientific choice" not in doc
