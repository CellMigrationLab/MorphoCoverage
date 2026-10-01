"""Regressions of the third audit, group I: the workflows (findings I-1 to I-18, J-2/16/17).

Each test names the finding it pins and fails on the code before the fix.
Constructed data only: every test checks a contract of the workflows (seeds,
keys, printed presets, refusals), never how well a rule performs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_audit_workflows import _wells
from test_workflows import FEATURE_COLS, READOUT, _acquisition, _setup

import morphocoverage.workflows._document as document_module
import morphocoverage.workflows._shared as shared_module
import morphocoverage.workflows.benchmarks as benchmarks_module
import morphocoverage.workflows.retrospective as workflow_module
from morphocoverage import RareTarget, validate_input_tables
from morphocoverage.strategies import (
    FixedFOVStrategy,
    SupportNoveltyEstimator,
    SupportNoveltyStrategy,
)
from morphocoverage.transforms import Standardize
from morphocoverage.workflows import (
    LiveTables,
    Plan,
    PositiveControl,
    Report,
    Setup,
    benchmark,
    default_families,
    default_novelty,
    document,
    live_session,
    optimise,
    report,
)
from morphocoverage.workflows._document import plan_settings
from morphocoverage.workflows._shared import order_seeds, orders, replay_rules, sufficiency_rules

KEY = ["condition", "biological_repeat", "technical_repeat", "image_file"]
QUICK = {
    "fov_counts": [2],
    "reservoir_fraction": 0.25,
    "n_splits": 1,
    "draws": 2,
    "permuted_orders": 1,
    "dispersed_orders": 0,
}


def _tables(n_fovs: int = 6, n_objects: int = 5):
    return _wells(*_acquisition(n_fovs=n_fovs, n_objects=n_objects))


def _rare_target() -> RareTarget:
    return RareTarget(
        fov_incidence=0.2, detection_probability=0.95, min_fov_occurrences=1, phenotype="big"
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
    assert "seeded from random_state); random_state 7." in runs[7].summary()


def test_i2_livetables_waits_for_the_first_fov_row_before_fixing_its_key():
    objects, fovs = _tables()
    setup = _setup(rules=(FixedFOVStrategy(4),))
    ordered = fovs.sort_values(["condition", "biological_repeat", "acquisition_order"])
    ordered = ordered.reset_index(drop=True)
    parts = [objects.merge(ordered.iloc[[i]][KEY], on=KEY) for i in range(len(ordered))]
    tables = LiveTables(live_session(setup))
    for i in range(len(ordered)):
        grown = pd.concat(parts[: i + 1], ignore_index=True)
        tables.update(grown, ordered[:i])  # the field's objects arrive first ...
        tables.update(grown, ordered[: i + 1])  # ... then its FOV row
    assert tables.key == ("condition", "biological_repeat", "technical_repeat", "image_file")
    assert [u.n_objects for u in tables.updates] == [len(p) for p in parts]


def test_i3_the_novelty_rule_in_rules_and_no_novelty_target_are_accepted():
    objects, fovs = _acquisition(n_fovs=6, n_objects=4)
    novelty = default_novelty(FEATURE_COLS)
    base = _setup(novelty=novelty, stage_cols=None)
    rule = base.novelty_rule()
    assert isinstance(rule, SupportNoveltyStrategy)
    declared = _setup(novelty=novelty, stage_cols=None, rules=(FixedFOVStrategy(4), rule))
    assert declared.novelty_rule_declared
    assert any(line.startswith("Novelty rule:") for line in declared.declaration())
    enough = report(objects, fovs, declared).results["sufficiency"].enough
    ours = enough[enough["config_id"] == rule.config_id]
    assert len(ours) and not ours["answer"].str.contains("report-only").any()
    # With a readout outside the features, the setup's rule is kept (as live) and no target
    # is passed, so sufficiency_report never builds the same rule twice.
    apart = _setup(
        features=FEATURE_COLS[1:],
        novelty=default_novelty(FEATURE_COLS[1:]),
        stage_cols=None,
    )
    apart = _setup(
        features=FEATURE_COLS[1:],
        novelty=apart.novelty,
        stage_cols=None,
        rules=(FixedFOVStrategy(4), apart.novelty_rule()),
    )
    rules, target = sufficiency_rules(apart)
    assert target is None and len(rules) == 2
    rules, target = sufficiency_rules(declared)
    assert target == 0.05 and [r.config_id for r in rules] == [FixedFOVStrategy(4).config_id]
    # novelty_target=None: no report-only rule, and the declaration says so.
    untargeted = _setup(novelty=novelty, novelty_target=None, stage_cols=None)
    assert untargeted.novelty_rule() is None
    assert any("novelty_target=None" in line for line in untargeted.declaration())
    report(objects, fovs, untargeted)


def test_i4_settings_print_the_fields_held_back_and_the_recorded_order():
    objects, fovs = _tables()
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


def test_i5_every_preset_is_printed_and_minimum_detection_reaches_the_report(monkeypatch):
    seen = {}
    assemble = document_module.microscopist_report

    def spy(**kwargs):
        seen.update(kwargs)
        return assemble(**kwargs)

    monkeypatch.setattr(document_module, "microscopist_report", spy)
    objects, fovs = _acquisition(n_fovs=6, n_objects=4)
    setup = _setup(stage_cols=None, minimum_detection=0.5)
    first = report(objects, fovs, setup)
    assert seen["minimum_detection"] == 0.5
    with pytest.raises(ValueError, match="minimum_detection"):
        _setup(minimum_detection=1.5)
    positive = _setup(
        novelty=default_novelty(FEATURE_COLS),
        positive=PositiveControl({"condition": "drug"}, fovs_seen=2),
        minimum_detection=0.8,
    )
    assert "fewer than 80% of its checks are recognized (minimum_detection" in " ".join(
        positive.declaration()
    )
    rules = [line for line in setup.declaration() if line.startswith("Stopping rules:")]
    assert len(rules) == 1 and "(the default, rules=None" in rules[0]
    chosen = _setup(rules=(FixedFOVStrategy(4),)).declaration()
    assert any("fixed 4 fields [fixed_fov (target_fovs=4)] (rules)" in line for line in chosen)
    assert "5 random orders" in first.summary()
    # document() keeps the plan's settings sentence (the protocol_threshold preset) and the
    # report's.
    plan = optimise(objects, fovs, setup, **{**QUICK, "fov_counts": [2]}, protocol_threshold=0.8)
    merged = document(first, plan).summary()
    assert "(protocol_threshold, a provisional preset)" in merged
    assert "'Did I image enough?' replays each rule" in merged
    # benchmark: n_boot, min_fovs, the grids and the families' seed.
    captured = {}
    families = benchmarks_module.default_families

    def quick_families(readout, n_resamples=200, random_state=0):
        captured["random_state"] = random_state
        return families(readout, 10, random_state)[:1]

    monkeypatch.setattr(benchmarks_module, "default_families", quick_families)
    text = benchmark(
        objects,
        fovs,
        _setup(stage_cols=None, random_state=3),
        reservoir_fraction=0.25,
        n_splits=1,
        permuted_orders=1,
        n_boot=20,
        min_fovs=2,
    ).summary()
    text = " ".join(text.split())  # the settings are wrapped
    assert "from 20 bootstrap resamples (n_boot)" in text
    assert "rules evaluated from 2 fields on (min_fovs)" in text
    assert "JCB per object (averaging_cv 0.0025 to 0.4, 9 settings)" in text
    assert "resampling rules seeded with random_state 3" in text


def test_i6_the_positive_control_pairs_within_repeats_by_default_and_prints_pooling():
    objects, fovs = _acquisition(n_fovs=10, n_objects=6)
    default = PositiveControl({"condition": "drug"}, fovs_seen=4)
    assert default.pair_cols is None and default.resolved_pair_cols(fovs) == ("biological_repeat",)
    assert default.resolved_pair_cols(fovs.drop(columns="biological_repeat")) == ()
    # A definition of new that these few fields can judge.
    novelty = SupportNoveltyEstimator(1.5, 1, Standardize(), 2, 2)
    paired = report(objects, fovs, _setup(novelty=novelty, positive=default, stage_cols=None))
    assert paired.results["controls"].pair_cols == ("biological_repeat",)
    pooled = PositiveControl({"condition": "drug"}, fovs_seen=4, pair_cols=())
    setup = _setup(novelty=novelty, positive=pooled, stage_cols=None)
    assert "every repeat pooled (pair_cols=())" in " ".join(setup.declaration())
    result = report(objects, fovs, setup)
    assert result.results["controls"].pair_cols == ()
    assert "paired across the whole table (pair_cols=())" in result.document.sections["controls"]


def test_i7_the_report_clusters_the_declared_rare_label():
    objects, fovs = _acquisition(n_fovs=6, n_objects=4)
    objects = objects.assign(state=objects[READOUT] > objects[READOUT].quantile(0.9))
    setup = _setup(control=None, rare=_rare_target(), rare_label="state", stage_cols=None)
    result = report(objects, fovs, setup)
    clustering = result.results["rare_clustering"]
    assert clustering.settings["label_col"] == "state"
    assert clustering.settings["states"] == [True]
    profile = result.document.sections["profile"]
    assert "Are the state" in profile and result.document.inputs["profile"] == "given"
    assert "- State objects: clustered in a few fields" in result.document.short["profile"]
    assert "fov_clustering" in set(result.document.profile["source"])


def test_i8_document_refuses_parts_of_different_setups():
    first, other = _setup(), _setup(tolerances=(0.1,))
    with pytest.raises(ValueError, match="different Setups"):
        document(Report({}, None, first), Plan({}, None, other))
    # n_jobs changes no result, so parts differing only there merge.
    merged = document(Report({}, None, first), Plan({}, None, _setup(n_jobs=2)))
    assert merged.declaration == first.declaration()


def test_i9_setup_random_state_seeds_the_default_families():
    for family in default_families(READOUT, n_resamples=10, random_state=7):
        parameters = family.strategy.parameters
        assert parameters.get("random_state", 7) == 7
    assert default_families(READOUT)[2].strategy.parameters["random_state"] == 0


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


def test_i13_the_unusual_labels_are_named_for_the_reader():
    objects, fovs = _acquisition(n_fovs=6, n_objects=4)
    result = report(objects, fovs, _setup(stage_cols=None, observation_unit="cell"))
    assert result.results["clustering"].settings["label_col"] == "unusual cells"
    assert "morphocoverage_unusual" not in result.summary()
    assert "Are the unusual cells spread over the fields?" in result.document.sections["profile"]


def test_i14_livetables_key_is_built_from_the_validator_resolvers():
    objects, fovs = _tables()
    one = fovs[(fovs["condition"] == "control") & (fovs["biological_repeat"] == "R1")].copy()
    mine = objects.merge(one[KEY], on=KEY)
    # A producer that writes every column leaves the optional ones empty.
    one["technical_repeat"], mine["technical_repeat"] = None, None
    one["position_id"], mine["position_id"] = None, None
    tables = LiveTables(live_session(_setup(rules=(FixedFOVStrategy(4),), stage_cols=None)))
    assert len(tables.update(mine, one)) == len(one)
    data = validate_input_tables(mine, fovs=one, feature_cols=[READOUT])
    assert tables.key == data.fov_key == ("condition", "biological_repeat", "image_file")


def test_i15_declaration_claims_new_phenotypes_only_with_a_definition_of_new():
    text = " ".join(_setup().declaration())
    assert "new phenotypes are not assessed (no definition of new" in text
    assert "and new phenotypes; definition of new: not declared" not in text
    assert "normalizes the mean-precision rule" not in Setup.__doc__
    assert "object_rule_check" in Setup.__doc__


def test_i16_one_observation_reads_in_the_singular():
    fov = pd.DataFrame({"position_id": ["P1"]})
    one = pd.DataFrame({"position_id": ["P1"], READOUT: [1.0]})
    session = live_session(Setup(readout=READOUT, rules=(FixedFOVStrategy(4),)))
    (update,) = LiveTables(session).update(one, fov)
    assert "P1: 1 object (read in" in update.summary()


def test_i17_setup_validates_the_novelty_target_seed_and_workers():
    novelty = default_novelty(FEATURE_COLS)
    for bad in (0.0, 1.5, -0.1, "5%", True):
        with pytest.raises(ValueError, match="novelty_target"):
            _setup(novelty=novelty, novelty_target=bad)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="random_state"):
            _setup(random_state=bad)
    for bad in (0, -2, 1.5):
        with pytest.raises(ValueError, match="n_jobs"):
            _setup(n_jobs=bad)
    setup = _setup(random_state=np.int64(3), n_jobs=np.int64(2), novelty_target=np.float32(0.1))
    assert type(setup.random_state) is int and type(setup.n_jobs) is int
    assert type(setup.novelty_target) is float


def test_i18_the_unusual_reference_states_that_it_pools_repeats():
    objects, fovs = _acquisition(n_fovs=6, n_objects=4)
    result = report(objects, fovs, _setup(stage_cols=None))
    profile = result.document.sections["profile"]
    assert "The reference of the unusual objects pools the control objects of 2 repeats" in profile
    assert "their repeats pooled" in " ".join(result.setup.declaration())


def test_j2_j16_j17_shared_helpers_are_named_listed_and_word_percentages_once():
    assert hasattr(shared_module, "check_workflow_inputs")
    assert not hasattr(shared_module, "check_tables")
    for module in (shared_module, document_module):
        assert module.__all__ and all(hasattr(module, name) for name in module.__all__)
        assert "Shared-private:" in module.__doc__
    settings = {
        "protocol_threshold": 0.999,
        "reservoir_fraction": 0.334,
        "reservoir_fovs": (2, 3),
        "n_splits": 2,
        "draws": 3,
        "permuted_orders": 1,
        "dispersed_orders": 0,
        "random_state": 0,
    }
    text = plan_settings(settings)
    assert "when 99.9% of random acquisitions" in text  # not "100%"
    assert "2-3 fields of each group held back (reservoir_fraction 33%" in text
