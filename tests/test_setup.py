"""Tests of ``workflows.setup``: the Setup, its parts and its declaration."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from _helpers_embedding import EMPTY, FEATURES, NOTHING_UNDER, _one_per_field
from _helpers_feature_tables import _two_tables
from _helpers_sampling_unit import CONTROL, _flat_text, _well_column_tables, _wells, _widths
from _helpers_workflows import (
    READOUT,
    _acquisition,
    _fingerprint,
    _homogeneous,
    _novelty_report,
    _rare,
    _setup,
    _uneven_tables,
    _well_acquisition,
)
from scenarios import FEATURE_COLS

from morphocoverage._style import CAUTION
from morphocoverage.objectives import ObjectiveConfig, SamplingObjective
from morphocoverage.sampling_unit import condition_contrast, object_rule_check
from morphocoverage.schema import validate_input_tables
from morphocoverage.strategies import (
    FixedFOVStrategy,
    SupportNoveltyEstimator,
    SupportNoveltyStrategy,
)
from morphocoverage.sufficiency_text import novelty_definition
from morphocoverage.transforms import Standardize
from morphocoverage.workflows import PositiveControl, Setup, default_novelty, report
from morphocoverage.workflows._declaration import declaration, declaration_short
from morphocoverage.workflows._feature_space import novelty_space
from morphocoverage.workflows._shared import control_checks, sufficiency_rules


def test_a_definition_that_calls_a_homogeneous_group_new_gives_no_chance():
    objects, fovs, names = _homogeneous(17, 6)
    report = _novelty_report(default_novelty(names), objects, fovs, names)
    row = report.next_fov.iloc[0]
    assert (
        isinstance(row["uninformative"], str)
        and "not informative with 6 fields" in row["uninformative"]
    )
    short = " ".join(report.summary().split())
    full = " ".join(report.summary(detail="full").split())
    assert "not informative" in short and "no chance of a new state" in short
    assert "One more field: not informative" in full
    assert "Fields still needed" not in full
    assert "chance" not in short.split("One more field")[1].split(".")[0].replace(
        "no chance of a new state", ""
    )
    assert report.next_fov["answer"].iloc[0].startswith("Not informative")


def test_the_normalised_sentence_names_the_declared_unit():
    _, fovs = _uneven_tables()
    declared = Setup(
        readout="area",
        control={"condition": "ctrl"},
        normalize="control_mean",
        observation_unit="image",
    )
    assert "control ctrl images of the same biological repeat" in declared.normalized_sentence(fovs)
    undeclared = Setup(readout="area", control={"condition": "ctrl"}, normalize="control_mean")
    assert "control ctrl objects of the same biological repeat" in undeclared.normalized_sentence(
        fovs
    )


def test_i9_the_report_only_novelty_rule_is_labelled_and_declared():
    objects, fovs = _well_acquisition()
    setup = _setup(novelty=default_novelty(("feature_a", "feature_b")), stage_cols=None)
    assert any(line.startswith("Report-only rule:") for line in setup.declaration())
    enough = report(objects, fovs, setup).results["sufficiency"].enough
    declared = {rule.config_id for rule in setup.stopping_rules()}
    extra = enough[~enough["config_id"].isin(declared)]
    assert len(extra) and extra["answer"].str.startswith("report-only rule").all()
    ours = enough[enough["config_id"].isin(declared)]
    assert not ours["answer"].str.contains("report-only").any()


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


def test_i15_declaration_claims_new_phenotypes_only_with_a_definition_of_new():
    text = " ".join(_setup().declaration())
    assert "new phenotypes are not assessed (no definition of new" in text
    assert "and new phenotypes; definition of new: not declared" not in text
    assert "normalizes the mean-precision rule" not in Setup.__doc__
    assert "object_rule_check" in Setup.__doc__


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


def test_p1_12_small_unit_d_plural_and_rare_declaration_wording():
    objects, fovs = _wells(2, shift=3.0)
    contrast = condition_contrast(objects, fovs, feature="x", control=CONTROL, n_draws=5)
    # From 2 well means a side a d can be very large by chance: it is not computed.
    assert contrast.table["d_technical"].isna().all()
    assert "unstable" not in " ".join(contrast.summary().split())
    assert " 2 vs 2 " in contrast.summary(detail="full")
    # A field rule asking for one field says "1 field", not "1 fields".
    # 5 fields a side: the check needs 5 since audit 10 (owner, 2026-10-06).
    spec = [("ctrl", "r1", "A", 5, 10.0), ("drug", "r1", "B", 5, 10.0)]
    objects, fovs = _well_column_tables(spec, per_field=200)
    fovs = pd.concat([fovs, fovs.assign(position_id=fovs["position_id"] + "_empty")])
    result = object_rule_check(objects, fovs, feature="x", control=CONTROL, n_draws=20)
    full = result.summary(detail="full")
    assert "fields with cells" in full and "1 fields" not in full
    row = result.table.iloc[0]
    assert f"{int(row['fov_rule_fovs'])} ({int(row['fov_rule_fovs_upper95'])})" in full
    from morphocoverage.objectives import RareTarget

    rare = RareTarget(
        detection_probability=0.9, min_fov_occurrences=2, phenotype="state", fov_incidence=0.2
    )
    setup = Setup("x", rare=rare, rare_label="r")
    rare_line = next(line for line in declaration(setup) if line.startswith("Rare phenotype"))
    assert "by optimise and benchmark" in rare_line


def test_p6_1_the_report_counts_unusual_fields_with_the_novelty_rules_m():
    rng = np.random.default_rng(0)
    rows, fields = [], []
    for condition in ("control", "drug"):
        for repeat in ("R1", "R2", "R3"):
            for j in range(6):
                name = f"{condition}{repeat}f{j}"
                fields.append(
                    {"condition": condition, "biological_repeat": repeat, "position_id": name}
                )
                for x, y in rng.normal(size=(20, 2)):
                    rows.append(
                        {"condition": condition, "biological_repeat": repeat, "position_id": name}
                        | {"a": x, "b": y}
                    )
    objects, fovs = pd.DataFrame(rows), pd.DataFrame(fields)
    setup = Setup("a", features=("a", "b"), control={"condition": "control"})
    results = control_checks(objects, fovs, setup, {})
    settings = results["clustering"].settings
    assert settings["min_objects_per_fov"] == default_novelty(["a", "b"]).min_novel_observations
    assert settings["reference"] == {"condition": "control"}


def test_p6_3_many_standardized_features_warn_and_the_definition_says_how_compact():
    features = tuple(f"f{i}" for i in range(8))
    with pytest.warns(UserWarning, match="small, ordinarily variable states are often missed"):
        Setup("f0", features=features, novelty=default_novelty(features))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Setup("f0", features=features[:4], novelty=default_novelty(features[:4]))
    text = novelty_definition(default_novelty(features[:4]), list(features[:4]))
    assert "each linked to another by a step within the distance" in text
    assert "sit about 2.8 apart" in text


def test_b3_several_controls_with_nothing_under_them_are_refused_naming_what_to_declare():
    with pytest.raises(ValueError, match=NOTHING_UNDER) as error:
        Setup(readout="area", control=EMPTY, master_control="cA")
    assert "control_parents" in str(error.value) and "'cA' and 'cB'" in str(error.value)
    # One member, or a parent, is enough: the declaration compares something.
    Setup(readout="area", control={"condition": {"cA": ["a1"], "cB": {}}})
    Setup(readout="area", control=EMPTY, control_parents={"cB": "cA"})


def test_seed3_b4_one_image_per_field_says_why_and_declares_only_what_runs():
    objects, fovs, _ = _one_per_field()
    setup = Setup(
        readout="f1",
        features=FEATURES,
        control={"condition": "ctrl"},
        novelty=default_novelty(FEATURES),
        positive=PositiveControl({"condition": "a"}, fovs_seen=5),
        observation_unit="image",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    full = " ".join(result.summary(detail="full").split())
    short = " ".join(result.summary().split())
    # One observation per field (owner, 2026-10-06, plan v2, replacing the earlier "not
    # applicable"): the positive control is judged by the field version of new, and the
    # unusual images run; only the clustering within fields is said not to apply.
    assert "no check could be made" not in full and "controls" in result.results
    assert result.results["controls"].status != "not assessed"
    for text in (full, short):
        assert "for unusual images" in text and "Positive control: fields of" in text
        assert "(new: an image that no other field shows), both exploratory" in text
        assert "unusual images are not assessed" not in text
    assert "cluster in a few fields is not applicable: each field holds one image" in short


def test_novelty_with_two_explainable_tables_does_not_depend_on_their_order():
    z, names = _two_tables(1)
    rows = pd.DataFrame(z, columns=names)
    rows["image_file"] = [f"f{i // 20}" for i in range(len(rows))]
    rows["label_id"] = np.arange(len(rows)) % 20 + 1
    rows["condition"], rows["biological_repeat"] = "c", "R1"
    fovs = rows[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    found = []
    for tables in ({"TA": names[:4], "TB": names[4:]}, {"TB": names[4:], "TA": names[:4]}):
        setup = Setup(
            readout="a0",
            features=tuple(names),
            novelty=default_novelty(names),
            explainable_tables=tables,
        )
        estimator, columns = novelty_space(setup)
        data = validate_input_tables(rows, fovs=fovs, feature_cols=columns)
        evidence = estimator.estimate(data, objective=ObjectiveConfig(SamplingObjective.POPULATION))
        found.append(evidence.quantities["lofo_novel_fraction"].value)
    assert found[0] == pytest.approx(found[1])


def test_h4_the_report_inspection_keeps_every_reason_with_novelty():
    from morphocoverage.workflows import default_novelty

    objects, fovs = _acquisition(n_fovs=8, n_objects=6)
    result = report(objects, fovs, _setup(novelty=default_novelty(FEATURE_COLS)))
    settings = result.results["inspection"].settings
    assert settings["novelty"] is not None and settings["novelty_cols"] == list(FEATURE_COLS)
    # With a contrast, a field's influence is on the result, not on a declared summary.
    on_result = "contrast" in result.results
    summaries = [] if on_result else [{"feature": READOUT, "statistic": "mean"}]
    assert settings["summaries"] == summaries
    assert ("influence" in settings["checked"]) and (
        settings.get("influence_on") == ("result" if on_result else None)
    )
    assert settings["count_ratio"] == 3.0 and settings["stage_cols"] == ["stage_x", "stage_y"]


def test_h12_h13_the_declaration_says_what_is_assessed():
    text = "\n".join(Setup(readout="x").declaration())
    assert "no feature columns declared, so unusual cells and new phenotypes are not" in text
    normalised = Setup(readout="x", control={"condition": "MOCK"}, normalize="control_mean")
    assert "(of the whole table without a repeat column)" in "\n".join(normalised.declaration())


def test_issue_143_the_declaration_states_the_support_that_makes_a_state_new():
    from dataclasses import replace

    from morphocoverage.workflows import default_novelty
    from morphocoverage.workflows._declaration import declaration_short

    novelty = default_novelty(FEATURE_COLS)
    for others, words in ((1, "no other field shows"), (3, "fewer than 3 other fields show")):
        setup = _setup(novelty=replace(novelty, min_other_fovs=others))
        short = " ".join(declaration_short(setup).split())
        assert f"at least 3 similar cells that {words}" in " ".join(setup.declaration())
        assert f"at least 3 similar cells that {words}" in short


def test_issue_144_the_default_rules_are_declared_as_they_run():
    for unit, sides in ((None, "per object and per field"), ("patch", "per field")):
        text = " ".join(Setup(readout="x", observation_unit=unit).declaration())
        assert f"(the default, rules=None: the mean-precision rule {sides} at 5%)" in text


@pytest.mark.parametrize(("unit", "rules"), [("cell", 2), ("patch", 1), ("track", 1)])
def test_the_rule_per_object_is_a_default_only_for_objects(unit, rules):
    chosen = Setup(readout="x", observation_unit=unit).stopping_rules()
    assert len(chosen) == rules and chosen[-1].resampling_unit == "fov"


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


# Setup.__post_init__: each message, and the first check that fires when two would.
SETUP_ERRORS = [
    ({"readout": ""}, TypeError, "readout must be one column name (None: no explainable features)"),
    ({"features": "area"}, TypeError, "features must be a sequence, not a string"),
    ({"intensity_features": "x"}, TypeError, "intensity_features must be a sequence, not a string"),
    ({"group_cols": "x"}, TypeError, "group_cols must be a sequence of column names"),
    ({"fields_independent": 1}, TypeError, "fields_independent must be True or False"),
    ({"fingerprint": (1,)}, TypeError, "fingerprint must be a sequence of OutcomeCriterion"),
    ({"rare": 3}, TypeError, "rare must be a RareTarget"),
    ({"rare_label": 3}, TypeError, "rare_label must be one object column name"),
    (
        {"rare_label": "m"},
        ValueError,
        "rare_label labels a declared rare target: declare rare too",
    ),
    (
        {"tolerances": (1.5,)},
        ValueError,
        "tolerances must be fractions between 0 and 1, such as 0.05",
    ),
    ({"control": {}}, TypeError, "control must map FOV-table columns to values"),
    (
        {"normalize": "x", "control": {"condition": "c"}},
        ValueError,
        "normalize must be one of ['control_mean'] or None",
    ),
    (
        {"intensity_features": ("b",)},
        ValueError,
        "intensity_features must be declared columns (readout or features)",
    ),
    # Two problems: the earlier check fires.
    (
        {"features": "area", "group_cols": "x"},
        TypeError,
        "features must be a sequence, not a string",
    ),
    ({"group_cols": "x", "rare": 3}, TypeError, "group_cols must be a sequence of column names"),
    ({"rare": 3, "tolerances": (2,)}, TypeError, "rare must be a RareTarget"),
    (
        {"tolerances": (2,), "control": {}},
        ValueError,
        "tolerances must be fractions between 0 and 1, such as 0.05",
    ),
    (
        {"control": {}, "control_parents": {"A": "B"}},
        TypeError,
        "control must map FOV-table columns to values",
    ),
    (
        {"control_parents": {"A": "B"}, "normalize": "x"},
        ValueError,
        (
            "control_parents link several controls: declare them as "
            "control={'condition': {control: {treatment: condition, ...}, ...}}"
        ),
    ),
    (
        {"normalize": "x", "intensity_features": ("b",)},
        ValueError,
        "normalize must be one of ['control_mean'] or None",
    ),
]


@pytest.mark.parametrize(("changes", "kind", "message"), SETUP_ERRORS)
def test_setup_refuses_with_its_messages_in_order(changes, kind, message):
    with pytest.raises(kind) as raised:
        Setup(**{"readout": "a", **changes})
    assert str(raised.value) == message


CONTROL_ERRORS = [
    (
        {"control": {"condition": "A"}, "master_control": "B"},
        ValueError,
        "master_control names one of several declared controls",
    ),
    (
        {"control": {"condition": {"A": ["a"]}, "well": "x"}},
        ValueError,
        (
            "several controls are named by the condition column alone: "
            "control={'condition': {control: {treatment: condition, ...}, ...}}"
        ),
    ),
    (
        {"control": {"condition": {"A": ["a"]}}, "control_parents": [("A", "B")]},
        TypeError,
        "control_parents must map a control to its parent condition",
    ),
    (
        {"control": {"condition": {"A": ["a"]}}, "control_parents": {"A": ("B", "t", "u")}},
        TypeError,
        "the parent of 'A' is a condition, or (condition, treatment)",
    ),
    (
        {
            "control": {"condition": {"A": {"t": "x"}, "B": {"t": "y"}}},
            "control_parents": {"B": ("A", "t")},
        },
        ValueError,
        "treatment 't' names two conditions of 'A'",
    ),
    (
        {
            "control": {"condition": {"A": ["a"], "B": ["b"]}},
            "control_parents": {"A": "B", "B": "A"},
        },
        ValueError,
        "the controls form a cycle through 'A'",
    ),
    (
        {"control": {"condition": {"A": ["a"], "B": ["A"]}}, "control_parents": {"A": "Z"}},
        ValueError,
        (
            "'A' is listed under control 'B' and linked to the parent 'Z': a control is compared "
            "with one reference, so keep one of the two"
        ),
    ),
]


@pytest.mark.parametrize(("changes", "kind", "message"), CONTROL_ERRORS)
def test_several_controls_are_refused_with_their_messages(changes, kind, message):
    with pytest.raises(kind) as raised:
        Setup(**{"readout": "a", **changes})
    assert str(raised.value) == message


def test_a_contrast_table_cannot_give_another_own_control():
    table = pd.DataFrame(
        {
            "condition": ["a"],
            "reference": ["A"],
            "kind": ["own_control"],
            "key": [None],
            "label": ["a vs A"],
        }
    )
    with pytest.raises(ValueError) as raised:
        Setup(readout="a", control={"condition": {"B": ["a"]}}, contrasts=table)
    assert str(raised.value) == (
        "the contrast table compares 'a' with 'A' as its own control, but the declared "
        "controls give 'B': a condition's own control comes from the declaration"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Setup(readout="a", control={"condition": {"A": ["a"]}}, contrasts=table)


def test_the_declaration_short_form_names_the_rules_and_states_each_caveat_once():
    setup = _setup(fingerprint=(_fingerprint(),), rare=_rare(), rare_label="state")
    text = declaration_short(setup)
    assert "Population summary: the mean of" in text
    assert (
        "Stopping rules: mean-precision rule, per object, 5% and mean-precision rule, per "
        "field, 5% (the default)" in _flat_text(text)
    )
    assert _flat_text(text).count("the standard error of the mean is 5% of the mean") == 1
    assert "the binomial plan asks for 14 representative fields" in _flat_text(text)
    assert all(width <= 101 for width in _widths(text))
    plain = declaration_short(_setup(acquisition_order_col=None, stage_cols=None))
    # Options left at their defaults are standing notes, not cautions.
    assert "No phenotypic fingerprint declared" in _flat_text(plain) and CAUTION not in plain
    assert "No rare phenotype declared" in _flat_text(plain)
    assert "the order is not checked" in _flat_text(plain)
    # The full sentences name each rule in words, with its strategy.
    rules = next(s for s in setup.declaration() if s.startswith("Stopping rules:"))
    assert "mean-precision rule, per object, 5% [scalar_precision_object_upper95 (" in rules
