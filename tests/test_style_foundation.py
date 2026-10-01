"""The readable-reports foundation: the shared style helpers, and question 2 and the opening.

The owner's decisions (2026-09-28): markers ✅ ⚠️ ❌ ➖ with fixed meanings, a
short default with every caveat once, and every number of the old text kept
in ``detail="full"``. These tests hold the helpers of ``morphocoverage._style``
to their contracts, the markers of question 2 and of the opening to their
rules (``retrospective.report_status``), and the short and full forms to what
each must contain.
"""

import re

import numpy as np
import pandas as pd
import pytest
from contract_registry import STRATEGIES_UNDER_CONTRACT
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage import (
    FixedFOVStrategy,
    ObjectiveConfig,
    SamplingObjective,
    ScalarPrecisionStrategy,
)
from morphocoverage._caveats import caveat_lines
from morphocoverage._style import (
    CAUTION,
    NOT_ASSESSED,
    OK,
    PROBLEM,
    Status,
    among,
    check_detail,
    group_columns,
    group_label,
    headline,
    merge_numbers,
    number_pattern,
    rule_label,
    span,
    table,
    width,
    worst,
    wrap,
)
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.retrospective import dataset_overview, microscopist_report
from morphocoverage.retrospective.report import GLANCE_TITLE, OVERVIEW_TITLE, QUESTIONS
from morphocoverage.retrospective.report_status import (
    data_check_status,
    enough_status,
    integrity_status,
)
from morphocoverage.strategies import (
    DescriptorSpreadStrategy,
    FOVBootstrapStrategy,
    SamplingStrategy,
    SupportNoveltyEstimator,
    SupportNoveltyStrategy,
    incidence_coverage,
    support_novelty,
)
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.workflows import default_families

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
PERMUTED = [AcquisitionOrder.permuted(seed) for seed in range(3)]

# --- Markers and statuses -----------------------------------------------------


def test_the_markers_are_the_owners_emoji():
    assert (OK, CAUTION, PROBLEM, NOT_ASSESSED) == ("✅", "⚠️", "❌", "➖")
    assert [s.marker for s in Status] == [NOT_ASSESSED, OK, CAUTION, PROBLEM]


def test_worst_takes_the_status_that_asks_most_and_ignores_not_assessed():
    assert worst([Status.OK, Status.CAUTION, Status.OK]) is Status.CAUTION
    assert worst([Status.CAUTION, Status.PROBLEM]) is Status.PROBLEM
    # A part not assessed does not lower an assessed one, nor raise it.
    assert worst([Status.OK, Status.NOT_ASSESSED]) is Status.OK
    assert worst([]) is Status.NOT_ASSESSED
    assert worst([Status.NOT_ASSESSED]) is Status.NOT_ASSESSED


def test_detail_is_short_or_full_and_anything_else_names_both():
    assert check_detail("short") == "short" and check_detail("full") == "full"
    with pytest.raises(ValueError, match="short.*full"):
        check_detail("long")


def test_headline_puts_the_marker_first():
    assert headline(Status.OK, "2. Did I image enough?") == "✅ 2. Did I image enough?"
    line = headline(Status.CAUTION, "2. Did I image enough?", "met in 3 of 4 groups")
    assert line == "⚠️ 2. Did I image enough? — met in 3 of 4 groups"


# --- Labels -------------------------------------------------------------------


def test_group_labels_join_the_values_and_name_the_columns_once():
    group = {"condition": "MOCK", "biological_repeat": "1A", "technical_repeat": "B6"}
    assert group_label(group) == "MOCK · 1A · B6"
    assert group_columns(list(group)) == "condition · biological_repeat · technical_repeat"
    assert group_label({}) == "all fields (pooled)"
    assert group_label({"condition": "MOCK", "biological_repeat": np.nan}) == "MOCK · (none)"


def test_among_counts_groups_in_words():
    assert among(12, 12) == "all 12 groups"
    assert among(7, 12) == "7 of 12 groups"
    assert among(0, 12) == "none of 12 groups"
    assert among(1, 1) == "the group" and among(0, 1) == "no group"
    assert among(2, 4, "stratum") == "2 of 4 strata"


# Every default rule family by its readable name (the unit is nucleus).
EXPECTED_LABELS = {
    "fixed_fov": "fixed 5 fields",
    "fixed_objects": "fixed 100 nuclei",
    "fixed_observations": "fixed 100 nuclei",
    "scalar_precision_object": "JCB rule, per nucleus, 5%, point SD",
    "scalar_precision_fov": "JCB rule, per field, 5%, point SD",
    "scalar_precision_object_upper95_pilot": "JCB rule, per nucleus, 5%, pilot of 3 fields",
    "descriptor_plateau": "JCS plateau, ≤0.1 over 3 fields",
    "descriptor_spread_object": "JCS spread, per nucleus, ≤0.3",
    "descriptor_spread_fov": "JCS spread, per field, ≤0.3",
    "scalar_precision_planned": "JCB plan, 122 nuclei",
    "fov_bootstrap_centroid": "field bootstrap of the mean, SE ≤0.05",
    "fov_bootstrap_quantiles": "field bootstrap of the quantiles, SE ≤0.1",
    "fov_bootstrap_covariance": "field bootstrap of the covariance, SE ≤0.1",
    "fov_incidence_plan": "rare-state field plan",
    "fov_bootstrap_eigenspectrum": "field bootstrap of the eigenspectrum, SE ≤0.05",
    "fov_bootstrap_subspace": "field bootstrap of the subspace, angle ≤0.2",
    "support_novelty": "nothing new (≤5%)",
    "incidence_coverage": "state coverage (≥95%)",
    "rare_incidence": "rare-state risk ≤5%",
    "fixed_area": "fixed imaged area 50000",
}


def test_every_default_rule_family_has_a_readable_name():
    names = {s.name: rule_label(s, "nucleus") for s in STRATEGIES_UNDER_CONTRACT}
    assert names == EXPECTED_LABELS
    families = {f.name: rule_label(f.strategy, "nucleus") for f in default_families("x")}
    assert families == {
        "JCB per object": "JCB rule, per nucleus, 0.25%",
        "JCB per field": "JCB rule, per field, 0.25%",
        "JCS spread per object": "JCS spread, per nucleus, ≤0.001",
        "JCS spread per field": "JCS spread, per field, ≤0.001",
        "FOV bootstrap of the mean": "field bootstrap of the mean, SE ≤0.0005",
    }
    # The report's own rules: the JCB rule with the upper-95 SD, as published.
    assert rule_label(ScalarPrecisionStrategy("x", 0.05, "object", "upper_95"), "nucleus") == (
        "JCB rule, per nucleus, 5%"
    )
    assert rule_label(FixedFOVStrategy(8)) == "fixed 8 fields"
    assert rule_label(DescriptorSpreadStrategy("x", 0.3, "fov", 20, 0)) == (
        "JCS spread, per field, ≤0.3"
    )


@pytest.mark.parametrize(
    "quantity, label",
    [
        ("lofo_novel_fov_upper", "nothing new (≤5%)"),
        ("lofo_novel_fov_fraction", "new fields (≤5%, point estimate)"),
        ("lofo_novel_fraction", "new nuclei (≤5%)"),
        ("lofo_mean_fov_novel_fraction", "new nuclei per field (≤5%)"),
    ],
)
def test_every_support_novelty_quantity_has_a_name(quantity, label):
    rule = SupportNoveltyStrategy(0.5, 0.05, 1, None, 3, 1, quantity)
    assert rule_label(rule, "nucleus") == label


def test_the_style_names_every_quantity_of_the_strategies():
    from morphocoverage import _style

    assert set(_style._NOVELTY) == set(support_novelty.QUANTITIES)
    assert set(_style._INCIDENCE) == set(incidence_coverage.QUANTITIES)
    for quantity in incidence_coverage.QUANTITIES:
        target = 2.0 if quantity == "expected_new_states_next_fov" else 0.5
        rule = incidence_coverage.IncidenceCoverageStrategy("state", quantity, target, 3)
        assert rule_label(rule) != rule.name, quantity


def test_an_unknown_rule_keeps_the_strategys_own_name():
    fixed, spread = FixedFOVStrategy(3), FOVBootstrapStrategy("centroid", 0.1, None, 5, 0, 2)
    combined = SamplingStrategy([*fixed.estimators, *spread.estimators], fixed.policies)
    assert rule_label(combined) == combined.name
    assert rule_label({"criterion": "my_rule", "settings": "q <= 1"}) == "my_rule"


def test_rows_of_result_tables_are_named_like_their_strategies():
    row = {"criterion": "scalar_precision_fov_upper95", "settings": "averaging_cv <= 0.05"}
    assert rule_label(row, "nucleus") == "JCB rule, per field, 5%"
    final = pd.Series({"strategy": "fixed_fov", "policies": "n_fovs >= 8", "config_id": "abc"})
    assert rule_label(final) == "fixed 8 fields"
    assert rule_label(final, detail="full") == "fixed 8 fields (fixed_fov: n_fovs >= 8; config abc)"
    # A row that carries its rule's name keeps it; a configuration's policies are read too.
    assert rule_label({"rule": "named", "criterion": "fixed_fov"}) == "named"
    configured = {
        "strategy": "fixed_fov",
        "policies": [{"parameters": {"effort": "n_fovs", "target": 4}}],
    }
    assert rule_label(configured) == "fixed 4 fields"
    strategy = FixedFOVStrategy(8)
    assert (
        rule_label(strategy, detail="full")
        == f"fixed 8 fields (fixed_fov, config {strategy.config_id})"
    )


# --- Layout -------------------------------------------------------------------


def test_tables_align_their_columns_with_emoji_counted_twice():
    text = table(
        [["fixed 8 fields", "12 of 12", OK], {"rule": "JCB", "met in": "3 of 12", "": CAUTION}],
        ["rule", "met in", ""],
        indent=2,
        align="lr",
    )
    lines = text.splitlines()
    assert lines[0].startswith("  rule") and all(line.startswith("  ") for line in lines)
    # The second column ends at the same display column in every line (right-aligned).
    ends = [width(line[: line.index("of") + 5]) if "of" in line else None for line in lines[1:]]
    assert len(set(ends)) == 1
    assert width(OK) == width(CAUTION) == 2 and width("abc") == 3
    assert table([[None, float("nan")]], ["a", "b"], indent=0).splitlines()[1] == "–  –"
    with pytest.raises(ValueError, match="cells"):
        table([["one"]], ["a", "b"])


def test_wrap_keeps_lines_within_100_characters_and_words_whole():
    long_word = "Nuclei_Intensity_IntegratedIntensityEdge_YAPTAZ"
    text = " ".join(["word"] * 60 + [long_word] * 3)
    wrapped = wrap(text, 2, hang=4)
    lines = wrapped.splitlines()
    assert all(width(line) <= 100 for line in lines)
    assert lines[0].startswith("  word") and all(line.startswith("    ") for line in lines[1:])
    assert wrapped.count(long_word) == 3
    assert wrap("a\n\nb", 2) == "  a\n\n  b"
    marked = wrap(f"{OK} " + "word " * 40, 0)
    assert all(width(line) <= 100 for line in marked.splitlines())


def test_numbers_that_differ_become_ranges_and_shared_ones_stay():
    texts = [
        "rho -0.43, p = 0.023, over 28 FOVs (68-88% of the time)",
        "rho -0.55, p = 0.0023, over 28 FOVs (68-88% of the time)",
        "rho -0.41, p = 0.032, over 28 FOVs (68-88% of the time)",
    ]
    assert merge_numbers(texts) == (
        "rho -0.55 to -0.41, p = 0.0023 to 0.032, over 28 FOVs (68-88% of the time)"
    )
    assert merge_numbers(texts[:1]) == texts[0]
    assert number_pattern(texts[0]) == number_pattern(texts[1])
    with pytest.raises(ValueError, match="more than their numbers"):
        merge_numbers(["2 fields", "2 wells"])


def test_cautions_are_stated_once_each_with_the_marker():
    lines = caveat_lines(["the fields are a random sample", "the fields are a random sample "])
    assert lines == [f"  {CAUTION} The fields are a random sample."]


def test_spans_write_one_value_or_the_lowest_and_highest():
    assert span([8, 8]) == "8" and span([3, 5, 4]) == "3-5"
    assert span([0.0, 0.2], "percent") == "0-20%" and span([0.04], "percent") == "4%"
    assert span([-0.5, -0.4]) == "-0.5 to -0.4"


# --- Question 2 and the opening ------------------------------------------------


def _groups_report(strategies, **kwargs):
    built = SCENARIOS["condition_repeat_groups"]()
    return sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=strategies,
        objective=POPULATION,
        orders=PERMUTED,
        **kwargs,
    )


def _novelty_report(**kwargs):
    rng = np.random.default_rng(0)
    rows = []
    for j in range(12):
        cells = rng.normal(10, 0.1, size=(20, 2))
        if j in (3, 7):
            cells = np.vstack([cells, rng.normal(20 + 5 * j, 0.05, size=(3, 2))])
        rows += [{"image_file": f"f{j}", "a": a, "b": b} for a, b in cells]
    return sufficiency_report(
        pd.DataFrame(rows),
        pd.DataFrame({"image_file": [f"f{j}" for j in range(12)]}),
        feature_cols=["a", "b"],
        strategies=[FixedFOVStrategy(4)],
        objective=POPULATION,
        orders=PERMUTED,
        novelty=SupportNoveltyEstimator(0.5, 1, None, 2, 3),
        few_fovs=3,
        **kwargs,
    )


def test_question_2_is_fine_only_when_every_rule_is_met_everywhere():
    met = _groups_report([ScalarPrecisionStrategy("feature_a", 0.5, "fov", "point")])
    assert set(met.enough["sampled_enough"]) == {"yes"}
    status, text = enough_status(met)
    # The point rule warns in some groups here, and a caution asks the reader to look.
    cautioned = met.final["warning_list"].map(bool).any()
    assert status is (Status.CAUTION if cautioned else Status.OK)
    assert "every rule said enough in every group" in text
    never = _groups_report([ScalarPrecisionStrategy("feature_a", 1e-6, "fov", "point")])
    assert set(never.enough["sampled_enough"]) == {"no"}
    assert enough_status(never)[0] is Status.PROBLEM
    both = _groups_report(
        [
            ScalarPrecisionStrategy("feature_a", 0.5, "fov", "point"),
            ScalarPrecisionStrategy("feature_a", 1e-6, "fov", "point"),
        ]
    )
    status, text = enough_status(both)
    assert status is Status.CAUTION and "1 of 2 rules said enough in every group" in text
    # A fixed amount of effort says nothing about the data: never a plain "fine".
    assert enough_status(_groups_report([FixedFOVStrategy(2)]))[0] is Status.CAUTION
    assert enough_status(None)[0] is Status.NOT_ASSESSED


def test_the_short_answer_has_the_verdicts_and_each_caveat_once():
    report = _groups_report(
        [FixedFOVStrategy(2), ScalarPrecisionStrategy("feature_a", 0.5, "fov", "point")]
    )
    short = report.summary()
    assert "fixed 2 fields" in short and "JCB rule, per field, 50%, point SD" in short
    assert "4 of 4" in short  # met in every group, in the table
    assert short.count("treats the fields as a random sample") == 1
    assert short.count("a fixed amount of effort reaching its target") == 1
    for pattern in ("condition=", "config "):
        assert pattern not in short
    # Every group's rule caution is kept, once, with its numbers as ranges.
    warned = report.final[report.final["verdict"].eq("stop")]["warning_list"]
    for texts in warned:
        for text in texts:
            assert number_pattern(text).split("#")[0] in short


def test_the_full_answer_keeps_every_number_of_every_answer():
    report = _novelty_report(novelty_target=0.2)
    full = report.summary(detail="full")
    answers = [*report.enough["answer"], *report.next_fov["answer"]]
    for answer in answers:
        for number in re.findall(r"\d+(?:\.\d+)?%?", answer):
            assert number in full
    assert f"config {report.enough['config_id'].iloc[0]}" in full
    short = report.summary()
    assert "nothing new (≤20%)" in short and "fixed 4 fields" in short
    assert short.count("treat fields as independent, representative draws") == 1
    # Short: what counts as new in one plain line; its exact definition in the full form.
    assert "Seen in few fields:" in short and "'New': at least 3 similar objects" in short
    assert "'New' means" not in short and "'New' means" in full
    with pytest.raises(ValueError, match="detail"):
        report.summary(detail="everything")


def test_the_opening_marks_the_data_checks_and_the_integrity():
    from morphocoverage import check_tables
    from morphocoverage.integrity import integrity_checks
    from morphocoverage.retrospective.report import with_opening

    built = SCENARIOS["condition_repeat_groups"]()
    objects, fovs = built.objects, built.fovs
    unchecked = dataset_overview(objects, fovs, feature_cols=list(FEATURE_COLS))
    assert data_check_status(unchecked)[0] is Status.NOT_ASSESSED
    checked = dataset_overview(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        data_check=check_tables(objects, fovs, feature_cols=list(FEATURE_COLS)),
    )
    assert data_check_status(checked)[0] in (Status.OK, Status.CAUTION)
    assert integrity_status(None)[0] is Status.NOT_ASSESSED
    clean = integrity_checks(objects, fovs, feature_cols=list(FEATURE_COLS))
    # The second field of the first group holds a copy of the first field's observations.
    first, second = fovs["image_file"].iloc[0], fovs["image_file"].iloc[1]
    group = fovs.iloc[0][["condition", "biological_repeat"]].to_dict()
    in_group = objects["condition"].eq(group["condition"]) & objects["biological_repeat"].eq(
        group["biological_repeat"]
    )
    source = objects[in_group & objects["image_file"].eq(first)].assign(image_file=second)
    copied = pd.concat([objects[~(in_group & objects["image_file"].eq(second))], source])
    dirty = integrity_checks(copied, fovs, feature_cols=list(FEATURE_COLS))
    assert integrity_status(clean)[0] is Status.OK
    assert integrity_status(dirty)[0] is Status.PROBLEM
    document = with_opening(microscopist_report(overview=checked), checked, dirty)
    short, full = document.summary(), document.summary(detail="full")
    opening = short.split(f"{OVERVIEW_TITLE}\n")[1].split("\n\n")[0]
    assert 3 <= len(opening.splitlines()) <= 8
    assert f"{PROBLEM} Integrity:" in opening and "Data checks" in opening
    assert full.startswith(f"{PROBLEM} {OVERVIEW_TITLE}\n")
    for number in (str(len(objects)), str(len(fovs))):
        assert number in opening and number in full
    # The full opening carries the integrity summary in full, with its own marker.
    assert f"{PROBLEM} Data integrity — " in full and "Exact checks for copied data" in full
    assert f"{PROBLEM} {PROBLEM}" not in full


def test_the_short_report_opens_with_a_title_and_one_line_per_question():
    built = SCENARIOS["condition_repeat_groups"]()
    overview = dataset_overview(built.objects, built.fovs, feature_cols=list(FEATURE_COLS))
    report = microscopist_report(
        overview=overview, sufficiency=_groups_report([FixedFOVStrategy(2)])
    )
    text = report.summary()
    title, glance = text.split("\n\n")[:2]
    assert title.startswith("Microscopist's report — ")
    lines = glance.splitlines()
    # The key opens the block (two lines); the answers follow, one per question.
    answers = [line for line in lines[1:] if not line.startswith(("     ", "    ", "  Key:"))]
    assert lines[0] == GLANCE_TITLE and len(answers) == 1 + len(QUESTIONS)
    assert answers[0].startswith(f"  {NOT_ASSESSED} The data — ")
    assert all(line[2] in "✅⚠❌➖" for line in answers)
    assert f"{CAUTION} 2. Did I image enough? — only fixed amounts of effort" in glance
    assert f"{NOT_ASSESSED} 5. What did the extra fields buy? — not assessed" in glance
    full = report.summary(detail="full")
    assert GLANCE_TITLE not in full and f"{CAUTION} 2. Did I image enough?\n" in full
    # The full form keeps the old full text of every answer.
    assert report.sections["enough"] in full
    with pytest.raises(ValueError, match="detail"):
        report.summary(detail="medium")


def test_a_weak_representation_marks_the_control_and_the_adequacy_answers():
    from test_microscopist_audit import _weak_control_report

    controls, sufficiency = _weak_control_report()
    report = microscopist_report(controls=controls, sufficiency=sufficiency)
    assert report.statuses["controls"] is Status.PROBLEM
    assert report.statuses["enough"] is Status.PROBLEM
    assert report.headlines["enough"].startswith("weak representation; ")
    short = report.summary()
    glance = short.split("\n\n")[1]
    assert f"{PROBLEM} 2. Did I image enough? — weak representation;" in glance
    assert f"{PROBLEM} Weak representation: fields of the declared positive control" in short
    relaxed = microscopist_report(controls=controls, sufficiency=sufficiency, minimum_detection=0.1)
    assert relaxed.statuses["enough"] is not Status.PROBLEM
    assert relaxed.statuses["controls"] is not Status.PROBLEM
