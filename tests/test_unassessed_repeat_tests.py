"""A paired test across repeats that cannot be computed is "not assessed", with its reason (#90).

With 3 or more biological repeats whose changes are all the same, the paired t-test has no
variance to test with: it gives no p. That is not a test that found nothing, and not a design
with too few repeats: the verdict, the table on the page, panel C of the figures and the plan
of the next experiment say it is not assessed and why, never "not significant", and never
leave the condition out without a word.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from morphocoverage.plots_pairwise import (
    PAIRED_NOTE,
    add_pairwise,
    pairwise_contrasts,
)
from morphocoverage.sampling_unit import reading_text
from morphocoverage.sampling_unit.contrast_repeats import NO_SPREAD
from morphocoverage.sampling_unit.reading_text import (
    CONFIRMED,
    NOT_SIGNIFICANT,
    NOT_TESTED,
    VERDICTS,
    condition_readings,
)
from morphocoverage.sampling_unit.repeat_plan import DEGENERATE, MISSING_VALUE
from morphocoverage.workflows import Setup, html_report, report
from morphocoverage.workflows._html_across import across_frame
from morphocoverage.workflows._html_plan import _difference_goal
from morphocoverage.workflows._html_plotly import _pairwise

# The verdict of a paired test that could not be computed (``reading_text.NOT_ASSESSED``).
NOT_ASSESSED = "not assessed across repeats"


def _same_change(repeats: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``A`` one unit above ``CTRL`` in every repeat, the same fields in each: the change is
    identical in every repeat, so the paired test has no spread."""
    rows, fovs = [], []
    for repeat in [f"r{i + 1}" for i in range(repeats)]:
        for condition, shift in (("CTRL", 0.0), ("A", 1.0)):
            for field in range(4):
                key = {
                    "condition": condition,
                    "biological_repeat": repeat,
                    "position_id": f"p{field}",
                }
                fovs.append({**key, "repeat_group": repeat})
                rows += [{**key, "x": 10.0 + shift + 0.3 * field + 0.1 * k} for k in range(5)]
    return pd.DataFrame(rows), pd.DataFrame(fovs)


def _row(condition: str, n: int, lower: int, higher: int, q: float) -> dict:
    return {
        "condition": condition,
        "control_condition": "CTRL",
        "repeats": n,
        "lower": lower,
        "higher": higher,
        "d_mean": 1.0,
        "q_paired_t": q,
        "q_paired_t_normalised": q,
    }


def _contrast(rows: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        table=pd.DataFrame(),
        settings={"repeat_col": "biological_repeat", "observation_unit": "cell"},
        repeats=pd.DataFrame(rows),
        cells=pd.DataFrame(),
        targets=pd.DataFrame(),
    )


@pytest.mark.parametrize("n", [3, 6])
def test_the_reading_of_equal_repeat_changes_is_not_assessed_never_not_significant(n) -> None:
    readings = condition_readings(_contrast([_row("A", n, 0, n, np.nan)])).set_index("condition")
    assert readings.loc["A", "verdict"] == NOT_ASSESSED
    assert getattr(reading_text, "NOT_ASSESSED", None) == NOT_ASSESSED
    assert NOT_ASSESSED in VERDICTS
    assert (
        "no paired test: the change is the same in every biological repeat"
        in (readings.loc["A", "reading"])
    )


def test_the_other_reading_verdicts_are_unchanged() -> None:
    contrast = _contrast(
        [_row("A", 3, 0, 3, 0.01), _row("B", 3, 0, 3, 0.4), _row("C", 2, 0, 2, np.nan)]
    )
    readings = condition_readings(contrast).set_index("condition")
    assert readings.loc["A", "verdict"] == CONFIRMED
    assert readings.loc["B", "verdict"] == NOT_SIGNIFICANT
    assert readings.loc["C", "verdict"] == NOT_TESTED


def _repeats_row(condition: str, n: int, higher: int, p: float) -> dict:
    return {
        "condition": condition,
        "control_condition": "CTRL",
        "repeats": n,
        "lower": n - higher,
        "higher": higher,
        "fold_change": 1.1,
        "mean_difference": 1.0,
        "p_paired_t": p,
        "q_paired_t": p,
        "p_paired_t_normalised": p,
        "q_paired_t_normalised": p,
        "d_mean": 1.0,
        "d_min": 1.0,
        "d_max": 1.0,
        "repeat_contrasts": "",
        "ci_low": np.nan,
        "ci_high": np.nan,
    }


def test_the_page_table_says_not_assessed_with_the_reason() -> None:
    contrast = SimpleNamespace(
        repeats=pd.DataFrame(
            [
                _repeats_row("A", 3, 3, np.nan),
                _repeats_row("B", 6, 6, np.nan),
                _repeats_row("C", 3, 3, 0.4),
                _repeats_row("D", 2, 2, np.nan),
            ]
        )
    )
    result = SimpleNamespace(
        discovery=None,
        results={"contrast": contrast},
        setup=SimpleNamespace(readout="x", unit_word="cell"),
    )
    verdicts = dict(zip(*across_frame(result)[["condition", "verdict"]].T.to_numpy(), strict=True))
    for condition in ("A", "B"):
        assert verdicts[condition] == f"{NOT_ASSESSED}: {NO_SPREAD}"
        assert "not significant" not in verdicts[condition]
    assert verdicts["C"] == NOT_SIGNIFICANT
    assert verdicts["D"] == "not tested (needs 3 repeats)"


def test_panel_c_marks_equal_repeat_changes_apart_from_too_few_repeats() -> None:
    objects, fovs = _same_change()
    table = pairwise_contrasts(objects, fovs, feature="x", control="CTRL", n_permutations=99)
    [pair] = table.itertuples()
    assert pair.shared_repeats == 3 and np.isnan(pair.p_paired)
    figure = Figure(figsize=(7, 5))
    figure.add_subplot()
    add_pairwise(figure, table, ["CTRL", "A"])
    panel_c = figure.axes[-1]
    cells = {t.get_text() for t in panel_c.texts}
    assert "=" in cells and "–" not in cells
    assert "=: not assessed" in PAIRED_NOTE
    assert "– : fewer than 3 repeats imaged both" in PAIRED_NOTE
    hovers = [h for line in _pairwise(table, ["CTRL", "A"])["c"]["hover"] for h in line if h]
    assert hovers and all("not assessed" in h and "fewer than" not in h for h in hovers)
    assert {t for line in _pairwise(table, ["CTRL", "A"])["c"]["text"] for t in line if t} == {"="}


def test_panel_c_still_says_too_few_repeats_with_two() -> None:
    objects, fovs = _same_change(repeats=2)
    table = pairwise_contrasts(objects, fovs, feature="x", control="CTRL", n_permutations=99)
    hovers = [h for line in _pairwise(table, ["CTRL", "A"])["c"]["hover"] for h in line if h]
    assert hovers and all("2 repeats imaged both, fewer than 3" in h for h in hovers)


def _plan_result(needed: list[float], status: list[str]) -> SimpleNamespace:
    discovery = SimpleNamespace(
        tests=pd.DataFrame(
            {
                "feature": ["area"] * 3,
                "condition": ["a", "b", "c"],
                "repeats": [3, 3, 3],
                "confirmed": [True, False, False],
            }
        ),
        ranking=pd.DataFrame({"feature": ["area"]}),
    )
    plan = pd.DataFrame(
        {
            "condition": ["a", "b", "c"],
            "repeats_needed": needed,
            "repeats_needed_twice_fields": needed,
            "planned_on": ["log ratio"] * 3,
            "change": [0.2, 0.1, -0.1],
            "status": status,
        }
    )
    contrast = SimpleNamespace(plan=plan, table=pd.DataFrame(), settings={})
    return SimpleNamespace(
        discovery=discovery,
        setup=SimpleNamespace(readout="area", columns=("area",)),
        results={"contrast": contrast},
    )


def test_a_plan_with_no_estimable_answer_is_shown_as_not_assessable_not_dropped() -> None:
    # D64: b and c have no plan; the panel said "Every condition is significant across
    # repeats" although only a is.
    result = _plan_result([4.0, np.nan, np.nan], ["ok", DEGENERATE, MISSING_VALUE])
    _, panel = _difference_goal(result)
    assert "Every condition is significant" not in panel
    assert "<td>b</td>" in panel and "<td>c</td>" in panel
    assert f"not assessable: {DEGENERATE}" in panel.replace("&#x27;", "'")
    assert f"not assessable: {MISSING_VALUE}" in panel.replace("&#x27;", "'")


def test_equal_repeat_changes_are_not_assessed_in_the_report_and_on_its_page(tmp_path) -> None:
    objects, fovs = _same_change()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(
            objects, fovs, Setup(readout="x", features=("x",), control={"condition": "CTRL"})
        )
        page = html_report(result, objects, fovs, tmp_path / "page").read_text(encoding="utf-8")
    full = " ".join(result.summary(detail="full").split())
    assert "A — not assessed across repeats" in full
    assert "A — not significant across repeats" not in full
    frame = pd.read_csv(tmp_path / "page" / "figures" / "tests_across_repeats.csv")
    assert list(frame["verdict"]) == [f"{NOT_ASSESSED}: {NO_SPREAD}"]
    # The glossary explains "not significant across repeats"; no verdict on the page reads it.
    assert "<strong>Not significant across repeats</strong>" not in page
    assert "<strong>Not assessed across repeats</strong>: A (higher)" in page
    assert f"<td>{NOT_ASSESSED}: {NO_SPREAD}</td>" in page
    assert "fewer than 3)" not in page
