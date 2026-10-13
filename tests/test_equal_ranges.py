"""A range whose ends read the same is one number: "34", never "34 to 34" (owner, 2026-10-12),
in ``_style.span`` and in the ranges written by hand."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from morphocoverage._style import low_to_high, span
from morphocoverage.retrospective.calibration import _not_preserved_cell
from morphocoverage.sampling_unit.distribution import spread_words
from morphocoverage.workflows._html_across import d_per_repeat
from morphocoverage.workflows._scan_text import _layout_lines


def test_low_to_high_and_span_write_equal_ends_once():
    assert low_to_high(34, 34) == "34" and low_to_high("3", "5") == "3 to 5"
    assert span([34, 34]) == "34" and span([3, 5, 4]) == "3 to 5"
    assert span([0.2, 0.2], "percent") == "20%" and span([0.0, 0.2], "percent") == "0 to 20%"


def test_the_scan_table_writes_one_number_when_every_repeat_holds_as_many():
    columns = [f"r{i}" for i in range(9)]  # more repeats than columns: one cell per condition
    layout = pd.DataFrame([[34] * 9, [30, 31, 32, 33, 34, 35, 36, 37, 38]], columns=columns)
    layout.index = ["siCTRL", "siKD"]
    scan = SimpleNamespace(layout=layout, settings={"observation_unit": "cell"})
    lines = _layout_lines(scan)
    rows = lines[1].splitlines()
    assert rows[1].split()[:3] == ["siCTRL", "9", "34"]
    assert " 30 to 38 " in rows[2]


def test_hand_written_ranges_with_equal_ends_are_one_number():
    same = {"mean": np.log(0.73), "low": np.log(0.701), "high": np.log(0.704)}
    assert spread_words(same) == "SD 0.73x the control (0.70)"
    row = SimpleNamespace(premature=0.2, not_preserved_by_stream=0.0004, not_preserved_at_stop=None)
    assert _not_preserved_cell(row) == "20%, 20% too early"
    alike = SimpleNamespace(d_per_repeat_mean=0.05, d_per_repeat_min=0.049, d_per_repeat_max=0.051)
    assert d_per_repeat(alike) == "+0.05"  # not "+0.05 (+0.05 to +0.05)"
