"""Tests of ``_style``: status markers, labels, tables and wrapping of the text output."""

from __future__ import annotations

import numpy as np
import pytest
from _helpers_retrospective import _acquisition, _inspect
from _helpers_sampling_unit import _flat_text

from morphocoverage._style import CAUTION, rule_label, without_provenance


def test_provenance_is_kept_in_the_full_report_only() -> None:
    """``[fixed_fov (...)]`` and ``(name: settings; config ...)``: the full text report keeps
    them, the short report and the page never (owner, 2026-10-06)."""
    text = (
        "Stopping rules: mean-precision rule, per cell, 5% [scalar_precision_object_upper95\n"
        "    (feature=Area, target_cv=0.05)]; fixed 8 fields [fixed_fov (target_fovs=8)].\n"
        "Rules: per cell, 5% (scalar_precision_object_upper95: averaging_cv <= 0.05;\n"
        "    config 8d1fdaa201845e06); 3.2 [1.9–4.2] (intensity_mean) kept."
    )
    assert without_provenance(text) == (
        "Stopping rules: mean-precision rule, per cell, 5%; fixed 8 fields.\n"
        "Rules: per cell, 5%; 3.2 [1.9–4.2] (intensity_mean) kept."
    )


def test_a_long_line_is_wrapped_with_its_continuation_lines() -> None:
    """``fitted`` wrapped a line of 101-104 columns alone, leaving its last word on a line of
    its own ("    the", "      ≤") in the committed full reports; it now wraps the line with
    its continuation lines, a list item or a deeper block left apart."""
    from morphocoverage._fitted import fitted

    long = "word " * 19 + "xx the"
    text = (
        f"  Header line that is short.\n  {long}\n    control, per repeat: a few words.\n"
        f"    - {long} xx\n      of 3 repeats.\n    - next item\n"
        "    feature   a    b\n"
    )
    lines = fitted(text).splitlines()
    assert all(len(line.strip()) > 3 for line in lines)
    assert all(len(line) <= 100 for line in lines)
    assert " ".join(fitted(text).split()) == " ".join(text.split())
    assert "    - next item" in lines and "    feature   a    b" in lines


def test_rule_label_keeps_the_name_of_an_unknown_variant():
    assert rule_label({"criterion": "scalar_precision_x", "settings": ""}, "cell") == (
        "scalar_precision_x"
    )
    assert rule_label({"criterion": "descriptor_spread_x", "settings": ""}, "cell") == (
        "descriptor_spread_x"
    )
    row = {"criterion": "support_novelty", "settings": "", "parameters": {"quantity": "other"}}
    assert rule_label(row, "cell") == "support_novelty"


def test_the_fields_to_reopen_are_one_table_and_the_full_form_keeps_every_reason():
    objects, fovs = _acquisition()
    fovs.loc[3, "stage_x"] = np.nan  # the stage neighbours cannot be compared
    result = _inspect(objects, fovs, stage_cols=("stage_x", "stage_y"))
    short = result.summary()
    assert short.splitlines()[0].startswith("3 fields to reopen in 1 group")
    # A long reason (an influence) does not fit a table row: one line per field.
    assert "control · repeat R1 · f07: removing it lowers" in short
    for name in ("f03", "f07", "f12"):
        assert name in short
    # Said once, and the reason not judged is a caveat.
    assert short.count("None of these is called an error") == 1
    assert f"{CAUTION} Stage neighbours were not compared in control · repeat R1" in short
    full = result.summary(detail="full")
    assert "Analysis groups: condition · biological repeat." in full
    assert "control · repeat R1: fields to reopen before trusting the result" in full
    for detail in result.fovs["detail"]:
        assert _flat_text(detail) in _flat_text(full)
    assert "f07 (stage_x 200.0, stage_y 100.0)" in full
    with pytest.raises(ValueError, match="short"):
        result.summary(detail="long")
