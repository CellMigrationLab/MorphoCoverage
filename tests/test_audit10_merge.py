"""Owner decisions of 2026-10-06 taken in with the merge of the audit 10 groups: the closing
section points to the sections, and the rules' provenance stays in the full text report."""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import pytest

from morphocoverage._style import without_provenance
from morphocoverage.workflows import Setup, html_report, report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "scenarios"))
import make_scenarios

CONFIG = re.compile(r"config [0-9a-f]{16}")


def _report(name: str):
    loaded = make_scenarios.load(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(
            loaded.objects,
            loaded.fovs,
            Setup(**loaded.setup),
            feature_tables=loaded.feature_tables,
        )
    return loaded, result


def _limits(text: str) -> str:
    return " ".join(text[text.index("What this dataset cannot tell me") :].split())


def test_the_closing_section_points_to_each_section_in_both_forms() -> None:
    """One line per limit pointing to its section, the reason only in the section (owner,
    2026-10-06; G3 had repeated the reasons in the short form)."""
    _, result = _report("a01_tagged_several_image_none_rare")
    reason = "each field holds one image, so a field holds a state or not"
    for detail in ("short", "full"):
        text = result.summary(detail=detail)
        limits = _limits(text)
        assert "3. What is limiting my sampling? Answered in part: see question 3." in limits
        assert "Answered in part: see the Foundation." in limits
        assert reason not in limits
        assert reason in " ".join(text.split())  # said once, in its section
    assert "no conclusion possible here, see question 10" in _limits(result.summary())
    # The page leaves out what it cannot answer, so its closing card keeps the reasons.
    page = " ".join(result.document.limits(None, "short", pointers=False).split())
    assert reason in page


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


@pytest.mark.slow
def test_the_page_and_the_short_report_carry_no_provenance(tmp_path: Path) -> None:
    loaded, result = _report("a06_expl_none_image_none_tol")
    assert CONFIG.search(result.summary(detail="full"))
    assert not CONFIG.search(result.summary())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        page = html_report(
            result, loaded.objects, loaded.fovs, tmp_path, feature_tables=loaded.feature_tables
        )
    html = page.read_text()
    assert not CONFIG.search(html)
    assert "_upper95" not in html


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
