"""Owner decisions of 2026-10-06 taken in with the merge of the audit 10 groups: the closing
section points to the sections, and the rules' provenance stays in the full text report."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

from morphocoverage.workflows import Setup, report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "scenarios"))
import make_scenarios


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
    assert "no group, see question 10" in _limits(result.summary())
    # The page leaves out what it cannot answer, so its closing card keeps the reasons.
    page = " ".join(result.document.limits(None, "short", pointers=False).split())
    assert reason in page
