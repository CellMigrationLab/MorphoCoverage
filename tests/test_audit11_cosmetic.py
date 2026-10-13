"""Audit A4 and A7: counts that add up, labels that do not fall on one another."""

from __future__ import annotations

from itertools import pairwise

from morphocoverage.plots_embedding import _spread_labels
from morphocoverage.workflows._enough_features import _needed


def _row(**changes):
    return {"needing": 0, "too_small": 0, "between_wells": 0, "unassessed": 0, **changes}


def test_comparisons_not_projected_are_counted_so_the_row_adds_up():
    """A4: 0 shown + 1 too small + 1 not assessable of 3 left one comparison unaccounted for."""
    said = _needed(_row(needing=1, flagged=False, too_small=1, unassessed=1), "images")
    assert "1 not projected (no candidate difference)" in said
    assert "1 too small to show" in said and "1 not assessable" in said


def test_a_flagged_feature_still_gets_its_projection():
    said = _needed(_row(needing=2, flagged=True, fields_low=12, fields_high=20), "images")
    assert "about 12 to 20 images a side" in said and "not projected" not in said


def test_labels_of_arrows_ending_together_are_moved_apart():
    """A7: arrows of duplicated features end at one point; their labels must not overprint."""
    spots = _spread_labels([(1.0, 1.0), (1.0, 1.0), (1.0, 1.01), (-5.0, 3.0)], 20.0, 10.0)
    near = [s for s in spots if s[0] == 1.0]
    ys = sorted(y for _, y in near)
    assert all(b - a >= 0.05 * 10.0 - 1e-9 for a, b in pairwise(ys))
    assert (-5.0, 3.0) in spots  # a label elsewhere stays where it was
