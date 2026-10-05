"""The owner's decisions of 2026-10-05 on the retrospective report's wording: a post-hoc
"best" allocation or rule family is described, not tested (#243, #244); the
representation-control rate is pooled over its strata, and said so (#253); a declared
planted check that could not run leaves the representation gate not assessed (#270);
assessed and not assessed groups are counted apart in the headlines (#272); and the
between-well spread names what it leaves out (#150)."""

import pandas as pd

from morphocoverage._style import Status
from morphocoverage.retrospective import effort_text
from morphocoverage.retrospective.frontier import BASELINE, FrontierResult
from morphocoverage.retrospective.frontier_text import frontier_status

# --- #243, #244: a post-hoc selected best is described ---------------------------------------


def _allocations(ratios):
    """Equal-total allocations, each resolved by its own interval, with these SE ratios."""
    return pd.DataFrame(
        {
            "samples": [10 * (i + 2) for i in range(len(ratios))],
            "fovs_per_sample": [4.0] * len(ratios),
            "ratio": ratios,
            "ratio_low": [r - 0.05 for r in ratios],
            "ratio_high": [r - 0.01 for r in ratios],
            "same_total": [True] * len(ratios),
            "resolved": [True] * len(ratios),
            "current_samples": [12] * len(ratios),
            "current_fovs_per_sample": [4.0] * len(ratios),
        }
    )


def test_the_smallest_resolved_allocation_is_described_not_called_the_answer():
    compared = _allocations([0.9, 0.8, 0.85])
    line = effort_text._allocation_line(compared, {}, 12, "biological_repeat", "sample")
    assert "the smallest standard error that resolved here is with 30 samples" in line
    assert "20% smaller than the acquired 12 samples" in line
    assert "would give" not in line and "best" not in line
    answer, reason = effort_text._allocation_cell(
        compared, {}, 12, "biological_repeat", "sample"
    )
    assert reason is None
    assert answer == "30 samples of about 4 fields: 20% smaller SE, the smallest that resolved here"


def _score(family, saved, low):
    return {
        "family": family,
        "alpha": 0.05,
        "setting": 0.1,
        "effort": 6.0 - saved,
        "risk": 0.04,
        "fields_saved": saved,
        "saved_low": low,
        "saved_high": saved + 1,
        "saved_reached": 1.0,
        "adaptivity": 0.3,
    }


def _frontier(rows):
    base = {**_score(BASELINE, 0.0, None), "setting": 6.0, "effort": 6.0}
    settings = {"label": "the mean", "units": 8, "unit_cols": ["biological_repeat"]}
    return FrontierResult(pd.DataFrame(), pd.DataFrame([base, *rows]), pd.DataFrame(), settings)


def test_the_family_that_saved_most_is_described_when_several_were_scored():
    one = _frontier([_score("saver", 2.0, 0.5)])
    assert frontier_status(one) == (Status.OK, "at ≤5% risk, saver saves 2.0 of 6 fields")
    several = _frontier([_score("saver", 2.0, 0.5), _score("other", 1.0, 0.2)])
    status, text = frontier_status(several)
    assert status is Status.OK
    assert "saver saves the most here, 2.0 of 6 fields" in text
    unresolved = _frontier([_score("saver", 2.0, -0.5), _score("other", 1.0, -0.2)])
    assert "saver may save the most here, 2.0 fields (not resolved)" in frontier_status(
        unresolved
    )[1]
