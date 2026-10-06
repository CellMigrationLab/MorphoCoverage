"""The guided session's rough runtime of a report and its page (private; re-exported by
``guided``), moved here to keep ``guided`` under 600 lines (audit 10, 2026-10-06)."""

from __future__ import annotations

import os
from collections.abc import Sequence

# A rough runtime of report() and html_report() together (owner, 2026-10-05: "acceptable,
# say it up front"), from the stress runs of 2026-10-02 and 2026-10-05 on 4 cores (5 to 300
# features, 3 to 151 conditions): seconds per feature and condition (the tests), per
# condition (the page's figures) and for the UMAP and the rest.
SECONDS_PER_PAIR = 0.6
SECONDS_PER_CONDITION = 8.0
SECONDS_FIXED = 60.0
# Each other feature table (owner, 2026-10-06): its populations, maps and copy checks, a
# fixed part and a part per column, from a report and page of 5,000 cells with a table of
# 400 and of 1,500 columns on 3 processes (65 s and 129 s more than without; 2026-10-06).
SECONDS_PER_TABLE = 40.0
SECONDS_PER_TABLE_COLUMN = 0.06


def rough_runtime(
    n_features: int, n_conditions: int, n_jobs: int = 4, tables: Sequence[int] = ()
) -> float:
    """The seconds a report and its page take, roughly: ``SECONDS_PER_PAIR`` for each
    feature and condition on 4 cores (fewer cores slower), ``SECONDS_PER_CONDITION`` for
    each condition and ``SECONDS_FIXED``; and for each other feature table (``tables``: its
    number of columns) ``SECONDS_PER_TABLE`` and ``SECONDS_PER_TABLE_COLUMN`` a column."""
    cores = (os.cpu_count() or 1) if n_jobs < 1 else n_jobs
    per_pair = SECONDS_PER_PAIR * 4 / max(1, min(cores, 4))
    conditions = max(1, n_conditions)
    others = sum(SECONDS_PER_TABLE + SECONDS_PER_TABLE_COLUMN * columns for columns in tables)
    return (
        per_pair * max(1, n_features) * conditions
        + SECONDS_PER_CONDITION * conditions
        + SECONDS_FIXED
        + others
    )


__all__ = [
    "SECONDS_FIXED",
    "SECONDS_PER_CONDITION",
    "SECONDS_PER_PAIR",
    "SECONDS_PER_TABLE",
    "SECONDS_PER_TABLE_COLUMN",
    "rough_runtime",
]
