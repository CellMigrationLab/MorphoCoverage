"""The cell-level gate of ``feature_discovery`` (module ``discovery``) corrected: the tests at
p <= alpha rerun with the shuffles their family needs (``contrast_cells.refined``), then
Benjamini-Hochberg within each sub-experiment (``_experiments``), the candidates and the
confirmed features, near-identical features and the screen-wide q.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .._seeds import group_rng
from ._discovery_tests import near_identical
from ._experiments import corrected, declared, experiments, screen_wide
from .contrast_cells import Retest, refined


def _gated(
    tests: pd.DataFrame,
    held: Mapping[Any, Any],
    retests: Mapping[Any, Retest],
    families: Mapping[Any, Any] | None,
    random_state: int,
    alpha: float,
    workers: int,
) -> pd.DataFrame:
    """``tests`` (one row per feature and condition) with each condition's sub-experiment,
    the gate's p refined and every q, ``passes``, ``confirmed``, ``near_identical`` and
    ``family_sd``; ``held``: the repeats each condition is compared in."""
    tests["experiment"] = tests["condition"].map(declared(experiments(held), families))
    # The tests at p <= alpha rerun with the shuffles their family needs (contrast_cells).
    tests["p_cells"], tests["permutations"] = refined(
        tests["p_cells"],
        tests["experiment"],
        tests["permutations"],
        retests,
        lambda index: group_rng(
            random_state,
            {
                "feature": tests.at[index, "feature"],
                "condition": tests.at[index, "condition"],
                "test": "refined",
            },
        ),
        alpha,
        n_jobs=workers,
    )
    # Corrected over every feature and condition of a sub-experiment (_experiments).
    tests["q_cells"] = corrected(tests["p_cells"], tests["experiment"])
    tests["q_paired"] = corrected(tests["p_paired"], tests["experiment"])
    tests["q_paired_raw"] = corrected(tests["p_paired_raw"], tests["experiment"])
    tests["passes"] = (tests["q_cells"] <= alpha).fillna(False) & tests["d"].notna()
    tests["confirmed"] = tests["passes"] & (tests["q_paired"] <= alpha).fillna(False)
    tests["near_identical"], tests["family_sd"] = near_identical(tests)
    tests["q_screen"] = screen_wide(
        tests["p_paired"], tests["control_condition"], tests["experiment"]
    )
    return tests
