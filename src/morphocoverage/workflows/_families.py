"""One correction family for the whole of "Do the conditions differ?" (private; owner,
2026-10-02, after an audit of the tests behind the reading).

With several features, ``feature_discovery`` corrects each test (Benjamini-Hochberg) over
every feature and condition of a sub-experiment, while ``condition_contrast`` corrects the
readout's over its conditions only: the same comparison would get two q-values on one
page, and a readout could read as significant in its own table and not in the summary.
``shared_family`` gives the readout's contrast the discovery's numbers for the tests they
share:

- across the biological repeats, the paired t-test on the repeat means, normalised (or raw
  when the discovery could not normalise) and raw: p and q;
- the cell-level screen, in the version both trust (``trusted_test``: balanced and
  normalised, balanced, normalised or as provided): d, p and q.

The other versions of the cell-level screen keep their own correction over the readout's
conditions (``settings["family_features"]`` records the shared family, and the page says
which is which). The p-values are the same tests; only the permutation draws of the
cell-level screen differ, so its p is the discovery's.

With several controls the secondary rows of the contrast table run for every feature
(``shared_across``, owner 2026-10-05): each family is corrected over every feature and its
rows, the readout's rows (``ConditionContrast.across``) take that family's q, and every
feature's are in ``settings["across_features"]``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from ..sampling_unit.contrast_across import across_features

# The cell-level columns of condition_contrast, by (normalised, balanced).
_CELL_VERSIONS = {
    (True, True): "balanced_normalised",
    (False, True): "balanced",
    (True, False): "cells_normalised",
    (False, False): "cells",
}


def _cell_columns(version: str) -> tuple[str, str, str]:
    return f"d_{version}", f"p_{version}", f"q_{version}"


def shared_family(contrast: Any, discovery: Any, readout: str) -> Any:
    """``contrast`` with the discovery's p, q (and the screen's d) for the readout (see the
    module); unchanged without a discovery test of the readout."""
    tests = getattr(discovery, "tests", None)
    if tests is None or not len(tests) or readout not in set(tests["feature"]):
        return contrast
    mine = tests[tests["feature"] == readout].set_index("condition")
    repeats = _shared_repeats(contrast.repeats.copy(), mine)
    cells = contrast.cells.copy() if contrast.cells is not None else None
    if cells is not None and len(cells):
        _shared_cells(cells, mine)
    features = int(tests["feature"].nunique())
    # A labelled state's share is tested with the features and counted apart (audit 8).
    share = discovery.settings.get("share")
    share = share if share in set(tests["feature"]) else None
    settings = {**contrast.settings, "family_features": features, "family_share": share}
    return replace(contrast, repeats=repeats, cells=cells, settings=settings)


def _shared_repeats(repeats: pd.DataFrame, mine: pd.DataFrame) -> pd.DataFrame:
    """The contrast's repeat rows (a copy, set in place) with the discovery's paired p and q
    of their condition, and its screen's q."""
    for index, row in repeats.iterrows():
        if row["condition"] not in mine.index:
            continue
        test = mine.loc[row["condition"]]
        if pd.notna(test["p_paired"]):
            target = "_normalised" if test["paired_values"] == "normalised" else ""
            repeats.loc[index, f"p_paired_t{target}"] = test["p_paired"]
            repeats.loc[index, f"q_paired_t{target}"] = test["q_paired"]
        if pd.notna(test.get("p_paired_raw")):
            repeats.loc[index, "p_paired_t"] = test["p_paired_raw"]
            repeats.loc[index, "q_paired_t"] = test["q_paired_raw"]
        if "q_screen" in test:  # over the features too (audit 10, owner, 2026-10-06)
            repeats.loc[index, "q_screen"] = test["q_screen"]
    return repeats


def _shared_cells(cells: pd.DataFrame, mine: pd.DataFrame) -> None:
    """The contrast's cell-level rows (a copy, set in place) with the discovery's d, p and q
    of their condition, in the columns of the version it tested."""
    for index, row in cells.iterrows():
        if row["condition"] not in mine.index:
            continue
        test = mine.loc[row["condition"]]
        if pd.isna(test["p_cells"]):
            continue
        version = _CELL_VERSIONS[(bool(test["normalised"]), bool(test["balanced"]))]
        d, p, q = _cell_columns(version)
        if p in cells:
            cells.loc[index, [d, p, q]] = [test["d"], test["p_cells"], test["q_cells"]]


def shared_across(
    contrast: Any, objects: pd.DataFrame, fovs: pd.DataFrame, features: list[str], setup: Any
) -> Any:
    """``contrast`` with its secondary rows tested for every feature (see the module);
    unchanged without them."""
    table = contrast.settings.get("contrasts")
    if table is None or not len(contrast.across):
        return contrast
    every = across_features(
        objects,
        fovs,
        features=features,
        rows=table,
        random_state=setup.random_state,
        scales=setup.scales,
        **setup.common,
    )
    if every.empty:
        return contrast
    mine = every[every["feature"] == setup.readout].drop(columns="feature")
    across = mine.reset_index(drop=True) if len(mine) else contrast.across
    across.attrs = contrast.across.attrs
    settings = {**contrast.settings, "across_features": every.reset_index(drop=True)}
    return replace(contrast, across=across, settings=settings)


__all__ = ["shared_across", "shared_family"]
