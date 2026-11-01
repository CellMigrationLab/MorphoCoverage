"""MorphoCoverage's copies of the MorphoBricks core constants equal the core's own values.

MorphoCoverage takes no MorphoBricks dependency (owner decision D-g, 2026-10-30): it reads
tables, never images, and keeps two constants of the core as literals. This test binds them to
the preprocessing brick v0.3.4 (which brings morphobricks-normalization v0.3.2). It runs where
preprocessing is installed and skips, saying why, where it is not.
"""

from importlib.metadata import PackageNotFoundError, version

import pytest

from morphocoverage._style import SINGLE_LEVEL
from morphocoverage.metadata import MORPHOBRICKS_GROUP_COLUMNS

CORE = "0.3.4"  # the preprocessing tag of the D86 core (MorphoBricks decision D86)


def _core():
    """Return the preprocessing and normalization modules, or skip when they are absent."""
    preprocessing = pytest.importorskip(
        "morphobricks_preprocessing",
        reason="morphobricks-preprocessing is not installed (MorphoCoverage does not depend on "
        "it, D-g); install its v0.3.4 tag to check the constants against the core",
    )
    try:
        installed = version("morphobricks-preprocessing")
    except PackageNotFoundError:  # fallback: importable but not installed as a distribution
        installed = "unknown"
    if installed != CORE:
        pytest.skip(f"morphobricks-preprocessing {installed} is installed; the check needs {CORE}")
    from morphobricks_normalization.records import SINGLE

    return preprocessing, SINGLE


def test_single_level_is_the_core_single():
    """``"single"``, CellPatcher's and the core's word for a level the folders did not have."""
    _, single = _core()
    assert SINGLE_LEVEL == single


def test_group_columns_and_fov_identity_are_the_core_fov_key():
    """The three group levels, then ``position_id``: preprocessing's ``FOV_KEY``."""
    preprocessing, _ = _core()
    assert (*MORPHOBRICKS_GROUP_COLUMNS, "position_id") == tuple(preprocessing.FOV_KEY)
