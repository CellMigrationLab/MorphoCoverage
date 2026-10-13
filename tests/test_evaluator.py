"""Tests of ``evaluator``: group iteration and the framework's own subsets."""

import numpy as np
import pandas as pd
import pytest

from morphocoverage import validate_input_tables
from morphocoverage.evaluator import FovSubsets


def test_b18_fovsubsets_take_refuses_repeated_rows():
    fovs = pd.DataFrame({"position_id": ["p0", "p1", "p2"]})
    objects = pd.DataFrame({"position_id": np.repeat(fovs.position_id, 4), "f": np.arange(12.0)})
    subsets = FovSubsets(validate_input_tables(objects, feature_cols=["f"], fovs=fovs))
    with pytest.raises(ValueError, match="distinct FOV rows"):
        subsets.take(np.array([0, 0, 1]))
    assert len(subsets.take(np.array([0, 2])).objects) == 8
