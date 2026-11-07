"""A contrast pairs repeats by their folder label; labels that share nothing are refused (#617).

MorphoCoverage pairs the biological repeats of two compared conditions by the value of the
``biological_repeat`` column (their folder label). When the repeats were confirmed as paired
under different folder names (CTRL/``day1`` with DrugA/``exp_A``), that pairing never reaches
this brick, so a label that matches nothing must not be read as "not compared" or paired by
accident. Until the Spec v1 reader carries the confirmed repeat group, such a contrast is
refused with a message naming both sets of labels. No pairing logic is added here.
"""

from __future__ import annotations

import pandas as pd
import pytest
from _helpers_sampling_unit import _two_conditions

from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.sampling_unit.discovery import feature_discovery
from morphocoverage.sampling_unit.ranking import feature_ranking

CONTROL = {"condition": "control"}
FAST = {"n_permutations": 99}
REFUSAL = r"repeat labels.*differ.*Spec v1"


def _renamed(renaming: dict[str, str], seed: int = 3):
    """Two conditions in R1 and R2, the treated one's repeats called by ``renaming``."""
    objects, fovs = _two_conditions(shift=2.0, field_sd=0.3, seed=seed)
    for table in (objects, fovs):
        treated = table["condition"] == "treated"
        table.loc[treated, "biological_repeat"] = table.loc[treated, "biological_repeat"].map(
            renaming
        )
    return objects, fovs


def _call(kind: str, objects: pd.DataFrame, fovs: pd.DataFrame):
    if kind == "contrast":
        return condition_contrast(
            objects, fovs, feature="x", control=CONTROL, n_draws=2, object_counts=(10,), **FAST
        )
    if kind == "ranking":
        return feature_ranking(objects, fovs, features=["x"], control=CONTROL, **FAST)
    return feature_discovery(objects, fovs, features=["x"], control=CONTROL, **FAST)


@pytest.mark.parametrize("kind", ["contrast", "ranking", "discovery"])
def test_repeat_labels_that_share_nothing_are_refused_with_both_label_sets(kind):
    objects, fovs = _renamed({"R1": "exp_A", "R2": "exp_B"})
    with pytest.raises(ValueError, match=REFUSAL) as caught:
        _call(kind, objects, fovs)
    message = str(caught.value)
    assert "treated" in message and "control" in message
    assert "'R1', 'R2'" in message and "'exp_A', 'exp_B'" in message


def test_declared_comparisons_are_refused_the_same_way():
    objects, fovs = _renamed({"R1": "exp_A", "R2": "exp_B"})
    with pytest.raises(ValueError, match=REFUSAL):
        condition_contrast(
            objects,
            fovs,
            feature="x",
            comparisons=[("treated", "control")],
            n_draws=2,
            object_counts=(10,),
            **FAST,
        )


def test_labels_that_match_are_compared_as_before():
    objects, fovs = _two_conditions(shift=2.0, field_sd=0.3)
    table = _call("contrast", objects, fovs).table
    assert sorted(table["biological_repeat"].astype(str)) == ["R1", "R2"]


def test_a_repeat_one_side_lacks_is_not_a_label_mismatch():
    # An uneven design (a condition not imaged in every repeat) keeps its shared labels.
    objects, fovs = _two_conditions(shift=2.0, field_sd=0.3)
    keep = ~((fovs["condition"] == "treated") & (fovs["biological_repeat"] == "R2"))
    keep_objects = ~((objects["condition"] == "treated") & (objects["biological_repeat"] == "R2"))
    table = _call("contrast", objects[keep_objects], fovs[keep]).table
    assert not table.empty


def test_a_table_without_a_repeat_column_is_not_affected():
    objects, fovs = _two_conditions(shift=2.0, field_sd=0.3)
    result = condition_contrast(
        objects,
        fovs,
        feature="x",
        control=CONTROL,
        repeat_col=None,
        n_draws=2,
        object_counts=(10,),
        **FAST,
    )
    assert not result.table.empty
