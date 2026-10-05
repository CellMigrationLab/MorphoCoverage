"""Groups are candidates on a UMAP, judged on all the features (owner, 2026-10-02, after
HDBSCAN on the principal components split a Gaussian and a straight line into populations
and missed a separate blob): a group has its cells' neighbours inside it, a continuum has
none; whether a group is a separate state is for the images (no gap test was reliable), and
each comes with the simplest rule on the features that picks out its cells."""

import numpy as np
import pandas as pd
import pytest

from morphocoverage.sampling_unit import find_populations
from morphocoverage.sampling_unit._population_rule import _folds, population_rule
from morphocoverage.sampling_unit.populations import _purity
from morphocoverage.workflows._populations import rule_words


def _tables(values: np.ndarray, seed: int = 0):
    """``values`` as cells of 2 conditions x 3 repeats, 20 cells a field."""
    n = len(values)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    frame = pd.DataFrame(values[order], columns=[f"f{i}" for i in range(values.shape[1])])
    frame["condition"] = np.where(np.arange(n) % 2, "a", "b")
    frame["biological_repeat"] = [f"r{i % 3}" for i in range(n)]
    frame["image_file"] = [
        f"{c}_{r}_{i // 120}.tif"
        for i, (c, r) in enumerate(zip(frame["condition"], frame["biological_repeat"], strict=True))
    ]
    fovs = frame[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return frame, fovs, [c for c in frame.columns if c.startswith("f")]


def _found(values: np.ndarray) -> pd.DataFrame:
    objects, fovs, features = _tables(values)
    return find_populations(objects, fovs, features=features, observation_unit="cell").table


def test_a_gaussian_has_no_population():
    rng = np.random.default_rng(1)
    assert _found(rng.normal(size=(2400, 4))).empty


def test_a_line_is_at_most_a_group_to_validate():
    rng = np.random.default_rng(1)
    t = rng.normal(size=(2400, 1))
    table = _found(np.hstack([t, 2 * t, -t]) + 0.3 * rng.normal(size=(2400, 3)))
    # Its end may read as a group: validated on the images, as the text says.
    assert set(table["kind"]) <= {"main", "group"}


def test_a_separate_blob_is_a_group_with_a_rule_that_finds_it():
    rng = np.random.default_rng(1)
    values = np.vstack([rng.normal(size=(2160, 4)), rng.normal(size=(240, 4)) + [5, 0, 0, 0]])
    table = _found(values).set_index("population")
    assert list(table["kind"]) == ["main", "group"]
    assert table.loc["G1", "observations"] == pytest.approx(240, abs=12)
    assert table.loc["G1", "purity"] >= 0.94
    # One threshold on the feature that sets it apart, right on held-out cells.
    assert table.loc["G1", "rule"].startswith("f0 >")
    assert table.loc["G1", "rule_recall"] > 0.95 and table.loc["G1", "rule_precision"] > 0.95


def test_the_purity_tells_a_group_from_a_tail_and_the_rule_reads_in_its_units():
    rng = np.random.default_rng(2)
    blob = np.vstack([rng.normal(size=(900, 2)), rng.normal(size=(100, 2)) + [6, 0]])
    member = np.arange(1000) >= 900
    assert _purity(blob, member) > 0.95
    rule = population_rule(blob * 10 + 100, member, ["area", "speed"])
    [(feature, op, threshold)] = rule.terms[:1]
    assert feature == "area" and op == ">" and 120 < threshold < 160
    assert rule.code().startswith('(objects["area"] > ')
    assert rule.recall > 0.95


def _shifted_fields():
    """10 fields of ordinary cells; the group in fields 0 and 1, sitting lower in field 1 (a
    shift of that field): a threshold fitted with field 1's cells finds them, one fitted
    without them does not."""
    rng = np.random.default_rng(0)
    values, fields, member = [], [], []
    for field in range(10):
        values.append(rng.normal(0, 1, (200, 2)))
        fields += [field] * 200
        member += [False] * 200
    for field, centre in ((0, 6.0), (1, 2.8)):
        values.append(np.c_[rng.normal(centre, 0.2, 40), rng.normal(0, 1, 40)])
        fields += [field] * 40
        member += [True] * 40
    return np.vstack(values), np.array(fields), np.array(member)


def test_the_rule_is_checked_on_whole_held_out_fields():
    """#88 (owner, 2026-10-05): cells of one field in both the fitting and the judging made
    the rule look transferable; held-out fields show it is not."""
    values, fields, member = _shifted_fields()
    cells = population_rule(values, member, ["f0", "f1"])
    grouped = population_rule(values, member, ["f0", "f1"], groups=fields)
    # The rule shown is the same, fitted on every cell; only its check differs.
    assert grouped.terms == cells.terms
    assert cells.held_out == "cells" and grouped.held_out == "fields"
    assert cells.recall > 0.9 and grouped.recall < 0.6
    # No field is in more than one held-out fold: never in the fitting and the judging both.
    folds, held_out = _folds(member, fields, 0)
    assert held_out == "fields" and len(folds) == 2
    assert sorted(np.concatenate(folds)) == list(range(len(member)))
    seen = [set(fields[test]) for test in folds]
    assert not seen[0] & seen[1]
    # Each fold holds out one of the group's two fields: the group stays evaluable.
    assert all(member[test].any() for test in folds)


def test_a_group_in_one_field_is_checked_on_held_out_cells_and_says_so():
    values, fields, member = _shifted_fields()
    keep = fields != 1
    rule = population_rule(values[keep], member[keep], ["f0", "f1"], groups=fields[keep])
    assert rule.held_out == "cells" and rule.recall > 0.95
    row = pd.Series(
        {
            "rule": rule.words(),
            "rule_recall": rule.recall,
            "rule_precision": rule.precision,
            "rule_held_out": rule.held_out,
        }
    )
    # The numbers are the procedure's, the rule shown fitted on every cell.
    words = rule_words(row, "cells")
    assert "fitted on all the cells" in words
    assert "refitted with cells held out (the group is in one field)" in words
    assert "whole fields" in rule_words(row.replace("cells", "fields"), "cells")


def test_a_group_in_one_field_is_seen_in_its_largest_field_share():
    """#79 (owner, 2026-10-05): the share of a group's cells in its largest field tells a
    group spread over the fields from one living in one image."""
    rng = np.random.default_rng(1)
    values = np.vstack([rng.normal(size=(2160, 4)), rng.normal(size=(240, 4)) + [5, 0, 0, 0]])
    objects, fovs, features = _tables(values)
    spread = find_populations(objects, fovs, features=features, observation_unit="cell")
    group = spread.table.set_index("population").loc["G1"]
    assert group["largest_field"] < 0.2 and group["fields"] > 5
    # Without a technical_repeat column there are none to count.
    assert group["technical_repeats"] is None
    # The same cells moved into one field of a technical repeat each.
    blob = objects["f0"] > 3
    one = objects.assign(technical_repeat="w1")
    one.loc[blob, ["condition", "biological_repeat", "image_file"]] = ["a", "r0", "a_r0_0.tif"]
    found = find_populations(
        one,
        one[[*fovs.columns, "technical_repeat"]].drop_duplicates(),
        features=features,
        observation_unit="cell",
    )
    group = found.table.set_index("population").loc["G1"]
    assert group["largest_field"] > 0.9
    inside = found.labels.reindex(one.index) == "G1"
    wells = one.loc[inside, ["condition", "biological_repeat", "technical_repeat"]]
    assert group["technical_repeats"] == len(wells.drop_duplicates())
