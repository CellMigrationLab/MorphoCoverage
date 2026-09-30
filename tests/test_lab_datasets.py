"""The lab datasets in data/lab: cards complete, tables valid, the paper's numbers reproduced."""

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morphocoverage import check_tables

LAB = Path(__file__).resolve().parents[1] / "data" / "lab"
sys.path.insert(0, str(LAB))
import catalog

IDS = catalog.ids()


@pytest.mark.parametrize("dataset_id", IDS)
def test_the_card_is_complete_and_follows_the_rules(dataset_id):
    assert catalog.check_card(catalog.read_card(dataset_id)) == []


@pytest.mark.parametrize("dataset_id", IDS)
def test_the_tables_pass_check_tables_without_errors(dataset_id):
    data = catalog.load(dataset_id)
    result = check_tables(
        *data.tables,
        feature_cols=data.feature_cols,
        observation_unit=data.observation_unit,
        control=data.control,
    )
    assert result.errors == []
    # Every warning is either fixed or explained in the card; this dataset has none.
    assert result.warnings == []


@pytest.mark.parametrize("dataset_id", IDS)
def test_dataset_md_is_the_rendering_of_the_card(dataset_id):
    assert (LAB / dataset_id / "DATASET.md").read_text() == catalog.render(dataset_id)


@pytest.mark.parametrize("dataset_id", IDS)
def test_the_design_counts_in_the_card_match_the_tables(dataset_id):
    data = catalog.load(dataset_id)
    counts = data.card["design"].get("counts", {})
    objects = data.objects
    assert len(objects) == counts["cells"]
    assert objects["condition"].nunique() == counts["conditions"]
    assert objects["biological_repeat"].nunique() == counts["biological_repeats"]
    assert len(data.fovs) == len(data.fovs.drop_duplicates())


@pytest.mark.parametrize("dataset_id", IDS)
def test_hierarchy_placeholders_are_the_agreed_words(dataset_id):
    data = catalog.load(dataset_id)
    for column in ("condition", "biological_repeat", "technical_repeat"):
        values = set(data.objects[column])
        assert "" not in values  # empty means "not recorded", which the tables never leave
        assert not {v for v in values if v.lower() in {"none", "nan", "na", "unknown"}}


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    return abs(a.mean() - b.mean()) / np.sqrt(pooled)


def test_cct8_screen_reproduces_the_papers_counts_means_and_effect_sizes():
    """Fig. 2C: n per condition, the pooled siCTRL, and the deposit's Cohen's d (4 decimals)."""
    objects = catalog.load("cct8_filopodia_screen").objects
    x = "Filopodia number"
    per = objects[objects.condition != "siCTRL"].groupby("condition")[x]
    assert per.size().min() == 76 and per.size().max() == 113  # "n = 76-113 cells per condition"
    control = objects.loc[objects.condition == "siCTRL", x].to_numpy(float)
    assert len(control) == 555 and objects["experiment"].nunique() == 6  # "pools ... 6 sets"
    # Cohen's d against the pooled siCTRL, from the deposit's Stats/ table.
    paper_d = {
        "siMAP4K4#5": 0.0962,
        "siMAP4K4#6": 0.5021,
        "siNF2#7": 0.2274,
        "siNF2#17": 0.1929,
        "siMINK1#3": 0.5115,
        "siMINK1#10": 0.7487,
        "siCCT8#5": 0.9926,
        "siCCT8#8": 0.9708,
        "siCSNK1A1#5": 1.2404,
        "siCSNK1A1#11": 1.1563,
        "siSCRIB#6": 1.0703,
        "siSCRIB#7": 1.0439,
    }
    for condition, d in paper_d.items():
        values = objects.loc[objects.condition == condition, x].to_numpy(float)
        assert round(_cohen_d(control, values), 4) == pytest.approx(d, abs=1e-4), condition
    # The claimed direction: the four hits lower, below the pooled control mean of 56.9.
    means = per.mean()
    for hit in ("siMINK1", "siCSNK1A1", "siSCRIB", "siCCT8"):
        assert (means[means.index.str.startswith(hit)] < control.mean()).all()


def test_cdm_morphology_reproduces_the_papers_counts_and_effect_sizes():
    """Fig. 5E: n per condition and the deposit's Cohen's d for area and aspect ratio."""
    objects = catalog.load("cct8_cdm_morphology").objects
    sizes = objects.groupby("condition").size()
    assert sizes.to_dict() == {"siCTRL": 53, "siCCT8#5": 63, "siCCT8#8": 58}  # "n > 53"
    assert objects["biological_repeat"].nunique() == 3
    # Cohen's d against siCTRL, from the deposit's Stats/ tables.
    paper_d = {"Area": (0.4362, 0.5682), "AR": (1.5631, 1.5430)}
    control = objects[objects.condition == "siCTRL"]
    for feature, (d5, d8) in paper_d.items():
        for condition, d in (("siCCT8#5", d5), ("siCCT8#8", d8)):
            values = objects.loc[objects.condition == condition, feature].to_numpy(float)
            got = _cohen_d(control[feature].to_numpy(float), values)
            assert round(got, 4) == pytest.approx(d, abs=1e-4), (feature, condition)
    # "smaller, more rounded cells": both siRNAs lower than siCTRL in every repeat.
    means = objects.groupby(["biological_repeat", "condition"])[["Area", "AR"]].mean()
    for feature in ("Area", "AR"):
        table = means[feature].unstack()
        assert (table[["siCCT8#5", "siCCT8#8"]].lt(table["siCTRL"], axis=0)).all().all()


def _convert_module(dataset_id):
    spec = importlib.util.spec_from_file_location("convert", LAB / dataset_id / "convert.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.network
@pytest.mark.parametrize("dataset_id", IDS)
def test_convert_rebuilds_the_stored_tables_from_the_pinned_source(dataset_id):
    module = _convert_module(dataset_id)
    objects, fovs = module.build_tables(module.read_source(module.fetch_member()))
    for frame, name in ((objects, "objects"), (fovs, "fovs")):
        stored = pd.read_csv(LAB / dataset_id / f"{name}.csv.gz", dtype=str, keep_default_na=False)
        rebuilt = pd.read_csv(
            io.StringIO(frame.to_csv(index=False)), dtype=str, keep_default_na=False
        )
        pd.testing.assert_frame_equal(stored, rebuilt)
