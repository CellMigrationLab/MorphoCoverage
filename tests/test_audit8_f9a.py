"""The owner's decisions after audit 8 on the feature spaces (F9a, 2026-10-06): the separation
of the repeats on raw values, MFA weights free of the tables' order, the populations' cells
without their own control, unusual observations on the readout alone, the label check's
scale, the combined map, the wide fingerprint's k and the values the fingerprint reads."""

import re
import warnings

from test_embedding_only import _cells
from test_feature_tables_batch import _batch, _setup

from morphocoverage.separation import separation
from morphocoverage.workflows import Setup, html_report, report
from morphocoverage.workflows._feature_space import centred_tables, separation_in_report

KEY = ["condition", "biological_repeat", "image_file", "label_id"]


def _visible(page: str) -> str:
    text = re.sub(r"<script.*?</script>", "", page, flags=re.DOTALL)
    return " ".join(re.sub(r"<[^>]+>", " ", text).split())


# --- 2. The separation of the repeats on raw values; one well per condition ----------------


def test_the_repeats_separation_reads_the_raw_values_and_the_conditions_the_centred_ones():
    merged, fovs, names = _batch()
    setup = _setup(names)
    centred = centred_tables(merged, fovs, setup)
    raw = separation_in_report(merged, fovs, setup, ["DINO"]).spaces.iloc[0]
    alone = separation_in_report(centred, fovs, setup, ["DINO"]).spaces.iloc[0]
    both = separation_in_report(centred, fovs, setup, ["DINO"], raw=merged).spaces.iloc[0]
    # A batch measured after the centring that removes it would be circular (audit 8 A2).
    assert both["repeat_share"] == raw["repeat_share"]
    assert both["repeat_share"] > alone["repeat_share"] + 0.3
    assert both["share"] == alone["share"]


def test_one_well_per_condition_says_a_well_difference_reads_as_a_condition_separation():
    merged, fovs, names = _batch()
    spaces = {"DINO": names}
    unknown = separation(merged, fovs, spaces=spaces).well_line()
    assert unknown.startswith("The wells are not given, so a difference between the wells")
    one = fovs.assign(technical_repeat=fovs["condition"] + fovs["biological_repeat"])
    assert separation(merged, one, spaces=spaces).well_line() == (
        "Each condition is imaged in one well per biological repeat, so a difference between "
        "wells reads here as a separation of the conditions."
    )
    field = fovs["image_file"].str.extract(r"_(\d+)\.tif$")[0].astype(int)
    two = one.assign(technical_repeat=one["technical_repeat"] + (field % 2).astype(str))
    assert separation(merged, two, spaces=spaces).well_line() == ""


def test_the_page_says_the_repeats_are_scored_on_raw_values_and_the_one_well_line(tmp_path):
    objects, fovs, table = _cells()
    setup = Setup(control={"condition": "ctrl"}, observation_unit="cell", normalize="control_mean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup, feature_tables={"DINO": table})
        page = html_report(result, objects, fovs, tmp_path, feature_tables={"DINO": table})
    visible = _visible(page.read_text())
    assert "Within a condition, the same with the repeat, always on the raw values" in visible
    assert "The wells are not given, so a difference between the wells" in visible
