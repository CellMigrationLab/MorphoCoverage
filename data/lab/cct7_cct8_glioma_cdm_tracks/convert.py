"""Build the U-87 MG tracks-on-CDM tables (CCT7 and CCT8 silencing) from the owner's file.

Source: ``Tracking_data_CCT8_CCT7_silencing_Glioma.csv``, provided by the owner on 2026-09-30
(unpublished, not deposited): the CellTracksColab track table of manually tracked U-87 MG
glioma cells migrating on cell-derived matrices, one row per track. See card.toml and
DATASET.md for the design, the mapping and the defects found in the source.

Usage::

    python data/lab/cct7_cct8_glioma_cdm_tracks/convert.py --source PATH_TO_CSV

The source is not downloadable, so ``--source`` is required. The output is deterministic (see
``convert_tools``). This is a data-preparation script, not part of the package.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
# The shared helpers live one folder up.
from convert_tools import check_columns, check_member, write_csv_gz

MEMBER = "Tracking_data_CCT8_CCT7_silencing_Glioma.csv"
MEMBER_SHA256 = "dec991b3fdd52956d01ee64c2348ee4ec64afe4f16b7d0f67c549350b064233e"
IDENTIFIERS = [
    "Unique_ID", "TRACK_ID", "File_name", "Condition", "experiment_nb", "File_name_raw", "Repeat",
]  # fmt: skip
# The CellTracksColab track metrics, in the source's order (names kept verbatim).
METRICS = [
    "Track Duration", "Mean Speed", "Median Speed", "Max Speed", "Min Speed",
    "Speed Standard Deviation", "Total Distance Traveled", "Mean Speed Rolling",
    "Median Speed Rolling", "Max Speed Rolling", "Min Speed Rolling Rolling",
    "Speed Standard Deviation Rolling", "Total Distance Traveled Rolling", "Directionality",
    "Directionality Rolling", "Tortuosity", "Tortuosity Rolling", "Total Turning Angle",
    "Total Turning Angle Rolling", "Spatial Coverage", "Spatial Coverage Rolling", "FMI_x_plus",
    "FMI_x_minus", "FMI_y_plus", "FMI_y_minus",
]  # fmt: skip
SOURCE_COLUMNS = [*IDENTIFIERS, *METRICS]
# "Results from Well B5_04-1 in px per 10min": the well, then the position within it.
RESULTS_FILE = re.compile(
    r"^Results from Well (?P<well>[A-H]\d+)_(?P<position>\S+) in px per 10min$"
)
UNRECORDED = "unrecorded"
FIELD_KEY = ["experiment", "condition", "biological_repeat", "technical_repeat", "position_id"]


def read_source(data: bytes) -> pd.DataFrame:
    """The source table, checked against the pinned checksum and its expected columns."""
    check_member(data, MEMBER, MEMBER_SHA256)
    source = pd.read_csv(io.BytesIO(data))
    check_columns(source, SOURCE_COLUMNS)
    # The file's line of each track (1 is the header), for tracing a row back to the source.
    return source.assign(source_row=range(2, len(source) + 2))


def _experiment(condition: str) -> str:
    """The experimental set of an siRNA label: its target gene (siCTRL_CCT7 belongs to CCT7)."""
    for gene in ("CCT7", "CCT8"):
        if gene in condition:
            return gene
    raise SystemExit(f"condition {condition!r} belongs to no experimental set")


def build_tables(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(objects, fovs)``: one row per track, MorphoBricks columns.

    - experiment: the experimental set (CCT7 or CCT8), each with its own siCTRL well.
    - condition: the siRNA, with both controls written ``siCTRL`` (the source label, e.g.
      ``siCTRL_CCT7``, is kept as source_condition).
    - biological_repeat: ``<set>_<Repeat>``, as the two sets were imaged for different lengths.
    - technical_repeat: the well (one well per siRNA per repeat).
    - position_id: ``<well>_<position>`` from the results file name, verbatim.
    - label_id: the track's number within its results file (TRACK_ID).
    Tracks that appear in two results files are kept, as in the source (see the card).
    """
    names = source["File_name_raw"].str.extract(RESULTS_FILE)
    if names.isna().any().any():
        raise SystemExit("a results file name does not name a well and a position")
    if not (source["experiment_nb"] == "R" + source["Repeat"].astype(str)).all():
        raise SystemExit("experiment_nb and Repeat disagree")
    experiment = source["Condition"].map(_experiment)
    control = source["Condition"].str.startswith("siCTRL")
    table = pd.DataFrame(
        {
            "experiment": experiment,
            "condition": source["Condition"].where(~control, "siCTRL"),
            "biological_repeat": experiment + "_" + source["Repeat"].astype(int).astype(str),
            "technical_repeat": names["well"],
            "position_id": names["well"] + "_" + names["position"],
            "image_file": UNRECORDED,
            "label_id": source["TRACK_ID"].astype(int),
            **{metric: source[metric] for metric in METRICS},
            "source_condition": source["Condition"],
            "source_file": source["File_name_raw"],
            "source_row": source["source_row"].astype(int),
        }
    )
    if table.duplicated([*FIELD_KEY, "label_id"]).any():
        raise SystemExit("a track number repeats within a results file")
    if (
        table.groupby(["experiment", "condition", "biological_repeat"])["technical_repeat"]
        .nunique()
        .max()
        != 1
    ):
        raise SystemExit("an siRNA spans several wells within a repeat")
    objects = table.sort_values([*FIELD_KEY, "label_id"], kind="stable").reset_index(drop=True)
    fovs = objects[[*FIELD_KEY, "image_file", "source_file"]].drop_duplicates()
    return objects, fovs.reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="the owner's CSV file")
    args = parser.parse_args(argv)
    objects, fovs = build_tables(read_source(args.source.read_bytes()))
    write_csv_gz(objects, HERE / "objects.csv.gz")
    write_csv_gz(fovs, HERE / "fovs.csv.gz")
    print(f"wrote {len(objects)} tracks in {len(fovs)} fields", file=sys.stderr)


if __name__ == "__main__":
    main()
