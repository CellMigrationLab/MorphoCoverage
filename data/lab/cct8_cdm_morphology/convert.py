"""Build the CCT8 cell-shape-on-CDM tables (Fig. 5D-E) from the pinned Zenodo deposit.

Source: Zenodo record 20704995 (CC BY 4.0), member
``Fig5 and FigS5/Fig5DE 231 CDM - shape/Cell Morphology 231s CDMs.xlsx`` of
``Fig5 and FigS5.zip``: one row per manually segmented MDA-MB-231 cell on a cell-derived
matrix, with the Fiji measurements of its outline, the siRNA (``Condition``) and the biological
repeat (``Repeat``). See card.toml and DATASET.md for the paper, the mapping and its
justification.

Usage::

    python data/lab/cct8_cdm_morphology/convert.py [--source PATH_TO_XLSX]

Without ``--source`` it downloads the archive (381 MB) to a temporary folder. It needs
``openpyxl`` (``pandas.read_excel``); the stored tables do not. The output is deterministic
(see ``convert_tools``). This is a data-preparation script, not part of the package.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
# The shared helpers live one folder up.
from convert_tools import (
    check_columns,
    check_member,
    fetch_zip_member,
    sirna_target,
    synthetic_positions,
    write_csv_gz,
)

RECORD = "20704995"
ARCHIVE_URL = f"https://zenodo.org/api/records/{RECORD}/files/Fig5%20and%20FigS5.zip/content"
ARCHIVE_MD5 = "b27b7c7ce0118f79e6b74b62955e0fe6"
MEMBER = "Fig5 and FigS5/Fig5DE 231 CDM - shape/Cell Morphology 231s CDMs.xlsx"
MEMBER_SHA256 = "944227bc9eda234b2b7eeca92fbced55258f8c8fa4df006e85b12617cc6c530e"
# The Fiji "Measure" columns, in the source's order; " " is Fiji's row number.
MEASUREMENTS = [
    "Area", "Mean", "StdDev", "Mode", "Min", "Max", "X", "Y", "XM", "YM", "Perim.", "BX", "BY",
    "Width", "Height", "Major", "Minor", "Angle", "Circ.", "Feret", "IntDen", "Median", "Skew",
    "Kurt", "%Area", "RawIntDen",
]  # fmt: skip
FERET = ["FeretX", "FeretY", "FeretAngle", "MinFeret", "AR", "Round", "Solidity"]
SOURCE_COLUMNS = ["Condition", "Repeat", " ", *MEASUREMENTS, "Ch", "Slice", *FERET]
# Fiji's bookkeeping columns, kept under a source_ prefix: they are not image axes of the
# stored tables (the channel of an RGB projection; the slice the outline was drawn on).
RENAMED = {" ": "source_roi", "Ch": "source_channel", "Slice": "source_slice"}
# One well per siRNA per repeat (owner, 2026-09-30); the images are not in the deposit.
SINGLE, UNRECORDED = "single", "unrecorded"
FIELD_KEY = ["condition", "biological_repeat", "technical_repeat", "position_id"]


def fetch_member() -> bytes:
    """The spreadsheet's bytes from the pinned archive, the archive's md5 verified."""
    return fetch_zip_member(ARCHIVE_URL, ARCHIVE_MD5, MEMBER)


def read_source(data: bytes) -> pd.DataFrame:
    """The source table, checked against the pinned checksum and its expected columns."""
    check_member(data, MEMBER, MEMBER_SHA256)
    source = pd.read_excel(io.BytesIO(data))
    check_columns(source, SOURCE_COLUMNS)
    # The spreadsheet row of each cell (1 is the header), for tracing a row back to the source.
    return source.assign(source_row=range(2, len(source) + 2))


def build_tables(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(objects, fovs)``: one cell per field (the owner's statement), MorphoBricks columns.

    - condition: the siRNA (``Condition``); biological_repeat: ``Repeat`` (1-3, one experiment).
    - technical_repeat: ``single`` (one well per siRNA per repeat, the owner's statement).
    - position_id: a synthetic, stable identifier of the field holding the cell, numbered by
      Fiji's row number within its condition and repeat; image_file is ``unrecorded``.
    - label_id: 1, one cell per field. The measurements keep their Fiji names and values.
    """
    table = source.rename(columns={"Condition": "condition", **RENAMED})
    table["condition"] = table["condition"].astype(str)
    table["biological_repeat"] = source["Repeat"].astype(int).astype(str)
    table["technical_repeat"] = SINGLE
    table["image_file"] = UNRECORDED
    table["label_id"] = 1
    if table.duplicated(["condition", "biological_repeat", "source_roi"]).any():
        raise SystemExit("Fiji's row number repeats within a condition and repeat")
    group = ["condition", "biological_repeat"]
    table["position_id"] = synthetic_positions(table, group, order="source_roi")
    table["target"] = table["condition"].map(sirna_target)
    table = table.sort_values(FIELD_KEY, kind="stable").reset_index(drop=True)
    columns = [
        *FIELD_KEY,
        "image_file",
        "target",
        "label_id",
        *MEASUREMENTS,
        *FERET,
        *RENAMED.values(),
        "source_row",
    ]
    objects = table[columns]
    fovs = objects[[*FIELD_KEY, "image_file", "target"]].drop_duplicates().reset_index(drop=True)
    if len(fovs) != len(objects):
        raise SystemExit("a field holds more than one cell")
    return objects, fovs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help="a local copy of the spreadsheet")
    args = parser.parse_args(argv)
    data = args.source.read_bytes() if args.source else fetch_member()
    objects, fovs = build_tables(read_source(data))
    write_csv_gz(objects, HERE / "objects.csv.gz")
    write_csv_gz(fovs, HERE / "fovs.csv.gz")
    print(f"wrote {len(objects)} cells in {len(fovs)} fields", file=sys.stderr)


if __name__ == "__main__":
    main()
