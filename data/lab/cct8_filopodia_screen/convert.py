"""Build the CCT8 filopodia siRNA screen tables (Fig. 2C) from the pinned Zenodo deposit.

Source: Zenodo record 20704995 (CC BY 4.0), member
``Fig2 and FigS2/Fig2C filopodia numbers after silencing/FILOPODIA NUMBER ALL.xlsx`` of
``Fig2 and FigS2.zip``: one row per cell with its manually counted number of MYO10-positive
filopodia, the siRNA (``Condition``), the biological repeat within its experimental set
(``Repeat``) and the experimental set (``dataset``, the targeted gene's screen). See card.toml
and DATASET.md for the paper, the mapping and its justification.

Usage::

    python data/lab/cct8_filopodia_screen/convert.py [--source PATH_TO_XLSX]

Without ``--source`` it downloads the archive (190 MB) to a temporary folder. It needs
``openpyxl`` (``pandas.read_excel``); the stored tables do not. The output is deterministic:
rows sorted by the field key, ``\\n`` line endings, integer values written as integers, gzip
headers without a timestamp. This is a data-preparation script, not part of the package.
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
    whole_numbers,
    write_csv_gz,
)

RECORD = "20704995"
ARCHIVE_URL = f"https://zenodo.org/api/records/{RECORD}/files/Fig2%20and%20FigS2.zip/content"
ARCHIVE_MD5 = "354a7ffd888bcb55e193bd7c6f9ae697"
MEMBER = "Fig2 and FigS2/Fig2C filopodia numbers after silencing/FILOPODIA NUMBER ALL.xlsx"
MEMBER_SHA256 = "4efb5c9913a47ad1f40be801ab12932ffdd697db4db6893b75b30cebfdb17b56"
SOURCE_COLUMNS = ["Condition", "Filopodia number", "Repeat", "dataset"]
FEATURE = "Filopodia number"
# One coverslip per siRNA per repeat (owner, 2026-09-30); the images are not in the deposit.
SINGLE, UNRECORDED = "single", "unrecorded"
FIELD_KEY = ["experiment", "condition", "biological_repeat", "technical_repeat", "position_id"]


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

    - experiment: the experimental set (``dataset``), one per targeted gene, each with its own
      siCTRL; the paper's figure pools the six siCTRL sets, these tables keep them apart.
    - biological_repeat: ``<set>_<Repeat>``, because repeat 1 of one set is not repeat 1 of
      another (each set was run as its own experiment with its own control).
    - technical_repeat: ``single`` (one coverslip per siRNA per repeat, the owner's statement).
    - position_id: a synthetic, stable identifier of the field holding the cell (the image
      names are not in the deposit); image_file is ``unrecorded``.
    - label_id: 1, one cell per field.
    """
    table = pd.DataFrame(
        {
            "experiment": source["dataset"].astype(str),
            "condition": source["Condition"].astype(str),
            "biological_repeat": source["dataset"].astype(str)
            + "_"
            + whole_numbers(source, "Repeat", 1).astype(str),
            "technical_repeat": SINGLE,
            "image_file": UNRECORDED,
            "label_id": 1,
            FEATURE: whole_numbers(source, FEATURE, 0),
            "source_row": source["source_row"].astype(int),
        }
    )
    # Number the cells of each experiment, condition and repeat in spreadsheet order.
    group = ["experiment", "condition", "biological_repeat"]
    table["position_id"] = synthetic_positions(table, group, order="source_row")
    table = table.sort_values(FIELD_KEY, kind="stable").reset_index(drop=True)
    table["target"] = table["condition"].map(sirna_target)
    columns = [*FIELD_KEY, "image_file", "target", "label_id", FEATURE, "source_row"]
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
