"""Helpers shared by the ``convert.py`` scripts of the lab datasets (data preparation only).

Each ``<id>/convert.py`` pins its source (record, archive checksum, member path and sha256) and
uses these helpers to fetch it, verify it and write deterministic tables.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_zip_member(url: str, archive_md5: str, member: str) -> bytes:
    """``member``'s bytes from the zip archive at ``url``, the archive's md5 verified, in
    chunks: the archive (hundreds of MB) is never held in memory whole (#577)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "archive.zip"
        urllib.request.urlretrieve(url, path)
        with open(path, "rb") as handle:
            md5 = hashlib.file_digest(handle, "md5").hexdigest()
        if md5 != archive_md5:
            raise SystemExit(f"{url}: md5 {md5}, expected {archive_md5}")
        with zipfile.ZipFile(path) as archive:
            return archive.read(member)


def check_member(data: bytes, member: str, expected_sha256: str) -> None:
    """Stop unless ``data`` is the pinned member."""
    digest = sha256(data)
    if digest != expected_sha256:
        raise SystemExit(f"{member}: sha256 {digest}, expected {expected_sha256}")


def check_columns(source: pd.DataFrame, expected: list[str]) -> None:
    """Stop unless the source has exactly the expected columns and no missing value."""
    if list(source.columns) != expected:
        raise SystemExit(f"unexpected columns {list(source.columns)}, expected {expected}")
    if source.isna().any().any():
        raise SystemExit("the source has missing values")


def whole_numbers(source: pd.DataFrame, column: str, minimum: int) -> pd.Series:
    """``source[column]`` as integers; stops unless every value is a finite whole number of at
    least ``minimum``, naming the source rows: ``astype(int)`` would truncate a repeat of
    1.8 to 1 or a count of 5.9 to 5 (#576)."""
    values = source[column]
    if pd.api.types.is_bool_dtype(values) or not pd.api.types.is_numeric_dtype(values):
        raise SystemExit(f"{column}: {values.dtype} values, not numbers")
    numbers = values.to_numpy(dtype=float)
    bad = ~np.isfinite(numbers) | (numbers != np.round(numbers)) | (numbers < minimum)
    if bad.any():
        rows = source["source_row"] if "source_row" in source else pd.Series(source.index)
        shown = "; ".join(
            f"row {row}: {value}"
            for row, value in zip(rows.to_numpy()[bad][:3], values.to_numpy()[bad][:3])
        )
        raise SystemExit(f"{column}: not a whole number of at least {minimum} in {shown}")
    return values.astype(int)


def synthetic_positions(table: pd.DataFrame, group: list[str], order: str) -> pd.Series:
    """A stable field identifier per row, ``<group values>|cellNNN``, numbered by ``order``.

    For sources that hold one cell per field but no image names: the field of each cell gets a
    readable, reproducible identifier from its group and its rank within the group.
    """
    ranked = table.sort_values([*group, order], kind="stable")
    ordinal = ranked.groupby(group, sort=False).cumcount() + 1
    prefix = ranked[group[0]].astype(str)
    for column in group[1:]:
        prefix = prefix + "|" + ranked[column].astype(str)
    return (prefix + "|cell" + ordinal.map("{:03d}".format)).reindex(table.index)


def sirna_target(condition: str, control: str = "siCTRL") -> str:
    """The gene an siRNA label silences (``siCCT8#5`` -> ``CCT8``); the control is
    ``non-targeting``. Stops on a label of another form."""
    if condition == control:
        return "non-targeting"
    match = re.fullmatch(r"si([A-Za-z0-9]+)(#\w+)?", condition)
    if match is None:
        raise SystemExit(f"cannot read a target from the siRNA label {condition!r}")
    return match.group(1)


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    """Deterministic gzip-compressed CSV (no timestamp in the gzip header, ``\\n`` endings)."""
    text = frame.to_csv(index=False, lineterminator="\n")
    with open(path, "wb") as handle, gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as gz:
        gz.write(text.encode("utf-8"))
