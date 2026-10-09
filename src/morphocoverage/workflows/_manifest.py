"""The live folder's files: ``write_fov`` writes a field, ``ManifestReader`` reads the manifest.

Private to ``workflows.live``, which re-exports ``write_fov`` and the names
below; the folder layout is described there.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .._wording import count

MANIFEST = "fovs.csv"
OBJECTS_DIR = "objects"
OBJECTS_FILE = "objects_file"


def write_fov(
    folder: str | Path,
    objects: pd.DataFrame,
    fov_metadata: Mapping[str, Any],
    *,
    name: str | None = None,
    suffix: str = ".csv",
) -> Path:
    """Add one completed field to a live folder: its object file, then its manifest line.

    ``name`` is the object file's stem (default: the field's position in the
    manifest, ``fov_000001``). The manifest's columns are fixed by its first
    line; a later field with other metadata keys raises. Returns the object
    file's path (written even when ``objects`` is empty: with its header, or as
    an empty file when it has no columns either).
    """
    folder = Path(folder)
    (folder / OBJECTS_DIR).mkdir(parents=True, exist_ok=True)
    manifest = folder / MANIFEST
    metadata = {str(k): v for k, v in fov_metadata.items()}
    if OBJECTS_FILE in metadata:
        raise ValueError(f"{OBJECTS_FILE!r} is written by write_fov; do not pass it")
    header = _manifest_header(manifest)
    if header is None and manifest.exists() and manifest.stat().st_size > 0:
        raise ValueError(f"{manifest} has no complete header line; fix or remove it")
    count = _manifest_rows(manifest)
    stem = name if name is not None else f"fov_{count + 1:06d}"
    relative = f"{OBJECTS_DIR}/{stem}{suffix}"
    target = folder / relative
    row = {**metadata, OBJECTS_FILE: relative}
    new_manifest = header is None  # absent, or an empty file: the header comes first
    if header is None:
        header = list(row)
    elif set(header) != set(row):
        # Checked before the object file is written, so a corrected retry finds no orphan.
        raise ValueError(f"manifest columns are {header}; this field has {list(row)}")
    if target.exists():
        raise FileExistsError(f"{target} exists; every field needs its own object file")
    temporary = target.with_name(f".{target.name}.tmp")
    if objects.shape[1] == 0:
        temporary.write_bytes(b"")  # a field without objects: an empty file, read as zero rows
    elif suffix in (".csv", ".csv.gz"):
        objects.to_csv(
            temporary, index=False, compression="gzip" if suffix.endswith(".gz") else None
        )
    elif suffix == ".parquet":
        objects.to_parquet(temporary, index=False)
    else:
        raise ValueError("suffix must be .csv, .csv.gz or .parquet")
    os.replace(temporary, target)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=header, lineterminator="\n")
    if new_manifest:
        writer.writeheader()
    writer.writerow(row)
    with open(manifest, "a", newline="") as handle:
        handle.write(buffer.getvalue())
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _manifest_header(manifest: Path) -> list[str] | None:
    if not manifest.exists():
        return None
    with open(manifest, newline="") as handle:
        first = handle.readline()
    return next(csv.reader([first])) if first.endswith("\n") else None


def _manifest_rows(manifest: Path) -> int:
    if not manifest.exists():
        return 0
    with open(manifest, newline="") as handle:
        return max(0, sum(1 for line in handle if line.endswith("\n")) - 1)


def value_kind(values: pd.Series) -> str:
    """``number``, ``text`` or ``other``: what the non-missing values of a key column are."""
    kind = pd.api.types.infer_dtype(values, skipna=True)
    if kind in ("integer", "floating", "mixed-integer-float", "decimal"):
        return "number"
    return "text" if kind == "string" else "other"


def read_coordinates(typed: Mapping[str, Any], stage_cols: Sequence[str]) -> dict[str, float]:
    """The stage columns the manifest holds as text, read once as numbers (the analyses refuse
    text coordinates, so the folder converts the declared ones here); a ``ValueError`` names
    the column and the text that is not a number."""
    found: dict[str, float] = {}
    for column in stage_cols:
        text = typed.get(column)
        if not isinstance(text, str):
            continue
        try:
            found[column] = float(text)
        except ValueError:
            raise ValueError(
                f"{MANIFEST}: stage column {column!r} holds {text!r}, which is not a number"
            ) from None
    return found


class ManifestReader:
    """The lines of a live folder's manifest, each read once, in order.

    ``new_rows`` reads the whole manifest and checks that its first bytes are
    the ones already read (their hash is kept): a manifest that shrank, or was
    rewritten or replaced (an acquisition restarted in the same folder), raises
    instead of silently skipping fields or parsing the middle of a line. A line
    counts as read only once ``accept`` or ``skip`` moves past it, so a field
    that raises is read again on the next call.
    """

    def __init__(self, folder: Path) -> None:
        self.path = folder / MANIFEST
        self.offset = 0
        self.lines = 0  # lines read so far, the header included
        self._header: list[str] | None = None
        self._read_hash = hashlib.sha256()
        self._accepted: dict[tuple[str, ...], int] = {}  # each accepted line: its number
        self._by_metadata: dict[tuple[str, ...], int] = {}  # the same, without the file

    def new_rows(self) -> list[tuple[dict[str, str], bytes]]:
        """The complete lines after the part read, each with its raw bytes (a blank one: ``{}``)."""
        if not self.path.exists():
            if self.offset:
                raise ValueError(self._changed("was removed"))
            return []
        with open(self.path, "rb") as handle:
            data = handle.read()
        if len(data) < self.offset:
            raise ValueError(self._changed("is shorter than the part already read"))
        if hashlib.sha256(data[: self.offset]).digest() != self._read_hash.digest():
            raise ValueError(self._changed("no longer starts with the lines already read"))
        data = data[self.offset :]
        complete = data[: data.rfind(b"\n") + 1]  # a line still being written waits
        rows = []
        for raw in complete.splitlines(keepends=True):
            fields = next(csv.reader([raw.decode("utf-8")]), None)
            if self._header is None:
                if OBJECTS_FILE not in (fields or []):
                    raise ValueError(f"{self.path} has no {OBJECTS_FILE!r} column")
                self._header = fields
                self.skip(raw)
                continue
            rows.append((dict(zip(self._header, fields, strict=True)) if fields else {}, raw))
        return rows

    def repeat_of(self, row: Mapping[str, str]) -> int | None:
        """The accepted line that ``row`` repeats exactly, if any."""
        return self._accepted.get(tuple(row.values()))

    def same_field_as(self, row: Mapping[str, str]) -> int | None:
        """The accepted line with the same metadata as ``row`` (another object file), if any."""
        return self._by_metadata.get(_metadata_text(row))

    def accept(self, row: Mapping[str, str], raw: bytes) -> None:
        """Move past a line whose field the session accepted."""
        self.skip(raw)
        self._accepted[tuple(row.values())] = self.lines
        self._by_metadata[_metadata_text(row)] = self.lines

    def skip(self, raw: bytes) -> None:
        """Move past a line without adding a field (the header, a blank or repeated line)."""
        self.offset += len(raw)
        self._read_hash.update(raw)
        self.lines += 1

    def _changed(self, how: str) -> str:
        fields = count(len(self._accepted), "field")
        return (
            f"{self.path} {how}: it was rewritten, truncated or replaced after {fields} were "
            f"read from it (its first {self.offset} bytes). The fields read stay in the "
            "session, and no field of the new manifest is added. If the acquisition restarted, "
            "write it to a new folder, or start a new session and LiveFolder on this one."
        )


def _metadata_text(row: Mapping[str, str]) -> tuple[str, ...]:
    """A manifest line's metadata as written, without its object file."""
    return tuple(value for key, value in row.items() if key != OBJECTS_FILE)
