"""``live``: run a stopping rule while the microscope acquires, one field at a time.

``live_session(setup)`` returns an ``OnlineSession`` for the setup's rules.
``LiveFolder`` watches a folder that the acquisition side fills, and feeds
the session each field as soon as it is complete.

**Folder layout: one file per field and an append-only manifest.** The field
is the atomic unit of MorphoCoverage (every object of a field enters
together), so it is also the unit of the files:

- ``objects/<name>.csv``: the objects of one field, written to a temporary
  name and renamed when complete (``os.replace`` is atomic), so a reader never
  sees half a field. A field with no objects has no file, or an empty one.
- ``fovs.csv``: one line per completed field, appended only after its object
  file is in place. It holds the field's metadata (identifier, condition,
  repeat, order, stage position, ...) and, in ``objects_file``, the object
  file's path relative to the folder. The manifest line is the signal that a
  field is complete; a field is never read before its line is.

**In memory.** When features are computed live and never written,
``LiveTables`` takes the two tables themselves: an object table and a FOV
table that grow by appended rows. Each ``update(objects, fovs)`` reads only
the rows appended since the last call. As on disk, the FOV row is the signal
that a field is complete: its objects must be appended before (or with) it,
and wait in a buffer until it arrives.

**A field that fails is retried, never skipped.** Each source moves past a
field only once the session has accepted it. If a field raises (an infinite
value, a missing object file), the updates so far are kept, and the next poll
or update raises on the same field again, so no later field is added past it
and acquisition effort is never under-counted.

``write_fov`` does both steps for the acquisition side. ``LiveFolder.poll()``
reads only the manifest lines added since the last poll (it keeps its byte
offset and ignores a last line that is not yet terminated), reads each new
field's object file, and returns one update per field; ``watch()`` repeats
the poll every ``interval`` seconds.

**Why CSV, and why chunks.** One growing object table has no signal that a
field is complete, can be read while half written, and must be re-read or
tailed as it grows (a 300-field table of 60 features, 49 MB, takes about
0.8 s to read again, and the time grows with every field). Per-field files
keep every read the size of one field: about 5 ms for 300 objects x 60
features and 36 ms for 3,000, well within the seconds between fields; about
0.2 s for 3,000 objects x 400 features (embedding-sized rows). CSV is what
the feature extractors write, needs no dependency and can be inspected by
hand. A binary format (Parquet, needing ``pyarrow``) pays off only for very
wide rows; ``reader`` accepts any function from a path to a DataFrame, for
example ``pandas.read_parquet``, and ``write_fov`` any file suffix pandas
writes. No chunking within a field is needed.
"""

from __future__ import annotations

import csv
import io
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..metadata import (
    EMPTY_MEANS_ABSENT,
    MORPHOBRICKS_FOV_CANDIDATES,
    MORPHOBRICKS_GROUP_COLUMNS,
    OPTIONAL_GROUP_COLUMNS,
    column_has_values,
)
from ..online import OnlineSession
from ..results import Decision, StrategyResult, decision_view
from ..strategies.base import SamplingStrategy
from .setup import Setup

MANIFEST = "fovs.csv"
OBJECTS_DIR = "objects"
OBJECTS_FILE = "objects_file"


def live_session(setup: Setup, rules: tuple[SamplingStrategy, ...] | None = None) -> OnlineSession:
    """An ``OnlineSession`` for ``rules`` (default: the setup's rules) on the setup's columns."""
    if not isinstance(setup, Setup):
        raise TypeError("setup must be a morphocoverage.workflows.Setup")
    return OnlineSession(
        strategies=list(setup.stopping_rules() if rules is None else rules),
        feature_cols=setup.columns,
        objective=setup.objective(),
        fov_col=setup.fov_col,
        group_cols=None if setup.group_cols is None else list(setup.group_cols),
        observation_unit=setup.observation_unit,
    )


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
    count = _manifest_rows(manifest)
    stem = name if name is not None else f"fov_{count + 1:06d}"
    relative = f"{OBJECTS_DIR}/{stem}{suffix}"
    target = folder / relative
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
    row = {**metadata, OBJECTS_FILE: relative}
    if header is None:
        header = list(row)
    elif set(header) != set(row):
        raise ValueError(f"manifest columns are {header}; this field has {list(row)}")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=header, lineterminator="\n")
    if count == 0 and not manifest.exists():
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


@dataclass(frozen=True)
class LiveUpdate:
    """One field added to the session: its metadata, object count and the rules' results."""

    fov_metadata: dict[str, Any]
    n_objects: int
    results: list[StrategyResult]
    read_seconds: float
    pending_objects: int = 0

    def summary(self) -> str:
        """One line per rule (the decision and why), then one view per target."""
        where = ", ".join(f"{k}={v}" for k, v in self.fov_metadata.items() if k != OBJECTS_FILE)
        head = f"{where}: {self.n_objects} objects (read in {1000 * self.read_seconds:.0f} ms)"
        lines = [head, *(f"  {r.explain()}" for r in self.results)]
        if self.results:
            lines.append(f"  {decision_view(self.results)}")
        if self.pending_objects:
            waiting = (
                "1 object row waits for its FOV row"
                if self.pending_objects == 1
                else f"{self.pending_objects} object rows wait for their FOV rows"
            )
            lines.append(f"  {waiting} (not yet counted in any field)")
        return "\n".join(lines)


def _value_kind(values: pd.Series) -> str:
    """``number``, ``text`` or ``other``: what the non-missing values of a key column are."""
    kind = pd.api.types.infer_dtype(values, skipna=True)
    if kind in ("integer", "floating", "mixed-integer-float", "decimal"):
        return "number"
    return "text" if kind == "string" else "other"


def print_update(update: LiveUpdate) -> None:
    """Print one update's summary (the default ``on_update`` of ``watch``)."""
    print(update.summary(), flush=True)


class LiveFolder:
    """Watch a live folder (see the module) and feed each completed field to a session.

    ``metadata_dtypes`` types manifest columns (default: every metadata value
    is read as text, so identifiers such as ``001`` keep their zeros; the
    session only compares them). ``reader`` reads one object file (default
    ``pandas.read_csv``).
    """

    def __init__(
        self,
        folder: str | Path,
        session: OnlineSession,
        *,
        reader: Callable[[Path], pd.DataFrame] | None = None,
        metadata_dtypes: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(session, OnlineSession):
            raise TypeError("session must be an OnlineSession (see live_session)")
        self.folder = Path(folder)
        self.session = session
        self.reader = pd.read_csv if reader is None else reader
        self.metadata_dtypes = dict(metadata_dtypes or {})
        self._offset = 0
        self._header: list[str] | None = None
        self.updates: list[LiveUpdate] = []

    def _new_rows(self) -> list[tuple[dict[str, str], int]]:
        """The complete manifest lines after the offset, each with the offset past it."""
        manifest = self.folder / MANIFEST
        if not manifest.exists():
            return []
        with open(manifest, "rb") as handle:
            handle.seek(self._offset)
            data = handle.read()
        complete = data[: data.rfind(b"\n") + 1]  # a line still being written waits
        rows, offset = [], self._offset
        for raw in complete.splitlines(keepends=True):
            offset += len(raw)
            fields = next(csv.reader([raw.decode("utf-8")]), None)
            if self._header is None:
                if OBJECTS_FILE not in (fields or []):
                    raise ValueError(f"{manifest} has no {OBJECTS_FILE!r} column")
                self._header, self._offset = fields, offset
                continue
            if fields:
                rows.append((dict(zip(self._header, fields, strict=True)), offset))
        return rows

    def _typed(self, row: dict[str, str]) -> dict[str, Any]:
        typed: dict[str, Any] = {}
        for key, value in row.items():
            if key == OBJECTS_FILE:
                continue
            kind = self.metadata_dtypes.get(key)
            typed[key] = value if kind is None else kind(value)
        return typed

    def poll(self) -> list[LiveUpdate]:
        """Add every field completed since the last poll; one update per field, in order."""
        updates = []
        for row, offset in self._new_rows():
            started = time.perf_counter()
            path = self.folder / row[OBJECTS_FILE] if row.get(OBJECTS_FILE) else None
            if path is not None and not path.exists():
                raise FileNotFoundError(
                    f"the manifest names {path}, which does not exist; an empty field has an "
                    f"empty {OBJECTS_FILE!r} entry or an empty file"
                )
            if path is not None and path.stat().st_size > 0:
                objects = self.reader(path)
            else:
                objects = pd.DataFrame(columns=self.session.feature_cols)
            read = time.perf_counter() - started
            metadata = self._typed(row)
            # The manifest is the authority on metadata: an object file that repeats a
            # metadata column (say a plate number read back as an integer) is compared
            # as text, so only a real disagreement raises.
            for key, value in metadata.items():
                if key in objects.columns and isinstance(value, str):
                    objects[key] = objects[key].astype(str)
            results = self.session.add_fov(objects, metadata)
            # Move past the field only once the session has it: a field that raises is
            # read again on the next poll instead of being skipped.
            self._offset = offset
            update = LiveUpdate(metadata, len(objects), results, read)
            updates.append(update)
            self.updates.append(update)
        return updates

    def watch(
        self,
        *,
        interval: float = 1.0,
        timeout: float | None = None,
        max_fovs: int | None = None,
        until_stop: bool = False,
        on_update: Callable[[LiveUpdate], None] | None = print_update,
    ) -> list[LiveUpdate]:
        """Poll every ``interval`` seconds until ``timeout``, ``max_fovs`` or, with
        ``until_stop``, until every rule says stop in the latest field's group.

        Stop with Ctrl-C at any time; the updates so far are kept in ``updates``.
        """
        if interval <= 0:
            raise ValueError("interval must be positive")
        started = time.monotonic()
        seen: list[LiveUpdate] = []
        try:
            while True:
                for update in self.poll():
                    seen.append(update)
                    if on_update is not None:
                        on_update(update)
                    if until_stop and all(r.decision == Decision.STOP for r in update.results):
                        return seen
                    if max_fovs is not None and len(seen) >= max_fovs:
                        return seen
                if timeout is not None and time.monotonic() - started >= timeout:
                    return seen
                time.sleep(interval)
        except KeyboardInterrupt:
            return seen


class LiveTables:
    """Feed a session from an object table and a FOV table held in memory (see the module).

    Call ``update(objects, fovs)`` with the current tables whenever rows were
    appended; rows already read must stay in place (the tables only grow).
    ``key`` names the columns that tie an object row to its FOV row (default:
    the session's FOV column and the condition/repeat columns present in both
    tables). Objects of a field must be appended no later than its FOV row; an
    object row for a field already added raises, since it would change a
    field the rules have already judged.
    """

    def __init__(self, session: OnlineSession, *, key: Sequence[str] | None = None) -> None:
        if not isinstance(session, OnlineSession):
            raise TypeError("session must be an OnlineSession (see live_session)")
        if isinstance(key, str):
            raise TypeError("key must be a sequence of column names")
        self.session = session
        self.key: tuple[str, ...] | None = None if key is None else tuple(key)
        self._n_objects = 0
        self._n_fovs = 0
        self._buffer: pd.DataFrame | None = None
        self._added: set[tuple] = set()
        self.updates: list[LiveUpdate] = []

    def _resolve_key(self, objects: pd.DataFrame, fovs: pd.DataFrame) -> tuple[str, ...]:
        if self.key is None:
            fov_col = self.session.fov_col or self.session._requested_fov_col
            if fov_col is None:
                fov_col = next(
                    (
                        c
                        for c in MORPHOBRICKS_FOV_CANDIDATES
                        if c in fovs.columns
                        and (c not in EMPTY_MEANS_ABSENT or column_has_values(fovs, c))
                    ),
                    None,
                )
            if fov_col is None:
                raise ValueError("could not find the FOV column; pass fov_col to the session")
            groups = self.session.group_cols or self.session._requested_group_cols
            if groups is None:
                # An optional group column counts only when it holds a value.
                groups = tuple(
                    g
                    for g in MORPHOBRICKS_GROUP_COLUMNS
                    if g not in OPTIONAL_GROUP_COLUMNS or column_has_values(fovs, g)
                )
            self.key = (*(g for g in groups if g in fovs and g in objects), fov_col)
        missing = [c for c in self.key if c not in objects.columns or c not in fovs.columns]
        if missing:
            raise ValueError(f"key columns {missing} must be in both the object and FOV tables")
        return self.key

    @staticmethod
    def _check_key_types(objects: pd.DataFrame, fovs: pd.DataFrame, key: tuple[str, ...]) -> None:
        # 1 and "1" never compare equal: the field would be counted as empty and its
        # objects would wait forever. Refuse numbers against text instead.
        for column in key:
            kinds = {_value_kind(objects[column]), _value_kind(fovs[column])}
            if kinds == {"number", "text"}:
                raise TypeError(
                    f"key column {column!r} holds numbers in one table and text in the other; "
                    "give it the same type in both (for example .astype(str))"
                )

    @staticmethod
    def _keys(table: pd.DataFrame, key: tuple[str, ...]) -> list[tuple]:
        return list(zip(*(table[c].tolist() for c in key), strict=True))

    def update(self, objects: pd.DataFrame, fovs: pd.DataFrame) -> list[LiveUpdate]:
        """Add every field whose FOV row was appended since the last call, in order."""
        if not isinstance(objects, pd.DataFrame) or not isinstance(fovs, pd.DataFrame):
            raise TypeError("objects and fovs must be pandas DataFrames")
        if len(objects) < self._n_objects or len(fovs) < self._n_fovs:
            raise ValueError("the tables only grow: rows already read were removed")
        new_objects = objects.iloc[self._n_objects :]
        new_fovs = fovs.iloc[self._n_fovs :]
        if new_fovs.empty and new_objects.empty:
            return []
        key = self._resolve_key(objects, fovs)
        self._check_key_types(new_objects if len(new_objects) else objects, fovs, key)
        if not new_objects.empty:
            late = [k for k in self._keys(new_objects, key) if k in self._added]
            if late:
                raise ValueError(
                    f"objects appended for field(s) already added {sorted(set(late))[:3]}; "
                    "append a field's objects before its FOV row"
                )
            parts = [p for p in (self._buffer, new_objects) if p is not None]
            self._buffer = pd.concat(parts) if len(parts) > 1 else new_objects
        self._n_objects = len(objects)
        updates = []
        for _, row in new_fovs.iterrows():
            started = time.perf_counter()
            metadata = row.to_dict()
            fov_key = tuple(metadata[c] for c in key)
            if fov_key in self._added:
                raise ValueError(f"field {fov_key} appears twice in the FOV table")
            if self._buffer is not None and len(self._buffer):
                mine = [k == fov_key for k in self._keys(self._buffer, key)]
                selected = self._buffer.loc[mine] if any(mine) else self._buffer.iloc[:0]
            else:
                mine, selected = [], objects.iloc[:0]
            read = time.perf_counter() - started
            results = self.session.add_fov(selected.reset_index(drop=True), metadata)
            # Consume the row and its objects only once the session has the field: a field
            # that raises stays in place and is tried again on the next update.
            if any(mine):
                self._buffer = self._buffer.loc[[not m for m in mine]]
            self._added.add(fov_key)
            self._n_fovs += 1
            update = LiveUpdate(metadata, len(selected), results, read, self.pending_objects)
            updates.append(update)
            self.updates.append(update)
        return updates

    @property
    def pending_objects(self) -> int:
        """Object rows read whose FOV row has not arrived yet."""
        return 0 if self._buffer is None else len(self._buffer)


__all__ = [
    "MANIFEST",
    "OBJECTS_DIR",
    "OBJECTS_FILE",
    "LiveFolder",
    "LiveTables",
    "LiveUpdate",
    "live_session",
    "print_update",
    "write_fov",
]
