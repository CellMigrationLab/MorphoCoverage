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

from .._wording import count
from ..metadata import (
    drop_empty_optional_columns,
    resolve_fov_col,
    resolve_group_cols,
    resolve_identity_cols,
)
from ..online import OnlineSession
from ..order_check import OrderCheck
from ..results import Decision, StrategyResult, decision_view
from ..strategies.base import SamplingStrategy
from ._live_order import _CheckedSession, _order_check
from .setup import Setup

MANIFEST = "fovs.csv"
OBJECTS_DIR = "objects"
OBJECTS_FILE = "objects_file"


def live_session(setup: Setup, rules: tuple[SamplingStrategy, ...] | None = None) -> OnlineSession:
    """An ``OnlineSession`` for ``rules`` (default: the setup's rules) on the setup's columns.

    The session also checks the acquisition order of each group's fields so
    far (``order_check``: the readout against ``acquisition_order_col``,
    neighbouring fields by ``stage_cols``), and ``LiveFolder``/``LiveTables``
    print the check's caution beside every stop; without either column, the
    stop states that it treats the fields as a random sample of the well's
    positions. The check flags; it never changes a decision.
    """
    if not isinstance(setup, Setup):
        raise TypeError("setup must be a morphocoverage.workflows.Setup")
    return _CheckedSession(
        strategies=list(setup.stopping_rules() if rules is None else rules),
        feature_cols=setup.columns,
        objective=setup.objective(),
        fov_col=setup.fov_col,
        group_cols=None if setup.group_cols is None else list(setup.group_cols),
        observation_unit=setup.observation_unit,
        readout=setup.readout,
        acquisition_order_col=setup.acquisition_order_col,
        stage_cols=setup.stage_cols,
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
    if header is None and manifest.exists() and manifest.stat().st_size > 0:
        raise ValueError(f"{manifest} has no complete header line; fix or remove it")
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
    new_manifest = header is None  # absent, or an empty file: the header comes first
    if header is None:
        header = list(row)
    elif set(header) != set(row):
        raise ValueError(f"manifest columns are {header}; this field has {list(row)}")
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


@dataclass(frozen=True)
class LiveUpdate:
    """One field added to the session: its metadata, object count and the rules' results.

    ``order_check`` is the acquisition-order check of the field's group so far
    (``live_session``), or ``None`` when none was run.
    """

    fov_metadata: dict[str, Any]
    n_objects: int
    results: list[StrategyResult]
    read_seconds: float
    pending_objects: int = 0
    observation_unit: str = "object"
    order_check: OrderCheck | None = None

    def summary(self) -> str:
        """One line per rule (the decision and why), then one view per target.

        Beside every stop, what it rests on: the order check's caution, or the
        assumption that the fields are a random sample of the well's positions.
        """
        where = ", ".join(f"{k}={v}" for k, v in self.fov_metadata.items() if k != OBJECTS_FILE)
        objects = count(self.n_objects, self.observation_unit)
        head = f"{where}: {objects} (read in {1000 * self.read_seconds:.0f} ms)"
        lines = [head, *(f"  {r.explain(self.order_check)}" for r in self.results)]
        if self.results:
            lines.append(f"  {decision_view(self.results, self.order_check)}")
        if self.pending_objects:
            unit = self.observation_unit
            waiting = (
                f"1 {unit} row waits for its FOV row"
                if self.pending_objects == 1
                else f"{self.pending_objects} {unit} rows wait for their FOV rows"
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

    def _read(self, path: Path, row: Mapping[str, str]) -> pd.DataFrame:
        """One object file; the default reader keeps metadata columns as text, like the manifest."""
        if self.reader is not pd.read_csv:
            return self.reader(path)
        # "001" in the manifest and in the object file stay "001", not 1.
        text = {k: str for k in row if k != OBJECTS_FILE and k not in self.metadata_dtypes}
        return pd.read_csv(path, dtype=text)

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
                objects = self._read(path, row)
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
            update = LiveUpdate(
                metadata,
                len(objects),
                results,
                read,
                observation_unit=self.session.observation_unit,
                order_check=_order_check(self.session, results),
            )
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

        "Says stop" is each rule's own verdict (``StrategyResult.rule_decision``),
        as in the replays: a rule that has no evidence about a declared target
        still stops when its own criterion is met, and the decision view of each
        update says which targets no rule decides.

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
                    stopped = [r.rule_decision == Decision.STOP for r in update.results]
                    if until_stop and stopped and all(stopped):
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
    the FOV identity, resolved with the validator's own resolvers
    (``metadata.resolve_fov_col``, ``resolve_group_cols``,
    ``resolve_identity_cols``, after ``drop_empty_optional_columns``): the
    session's analysis-group columns, then every MorphoBricks group column
    (condition, repeat and, when it holds values, technical_repeat), each when
    the object table has it too, then the FOV column; so a well ``B02`` of two
    repeats is two fields even when ``group_cols`` pool the repeats). The
    default key is resolved once the FOV table has a row, so objects appended
    before the first FOV row wait without fixing it. Objects of a field must
    be appended no later than its FOV row; an object row for a field already
    added raises, since it would change a field the rules have already judged.
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
            self.key = self._default_key(objects, fovs)
        missing = [c for c in self.key if c not in objects.columns or c not in fovs.columns]
        if missing:
            raise ValueError(f"key columns {missing} must be in both the object and FOV tables")
        return self.key

    def _default_key(self, objects: pd.DataFrame, fovs: pd.DataFrame) -> tuple[str, ...]:
        """The FOV identity the validator resolves for these tables, as the key.

        Resolved on the FOV table's columns, where every field's metadata is:
        a group column only the FOV table holds is attached to the objects by
        the validator when the FOV identifier maps to one value, so it is left
        out of the key (a reused identifier then reads as a field appearing
        twice, which raises). An optional column empty in both tables is
        absent, as for the validator.
        """
        requested_fov = self.session.fov_col or self.session._requested_fov_col
        groups = self.session.group_cols
        if groups is None:
            groups = self.session._requested_group_cols
        explicit = [*(groups or ()), *([requested_fov] if requested_fov else [])]
        observed, table = drop_empty_optional_columns(objects, fovs, explicit)
        assert table is not None
        try:
            fov_col = resolve_fov_col(table, None, requested_fov)
        except ValueError as error:
            raise ValueError(
                f"could not find the FOV column; pass fov_col to the session ({error})"
            ) from None
        resolved = resolve_group_cols(table, None, groups)
        identity = resolve_identity_cols(table, None, resolved, fov_col)
        return (*(c for c in identity if c in observed.columns), fov_col)

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
        if fovs.empty:
            # No field yet: the objects wait, and the key is resolved with the first
            # FOV row (an empty FOV table would read every optional column as absent).
            self._buffer = self._appended(new_objects)
            self._n_objects = len(objects)
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
            self._buffer = self._appended(new_objects)
        self._n_objects = len(objects)
        updates = []
        # Records keep each column's type; iterrows would turn integers into floats
        # when every other column is numeric.
        for metadata in new_fovs.to_dict("records"):
            started = time.perf_counter()
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
            update = LiveUpdate(
                metadata,
                len(selected),
                results,
                read,
                self.pending_objects,
                self.session.observation_unit,
                _order_check(self.session, results),
            )
            updates.append(update)
            self.updates.append(update)
        return updates

    def _appended(self, new_objects: pd.DataFrame) -> pd.DataFrame:
        """The buffer with ``new_objects`` appended."""
        parts = [p for p in (self._buffer, new_objects) if p is not None]
        return pd.concat(parts) if len(parts) > 1 else new_objects

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
