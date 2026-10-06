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

``write_fov`` does both steps for the acquisition side (it and the manifest
reader live in ``workflows._manifest``). ``LiveFolder.poll()`` reads the
manifest lines added since the last poll (it keeps its byte offset and
ignores a last line that is not yet terminated), reads each new field's
object file, and returns one update per field; ``watch()`` repeats the poll
every ``interval`` seconds, and with ``until_stop`` returns once the
objective's overall decision is stop in every group seen so far
(``"latest_group"``: in the latest field's group). The manifest may only
grow: one rewritten, truncated or replaced under the watcher (its first
bytes are no longer the ones read) raises, rather than skipping fields. A
line identical to one already read (an acquisition retry) is skipped with a
warning; a line naming an accepted field with another object file raises,
naming both lines.

**Checks on the fields.** Each field passes the session's validation and,
as ``check_tables`` warns, a warning for rows that are exact copies of
another row or a ``label_id`` twice (an object file appended twice).
``LiveTables`` refuses, before adding any field of an update, a key that
names two fields: a FOV-table group column the key leaves out that holds two
values for one key (file names that restart in each well), or a key twice.

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

import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .._former_repeat import Reader, one_warning
from ..metadata import (
    MORPHOBRICKS_GROUP_COLUMNS,
    drop_empty_optional_columns,
    resolve_fov_col,
    resolve_group_cols,
    resolve_identity_cols,
)
from ..objectives import ObjectiveConfig
from ..online import OnlineSession
from ..order_check import OrderCheck
from ..results import Decision, StrategyResult, _target_decisions
from ..strategies.base import SamplingStrategy
from ._live_order import _CheckedSession, _order_check, embedding_columns
from ._live_text import live_summary
from ._manifest import MANIFEST, OBJECTS_DIR, OBJECTS_FILE, ManifestReader, write_fov
from .setup import Setup


def live_session(
    setup: Setup,
    rules: tuple[SamplingStrategy, ...] | None = None,
    *,
    objective: ObjectiveConfig | None = None,
) -> OnlineSession:
    """An ``OnlineSession`` for ``rules`` (default: the setup's rules) on the setup's columns.

    ``objective`` (default: ``setup.objective()``, every declared target) narrows what a stop
    needs, for example ``ObjectiveConfig("rare_discovery", rare=target)`` to stop each group
    once a rare phenotype is settled, its mean not waited for.

    The session also checks the acquisition order of each group's fields so
    far (``order_check``: the readout against ``acquisition_order_col``,
    neighbouring fields by ``stage_cols``), and ``LiveFolder``/``LiveTables``
    print the check's caution beside every stop; without either column, the
    stop states that it treats the fields as a random sample of the well's
    positions. The check flags; it never changes a decision.
    """
    if not isinstance(setup, Setup):
        raise TypeError("setup must be a morphocoverage.workflows.Setup")
    if objective is not None and not isinstance(objective, ObjectiveConfig):
        raise TypeError("objective must be a morphocoverage.ObjectiveConfig")
    columns = embedding_columns(setup, rules, objective) if setup.embedding_only else None
    return _CheckedSession(
        strategies=list(setup.stopping_rules() if rules is None else rules),
        feature_cols=columns or setup.columns,
        objective=setup.objective() if objective is None else objective,
        fov_col=setup.fov_col,
        group_cols=None if setup.group_cols is None else list(setup.group_cols),
        observation_unit=setup.observation_unit,
        readout=setup.readout,
        acquisition_order_col=setup.acquisition_order_col,
        stage_cols=setup.stage_cols,
    )


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

    def summary(self, detail: str = "short") -> str:
        """The field's block of the live view: ``"short"`` (the default) or ``"full"``.

        Short: ``✅ field 8 · MOCK · 1A · B6 — stop: all 3 rules say enough``
        (``⏳ continue: ...`` while a target is not yet met, with what it still
        needs), the field's counts, one table of the rules (each one's verdict,
        evidence and threshold), and at a stop what it rests on (the order
        check's caution, or the assumption that the fields are a random sample
        of the well's positions), each caveat once. Full: every rule's
        ``StrategyResult.explain`` line and every entry of ``decision_view``,
        one per line with its marker (``workflows._live_text``). A pure function
        of the update.
        """
        return live_summary(self, detail)


def _value_kind(values: pd.Series) -> str:
    """``number``, ``text`` or ``other``: what the non-missing values of a key column are."""
    kind = pd.api.types.infer_dtype(values, skipna=True)
    if kind in ("integer", "floating", "mixed-integer-float", "decimal"):
        return "number"
    return "text" if kind == "string" else "other"


def print_update(update: LiveUpdate, detail: str = "short") -> None:
    """Print one update's summary (the default ``on_update`` of ``watch``; ``detail`` as
    ``LiveUpdate.summary``)."""
    print(update.summary(detail), flush=True)


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
        self._manifest = ManifestReader(self.folder)
        self.updates: list[LiveUpdate] = []

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

    @one_warning
    def poll(self) -> list[LiveUpdate]:
        """Add every field completed since the last poll; one update per field, in order.

        A line identical to one already accepted (an acquisition that retried
        and wrote its line twice) is skipped with a warning. A field that
        raises is read again on the next poll; the error's note names its
        manifest line.
        """
        return list(self._fields())

    def _fields(self) -> Iterator[LiveUpdate]:
        """Add the completed fields one at a time, each only when the next is asked for.

        ``watch`` stops asking once its condition is met, so the session holds
        exactly the fields of the updates it returns; the rest wait for the
        next poll.
        """
        manifest = self._manifest
        for row, raw in manifest.new_rows():
            line = manifest.lines + 1
            repeated = manifest.repeat_of(row)
            if repeated is not None:
                warnings.warn(
                    f"{MANIFEST} line {line} repeats line {repeated} exactly "
                    f"({OBJECTS_FILE} {row.get(OBJECTS_FILE)!r}); skipped, since a field is "
                    "added once (an acquisition that retried may write its line twice).",
                    UserWarning,
                    stacklevel=3,
                )
            if not row or repeated is not None:
                manifest.skip(raw)
                continue
            update = self._add(row, line)
            # Move past the field only once the session has it: a field that raises is
            # read again on the next poll instead of being skipped.
            manifest.accept(row, raw)
            self.updates.append(update)
            yield update

    def _add(self, row: dict[str, str], line: int) -> LiveUpdate:
        """Read one manifest line's field and add it to the session."""
        started = time.perf_counter()
        # The former name of the biological repeat, read as the new one (``_former_repeat``).
        former = Reader()
        metadata = former.mapping(self._typed(row), MANIFEST)
        same = self._manifest.same_field_as(row)
        if same is not None:
            raise ValueError(
                f"{MANIFEST} line {line} lists the field of line {same} (the same metadata) "
                f"with another object file, {row.get(OBJECTS_FILE)!r}; if it is a retry, remove "
                f"line {line} from {self.folder / MANIFEST} (the watcher reads it again on the "
                "next poll; the lines already read must stay as they are)"
            )
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
        objects = former.table(objects, str(row[OBJECTS_FILE]) if path else "objects")
        former.warn()
        read = time.perf_counter() - started
        # The manifest is the authority on metadata: an object file that repeats a
        # metadata column (say a plate number read back as an integer) is compared
        # as text, so only a real disagreement raises.
        for key, value in metadata.items():
            if key in objects.columns and isinstance(value, str):
                objects[key] = objects[key].astype(str)
        try:
            results = self.session.add_fov(objects, metadata)
        except Exception as error:
            error.add_note(
                f"While adding {MANIFEST} line {line} ({OBJECTS_FILE} "
                f"{row.get(OBJECTS_FILE)!r}): fix that field or its line in "
                f"{self.folder / MANIFEST}; the watcher reads it again on the next poll."
            )
            raise
        return LiveUpdate(
            metadata,
            len(objects),
            results,
            read,
            observation_unit=self.session.observation_unit,
            order_check=_order_check(self.session, results),
        )

    @one_warning
    def watch(
        self,
        *,
        interval: float = 1.0,
        timeout: float | None = None,
        max_fovs: int | None = None,
        until_stop: bool | str = False,
        on_update: Callable[[LiveUpdate], None] | None = print_update,
        stop_on: str = "decision",
    ) -> list[LiveUpdate]:
        """Poll every ``interval`` seconds until ``timeout``, ``max_fovs`` or ``until_stop``.

        ``until_stop=True`` (or ``"all_groups"``) returns once a group is done in
        every group seen so far, each group judged by its latest field: on a
        plate acquired well by well in turn, the watch goes on while any well
        still needs fields. ``until_stop="latest_group"`` returns as soon as the
        latest field's group is done (one well done).

        A group is done when the objective's overall decision is stop, the last
        line of the update's decision view (``stop_on="decision"``, the default,
        as in ``benchmark``): a declared target that no rule decides, or one
        still continuing or undefined, keeps the watch going, so the watch and
        the live text agree (owner, 2026-10-05; from 2026-09-26 it read each
        rule's own verdict). ``stop_on="rule"`` is the lower-level stop when
        every rule's own verdict (``StrategyResult.rule_decision``) is stop,
        whatever the targets it does not decide.

        The condition is checked after each field, and fields are added one at a
        time: when it is met, the fields still waiting in the folder (a backlog,
        or a restart on a full folder) are not added, so the session holds
        exactly the fields of the returned updates; the next ``poll`` or
        ``watch`` reads the rest.

        Stop with Ctrl-C at any time; the updates so far are kept in ``updates``.
        """
        if interval <= 0:
            raise ValueError("interval must be positive")
        if until_stop not in (False, True, "all_groups", "latest_group"):
            raise ValueError("until_stop must be False, True, 'all_groups' or 'latest_group'")
        if stop_on not in ("decision", "rule"):
            raise ValueError(f"stop_on must be 'decision' or 'rule', not {stop_on!r}")
        started = time.monotonic()
        seen: list[LiveUpdate] = []
        latest: dict[tuple, bool] = {}  # each group's latest field: done
        try:
            while True:
                for update in self._fields():
                    seen.append(update)
                    if on_update is not None:
                        on_update(update)
                    group = tuple(update.results[0].group.items()) if update.results else ()
                    latest[group] = _done(update.results, stop_on)
                    if until_stop == "latest_group" and latest[group]:
                        return seen
                    if until_stop in (True, "all_groups") and all(latest.values()):
                        return seen
                    if max_fovs is not None and len(seen) >= max_fovs:
                        return seen
                if timeout is not None and time.monotonic() - started >= timeout:
                    return seen
                time.sleep(interval)
        except KeyboardInterrupt:
            return seen


def _done(results: Sequence[StrategyResult], stop_on: str) -> bool:
    """Whether one field's results end its group's acquisition (see ``LiveFolder.watch``)."""
    if not results:
        return False
    if stop_on == "rule":
        return all(r.rule_decision is Decision.STOP for r in results)
    return _target_decisions(results)[-1].decision is Decision.STOP


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
        former = Reader()
        key = former.columns(key, "key")
        former.warn()
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

    def _check_fov_keys(
        self, fovs: pd.DataFrame, new_fovs: pd.DataFrame, key: tuple[str, ...]
    ) -> None:
        """Refuse, before adding any field, a key that does not name one field.

        A group column of the FOV table that the key leaves out (the objects
        do not carry it) must hold one value per key, as the validator requires
        to attach it; and no key may appear twice in the FOV table.
        """
        requested = self.session.group_cols or self.session._requested_group_cols or ()
        groups = [
            c
            for c in dict.fromkeys((*requested, *MORPHOBRICKS_GROUP_COLUMNS))
            if c in fovs.columns and c not in key
        ]
        for column in groups:
            pairs = fovs.loc[:, [*key, column]].drop_duplicates()
            if pairs.duplicated(subset=list(key)).any():
                raise ValueError(
                    f"cannot attach {column!r} from the FOV table because the key {list(key)} "
                    "is reused across different metadata values; include the grouping column "
                    "in the object table (or pass it in key) or use a globally unique FOV "
                    "identifier. No field of this update was added."
                )
        seen = set(self._added)
        for fov_key in self._keys(new_fovs, key):
            if fov_key in seen:
                raise ValueError(
                    f"field {fov_key} appears twice in the FOV table; no field of this update "
                    "was added"
                )
            seen.add(fov_key)

    @staticmethod
    def _keys(table: pd.DataFrame, key: tuple[str, ...]) -> list[tuple]:
        return list(zip(*(table[c].tolist() for c in key), strict=True))

    @one_warning
    def update(self, objects: pd.DataFrame, fovs: pd.DataFrame) -> list[LiveUpdate]:
        """Add every field whose FOV row was appended since the last call, in order.

        A ``repeat`` column (the former name of ``biological_repeat``) is read as
        ``biological_repeat``, with one ``DeprecationWarning`` per call.
        """
        if not isinstance(objects, pd.DataFrame) or not isinstance(fovs, pd.DataFrame):
            raise TypeError("objects and fovs must be pandas DataFrames")
        former = Reader()
        objects, fovs = former.table(objects, "objects"), former.table(fovs, "fovs")
        former.warn()
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
        # Before any field is added: a key that names two fields would pool their objects.
        self._check_fov_keys(fovs, new_fovs, key)
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
