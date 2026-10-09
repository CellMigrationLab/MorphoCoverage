"""The identity of the acquisition a result was computed on (private).

``report`` and ``optimise`` record ``source_identity(objects, fovs)`` on their result;
``html_report`` and ``document`` check it, so one page or one document cannot mix conclusions
from one acquisition with figures, downloads or a plan from another (issue 359).

The identity is a SHA-256 digest of both tables, independent of the order of rows and columns
and of the former name ``repeat`` of ``biological_repeat``, and type-preserving: integer ``1``
and text ``"1"`` give different digests, as do ``1`` and ``1.0`` held in different dtypes.
Object columns are hashed through their distinct values, so the cost is one Python pass over
the distinct values, not over the rows.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..metadata import REPEAT_COLUMN

_FORMER_REPEAT = "repeat"
# What stands for a missing value in an object column (pandas codes it -1 when factorizing).
_MISSING = hashlib.sha256(b"missing").digest()


def _text_hash(text: str) -> np.uint64:
    return np.frombuffer(hashlib.sha256(text.encode()).digest()[:8], dtype=np.uint64)[0]


def _column_hash(values: pd.Series) -> np.ndarray:
    """One uint64 per row; equal values of equal type hash alike."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.astype(object)
    if values.dtype == object:
        codes, uniques = pd.factorize(values, use_na_sentinel=True)
        typed = np.array(
            [_text_hash(f"{type(u).__module__}.{type(u).__name__}:{u!r}") for u in uniques]
            + [np.frombuffer(_MISSING[:8], dtype=np.uint64)[0]],
            dtype=np.uint64,
        )
        return typed[np.where(codes < 0, len(uniques), codes)]
    return pd.util.hash_array(values.to_numpy())


def _table_digest(table: pd.DataFrame) -> bytes:
    if _FORMER_REPEAT in table.columns and REPEAT_COLUMN not in table.columns:
        table = table.rename(columns={_FORMER_REPEAT: REPEAT_COLUMN})
    names = sorted(str(c) for c in table.columns)
    by_name = {str(c): table[c] for c in table.columns}
    rows = np.zeros(len(table), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for name in names:
            rows = rows * np.uint64(1000003) + _column_hash(by_name[name]) + _text_hash(name)
    return hashlib.sha256(repr(names).encode() + np.sort(rows).tobytes()).digest()


def source_identity(objects: pd.DataFrame, fovs: pd.DataFrame) -> str:
    """The digest of the observation table and the FOV table (see the module)."""
    combined = hashlib.sha256()
    combined.update(_table_digest(objects))
    combined.update(_table_digest(fovs))
    return combined.hexdigest()


def same_source(first: str | None, other: str | None) -> bool:
    """Whether two recorded identities are the same acquisition; ``None`` (no identity
    recorded) is never the same as anything."""
    return first is not None and first == other


def verify_source(recorded: str | None, objects: pd.DataFrame, fovs: pd.DataFrame) -> None:
    """Refuse tables other than the ones a result was computed on (issue 359): ``ValueError``
    naming why, also when the result records no source."""
    if recorded is None:
        raise ValueError(
            "this report records no source tables, so the page cannot check that objects and "
            "fovs are the ones it was run on; pass check_source=False to build the page anyway"
        )
    if not same_source(recorded, source_identity(objects, fovs)):
        raise ValueError(
            "objects and fovs are not the tables the report was run on (their contents differ: "
            "a value, a row or a column changed); the page would mix the report's answers with "
            "figures and data from other tables. Pass the tables given to report()"
        )


__all__ = ["same_source", "source_identity", "verify_source"]
