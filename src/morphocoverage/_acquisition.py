"""The identity of the acquisition a result was computed on, so results can be checked together.

Results assembled into one report must describe one acquisition (issue 277). The identity is
the digest of the typed FOV universe, the FOV key of every acquired field (``_identity``), so it
is stable under row order and under normalisation of values, and tells ``1`` from ``"1"``.
Only that universe is compared: analysis settings, groups, features and values may differ
legitimately between the results of one acquisition. Every result records it, and a result
that records none is refused (``check_one_acquisition``), never assembled with a note.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from ._identity import row_keys
from .metadata import MORPHOBRICKS_GROUP_COLUMNS
from .schema import ValidatedInput

ACQUISITION = "acquisition"


def acquisition_identity(data: ValidatedInput) -> str:
    """The SHA-256 of the sorted typed FOV keys of ``data``.

    The key is the FOV column with the MorphoBricks group columns the FOV table holds, whether
    or not the analysis groups by them, so the same acquisition read with other MorphoBricks
    groups (or with identity columns only) gives the same identity; and with every other
    column of ``data.fov_key`` (a custom ``experiment``, ``plate`` or ``well`` named in
    ``group_cols``), since such a column tells fields apart that share a FOV name (#416): two
    acquisitions with field ``f1`` of ``experiment`` ``run_A`` and ``run_B`` must not share
    an identity. Results read with and without a custom identity column therefore differ."""
    fov_col = data.fov_key[-1]
    named = [c for c in data.fovs.columns if c in MORPHOBRICKS_GROUP_COLUMNS or c in data.fov_key]
    # Sorted by name, so the column order of the FOV table does not change the identity.
    columns = sorted((c for c in named if c != fov_col), key=str) + [fov_col]
    keys = sorted([list(part) for part in key] for key in row_keys(data.fovs, columns))
    text = json.dumps(keys, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def recorded(settings: Mapping[str, Any] | None) -> str | None:
    """The acquisition identity a result's settings carry, or ``None`` when it has none."""
    if not settings:
        return None
    value = settings.get(ACQUISITION)
    return value if isinstance(value, str) else None


def result_identity(result: Any) -> str | None:
    """The acquisition identity of one analysis result: its settings' record, or the
    ``acquisition`` field of a result without settings (``SufficiencyReport``)."""
    found = recorded(getattr(result, "settings", None))
    if found is not None:
        return found
    field_value = getattr(result, "acquisition", None)
    return field_value if isinstance(field_value, str) else None


def check_one_acquisition(results: Mapping[str, Any]) -> None:
    """Refuse results that record no acquisition identity, or that came from different ones.

    ``results`` maps each supplied result's name to it (``None`` for one not supplied). Every
    analysis result records the identity of the tables it was computed on, so a result without
    one (rebuilt by hand, or edited) cannot be checked against the others and is refused, even
    alone: it is never assembled with a note. Raises when two recorded identities differ."""
    found = {name: result_identity(r) for name, r in results.items() if r is not None}
    missing = [name for name, ident in found.items() if ident is None]
    if missing:
        verb, it = ("records", "it") if len(missing) == 1 else ("record", "they")
        raise ValueError(
            f"{', '.join(missing)} {verb} no acquisition identity, so {it} cannot be checked "
            "against the other results: pass results as the analysis functions return them"
        )
    if len(set(found.values())) > 1:
        by_identity: dict[str, list[str]] = {}
        for name, ident in found.items():
            by_identity.setdefault(ident, []).append(name)
        groups = " and ".join("/".join(names) for names in by_identity.values())
        raise ValueError(
            f"results from different acquisitions cannot be assembled into one report: "
            f"{groups} were computed on different sets of fields"
        )
