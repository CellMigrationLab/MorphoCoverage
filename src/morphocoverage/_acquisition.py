"""The identity of the acquisition a result was computed on, so results can be checked together.

Results assembled into one report must describe one acquisition (issue 277). The identity is
the digest of the typed FOV universe, the FOV key of every acquired field (``_identity``), so it
is stable under row order and under normalisation of values, and tells ``1`` from ``"1"``.
Only that universe is compared: analysis settings, groups, features and values may differ
legitimately between the results of one acquisition.
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

    The key is the FOV column with the MorphoBricks group columns the FOV table holds, never
    the analysis's own ``group_cols``: the same acquisition read with other groups (or with
    identity columns only) must give the same identity."""
    columns = [c for c in data.fovs.columns if c in MORPHOBRICKS_GROUP_COLUMNS]
    columns = [c for c in columns if c != data.fov_key[-1]] + [data.fov_key[-1]]
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


def check_one_acquisition(results: Mapping[str, Any]) -> list[str]:
    """Refuse results that came from different acquisitions; name those that cannot be told.

    ``results`` maps each supplied result's name to it. Returns the names of results that
    record no identity (rebuilt or older) while others do, so the report can say they were not
    checked; raises when two recorded identities differ."""
    found = {name: result_identity(r) for name, r in results.items() if r is not None}
    known = {name: ident for name, ident in found.items() if ident is not None}
    if len(set(known.values())) > 1:
        by_identity: dict[str, list[str]] = {}
        for name, ident in known.items():
            by_identity.setdefault(ident, []).append(name)
        groups = " and ".join("/".join(names) for names in by_identity.values())
        raise ValueError(
            f"results from different acquisitions cannot be assembled into one report: "
            f"{groups} were computed on different sets of fields"
        )
    return [name for name, ident in found.items() if ident is None] if known else []


def unchecked_note(names: list[str]) -> str:
    """The sentence naming the results that could not be checked against the others."""
    verb = "records" if len(names) == 1 else "record"
    return (
        f"{', '.join(names)} {verb} no acquisition identity, so it could not be checked "
        "against the other results."
    )
