"""The copy checks on each feature table apart (private; the report and the scan).

A copy confined to one table, an embedding computed twice for one image while its
explainable features differ, is missed when every column is compared at once: the two
fields differ in the other columns (owner, 2026-10-06, after an audit). So
``integrity_by_table`` runs ``integrity.integrity_checks`` on the explainable features and on
each other table apart, and gives the union: each finding once, naming the table or tables in
which it was found (``table``); a check is ``checked`` when it applied to any table. With the
explainable features alone it is ``integrity_checks`` itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ._wording import count, listing
from .integrity import CHECK_COLUMNS, FINDING_COLUMNS, IntegrityResult, integrity_checks

# A finding is one check on one pair of fields (or one field): found in one table or several.
_SAME = ["check", "field", "other_field"]


def _checks(results: Mapping[str, IntegrityResult]) -> pd.DataFrame:
    """Per check, ``checked`` when it applied to any table (then the most compared), else the
    first table's reason."""
    rows = []
    first = next(iter(results.values())).checks
    for check in first["check"]:
        rows_here = [r.checks.set_index("check").loc[check] for r in results.values()]
        checked = [r for r in rows_here if r["status"] == "checked"]
        chosen = rows_here[0] if not checked else checked[0]
        compared = max(int(r["compared"]) for r in (checked or rows_here))
        rows.append(
            {
                "check": check,
                "status": chosen["status"],
                "reason": None if checked else chosen["reason"],
                "compared": compared,
            }
        )
    return pd.DataFrame(rows, columns=list(CHECK_COLUMNS))


def _findings(results: Mapping[str, IntegrityResult]) -> pd.DataFrame:
    """Every finding once, the tables that found it named in ``table``."""
    found = [r.findings.assign(table=name) for name, r in results.items() if len(r.findings)]
    columns = [*FINDING_COLUMNS, "table"]
    if not found:
        return pd.DataFrame(columns=columns)
    every = pd.concat(found, ignore_index=True)
    tables = every.groupby(_SAME, sort=False)["table"].agg(lambda names: listing(list(names)))
    first = every.drop_duplicates(_SAME).drop(columns="table")
    return first.join(tables, on=_SAME).reset_index(drop=True)[columns]


def integrity_by_table(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    tables: Mapping[str, Sequence[str]],
    **options: Any,
) -> IntegrityResult:
    """The copy checks on each of ``tables`` apart (a name and its columns, the explainable
    features first), and their union (see the module); ``options`` are
    ``integrity_checks``'s."""
    named = {name: list(columns) for name, columns in tables.items() if len(columns)}
    if len(named) == 1:
        return integrity_checks(objects, fovs, feature_cols=next(iter(named.values())), **options)
    results = {
        name: integrity_checks(objects, fovs, feature_cols=columns, **options)
        for name, columns in named.items()
    }
    first = next(iter(results.values()))
    settings = dict(first.settings)
    for key in ("field_labels", "group_labels"):
        settings[key] = {k: v for r in results.values() for k, v in r.settings[key].items()}
    settings["feature_cols"] = [c for columns in named.values() for c in columns]
    settings["continuous_features"] = [
        c for r in results.values() for c in r.settings["continuous_features"]
    ]
    settings["tables"] = {name: len(columns) for name, columns in named.items()}
    settings["features_compared"] = "each feature table apart: " + listing(
        [f"{name} ({count(n, 'column')})" for name, n in settings["tables"].items()], 8
    )
    return IntegrityResult(_findings(results), _checks(results), settings)


__all__ = ["integrity_by_table"]
