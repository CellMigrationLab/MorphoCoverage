"""Acquisition anomalies: fields far from the rest of their own condition and repeat
(quality control; owner, 2026-10-06, plan v2 and audit 10).

Unusual observations are split in two. Biological unusualness reads values made relative to
each observation's own control in its repeat (``unusual_objects`` on those values, in the
report). An acquisition anomaly reads the values themselves, as quality control, and is
reported apart, in "What is in the dataset": never a biological verdict, since a field imaged
out of focus, at another exposure or on debris differs from its neighbours whatever the
biology.

Per condition and biological repeat (a field's own group, whatever the analysis groups),
each field's median of every declared feature is compared with the median of the group's
other fields, on the feature's scale: the log of the medians for a feature whose every value
is positive, else the medians themselves, or the scale declared in the Setup (``scales``),
as the tests across repeats read it (``_repeat_statistics.log_scale``; owner, 2026-10-06,
audit 10: on raw skewed values a null log-normal flagged about 5 of 174 fields, on the log
scale none). The difference ``r`` of a field from the median of its group's other fields is
divided by one robust scale per feature, pooled over every judged group: the robust SD
(1.4826 times the median absolute deviation) of the ``r`` of all their fields, or their SD
where the median absolute deviation is 0, most of them tied: never the SD otherwise, which a
far field among a few would widen enough to hide itself (owner, 2026-10-06, audit 10;
before, the larger of the robust SD and SD of the 7 other fields, which read a null field at
14.9 SDs: the ratio of a difference to the SD of 7 values is about a t with 6 degrees of
freedom, above 5 in about 0.3% of fields and features). A field is an anomaly when
``|r| / scale`` reaches ``ANOMALY_Z`` on some feature, the feature of the largest named with
its relative difference (its difference on a feature that is not on the log scale), so a
flag of a 1% difference reads as one. Leaving the field
out of its own centre keeps one far field from moving the centre that judges it, and the
pooled scale is robust to the few far fields among it. A group needs ``MIN_OTHER_FIELDS``
other fields with a value to judge a field; smaller groups are counted, not judged. With one
observation per field the field median is the observation, so this reads single
observations, and the line says so.

``ANOMALY_Z`` and ``MIN_OTHER_FIELDS`` are engineering thresholds for quality control, not a
calibrated test (owner, 2026-10-06). The count expected by chance is stated beside the
finding: ``EXPECTED_RATE`` (``2 P(Z >= ANOMALY_Z)``, 5.7e-7) per field and feature judged,
were the differences Gaussian on these scales. Missing values are left out of a field's
median; a field with no value of a feature is not judged on it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .._frozen import owned
from .._repeat_statistics import log_scale
from .._style import check_detail, group_label, table, wrap
from .._wording import count, listing, number, percent_fine
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..schema import fov_positions, fov_values, observation_words
from ._common import validated

# SDs from the other fields of the group: the quality-control threshold (see module).
ANOMALY_Z = 5.0
# The other fields with a value a field is compared with, at least (groups of 8 fields).
MIN_OTHER_FIELDS = 7
# The share of fields and features at ANOMALY_Z or more by chance, were the differences
# Gaussian: the expected-by-chance count is this times the values judged.
EXPECTED_RATE = float(2 * stats.norm.sf(ANOMALY_Z))
# 1.4826 times the median absolute deviation estimates the SD of Gaussian data.
_MAD_SD = 1.4826
# Fields named in the short form; the rest are counted.
_NAMED = 4
TITLE = "Acquisition anomalies"


@owned("fields", "groups", "settings")
@dataclass(frozen=True)
class AcquisitionAnomalies:
    """Fields far from the other fields of their condition and repeat (see the module).

    - ``fields``: one row per field judged anomalous: its key, its group, the feature of
      the largest ``|z|``, the field's median, the other fields' median, ``z``, the fold
      between the two medians (when both are positive), the relative difference
      (``relative``, on a feature on the log scale) and the difference;
    - ``groups``: one row per condition and repeat: its fields, the fields judged and the
      anomalies;
    - ``settings``: the features, their scales and pooled scales, the threshold, the minimum
      of other fields, the values judged and the count expected by chance.
    """

    fields: pd.DataFrame
    groups: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any]

    @property
    def found(self) -> int:
        """How many fields are anomalies."""
        return len(self.fields)

    @property
    def judged(self) -> int:
        """How many fields were judged (in groups of ``MIN_OTHER_FIELDS + 1`` fields or more)."""
        return int(self.groups["judged"].sum()) if len(self.groups) else 0

    def _scope(self) -> str:
        """``48 fields judged in 6 groups``, with the fields of the groups too small to judge
        counted apart."""
        judged = self.judged
        groups = self.groups
        within = int((groups["judged"] > 0).sum()) if len(groups) else 0
        small = groups[groups["judged"] == 0] if len(groups) else groups
        apart = (
            f"; {count(int(small['fields'].sum()), 'field')} in {count(len(small), 'group')} "
            f"of fewer than {MIN_OTHER_FIELDS + 1} fields counted, not judged"
            if len(small) and judged
            else ""
        )
        if not judged:
            fields = MIN_OTHER_FIELDS + 1
            return f"no field judged, as no condition and repeat holds {fields} fields"
        return f"{count(judged, 'field')} judged in {count(within, 'group')}{apart}"

    def _how(self) -> str:
        """The scales read: ``on the log scale``, or which features are."""
        scales = self.settings.get("scales", {})
        logged = [f for f, s in scales.items() if s == "log"]
        if logged and len(logged) == len(scales):
            return "on the log scale"
        return f"on the log scale for {listing(logged)}, else on the values" if logged else ""

    def _single(self) -> str:
        """With one observation per field, that this reads single observations."""
        if not self.settings.get("one_per_field"):
            return ""
        unit, units = self.settings.get("unit", ("observation", "observations"))
        return f"; each field holds one {unit}, so this reads single {units}"

    def _chance(self) -> str:
        """``about 0.0017 expected by chance in 3,000 values judged ...``."""
        values = int(self.settings.get("values_judged", 0))
        expected = float(self.settings.get("expected_by_chance", 0.0))
        return (
            f"about {expected:.2g} expected by chance in the {values} values of a field and "
            "feature judged, were the differences Gaussian on these scales"
        )

    def line(self) -> str:
        """The finding in one sentence (the report's short block): ``none: ...`` or ``2 fields
        ...: <the fields>; ...``, the first clause the finding (``The data``'s row)."""
        far = f"{number(ANOMALY_Z)} SDs or more from the other fields of"
        how = f" {self._how()}" if self._how() else ""
        scope = f"{self._scope()}{self._single()}"
        if not self.found:
            return f"none: no field {far} its condition and repeat{how}, {scope}"
        named = [_field_words(row) for row in self.fields.head(_NAMED).to_dict("records")]
        more = self.found - len(named)
        listed = listing([*named, *([f"{more} more"] if more > 0 else [])])
        return (
            f"{count(self.found, 'field')} {far} {'its' if self.found == 1 else 'their'} "
            f"condition and repeat{how}: {listed}; {self._chance()}; an engineering "
            "quality-control threshold, never a biological verdict: "
            f"open {'its image' if self.found == 1 else 'their images'}, {scope}"
        )

    def summary(self, detail: str = "short") -> str:
        """Short: the line; full: the line and one row per anomalous field."""
        line = wrap(f"{TITLE}: {self.line()}.", 0, hang=2)
        if check_detail(detail) != "full" or not self.found:
            return line
        return "\n".join([line, self.table(indent=2)])

    def table(self, indent: int = 0) -> str:
        """One row per anomalous field: its medians, the difference and the SDs."""
        cells = [
            [
                row["field"],
                row["feature"],
                f"{row['value']:.3g}",
                f"{row['others']:.3g}",
                difference_words(row),
                f"{row['z']:+.1f}",
            ]
            for row in self.fields.to_dict("records")
        ]
        columns = ["field", "feature", "field median", "other fields", "difference", "SDs"]
        return table(cells, columns, indent=indent, align="llrrrr")


def difference_words(row: Mapping[str, Any]) -> str:
    """``+25%`` (relative, a feature on the log scale) or ``+0.31`` (the difference)."""
    relative = row.get("relative")
    if relative is not None and not pd.isna(relative):
        if relative >= 1 or relative <= -0.5:  # twice or half the others: a fold reads better
            fold = 1 + float(relative)
            return f"×{fold:.0f}" if fold >= 10 else f"×{fold:.2g}"
        return f"{'+' if relative >= 0 else '−'}{percent_fine(abs(float(relative)))}"
    return f"{float(row['difference']):+.3g}"


def _field_words(row: dict[str, Any]) -> str:
    """``siCTRL · repeat 1 · 7 (intensity, 9.3 SDs above, +25%)``."""
    side = "above" if row["z"] > 0 else "below"
    sds = f"{number(abs(row['z']))} SDs {side}"
    return f"{row['field']} ({row['feature']}, {sds}, {difference_words(row)})"


def _scale(residuals: np.ndarray) -> float:
    """The robust SD of the pooled differences, or their SD where the median absolute
    deviation is 0 (most differences tied, see the module); 0 when undefined."""
    residuals = residuals[~np.isnan(residuals)]
    if len(residuals) < 2:
        return 0.0
    mad = _MAD_SD * float(np.median(np.abs(residuals - np.median(residuals))))
    return mad if mad > 0 else float(np.std(residuals, ddof=1))


def _residuals(medians: np.ndarray) -> np.ndarray:
    """Each field's difference from the median of the group's other fields (NaN when the
    group has fewer than ``MIN_OTHER_FIELDS`` other fields with a value)."""
    residuals = np.full(len(medians), np.nan)
    present = np.flatnonzero(~np.isnan(medians))
    if len(present) < MIN_OTHER_FIELDS + 1:
        return residuals
    for i in present:
        residuals[i] = medians[i] - float(np.median(medians[present[present != i]]))
    return residuals


def _field_medians(data: Any) -> np.ndarray:
    """Each field's median of every feature, one row per row of ``data.fovs`` (NaN for a
    field with no value of it)."""
    return np.column_stack(
        [
            [float(np.median(v)) if len(v) else np.nan for v in fov_values(data, values)]
            for values in (data.feature_values(f) for f in data.feature_cols)
        ]
    )


def _logged(data: Any, scales: Mapping[str, str | None] | None) -> np.ndarray:
    """Per feature, whether it is read on the log scale (see the module)."""
    declared = dict(scales or {})
    return np.array(
        [log_scale(data.feature_values(f), declared.get(f), repr(f)) for f in data.feature_cols],
        dtype=bool,
    )


def _one_per_field(data: Any) -> bool:
    """Whether every field with observations holds exactly one."""
    positions = fov_positions(data, data.fovs)
    counts = np.bincount(positions[positions >= 0])
    return bool(len(counts)) and int(counts.max()) == 1


def acquisition_anomalies(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
    scales: Mapping[str, str | None] | None = None,
) -> AcquisitionAnomalies:
    """The fields far from the other fields of their condition and repeat (see the module).
    ``group_cols`` are those of ``validate_input_tables``: they name a field (a well beside
    its position, say), whatever the condition and repeat the fields are compared within.
    ``scales`` are the features' declared scales (``"ratio"`` or ``"difference"``, the
    Setup's ``scales``); a feature not named keeps the automatic rule."""
    data = validated(objects, fovs, feature_cols, fov_col, group_cols, observation_unit)
    own = [c for c in (CONDITION_COLUMN, REPEAT_COLUMN) if c in data.fovs.columns]
    key = list(dict.fromkeys([*data.group_cols, *data.fov_key]))
    raw = _field_medians(data)
    logged = _logged(data, scales)
    # fallback: the log of a missing or non-positive median: NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        medians = np.where(logged, np.log(raw), raw)
    blocks = data.fovs.groupby(own, sort=False, dropna=False).indices if own else {(): None}
    members_of = {
        values: np.arange(len(data.fovs)) if members is None else np.asarray(members)
        for values, members in blocks.items()
    }
    residuals = np.full(medians.shape, np.nan)
    for members in members_of.values():
        for j in range(medians.shape[1]):
            residuals[members, j] = _residuals(medians[members, j])
    pooled = np.array([_scale(residuals[:, j]) for j in range(medians.shape[1])])
    # fallback: a feature without spread (pooled scale 0): NaN, not scored
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(pooled > 0, residuals / pooled, np.nan)
    identity = data.fovs[key].to_dict("records")
    found, rows = [], []
    for values, members in members_of.items():
        group = dict(zip(own, values if isinstance(values, tuple) else (values,), strict=True))
        mine = z[members]
        judged = ~np.isnan(mine).all(axis=1)
        far = (np.nan_to_num(np.abs(mine)) >= ANOMALY_Z).any(axis=1)
        for i in np.flatnonzero(judged & far):
            j = int(np.nanargmax(np.abs(mine[i])))
            field = identity[members[i]]
            found.append(
                {**field, "field": group_label(field), **_flag(raw[members, j], i, mine[i, j])}
            )
            found[-1].update(feature=data.feature_cols[j], log_scale=bool(logged[j]))
            if not logged[j]:
                found[-1]["relative"] = None
        rows.append({**group, "fields": len(members), "judged": int(judged.sum())})
        rows[-1]["anomalies"] = int((judged & far).sum())
    fields = pd.DataFrame(found)
    if len(fields):
        fields = fields.reindex(fields["z"].abs().sort_values(ascending=False).index)
    judged_values = int(np.isfinite(z).sum())
    unit = observation_words(data)
    settings = {
        "feature_cols": list(data.feature_cols),
        "anomaly_z": ANOMALY_Z,
        "min_other_fields": MIN_OTHER_FIELDS,
        "fov_key": key,
        "values": "log scale where positive",
        "scales": {
            f: "log" if g else "values" for f, g in zip(data.feature_cols, logged, strict=True)
        },
        "pooled_scale": dict(zip(data.feature_cols, map(float, pooled), strict=True)),
        "values_judged": judged_values,
        "expected_by_chance": EXPECTED_RATE * judged_values,
        "one_per_field": _one_per_field(data),
        "unit": unit,
    }
    return AcquisitionAnomalies(fields, pd.DataFrame(rows), tuple(own), settings)


def _flag(raw: np.ndarray, i: int, z: float) -> dict[str, Any]:
    """A flagged field's medians, difference, fold and relative difference on the raw values
    (``raw`` the group's medians of the feature, ``i`` the field)."""
    value = float(raw[i])
    centre = float(np.nanmedian(np.delete(raw, i)))
    low, high = sorted((value, centre))
    positive = low > 0
    return {
        "value": value,
        "others": centre,
        "z": float(z),
        "fold": high / low if positive else None,
        "relative": value / centre - 1 if positive else None,
        "difference": value - centre,
    }


__all__ = [
    "ANOMALY_Z",
    "EXPECTED_RATE",
    "MIN_OTHER_FIELDS",
    "TITLE",
    "AcquisitionAnomalies",
    "acquisition_anomalies",
    "difference_words",
]
