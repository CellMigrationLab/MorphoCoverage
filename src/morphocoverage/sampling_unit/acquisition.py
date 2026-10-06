"""Acquisition anomalies: fields far from the rest of their own condition and repeat, on the
raw values (quality control; owner, 2026-10-06, plan v2).

Unusual observations are split in two. Biological unusualness reads values made relative to
each observation's own control in its repeat (``unusual_objects`` on those values, in the
report). An acquisition anomaly reads the raw values, as quality control, and is reported
apart, in "What is in the dataset": never a biological claim, since a field imaged out of
focus, at another exposure or on debris differs from its neighbours whatever the biology.

Per condition and biological repeat (a field's own group, whatever the analysis groups),
each field's median of every declared feature is compared with the medians of the group's
other fields: ``z = (median - centre) / scale`` with the centre the other fields' median and
the scale the larger of their robust SD (1.4826 times their median absolute deviation) and
their SD, since the median absolute deviation of a few fields is too unstable alone (a feature
constant over the other fields is not judged). A field is an anomaly when ``|z|`` reaches
``ANOMALY_Z`` such SDs on some feature, the feature of the largest ``|z|`` named. Leaving
the field out of its own reference keeps one far field from widening the scale that judges
it. A group needs ``MIN_OTHER_FIELDS`` other fields with a value to judge a field; smaller
groups are counted, not judged. With one observation per field the field median is the
observation, so this reads the observations themselves.

``ANOMALY_Z`` and ``MIN_OTHER_FIELDS`` are engineering thresholds for quality control, not a
calibrated test (owner, 2026-10-06): on Gaussian field medians they flagged 0.26% of the fields
per feature with 7 other fields, 0.15% with 9 and under 0.1% from 11, and a field 10 times
brighter than fields of CV 30% 88-93% of the time (4,000 and 500 simulated groups); with 5
other fields they flagged 0.8% per feature, too many for a list of fields to open. Missing
values are left out of a field's median; a field with no value of a feature is not judged on
it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._style import check_detail, group_label, table, wrap
from .._wording import count, listing, number
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..schema import fov_values
from ._common import groups, validated

# SDs from the other fields of the group: the quality-control threshold (see module).
ANOMALY_Z = 5.0
# The other fields with a value a field is compared with, at least (groups of 8 fields).
MIN_OTHER_FIELDS = 7
# 1.4826 times the median absolute deviation estimates the SD of Gaussian data.
_MAD_SD = 1.4826
# Fields named in the short form; the rest are counted.
_NAMED = 4
TITLE = "Acquisition anomalies"


@dataclass(frozen=True)
class AcquisitionAnomalies:
    """Fields far from the other fields of their condition and repeat, on raw values.

    - ``fields``: one row per field judged anomalous: its key, its group, the feature of
      the largest ``|z|``, the field's median, the other fields' median, ``z`` and the
      fold between the two medians (when both are positive);
    - ``groups``: one row per condition and repeat: its fields, the fields judged and the
      anomalies;
    - ``settings``: the features, the threshold and the minimum of other fields.
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

    def line(self) -> str:
        """The finding in one sentence (the report's short block): ``none: ...`` or ``2 fields
        ...: <the fields>; ...``, the first clause the finding (``The data``'s row)."""
        judged = self.judged
        within = int((self.groups["judged"] > 0).sum()) if len(self.groups) else 0
        scope = (
            f"{count(judged, 'field')} judged in {count(within, 'group')}"
            if judged
            else f"no field judged, as no condition and repeat holds {MIN_OTHER_FIELDS + 1} fields"
        )
        far = f"{number(ANOMALY_Z)} SDs or more from the other fields of"
        if not self.found:
            return f"none: no field {far} its condition and repeat on the raw values, {scope}"
        named = [_field_words(row) for row in self.fields.head(_NAMED).to_dict("records")]
        more = self.found - len(named)
        listed = listing([*named, *([f"{more} more"] if more > 0 else [])])
        return (
            f"{count(self.found, 'field')} {far} their condition and repeat on the raw values: "
            f"{listed}; quality control, never a biological claim: open their images, {scope}"
        )

    def summary(self, detail: str = "short") -> str:
        """Short: the line; full: the line and one row per anomalous field."""
        line = wrap(f"{TITLE}: {self.line()}.", 0, hang=2)
        if check_detail(detail) != "full" or not self.found:
            return line
        cells = [
            [
                row["field"],
                row["feature"],
                f"{row['value']:.3g}",
                f"{row['others']:.3g}",
                f"{row['z']:+.1f}",
            ]
            for row in self.fields.to_dict("records")
        ]
        columns = ["field", "feature", "field median", "other fields", "SDs from them"]
        return "\n".join([line, table(cells, columns, indent=2, align="llrrr")])


def _field_words(row: dict[str, Any]) -> str:
    """``siCTRL · repeat 1 · 7 (intensity, 9.3 SDs above)``."""
    side = "above" if row["z"] > 0 else "below"
    fold = row.get("fold")
    big = fold is not None and not pd.isna(fold) and fold >= 2
    times = f", {number(fold)} times" if big else ""
    return f"{row['field']} ({row['feature']}, {number(abs(row['z']))} SDs {side}{times})"


def _scale(others: np.ndarray) -> float:
    """The larger of the other fields' robust SD and SD (see the module)."""
    centre = float(np.median(others))
    mad = _MAD_SD * float(np.median(np.abs(others - centre)))
    return max(mad, float(np.std(others, ddof=1)))


def _judge(medians: np.ndarray) -> np.ndarray:
    """Each field's ``z`` against the other fields (NaN when not judged)."""
    z = np.full(len(medians), np.nan)
    present = np.flatnonzero(~np.isnan(medians))
    if len(present) < MIN_OTHER_FIELDS + 1:
        return z
    for i in present:
        others = medians[present[present != i]]
        scale = _scale(others)
        if scale > 0:
            z[i] = (medians[i] - float(np.median(others))) / scale
    return z


def acquisition_anomalies(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    fov_col: str | None = None,
    observation_unit: str | None = None,
) -> AcquisitionAnomalies:
    """The fields far from the other fields of their condition and repeat (see the module)."""
    own = [c for c in (CONDITION_COLUMN, REPEAT_COLUMN) if c in fovs.columns]
    data = validated(objects, fovs, feature_cols, fov_col, own, observation_unit)
    key = list(data.fov_key)
    found, rows = [], []
    for group, group_data in groups(data):
        medians = np.column_stack(
            [
                [float(np.median(v)) if len(v) else np.nan for v in fov_values(group_data, vals)]
                for vals in (group_data.feature_values(f) for f in data.feature_cols)
            ]
        )
        before = len(found)
        z = np.column_stack([_judge(medians[:, j]) for j in range(medians.shape[1])])
        judged = ~np.isnan(z).all(axis=1)
        identity = group_data.fovs[key].to_dict("records")
        far = (np.nan_to_num(np.abs(z)) >= ANOMALY_Z).any(axis=1)
        for i in np.flatnonzero(judged & far):
            j = int(np.nanargmax(np.abs(z[i])))
            others = np.delete(medians[:, j], i)
            centre = float(np.nanmedian(others))
            value = float(medians[i, j])
            low, high = sorted((value, centre))
            found.append(
                {
                    **identity[i],
                    "field": group_label(identity[i]),
                    "feature": data.feature_cols[j],
                    "value": value,
                    "others": centre,
                    "z": float(z[i, j]),
                    "fold": high / low if low > 0 else None,
                }
            )
        judged_fields = int(judged.sum())
        rows.append({**group, "fields": len(group_data.fovs), "judged": judged_fields})
        rows[-1]["anomalies"] = len(found) - before
    fields = pd.DataFrame(found)
    if len(fields):
        fields = fields.reindex(fields["z"].abs().sort_values(ascending=False).index)
    settings = {
        "feature_cols": list(data.feature_cols),
        "anomaly_z": ANOMALY_Z,
        "min_other_fields": MIN_OTHER_FIELDS,
        "fov_key": key,
        "values": "raw",
    }
    return AcquisitionAnomalies(fields, pd.DataFrame(rows), tuple(own), settings)


__all__ = [
    "ANOMALY_Z",
    "MIN_OTHER_FIELDS",
    "TITLE",
    "AcquisitionAnomalies",
    "acquisition_anomalies",
]
