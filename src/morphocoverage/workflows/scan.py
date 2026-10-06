"""Scan the experimental design before any analysis: what the dataset is, whether it was
loaded correctly, which columns would say more, and what to ask before the analysis.

``scan_design(objects, fovs)`` reads the tables as given, with as little declared as
possible, and returns a ``DesignScan``:

- **what the dataset is**: the conditions (the control, declared or guessed from its
  name), the biological repeats and which conditions each holds (one experiment, or
  several acquisitions sharing the control), the technical repeats, the fields and the
  observations per field, the planes, the feature columns;
- **whether it was loaded correctly**: the data check (``check_tables``), the copy checks
  (``integrity_checks``: a field read twice, rows pasted into another field), features that
  hold one value everywhere, and how the observations fall across the biological repeats
  (unbalanced repeats, which the report then tests both as provided and balanced, and sparse
  repeats, which the balanced test leaves out);
- **columns that would say more**: an image identifier (without one, each row is assumed
  to be its own image), a ``target`` when several conditions are compared (reagents of one
  gene that should agree), a ``technical_repeat`` when a condition was imaged in more than
  one well, a FOV table;
- **questions before the analysis**: the goal (compare population averages, the default;
  a phenotypic fingerprint; a rare population), the readout, and the control, the features
  and the observation unit when they were not declared.

Nothing is estimated and nothing is tested: the scan counts and checks. Its findings follow
the owner's rule for usual designs (three repeats, one well each): what is usual is stated,
not warned about. ``DesignScan.setup(readout, ...)`` turns the answers into a ``Setup``,
and ``DesignScan.objects`` / ``.fovs`` are the tables as scanned (with the image identifier
the scan added, when it added one), ready for ``report`` and ``optimise``.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import pandas as pd

from .._former_repeat import one_warning, read_former, renamed_tables
from .._integrity_tables import integrity_by_table
from .._style import Status, check_detail
from ..data_check import DataCheck, check_tables
from ..feature_tables import EXPLAINABLE, FeatureTables, attach_feature_tables
from ..integrity import IntegrityResult
from ..metadata import (
    CONDITION_COLUMN,
    MORPHOBRICKS_FOV_CANDIDATES,
    MORPHOBRICKS_SHARED_COLUMNS,
    REPEAT_COLUMN,
    TARGET_COLUMN,
    TECHNICAL_REPEAT_COLUMN,
    WITHIN_FOV_METADATA_COLUMNS,
    column_has_values,
    sorted_labels,
)
from ..sampling_unit.contrast_cells import IMBALANCE, sparse
from .setup import Setup

T = TypeVar("T")

# The image identifier the scan adds when the tables have none: one image per row.
ADDED_FOV_COLUMN = "position_id"
# Columns that describe a row rather than measure it; never offered as features.
_METADATA = {
    *MORPHOBRICKS_SHARED_COLUMNS,
    *WITHIN_FOV_METADATA_COLUMNS,
    TARGET_COLUMN,
    "repeat",
    "label_id",
    "fov_id",
    "source_row",
    "acquisition_order",
    "valid",
    "experiment",
    "plate",
    "well",
    "site",
    "field",
    "frame",
    "index",
}
_STAGE = re.compile(r"stage", re.IGNORECASE)
# Where a patch sits in its image and how it was cut, not what it shows: CellPatcher's
# coordinates (y0, center_x, output_height) and fiberfeatures' copies of them (patch_y0,
# patch_raw_x1, patch_center_y, patch_was_clipped).
_PATCH_GEOMETRY = re.compile(
    r"^(?:patch_(?:raw_)?)?(?:[yxz][01]|center_[yxz]|output_(?:height|width))$"
    r"|^patch_(?:height|width|coverage|original_coverage|touches_border|(?:original_|measured_)?"
    r"area_px|clipped_fraction|was_clipped)$",
    re.IGNORECASE,
)
# A control named as labs name it: siCTRL, Control, MOCK, DMSO, NT, scramble, vehicle.
_CONTROL_WORDS = re.compile(
    r"^(ctrl|control|mock|dmso|untreated|scr|scramble|scrambled|nt|nontargeting|non_targeting|"
    r"vehicle|neg|negative|empty|luc|gfp)$"
)
# Two reagents of one target: siCCT8#5 and siCCT8#8, shRNA_1 and shRNA_2.
_REAGENT = re.compile(r"^(?P<stem>.+?)[#_\-. ](?P<number>\w*\d\w*)$")
_PREFIX = re.compile(r"^(si|sh|sg)(?=[A-Z0-9])")
GOALS = ("averages", "fingerprint", "rare")


@dataclass(frozen=True)
class Finding:
    """One line of the scan: its ``status`` (✅, ⚠️ or ❌), what it is ``about`` and the text."""

    status: Status
    about: str
    text: str


@dataclass(frozen=True)
class Question:
    """A question to answer before the analysis: its ``key`` (the ``Setup`` field it sets, or
    ``goal``), the ``text``, the ``options``, the ``suggested`` answer (or ``None``) and
    ``why`` it matters."""

    key: str
    text: str
    options: tuple[str, ...]
    suggested: str | None
    why: str


@dataclass(frozen=True)
class DesignScan:
    """What ``scan_design`` found (see the module).

    - ``design``: the facts, counted (conditions, the control, repeats, technical repeats,
      fields, observations per field, planes, targets, reagent groups).
    - ``layout``: observations per condition (rows) and biological repeat (columns).
    - ``findings``: whether the tables were loaded correctly, one ``Finding`` each.
    - ``suggestions``: the columns that would say more, in sentences.
    - ``questions``: what to answer before the analysis.
    - ``data_check``, ``integrity``: the checks the scan ran (``integrity`` is ``None`` when
      the tables could not be read or hold no feature).
    - ``objects``, ``fovs``: the tables as scanned, ready for the workflows.
    - ``settings``: the features (and whether declared), the field identifier (and whether
      the scan added it), the analysis groups, the observation unit, the control.
    """

    design: dict[str, Any]
    layout: pd.DataFrame
    findings: tuple[Finding, ...]
    suggestions: tuple[str, ...]
    questions: tuple[Question, ...]
    data_check: DataCheck
    integrity: IntegrityResult | None
    objects: pd.DataFrame
    fovs: pd.DataFrame
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> Status:
        """❌ when the tables cannot be analysed as they are, ⚠️ when a finding needs a look,
        ✅ otherwise."""
        statuses = [f.status for f in self.findings]
        if Status.PROBLEM in statuses:
            return Status.PROBLEM
        return Status.CAUTION if Status.CAUTION in statuses else Status.OK

    def summary(self, detail: str = "short", *, questions: bool = True) -> str:
        """``"short"``: the design, the checks, the columns that would say more and the
        questions; ``"full"`` adds the data check and the copy checks in full. With
        ``questions=False`` the questions are left out (the guided session asks them)."""
        from ._scan_text import scan_summary

        return scan_summary(self, check_detail(detail), questions=questions)

    def setup(self, readout: str, *, goal: str = "averages", **fields: Any) -> Setup:
        """A ``Setup`` from the scan: the features, the control, the field identifier, the
        analysis groups and the observation unit it found, with ``readout`` and any other
        ``Setup`` field in ``fields`` (which win over the scan's).

        ``goal="averages"`` compares the population averages (the readout's mean);
        ``"fingerprint"`` needs ``fingerprint=(OutcomeCriterion(...), ...)`` and ``"rare"``
        needs ``rare=RareTarget(...)`` in ``fields``, since their tolerances are the user's
        choice (``docs/common/setup.md``). The control must be declared or answered: a guessed
        control is never used silently.
        """
        if goal not in GOALS:
            raise ValueError(f"goal must be one of {GOALS}, not {goal!r}")
        needed = {"fingerprint": "fingerprint", "rare": "rare"}.get(goal)
        if needed and not fields.get(needed):
            raise ValueError(
                f"goal={goal!r} needs {needed}=... in the Setup: its tolerance is a scientific "
                "choice with no default (see docs/common/setup.md, 'What must the acquisition "
                "preserve?')"
            )
        settings = self.settings
        if "control" not in fields and not settings["control_declared"]:
            guess = self.design.get("control_guess")
            raise ValueError(
                "declare the control: control={'condition': ...}"
                + (f" (the scan suggests {guess!r})" if guess else "")
            )
        values: dict[str, Any] = {
            "features": tuple(settings["feature_cols"]),
            "control": settings["control"],
            "fov_col": settings["fov_col"],
            "group_cols": tuple(settings["group_cols"]),
            "observation_unit": settings["observation_unit"],
            "feature_tables": settings.get("feature_tables"),
        }
        values.update(fields)
        return Setup(readout=readout, **values)


def _feature_candidates(objects: pd.DataFrame, taken: set[str]) -> list[str]:
    """Numeric columns that may be features: not metadata, not stage coordinates, not where a
    patch was cut (``patch_y0``, ``center_x``), not a row number (an integer column with a
    different value in every row), not one value in every row."""
    found = []
    for column in objects.columns:
        name = str(column)
        if column in taken or name.lower() in _METADATA or name.lower().endswith("_id"):
            continue
        if _STAGE.search(name) or _PATCH_GEOMETRY.match(name):
            continue
        if name.startswith(("Unnamed", "source")):
            continue
        values = objects[column]
        if values.dtype == bool or not pd.api.types.is_numeric_dtype(values):
            continue
        if values.nunique() < 2:
            continue  # one value everywhere: a setting carried along, not a measurement
        if (
            pd.api.types.is_integer_dtype(values)
            and len(values) > 1
            and values.nunique() == len(values)
        ):
            continue
        found.append(column)
    return found


def _control_guess(conditions: Sequence[Any]) -> list[Any]:
    """The conditions whose name reads like a negative control (siCTRL, MOCK, DMSO)."""
    guesses = []
    for condition in conditions:
        name = _PREFIX.sub("", str(condition).strip())
        words = [w for w in re.split(r"[^A-Za-z0-9]+", name.lower()) if w]
        if any(_CONTROL_WORDS.match(w) for w in words) or _CONTROL_WORDS.match(
            name.lower().replace(" ", "").replace("-", "")
        ):
            guesses.append(condition)
    return guesses


def reagent_groups(conditions: Sequence[Any], control: Any = None) -> dict[str, list[Any]]:
    """Conditions that look like several reagents of one target, by name: ``siCCT8#5`` and
    ``siCCT8#8`` give ``{"CCT8": ["siCCT8#5", "siCCT8#8"]}``. Only groups of two or more,
    never the control."""
    groups: dict[str, list[Any]] = {}
    for condition in conditions:
        if condition == control:
            continue
        match = _REAGENT.match(str(condition).strip())
        if match is None:
            continue
        target = _PREFIX.sub("", match["stem"])
        groups.setdefault(target, []).append(condition)
    return {t: members for t, members in groups.items() if len(members) > 1}


def _fov_column(
    objects: pd.DataFrame, fovs: pd.DataFrame | None, fov_col: str | None
) -> str | None:
    """The field identifier: ``fov_col``, else the first MorphoBricks candidate with values."""
    if fov_col is not None:
        return fov_col
    for name in MORPHOBRICKS_FOV_CANDIDATES:
        if column_has_values(objects, name):
            return name
    return None


def _repeat_sets(objects: pd.DataFrame, conditions: list[Any]) -> list[dict[str, Any]]:
    """The biological repeats grouped by the conditions they hold, largest group first."""
    by_repeat = objects.groupby(REPEAT_COLUMN, sort=False)[CONDITION_COLUMN].agg(
        lambda values: tuple(c for c in conditions if c in set(values))
    )
    sets: dict[tuple, list[Any]] = {}
    for repeat in sorted_labels(by_repeat.index):
        sets.setdefault(by_repeat[repeat], []).append(repeat)
    return [
        {"conditions": list(held), "repeats": repeats}
        for held, repeats in sorted(sets.items(), key=lambda item: -len(item[1]))
    ]


def _per_field(
    objects: pd.DataFrame, key: list[str], fovs: pd.DataFrame | None = None
) -> tuple[int, int, float, int]:
    """The fields and the observations per field (fewest, median, most); with the FOV table,
    every field it lists, an empty one as 0, as the report counts them."""
    sizes = objects.groupby(key, sort=False, dropna=False).size().rename("n").reset_index()
    if fovs is not None and set(key) <= set(fovs.columns):
        sizes = fovs[key].drop_duplicates().merge(sizes, on=key, how="left").fillna({"n": 0})
    if sizes.empty:
        return 0, 0, 0.0, 0
    n = sizes["n"]
    return len(sizes), int(n.min()), float(n.median()), int(n.max())


def _technical(objects: pd.DataFrame, keys: list[str]) -> dict[str, Any] | None:
    """Technical repeats per condition and biological repeat: ``None`` when no column holds
    a label other than ``single``."""
    if not column_has_values(objects, TECHNICAL_REPEAT_COLUMN):
        return None
    labels = objects[TECHNICAL_REPEAT_COLUMN].astype(str).str.strip().str.lower()
    if set(labels) <= {"single", ""}:
        return None
    per = objects.groupby(keys, sort=False)[TECHNICAL_REPEAT_COLUMN].nunique() if keys else None
    if per is None or per.empty:
        n = int(objects[TECHNICAL_REPEAT_COLUMN].nunique())
        return {"low": n, "high": n}
    return {"low": int(per.min()), "high": int(per.max())}


def _design(
    objects: pd.DataFrame,
    fov_key: list[str],
    control: Any,
    control_declared: bool,
    fovs: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """The facts of the design and the observations per condition and repeat; with the FOV
    table (``fovs``), its conditions and repeats, those held only by empty fields included."""
    design: dict[str, Any] = {"observations": len(objects)}
    has_condition = column_has_values(objects, CONDITION_COLUMN)
    has_repeat = column_has_values(objects, REPEAT_COLUMN)
    listed = fovs is not None and set(fov_key) <= set(fovs.columns)
    groups = fovs if listed else objects
    fields, *per_field = (
        _per_field(objects, fov_key, fovs) if fov_key else (len(objects), 1, 1.0, 1)
    )
    design["fields"], design["per_field"] = fields, tuple(per_field)
    conditions: list[Any] = []
    design["control_guesses"], design["control_guess"] = [], None
    if has_condition:
        guesses = _control_guess(list(dict.fromkeys(groups[CONDITION_COLUMN])))
        design["control_guesses"] = guesses
        design["control_guess"] = (
            control if control_declared else (guesses[0] if len(guesses) == 1 else None)
        )
        first = control if control_declared else design["control_guess"]
        conditions = sorted_labels(groups[CONDITION_COLUMN], first)
    design["conditions"] = conditions
    design["control"] = control if control_declared else None
    design["repeats"] = sorted_labels(groups[REPEAT_COLUMN]) if has_repeat else []
    keys = [c for c, has in ((CONDITION_COLUMN, has_condition), (REPEAT_COLUMN, has_repeat)) if has]
    design["technical"] = _technical(objects, keys)
    design["repeat_sets"] = _repeat_sets(groups, conditions) if has_condition and has_repeat else []
    planes = {}
    for axis in WITHIN_FOV_METADATA_COLUMNS:
        if column_has_values(objects, axis) and fov_key:
            per = objects.groupby(fov_key, sort=False, dropna=False)[axis].nunique()
            if int(per.max()) > 1:
                planes[axis] = (int(per.min()), int(per.max()))
    design["planes"] = planes
    if column_has_values(objects, TARGET_COLUMN) and has_condition:
        targets = objects.groupby(TARGET_COLUMN, sort=False)[CONDITION_COLUMN].agg(
            lambda values: sorted_labels(values)
        )
        design["targets"] = {t: targets[t] for t in sorted_labels(targets.index)}
    else:
        design["targets"] = None
    design["reagent_groups"] = reagent_groups(conditions, design["control_guess"])
    if has_condition and has_repeat:
        layout = objects.groupby([CONDITION_COLUMN, REPEAT_COLUMN], sort=False).size().unstack()
        layout = layout.reindex(index=conditions, columns=design["repeats"])
    elif has_condition:
        layout = objects.groupby(CONDITION_COLUMN, sort=False).size().reindex(conditions)
        layout = layout.to_frame("observations")
    else:
        layout = pd.DataFrame()
    if listed and not layout.empty:
        # A condition and repeat whose fields are all empty holds 0 observations, not none.
        layout = layout.where(_layout(fovs, layout).isna(), layout.fillna(0))
    return design, layout


def _layout(table: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``table`` per condition (and repeat), shaped as the layout ``like``."""
    if list(like.columns) == ["observations"]:
        counts = table.groupby(CONDITION_COLUMN, sort=False).size().to_frame("observations")
    else:
        counts = table.groupby([CONDITION_COLUMN, REPEAT_COLUMN], sort=False).size().unstack()
    return counts.reindex(index=like.index, columns=like.columns)


def _balance(layout: pd.DataFrame, repeats: list[Any]) -> list[dict[str, Any]]:
    """Per condition held in two or more repeats: the observations per repeat, the spread
    (largest over smallest) and the sparse repeats the balanced test leaves out."""
    rows = []
    if len(repeats) < 2:
        return rows
    for condition, counts in layout.iterrows():
        sizes = {r: int(n) for r, n in counts.items() if pd.notna(n) and n > 0}
        if len(sizes) < 2:
            continue
        spread = max(sizes.values()) / min(sizes.values())
        rows.append(
            {
                "condition": condition,
                "sizes": sizes,
                "spread": spread,
                "unbalanced": spread > IMBALANCE,
                "sparse": sparse(sizes),
            }
        )
    return rows


def _constant(objects: pd.DataFrame, features: Sequence[str]) -> list[str]:
    """Features holding one value in every row (a column exported wrong, or a placeholder)."""
    if len(objects) < 2:
        return []
    found = []
    for name in features:
        values = pd.to_numeric(objects[name], errors="coerce").dropna()
        if len(values) > 1 and values.nunique() == 1:
            found.append(name)
    return found


def _with_tables(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    tables: Mapping[str, pd.DataFrame] | None,
    fov_col: str,
    added: bool,
) -> tuple[pd.DataFrame, FeatureTables | None, list[Finding]]:
    """``objects`` with the other feature tables attached, what was checked, and the finding
    that says why they cannot be used (then ``objects`` as given)."""
    if not tables:
        return objects, None, []
    reason = None
    if added:
        reason = (
            "the other feature tables are matched to observations by their image, and the "
            "observation table names none"
        )
    else:
        try:
            attached = attach_feature_tables(objects, fovs, tables, fov_col=fov_col)
        except (ValueError, TypeError) as error:
            reason = str(error)
        else:
            return attached.objects, attached, []
    text = f"{reason[:1].upper()}{reason[1:]}."
    return objects, None, [Finding(Status.PROBLEM, "feature_tables", text)]


def _quiet(function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """``function`` with its ``UserWarning`` s caught: the scan reports what they say."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return function(*args, **kwargs)


@one_warning
def scan_design(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None = None,
    *,
    feature_cols: Sequence[str] | None = None,
    control: Mapping[str, Any] | None = None,
    observation_unit: str | None = None,
    fov_col: str | None = None,
    feature_tables: Mapping[str, pd.DataFrame] | None = None,
) -> DesignScan:
    """Scan the design of ``objects`` (and ``fovs``, when there is a FOV table) before any
    analysis (see the module).

    Everything may be left undeclared: without ``feature_cols`` the numeric columns that are
    not metadata are read as candidate features (for the copy checks only, and said so);
    without an image identifier (``position_id``, ``image_file``, ``fov_id`` or ``fov_col``)
    each row is assumed to be its own image; without ``control`` a condition named like a
    control is suggested, never used. ``feature_tables`` are other feature tables for the same
    observations, by name (a DINO embedding, owner 2026-10-05): checked and attached
    (``attach_feature_tables``), they are listed with the design, read by the copy checks,
    and named in ``DesignScan.setup``; one that cannot be used is a finding. Never raises
    for bad data: what is wrong is in the findings.
    """
    from ._scan_rules import findings, questions, suggestions

    if not isinstance(objects, pd.DataFrame):
        raise TypeError(f"objects must be a pandas DataFrame, not {type(objects).__name__}")
    objects, fovs = renamed_tables(objects, fovs)
    (control,) = read_former(control=control)
    objects = objects.copy()
    control_declared = control is not None
    control_value = dict(control).get(CONDITION_COLUMN) if control else None

    found_fov = _fov_column(objects, fovs, fov_col)
    added_fov = found_fov is None
    if added_fov:
        found_fov = ADDED_FOV_COLUMN
        objects[found_fov] = [f"row {i + 1}" for i in range(len(objects))]
        fovs = None
    groups = [
        c
        for c in (CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN)
        if column_has_values(objects, c)
    ]
    fov_key = [*groups, found_fov]
    fovs_given = fovs is not None
    if not fovs_given:
        fovs = objects[fov_key].drop_duplicates().reset_index(drop=True)

    features_declared = feature_cols is not None
    taken = {*fov_key, *groups}
    features = list(feature_cols) if features_declared else _feature_candidates(objects, taken)
    objects, tables, refused = _with_tables(objects, fovs, feature_tables, found_fov, added_fov)
    data_check = _quiet(
        check_tables,
        objects,
        fovs,
        feature_cols=features,
        fov_col=found_fov,
        observation_unit=observation_unit,
        control=control,
    )
    integrity = None
    if data_check.ok and features:
        # On each feature table apart, the union said (owner, 2026-10-06).
        integrity = _quiet(
            integrity_by_table,
            objects,
            fovs,
            tables={EXPLAINABLE: features, **(tables.columns if tables else {})},
            fov_col=found_fov,
            observation_unit=observation_unit,
        )
    # A FOV table that passed the data check lists every field acquired, empty ones included.
    listed = fovs if fovs_given and data_check.ok else None
    design, layout = _design(objects, fov_key, control_value, control_declared, listed)
    design["balance"] = _balance(layout, design["repeats"])
    design["constant"] = _constant(objects, features) if data_check.ok and features_declared else []
    columns = [*objects.columns, *(fovs.columns if fovs is not None else ())]
    design["stage"] = list(dict.fromkeys(str(c) for c in columns if _STAGE.search(str(c))))
    design["order"] = "acquisition_order" in objects.columns or (
        fovs is not None and "acquisition_order" in fovs.columns
    )
    settings = {
        "feature_cols": features,
        "features_declared": features_declared,
        "fov_col": found_fov,
        "fov_added": added_fov,
        "fovs_given": fovs_given,
        "group_cols": groups,
        "observation_unit": observation_unit,
        "control": dict(control) if control else None,
        "control_declared": control_declared,
        "feature_tables": tables.columns if tables else None,
    }
    design["feature_tables"] = tables.lines() if tables else []
    scan = DesignScan(
        design=design,
        layout=layout,
        findings=(),
        suggestions=(),
        questions=(),
        data_check=data_check,
        integrity=integrity,
        objects=objects,
        fovs=fovs,
        settings=settings,
    )
    return DesignScan(
        design=design,
        layout=layout,
        findings=(*findings(scan), *refused),
        suggestions=tuple(suggestions(scan)),
        questions=tuple(questions(scan)),
        data_check=data_check,
        integrity=integrity,
        objects=objects,
        fovs=fovs,
        settings=settings,
    )


__all__ = ["DesignScan", "Finding", "Question", "reagent_groups", "scan_design"]
