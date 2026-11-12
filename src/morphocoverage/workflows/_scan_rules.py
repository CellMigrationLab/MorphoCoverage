"""The rules of the design scan (private): the findings on how the tables were loaded, the
columns that would say more, and the questions to answer before the analysis
(``scan.scan_design``; the text is ``_scan_text``)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

from .._integrity_text import integrity_status
from .._repeat_groups import PairingRefused, named_by_id
from .._style import Status
from .._wording import count, listing, percent
from ..feature_tables import EXPLAINABLE, OBSERVATION_TABLE, FeatureTables, attach_feature_tables
from ..sampling_unit.contrast_cells import IMBALANCE, MIN_BALANCED, SPARSE
from ..schema import unit_plural
from .scan import GOALS, DesignScan, Finding, Question

# Data-check warnings the scan turns into a suggestion or a question instead.
_ASKED = {"No FOV table was given.", "The observation unit is not declared."}
_UNITS = ("cell", "nucleus", "track", "patch", "object")
_UNIT_HINTS = (
    ("track", re.compile(r"track|speed|displacement", re.IGNORECASE)),
    ("nucleus", re.compile(r"nucle", re.IGNORECASE)),
)


def with_tables(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    tables: Mapping[str, pd.DataFrame | tuple[pd.DataFrame, str]] | None,
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
        # fallback: the error is the reason the scan reports
        except (ValueError, TypeError) as error:
            reason = str(error)
        else:
            return attached.objects, attached, []
    text = f"{reason[:1].upper()}{reason[1:]}."
    return objects, None, [Finding(Status.PROBLEM, "feature_tables", text)]


def named_tables(
    observations: pd.DataFrame, fovs: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[Finding]]:
    """The tables the design reads, each condition named by its ``condition_id`` when they
    hold it, as the analyses and the report name it (#628): the scan's conditions, its control
    guess and the ``Setup`` it builds then use the IDs (the folder labels stay in
    ``DesignScan.objects`` and ``.fovs``). IDs that cannot be read leave the folder labels
    and are a finding."""
    try:
        named, named_fovs, _ = named_by_id(observations, fovs)
    # fallback: the scan reports the refusal and describes the folders as given
    except (PairingRefused, TypeError) as refused:
        text = f"The conditions cannot be named by their IDs: {refused}."
        return observations, fovs, [Finding(Status.PROBLEM, "condition_id", text)]
    return named, named_fovs, []


def copy_tables(
    features: list[str], explainable: dict[str, tuple[str, ...]], others: dict[str, Any]
) -> dict[str, Any]:
    """The tables the copy checks read apart: the observation table's features and each
    explainable table when there are explainable tables, else the features as one; then
    each other table (owner, 2026-10-06)."""
    inside = {c for cols in explainable.values() for c in cols}
    own = [c for c in features if c not in inside]
    first = {OBSERVATION_TABLE: own, **explainable} if explainable else {EXPLAINABLE: features}
    return {**first, **others}


def unit_words(scan: DesignScan) -> tuple[str, str]:
    """The observation unit in the singular and plural (``observation`` when not declared)."""
    unit = scan.settings["observation_unit"] or "observation"
    return unit, unit_plural(unit)


def compared(scan: DesignScan) -> list[Any]:
    """The conditions compared with a control (all of them when there is no control)."""
    controls = scan.design.get("controls") or [scan.design.get("control_guess")]
    return [c for c in scan.design["conditions"] if c not in controls]


def _check_findings(scan: DesignScan) -> list[Finding]:
    check = scan.data_check
    found = [
        Finding(Status.PROBLEM, "loaded", f"{p.title} Where: {p.where} Fix: {p.fix}")
        for p in check.errors
    ]
    found += [
        Finding(Status.CAUTION, "loaded", f"{p.title} Fix: {p.fix}")
        for p in check.warnings
        if p.title not in _ASKED
    ]
    if not found:
        found.append(Finding(Status.OK, "loaded", "The data check found nothing to fix."))
    return found


def _copy_finding(scan: DesignScan) -> Finding | None:
    integrity = scan.integrity
    if integrity is None:
        if not scan.data_check.ok:
            return None  # the errors above come first
        return Finding(
            Status.NOT_ASSESSED, "copies", "Copies were not checked: no numeric feature column."
        )
    status = integrity_status(integrity)
    if status is Status.PROBLEM:
        involved = integrity.field_labels()
        return Finding(
            status,
            "copies",
            f"{count(len(integrity.findings), 'copy finding')} in "
            f"{count(len(involved), 'field')} ({listing(involved, 4)}): structure to inspect, "
            "not a judgment; scan.integrity.summary() names each.",
        )
    if status is Status.NOT_ASSESSED:
        return Finding(
            status,
            "copies",
            "Copies could not be checked: no feature varies enough to tell observations apart.",
        )
    checked = int((integrity.checks["status"] == "checked").sum())
    return Finding(
        Status.OK,
        "copies",
        f"No copied field or row ({checked} copy checks: fields read twice, rows pasted into "
        "another field, units changed).",
    )


def _balance_findings(scan: DesignScan) -> list[Finding]:
    rows = scan.design["balance"]
    if not rows:
        return []
    word, words = unit_words(scan)
    found = []
    unbalanced = [r for r in rows if r["unbalanced"]]
    if unbalanced:
        widest = max(unbalanced, key=lambda r: r["spread"])
        found.append(
            Finding(
                Status.OK,
                "balance",
                f"The {words} per biological repeat differ by more than "
                f"{percent(IMBALANCE - 1)} in {count(len(unbalanced), 'condition')} of "
                f"{len(rows)} (up to {widest['spread']:.1f}× in {widest['condition']}); the "
                f"report tests the {words} both as provided and balanced over the repeats.",
            )
        )
    else:
        found.append(
            Finding(
                Status.OK,
                "balance",
                f"The {words} per biological repeat are balanced (within "
                f"{percent(IMBALANCE - 1)}) in every condition.",
            )
        )
    by_repeat: dict[str, list[str]] = {}
    for row in rows:
        for repeat in row["sparse"]:
            label = f"{row['condition']} ({count(row['sizes'][repeat], word, words)})"
            by_repeat.setdefault(str(repeat), []).append(label)
    for repeat, entries in by_repeat.items():
        found.append(
            Finding(
                Status.CAUTION,
                "balance",
                f"Few {words} in biological repeat {repeat}: {listing(entries)}, under "
                f"{percent(SPARSE)} of the median repeat or under {MIN_BALANCED}. The balanced "
                "test leaves it out rather than draw every repeat down to it; check it was not "
                "cut short or filtered by mistake.",
            )
        )
    return found


def findings(scan: DesignScan) -> list[Finding]:
    """How the tables were loaded: the data check, copies, constant features, balance."""
    found = _check_findings(scan)
    copy = _copy_finding(scan)
    if copy is not None:
        found.append(copy)
    for name in scan.design["constant"]:
        found.append(
            Finding(
                Status.CAUTION,
                "features",
                f"Feature {name!r} holds one value in every row: check the column was exported "
                "as intended.",
            )
        )
    found += _balance_findings(scan)
    return found + _described_findings(scan)


def _described_findings(scan: DesignScan) -> list[Finding]:
    """Where the controls come from on a described export (MC-2): the description's
    references, or why none is defaulted."""
    described = scan.design.get("described")
    if described is None:
        return _by_id_findings(scan)
    if described.block is None:
        return [
            Finding(Status.CAUTION, "controls", f"No control is defaulted: {described.missing}.")
        ]
    compared = [f"{c} vs {r}" for c, r in described.pairs] or ["none (no reference declared)"]
    text = (
        f"Controls from the experiment description ({described.block.description_sha256[:12]}): "
        f"{listing(compared, 8)}; the report uses them unless you declare others, and then says "
        "so beside them."
    )
    return [Finding(Status.OK, "controls", " ".join([text, *described.notes]))]


def _by_id_findings(scan: DesignScan) -> list[Finding]:
    """A described table (``condition_id``) scanned without its description: no control is
    guessed from the names (#631), and the scan says where the controls come from."""
    design = scan.design
    if not design.get("by_id") or scan.settings["control_declared"] or not design["conditions"]:
        return []
    text = (
        "Controls: the tables carry condition_id (a described run), so no control is guessed "
        "from the condition names; the controls come from the experiment description "
        "(scan_design(..., description=read_experiment(coverage_json, fovs))) or must be "
        "declared by their ID (control={'condition': ...})."
    )
    return [Finding(Status.OK, "controls", text)]


def _target_suggestion(scan: DesignScan) -> str | None:
    suggested = scan.design.get("target_suggestions")
    if suggested and scan.design["targets"] is None:  # a described export (MC-2)
        return (
            f"The experiment description suggests targets ({'; '.join(suggested)}), as "
            "written, descriptive only: confirm or edit them, "
            "Setup(targets=experiment.target_suggestions.confirm()), and the report then says "
            "whether the reagents of one target agree; unconfirmed, they are not used."
        )
    others = compared(scan)
    if len(others) < 2 or scan.design["targets"] is not None:
        return None
    text = (
        f"Several conditions are compared ({len(others)}"
        + (" besides the control" if scan.design.get("control_guess") is not None else "")
        + ("s" if len(scan.design.get("controls") or []) > 1 else "")
        + "): add a 'target' column naming what each acts on, if some should have the same "
        "effect; the report then says whether they agree."
    )
    groups = scan.design["reagent_groups"]
    if groups:
        named = [
            f"{listing(members)} look like reagents of {target}"
            for target, members in groups.items()
        ]
        text += " By their names, " + "; ".join(named) + "."
    return text


def suggestions(scan: DesignScan) -> list[str]:
    """The columns that would say more, most useful first."""
    design, settings = scan.design, scan.settings
    unit, words = unit_words(scan)
    found = []
    if not design["conditions"]:
        found.append(
            "Add a 'condition' column naming the experimental group of each row; without it "
            "nothing is compared."
        )
    if not design["repeats"]:
        found.append(
            "Add a 'biological_repeat' column (the independent experiment, often its date); "
            "without it the conditions cannot be compared across repeats."
        )
    if settings["fov_added"]:
        found.append(
            "Add 'image_file' (or 'position_id', one value per stage position): no image "
            f"identifier was found, so each {unit} is assumed to be on its own image. With it "
            f"the report tells images from {words} (how many fields, copies between images)."
        )
    elif not settings["fovs_given"]:
        found.append(
            "Add a FOV table listing every imaged field, empty ones included: without it, "
            f"fields with no {unit} are not known and the sampling answers are optimistic."
        )
    target = _target_suggestion(scan)
    if target:
        found.append(target)
    if design["technical"] is None and design["repeats"]:
        found.append(
            "No technical repeat found: add 'technical_repeat' if a condition was imaged in "
            "more than one well, dish or coverslip per biological repeat; the report then "
            "tells wells from fields."
        )
    missing = []
    if not design["order"]:
        missing.append("'acquisition_order'")
    if not design["stage"]:
        missing.append("stage coordinates ('stage_x_um', 'stage_y_um')")
    if missing and not settings["fov_added"]:
        found.append(
            f"Optional, for planning the next experiment: {listing(missing)} let optimise check "
            "whether the order of the fields mattered."
        )
    return found


def _unit_hint(scan: DesignScan) -> str | None:
    names = " ".join(str(c) for c in scan.settings["feature_cols"])
    for unit, pattern in _UNIT_HINTS:
        if pattern.search(names):
            return unit
    return None


def questions(scan: DesignScan) -> list[Question]:
    """What to answer before the analysis, the goal first."""
    design, settings = scan.design, scan.settings
    features = settings["feature_cols"]
    asked = [
        Question(
            "goal",
            "What should the analysis answer: compare the population averages of the "
            "conditions, compare their phenotypic fingerprints, or check a rare population?",
            GOALS,
            "averages",
            "averages: the readout's mean per condition across biological repeats (the "
            "default); fingerprint: how several features vary together, the covariance "
            "fingerprint, declared as fingerprint=(OutcomeCriterion(...), ...); rare: whether "
            "a rare population was sampled enough, declared as rare=RareTarget(...).",
        ),
        Question(
            "readout",
            "Which feature is the readout you report?",
            tuple(str(f) for f in features),
            str(features[0]) if len(features) == 1 else None,
            "Every answer is about the readout; the other features describe the phenotype.",
        ),
    ]
    described = design.get("described")
    defaulted = described is not None and described.block is not None and described.pairs
    if not settings["control_declared"] and design["conditions"] and not defaulted:
        guesses = design.get("control_guesses", [])
        asked.append(
            Question(
                "control",
                "Which condition is the negative control?",
                tuple(str(c) for c in design["conditions"]),
                None if design["control_guess"] is None else str(design["control_guess"]),
                "The conditions are compared with it, within each biological repeat"
                + (f"; {listing(guesses)} are all named like a control" if len(guesses) > 1 else "")
                + ".",
            )
        )
    if not settings["features_declared"]:
        asked.append(
            Question(
                "features",
                "Which columns are the features?",
                tuple(str(f) for f in features),
                None,
                f"The scan read {count(len(features), 'numeric column')} as candidates, for "
                "the copy checks only; declare the ones that describe the phenotype.",
            )
        )
    if settings["observation_unit"] is None:
        asked.append(
            Question(
                "observation_unit",
                "What is one row: a cell, a nucleus, a track?",
                _UNITS,
                _unit_hint(scan),
                "The report names it in every sentence.",
            )
        )
    return asked


def repeat_lines(scan: DesignScan) -> list[str]:
    """How the conditions fall across the biological repeats, in sentences."""
    design = scan.design
    repeats, sets = design["repeats"], design["repeat_sets"]
    if not repeats:
        return []
    head = f"{count(len(repeats), 'biological repeat')} ({listing(repeats, 6)})"
    if len(repeats) == 1:
        return [f"{head}: a pilot; the report describes it, and nothing is tested across repeats."]
    if len(sets) == 1:
        return [f"{head}; every condition in every repeat."]
    lines = [f"{head}, holding different conditions (separate acquisitions):"]
    control = design.get("control_guess")
    controls = design.get("controls") or [control]
    for entry in sets:
        held = [c for c in entry["conditions"] if c not in controls]
        mine = [c for c in controls if c in entry["conditions"]]
        shared = f" with {listing(mine)}" if len(controls) > 1 and mine else ""
        shared = shared or (" with the control" if control in entry["conditions"] else "")
        what = f"{listing(held, 6)}{shared}" if held else "the control only"
        lines.append(f"  {listing(entry['repeats'], 4)}: {what}")
    held_in = scan.layout.notna().sum(axis=1)
    single = [c for c in design["conditions"] if held_in.get(c, 0) == 1]
    if single:
        lines.append(f"  In one repeat only: {listing(single)}.")
    return lines


def fields_line(scan: DesignScan) -> str:
    """The fields and the observations per field, in a sentence."""
    unit, words = unit_words(scan)
    low, median, high = scan.design["per_field"]
    fields = count(scan.design["fields"], "field")
    if scan.settings["fov_added"]:
        return f"{fields}, assumed: one per row (no image identifier)."
    if high == 1:
        return f"{fields}, one {unit} per field."
    if low == high:
        return f"{fields}, {low} {words} per field."
    return f"{fields}, {low} to {high} {words} per field (median {median:g})."


def technical_line(scan: DesignScan) -> str | None:
    """What the scan found about technical repeats (wells), or ``None`` without repeats."""
    technical = scan.design["technical"]
    if not scan.design["repeats"]:
        return None
    if technical is None:
        return "Technical repeats: none recorded (one well per condition and repeat)."
    low, high = technical["low"], technical["high"]
    spread = f"{low}" if low == high else f"{low} to {high}"
    return f"Technical repeats: {spread} per condition and biological repeat."


__all__ = [
    "compared",
    "copy_tables",
    "fields_line",
    "findings",
    "questions",
    "repeat_lines",
    "suggestions",
    "technical_line",
    "unit_words",
    "with_tables",
]
