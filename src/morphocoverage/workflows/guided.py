"""A guided session: load the tables, scan them, ask a few questions, save the answers, report.

``guide()`` walks a user through a dataset in this order:

1. **Load** the observation table (and the FOV table, when there is one), from a path or
   a DataFrame, and choose the **results folder**.
2. If the folder already holds saved answers (``ANSWERS_FILE``) for the same table, offer
   to **continue** with them or to **delete them and start over**; answers saved for
   another table are never reused.
3. **Scan** the design (``scan_design``) and print what it found; the scan is saved as
   ``scan.txt``.
4. **Ask**: did the dataset load correctly (if not, stop and fix the tables); the control
   to log (the scan's guess is the default); an optional positive control (a condition known
   to change the phenotype, for question 1); the aim (population averages, a phenotypic
   fingerprint, or both); the features to assess (several, or all the numeric columns);
   the readout among them; what one row is, when not declared; and for a fingerprint, its
   tolerance (no default has a published meaning, so the user states it).
5. **Save** the answers to ``ANSWERS_FILE`` in the folder, then run the **report** with the
   ``Setup`` they make, and save it: ``report.html`` (``html_report``: every answer and
   what to do next, the best-ranked features, the interactive PCA and UMAP, the glossary,
   and every file inside it for download, so the page alone can be shared), the figures in
   ``figures/`` as PNG, PDF and their data as CSV, ``report_short.txt``,
   ``report_full.txt``, and the whole folder as ``<folder>.zip`` (``zip_results``).

Every question has a default the user takes with Enter. ``ask`` and ``say`` are the
input and output functions (``input`` and ``print`` by default; a notebook cell works the
same way), so a script or a test can answer for the user. ``Answers`` can also be written
and read without the questions (``save_answers``, ``load_answers``), and
``answers.setup(scan)`` builds the ``Setup`` from them. From a terminal:
``python -m morphocoverage.workflows cells.csv [fovs.csv] --folder results``.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .._wording import percent
from ..multivariate_outcomes import CovarianceError
from ..retrospective.criteria import OutcomeCriterion
from . import _guided_ask as q
from .html_report import html_report, zip_results
from .retrospective import Report, report
from .scan import DesignScan, scan_design
from .setup import PositiveControl, Setup, default_novelty

ANSWERS_FILE = "morphocoverage_answers.json"
ANSWERS_VERSION = 1
DEFAULT_FOLDER = "morphocoverage_results"
# The reference fields seen before a positive-control field is judged (PositiveControl).
POSITIVE_FIELDS = 10


def table_checksum(table: pd.DataFrame) -> str:
    """A short checksum of a table's columns and values, so saved answers are reused only for
    the table they were given for."""
    digest = hashlib.sha256("\x1f".join(map(str, table.columns)).encode())
    digest.update(pd.util.hash_pandas_object(table, index=False).to_numpy().tobytes())
    return digest.hexdigest()[:16]


@dataclass
class Answers:
    """What the user answered, as saved in ``ANSWERS_FILE``.

    ``aim`` is ``"population"``, ``"fingerprint"`` or ``"both"``; ``control`` is
    ``{"condition": ...}`` or ``None``; ``fingerprint_tolerance`` is the share the covariance
    of the standardised features must stay within (``None`` for the population aim);
    ``dataset`` records the tables the answers were given for (paths, rows, checksum).
    """

    loaded_correctly: bool
    control: dict[str, Any] | None
    aim: str
    features: list[str]
    readout: str
    observation_unit: str
    fingerprint_tolerance: float | None = None
    positive: dict[str, Any] | None = None
    dataset: dict[str, Any] = field(default_factory=dict)
    saved: str | None = None
    version: int = ANSWERS_VERSION

    def __post_init__(self) -> None:
        if self.aim not in q.AIMS:
            raise ValueError(f"aim must be one of {tuple(q.AIMS)}, not {self.aim!r}")
        if not self.features:
            raise ValueError("choose at least one feature")
        if self.readout not in self.features:
            raise ValueError(f"the readout {self.readout!r} must be one of the features")
        if self.aim != "population" and self.fingerprint_tolerance is None:
            raise ValueError(f"aim={self.aim!r} needs fingerprint_tolerance")

    def fingerprint(self) -> tuple[OutcomeCriterion, ...]:
        """The fingerprint criterion the answers declare (none for the population aim)."""
        if self.aim == "population":
            return ()
        tolerance = float(self.fingerprint_tolerance)
        return (
            OutcomeCriterion(
                CovarianceError(tuple(self.features), True),
                tolerance,
                "lower_is_better",
                f"the covariance of the standardised features within {percent(tolerance)}",
            ),
        )

    def setup(self, scan: DesignScan, **fields: Any) -> Setup:
        """The ``Setup`` these answers make for ``scan`` (``fields`` add or override any
        ``Setup`` field, e.g. ``n_jobs=4``)."""
        goal = q.AIMS[self.aim]
        values: dict[str, Any] = {
            "features": tuple(self.features),
            "control": self.control,
            "observation_unit": self.observation_unit,
        }
        if goal == "fingerprint":
            values["fingerprint"] = self.fingerprint()
        if goal == "fingerprint" or self.positive is not None:
            # A positive control is judged by the definition of new.
            values["novelty"] = default_novelty(tuple(self.features))
        if self.positive is not None:
            values["positive"] = PositiveControl(dict(self.positive), fovs_seen=POSITIVE_FIELDS)
        values.update(fields)
        with warnings.catch_warnings():
            # A control the user chose to leave out is an answer, not a guess to warn about.
            warnings.simplefilter("ignore", UserWarning)
            return scan.setup(self.readout, goal=goal, **values)

    def lines(self) -> list[str]:
        """The answers in sentences."""
        control = "none" if self.control is None else self.control["condition"]
        lines = [
            f"Control: {control}.",
            f"Aim: {q.AIM_TEXT[self.aim]}.",
            f"Features ({len(self.features)}): {', '.join(self.features)}".rstrip(".") + ".",
            f"Readout: {self.readout}.",
            f"One row is a {self.observation_unit}.",
        ]
        if self.positive is not None:
            lines.append(f"Positive control: {self.positive['condition']}.")
        if self.fingerprint_tolerance is not None:
            lines.append(f"Fingerprint tolerance: {percent(self.fingerprint_tolerance)}.")
        return lines


def save_answers(folder: str | Path, answers: Answers) -> Path:
    """Write ``answers`` to ``folder/ANSWERS_FILE`` (the folder is created) and return the path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    answers.saved = datetime.now(UTC).isoformat(timespec="seconds")
    path = folder / ANSWERS_FILE
    path.write_text(json.dumps(asdict(answers), indent=2, default=str) + "\n")
    return path


def load_answers(folder: str | Path) -> Answers | None:
    """The answers saved in ``folder``, or ``None`` when there are none.

    Raises ``ValueError`` for a file that is not a saved-answers file of this version.
    """
    path = Path(folder) / ANSWERS_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not a saved-answers file ({error})") from error
    if not isinstance(data, dict) or data.get("version") != ANSWERS_VERSION:
        raise ValueError(f"{path} was not written by this version of the guided session")
    return Answers(**data)


@dataclass(frozen=True)
class GuidedResult:
    """What a guided session produced: the ``scan``, the ``answers`` (``None`` when the user
    stopped), the ``setup`` and the ``report`` (``None`` when not run), and the ``folder``."""

    folder: Path
    scan: DesignScan | None
    answers: Answers | None
    setup: Setup | None
    report: Report | None
    stopped: str | None = None


def _read(source: Any, name: str) -> tuple[pd.DataFrame | None, str | None]:
    if source is None:
        return None, None
    if isinstance(source, pd.DataFrame):
        return source, None
    path = Path(str(source)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"the {name} {str(path)!r} does not exist")
    return pd.read_csv(path), str(path)


def _load(
    ask: q.Ask,
    say: q.Say,
    objects: pd.DataFrame | str | Path | None,
    fovs: pd.DataFrame | str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    if objects is None:
        path = q.text(ask, "Path to the observation table (one row per cell, track, ...):")
        objects = path
        if fovs is None:
            fovs = q.text(ask, "Path to the FOV table (Enter if there is none):") or None
    objects, objects_path = _read(objects, "observation table")
    fovs, fovs_path = _read(fovs, "FOV table")
    say(f"Loaded {len(objects)} rows and {len(objects.columns)} columns.")
    dataset = {
        "objects": objects_path,
        "fovs": fovs_path,
        "rows": len(objects),
        "checksum": table_checksum(objects),
    }
    return objects, fovs, dataset


def _saved(
    ask: q.Ask, say: q.Say, folder: Path, dataset: dict[str, Any]
) -> tuple[Answers | None, str | None]:
    """Saved answers to continue with (or ``None`` after starting over), and the reason to
    stop when the user keeps a file that cannot be used."""
    path = folder / ANSWERS_FILE
    try:
        saved = load_answers(folder)
    except (ValueError, TypeError) as error:
        say(f"{path} cannot be read ({error}).")
        saved, usable = None, False
    else:
        if saved is None:
            return None, None
        usable = saved.dataset.get("checksum") == dataset["checksum"]
    if saved is not None and usable:
        say(f"Answers saved on {saved.saved or 'an earlier session'} were found for this table:")
        say("\n".join(f"  {line}" for line in saved.lines()))
        choice = q.choose(
            ask, say, "Continue with them, or start over?", ["continue", "start over"], "continue"
        )
        if choice == "continue":
            return saved, None
    elif saved is not None:
        say(
            f"{path} holds answers for another table ({saved.dataset.get('rows')} rows, saved "
            f"{saved.saved or 'earlier'}); they cannot be reused for this one."
        )
    unusable = saved is None or not usable
    if unusable and not q.yes_no(ask, say, "Delete them and start over?", True):
        return None, f"{path} was kept; choose another folder for this table"
    path.unlink()
    say("The saved answers were deleted; starting over.")
    return None, None


def _questions(ask: q.Ask, say: q.Say, scan: DesignScan, dataset: dict[str, Any]) -> Answers | None:
    if not scan.data_check.ok:
        say("The tables cannot be analysed until the errors above are fixed.")
        return None
    if not q.yes_no(ask, say, "Here is what we found. Did your dataset load correctly?"):
        say("Fix the tables (the scan above says what it read) and start again.")
        return None
    control = q.ask_control(ask, say, scan)
    positive = q.ask_positive(ask, say, scan, control)
    aim = q.ask_aim(ask, say)
    features = q.ask_features(ask, say, scan.settings["feature_cols"])
    readout = q.ask_readout(ask, say, features)
    unit = scan.settings["observation_unit"]
    if unit is None:
        hint = next((x.suggested for x in scan.questions if x.key == "observation_unit"), None)
        unit = q.ask_unit(ask, say, hint)
    tolerance = q.ask_tolerance(ask, say, features) if aim != "population" else None
    return Answers(
        loaded_correctly=True,
        control=control,
        aim=aim,
        features=list(features),
        readout=readout,
        observation_unit=unit,
        fingerprint_tolerance=tolerance,
        positive=positive,
        dataset=dataset,
    )


def guide(
    objects: pd.DataFrame | str | Path | None = None,
    fovs: pd.DataFrame | str | Path | None = None,
    *,
    folder: str | Path | None = None,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
    run_report: bool = True,
    **setup_fields: Any,
) -> GuidedResult:
    """Run the guided session (see the module) and return what it produced.

    ``objects`` and ``fovs`` are DataFrames or CSV paths (asked for when ``objects`` is
    ``None``); ``folder`` is the results folder (asked for when ``None``). ``setup_fields``
    go to the ``Setup`` (``n_jobs=4``, ``normalize="control_mean"``). With
    ``run_report=False`` the session stops after saving the answers.
    """
    objects, fovs, dataset = _load(ask, say, objects, fovs)
    if folder is None:
        folder = q.text(ask, "Folder for the results:", DEFAULT_FOLDER)
    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    answers, stop = _saved(ask, say, folder, dataset)
    if stop:
        say(f"Stopped: {stop}.")
        return GuidedResult(folder, None, None, None, None, stop)
    observation_unit = answers.observation_unit if answers else None
    scan = scan_design(objects, fovs, observation_unit=observation_unit)
    (folder / "scan.txt").write_text(scan.summary() + "\n")
    if answers is None:
        say(scan.summary(questions=False))
        answers = _questions(ask, say, scan, dataset)
        if answers is None:
            return GuidedResult(folder, scan, None, None, None, "the dataset did not load")
        path = save_answers(folder, answers)
        say(f"Answers saved to {path}; a later session in this folder can continue from them.")
    setup = answers.setup(scan, **setup_fields)
    if not run_report:
        return GuidedResult(folder, scan, answers, setup, None)
    say("Running the report...")
    result = report(scan.objects, scan.fovs, setup, integrity=True)
    (folder / "report_short.txt").write_text(result.summary() + "\n")
    (folder / "report_full.txt").write_text(result.summary(detail="full") + "\n")
    say(result.summary())
    page = html_report(
        result, scan.objects, scan.fovs, folder, attach=[folder / "scan.txt", folder / ANSWERS_FILE]
    )
    archive = zip_results(folder)
    say(
        f"Saved in {folder}: {page.name}, one file to open with a double click (every figure "
        "and file is inside it, to share as it is); scan.txt, report_short.txt, "
        f"report_full.txt, {ANSWERS_FILE}, the figures in figures/ as PNG and PDF; and all of "
        f"it in {archive.name}, to archive or send as one file."
    )
    return GuidedResult(folder, scan, answers, setup, result)


__all__ = [
    "ANSWERS_FILE",
    "Answers",
    "GuidedResult",
    "guide",
    "load_answers",
    "save_answers",
    "table_checksum",
]
