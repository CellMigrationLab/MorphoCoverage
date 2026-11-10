"""A guided session: load the tables, scan them, ask a few questions, save the answers, report.

``guide()`` walks a user through a dataset in this order:

1. **Load** the observation table (and the FOV table, when there is one), from a path or
   a DataFrame; with paths (asked or given), ask too for **other feature tables** for the same
   observations (a DINO embedding, owner 2026-10-05: each with a path, a name and whether it
   is explainable, owner 2026-10-06, checked against the observation table,
   ``_guided_tables``; or ``guide(feature_tables=...)``);
   and choose the **results folder**.
2. If the folder already holds saved answers (``ANSWERS_FILE``) for the same tables (the
   observation table and the FOV table, or no FOV table, by their checksums), offer to
   **continue** with them or to **delete them and start over**; answers saved for other
   tables (the other feature tables by name, checksum and tag included) are never reused, and
   answers written by another version of this session are not read (they are reported as
   such and offered for deletion).
3. **Scan** the design (``scan_design``, the other feature tables listed with it) and print
   what it found; the scan is saved as ``scan.txt``.
4. **Ask** (owner, 2026-10-05, for the order and the wording): did the dataset load
   correctly (if not, stop and fix the tables); what one row is, when not declared (the scan
   is redone with it, its words follow the unit); whether all the conditions are compared
   with the same control (``_guided_controls``: Yes, the control to log, the scan's guess
   the default; No, several controls with their conditions, parents and a master, each
   proposed to confirm), then the tests that will run, one per line with its family; an
   optional positive control (a condition known to change the phenotype, for question 1);
   whether the average of each feature differs, the whole distribution (shape, spread,
   tails; no mean target, ``tolerances=()``, owner 2026-10-06), or both (stored as
   ``population``, ``fingerprint``, ``both``); the features to
   assess (several, or all the numeric columns); the readout among them; an optional column
   that marks a phenotype (true/false or 0/1 per observation: ``Setup.rare_label``, its
   share the rare target's prevalence, seen in ``RARE_FIELDS`` fields with
   ``RARE_DETECTION``); and for the whole distribution, its tolerance (no default has a
   published meaning, so the user states it).
5. **Save** the answers to ``ANSWERS_FILE`` in the folder, ask whether to run the report now
   (the default) or stop there (as ``run_report=False``), then run the **report** with the
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
import os
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .._wording import count, duration, listing, percent
from ..metadata import CONDITION_COLUMN
from ..multivariate_outcomes import CovarianceError
from ..objectives import RareTarget
from ..retrospective.criteria import OutcomeCriterion
from . import _guided_ask as q
from ._feature_space import features_words
from ._guided_controls import ask_controls, ask_experiment, contrast_lines
from ._guided_runtime import rough_runtime
from ._guided_tables import (
    Tables,
    ask_tables,
    changed,
    entries,
    identity,
    read_table,
    read_tables,
    tag_of,
    tagged,
)
from ._html_data import EMBED_BUDGET
from .html_report import html_report, zip_results
from .retrospective import Report, report
from .scan import DesignScan, scan_design
from .setup import PositiveControl, Setup, default_novelty

ANSWERS_FILE = "morphocoverage_answers.json"
ANSWERS_VERSION = 2  # 2: the FOV table's checksum (owner, 2026-10-05)
DEFAULT_FOLDER = "morphocoverage_results"
# The reference fields seen before a positive-control field is judged (PositiveControl).
POSITIVE_FIELDS = 10
# The rare target of a phenotype column (presets, as the page's suggested code uses them).
RARE_DETECTION = 0.95
RARE_FIELDS = 3


def table_checksum(table: pd.DataFrame) -> str:
    """A short checksum of a table's columns and values, so saved answers are reused only for
    the tables they were given for."""
    digest = hashlib.sha256("\x1f".join(map(str, table.columns)).encode())
    digest.update(pd.util.hash_pandas_object(table, index=False).to_numpy().tobytes())
    return digest.hexdigest()[:16]


@dataclass
class Answers:
    """What the user answered, as saved in ``ANSWERS_FILE``.

    ``aim`` is ``"population"``, ``"fingerprint"`` or ``"both"``; ``control`` is ``{"condition":
    ...}`` (one control, or several with their conditions, then with ``control_parents`` and
    ``master_control`` as ``Setup`` takes them) or ``None``; ``fingerprint_tolerance`` is the share
    the covariance of the standardised features must stay within (``None`` for the population aim);
    ``dataset`` records the tables the answers were given for: paths, rows, the observation table's
    ``checksum`` and the FOV table's ``fovs_checksum`` (``None`` without one). ``scales``: the
    features named like an intensity or an area and the scale confirmed (``ask_scales``; plan v2,
    owner, 2026-10-06). ``experiment``: each condition's experiment when the user said sets are
    separate experiments, or one screen (``ask_experiment``; audit 10, owner, 2026-10-06).
    """

    loaded_correctly: bool
    control: dict[str, Any] | None
    aim: str
    features: list[str]
    readout: str | None
    observation_unit: str
    fingerprint_tolerance: float | None = None
    positive: dict[str, Any] | None = None
    dataset: dict[str, Any] = field(default_factory=dict)
    saved: str | None = None
    version: int = ANSWERS_VERSION
    control_parents: dict[str, Any] | None = None
    master_control: Any = None
    rare_label: str | None = None
    scales: dict[str, str] | None = None
    experiment: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.control_parents:
            # JSON keeps a (parent, treatment) pair as a list.
            self.control_parents = {
                c: tuple(p) if isinstance(p, list) else p for c, p in self.control_parents.items()
            }
        if self.control is not None:  # controls nothing could be compared with are refused
            parents, master = self.control_parents, self.master_control
            Setup(readout="_", control=self.control, control_parents=parents, master_control=master)
        if self.aim not in q.AIMS:
            raise ValueError(f"aim must be one of {tuple(q.AIMS)}, not {self.aim!r}")
        if not self.features:
            # Embedding-only answers (owner, 2026-10-06): the other tables alone, no readout.
            if self.readout is not None or self.aim not in q.TABLE_AIMS:
                raise ValueError(
                    "choose at least one feature, or, with other feature tables and no "
                    f"explainable feature, readout=None and an aim among {q.TABLE_AIMS}"
                )
            if "label" in self.aim and self.rare_label is None:
                raise ValueError(f"aim={self.aim!r} needs the phenotype column (rare_label)")
        elif self.readout not in self.features:
            raise ValueError(f"the readout {self.readout!r} must be one of the features")
        if self.aim not in ("population", "label") and self.fingerprint_tolerance is None:
            raise ValueError(f"aim={self.aim!r} needs fingerprint_tolerance")

    def fingerprint(
        self, tables: Mapping[str, Sequence[str]] | None = None
    ) -> tuple[OutcomeCriterion, ...]:
        """The fingerprint criteria the answers declare (none for the population aim): the
        explainable features', and with other feature tables (``tables``: each name and its
        columns) one per table, each with its own verdict at the same tolerance (owner,
        2026-10-06)."""
        if self.aim in ("population", "label"):
            return ()
        tolerance = float(self.fingerprint_tolerance)
        within = f"within {percent(tolerance)}"
        criteria = [
            OutcomeCriterion(
                CovarianceError(tuple(self.features), True),
                tolerance,
                "lower_is_better",
                f"the covariance of the standardised features {within}",
            )
            for _ in ([1] if self.features else [])  # none without explainable features
        ]
        for name, columns in (tables or {}).items():
            criteria.append(
                OutcomeCriterion(
                    CovarianceError(tuple(columns), True),
                    tolerance,
                    "lower_is_better",
                    f"the covariance of the standardised {name} columns {within}",
                )
            )
        return tuple(criteria)

    def setup(self, scan: DesignScan, **fields: Any) -> Setup:
        """The ``Setup`` these answers make for ``scan`` (``fields`` add or override any
        ``Setup`` field, e.g. ``n_jobs=4``)."""
        goal = q.AIMS[self.aim]
        values: dict[str, Any] = {
            "features": tuple(self.features),
            "control": self.control,
            "control_parents": self.control_parents,
            "master_control": self.master_control,
            "observation_unit": self.observation_unit,
        }
        if goal == "fingerprint":
            values["fingerprint"] = self.fingerprint(scan.settings.get("feature_tables"))
        if self.aim == "fingerprint":  # the whole distribution: no mean target (owner, 2026-10-06)
            values["tolerances"] = ()
        if (goal == "fingerprint" and self.features) or self.positive is not None:
            # A positive control is judged by the definition of new.
            values["novelty"] = default_novelty(tuple(self.features))
        if self.positive is not None:
            values["positive"] = PositiveControl(dict(self.positive), fovs_seen=POSITIVE_FIELDS)
        if self.rare_label is not None:
            values["rare"], values["rare_label"] = self.rare_target(scan), self.rare_label
        values["scales"] = self.scales or None
        values["experiment"] = self.experiment or None
        values.update(fields)
        with warnings.catch_warnings():
            # fallback: a control the user chose to leave out is an answer, not a guess to warn
            warnings.simplefilter("ignore", UserWarning)
            return scan.setup(self.readout, goal=goal, **values)

    def rare_target(self, scan: DesignScan) -> RareTarget:
        """The rare target of ``rare_label``: its share of the labelled observations as the
        prevalence, seen in ``RARE_FIELDS`` fields with ``RARE_DETECTION`` (presets the lines
        state)."""
        labels = scan.objects[self.rare_label].dropna().astype(bool)
        return RareTarget(
            detection_probability=RARE_DETECTION,
            min_fov_occurrences=RARE_FIELDS,
            phenotype=str(self.rare_label),
            object_prevalence=float(labels.mean()),
        )

    def _control_words(self) -> Any:
        """The control in words: ``none``, its value, or several with their conditions,
        parents and master."""
        control = "none" if self.control is None else self.control["condition"]
        if isinstance(control, dict):  # several controls, each with its conditions
            control = "; ".join(
                f"{c} for {', '.join(map(str, m.values() if isinstance(m, dict) else m)) or '-'}"
                for c, m in control.items()
            )
            control += "".join(
                f"; {c} under {p[0] if isinstance(p, tuple) else p}"
                for c, p in (self.control_parents or {}).items()
            )
            control += f"; master {self.master_control}" if self.master_control else ""
        return control

    def lines(self) -> list[str]:
        """The answers in sentences."""
        lines = [
            f"Control: {self._control_words()}.",
            f"Aim: {q.AIM_TEXT[self.aim]}.",
            (
                f"Features ({len(self.features)}): {', '.join(self.features)}".rstrip(".") + "."
                if self.features
                else "Features: none explainable; the other feature tables alone."
            ),
            f"Readout: {self.readout or 'none'}.",
            f"One row is a {self.observation_unit}.",
        ]
        tables = [
            f"{e['name']} ({tag_of(e)}, {e['path'] or 'a DataFrame'})"
            for e in self.dataset.get("feature_tables") or []
        ]
        if tables:
            lines.append(f"Other feature tables: {', '.join(tables)}.")
        if self.positive is not None:
            lines.append(f"Positive control: {self.positive['condition']}.")
        if self.rare_label is not None:
            lines.append(
                f"Phenotype column: {self.rare_label} (its share as the rare target, seen in "
                f"{RARE_FIELDS} fields with {percent(RARE_DETECTION)} chance)."
            )
        if self.fingerprint_tolerance is not None:
            lines.append(f"Fingerprint tolerance: {percent(self.fingerprint_tolerance)}.")
        if self.experiment:
            named = listing(sorted({str(e) for e in self.experiment.values()}), 4)
            lines.append(f"Experiments, each corrected on its own: {named}.")
        return lines + q.scale_lines(self.scales or {})


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
    return read_table(source, name)


def _load(
    ask: q.Ask,
    say: q.Say,
    objects: pd.DataFrame | str | Path | None,
    fovs: pd.DataFrame | str | Path | None,
    given: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, Tables, dict[str, Any]]:
    # Interactive (paths asked or given, as from a terminal): the other tables are asked too.
    asked = not isinstance(objects, pd.DataFrame)
    if objects is None:
        path = q.text(ask, "Path to the observation table (one row per cell, track, ...):")
        objects = path
        if fovs is None:
            fovs = q.text(ask, "Path to the FOV table (Enter if there is none):") or None
    objects, objects_path = _read(objects, "observation table")
    fovs, fovs_path = _read(fovs, "FOV table")
    say(f"Loaded {len(objects)} rows and {len(objects.columns)} columns.")
    # Other feature tables: asked for with the paths, else as given (owner, 2026-10-05).
    tables = ask_tables(ask, say, objects, fovs) if asked and given is None else read_tables(given)
    dataset = {
        "objects": objects_path,
        "fovs": fovs_path,
        "rows": len(objects),
        "checksum": table_checksum(objects),
        "fovs_checksum": None if fovs is None else table_checksum(fovs),
        "feature_tables": entries(tables, objects),
    }
    return objects, fovs, tables, dataset


# The dataset entries that must match for saved answers to be reused (paths are provenance).
_IDENTITY = ("checksum", "fovs_checksum")


def _saved(
    ask: q.Ask, say: q.Say, folder: Path, dataset: dict[str, Any]
) -> tuple[Answers | None, str | None]:
    """Saved answers to continue with (or ``None`` after starting over), and the reason to
    stop when the user keeps a file that cannot be used."""
    path = folder / ANSWERS_FILE
    try:
        saved = load_answers(folder)
    # fallback: said to the user; the questions are asked again
    except (ValueError, TypeError) as error:
        say(f"{path} cannot be read ({error}).")
        saved, usable = None, False
    else:
        if saved is None:
            return None, None
        usable = all(saved.dataset.get(key, "missing") == dataset[key] for key in _IDENTITY)
        usable = usable and identity(saved.dataset) == identity(dataset)
    if saved is not None and usable:
        say(f"Answers saved on {saved.saved or 'an earlier session'} were found for this table:")
        say("\n".join(f"  {line}" for line in saved.lines()))
        choice = q.choose(
            ask, say, "Continue with them, or start over?", ["continue", "start over"], "continue"
        )
        if choice == "continue":
            return saved, None
    elif saved is not None:
        differs = changed(saved.dataset, dataset) or ["the tables"]
        say(
            f"{path} holds answers for another table ({saved.dataset.get('rows')} rows, saved "
            f"{saved.saved or 'earlier'}); what differs: {listing(differs, 6)}. They cannot be "
            "reused for this one."
        )
    unusable = saved is None or not usable
    if unusable and not q.yes_no(ask, say, "Delete them and start over?", False):
        return None, f"{path} was kept; choose another folder for this table"
    path.unlink()
    say("The saved answers were deleted; starting over.")
    return None, None


def _questions(
    ask: q.Ask,
    say: q.Say,
    scan: DesignScan,
    dataset: dict[str, Any],
    rescan: Callable[[str], DesignScan],
) -> tuple[Answers | None, DesignScan]:
    if not scan.data_check.ok:
        say("The tables cannot be analysed until the errors above are fixed.")
        return None, scan
    if not q.yes_no(ask, say, "Here is what we found. Did your dataset load correctly?"):
        say("Fix the tables (the scan above says what it read) and start again.")
        return None, scan
    # What one row is, first: the scan's words follow it (owner, 2026-10-05).
    unit = scan.settings["observation_unit"]
    if unit is None:
        hint = next((x.suggested for x in scan.questions if x.key == "observation_unit"), None)
        unit = q.ask_unit(ask, say, hint)
        scan = rescan(unit)
    control, parents, master = ask_controls(ask, say, scan)
    say("The tests that will run:\n" + "\n".join(contrast_lines(control, parents, master)))
    experiment = ask_experiment(ask, say, scan, control)
    positive = q.ask_positive(ask, say, scan, control)
    tables = scan.settings.get("feature_tables") or {}
    taken = [*scan.settings["group_cols"], scan.settings["fov_col"], *_table_columns(tables)]
    labels = q.label_columns(scan.objects, taken)
    numeric = [c for c in scan.settings["feature_cols"] if c not in labels]
    if not numeric and tables:
        # No explainable feature, other tables given (owner, 2026-10-06): no feature or
        # readout question, the aim among the targets that need no readout.
        features, readout, scales = [], None, {}
        rare_label = q.ask_rare_label(ask, say, labels)
        aim = q.ask_table_aim(ask, say, rare_label)
    else:
        aim = q.ask_aim(ask, say)
        features = q.ask_features(ask, say, scan.settings["feature_cols"])
        readout = q.ask_readout(ask, say, features)
        scales = q.ask_scales(ask, say, scan.objects, features)
        rare_label = q.ask_rare_label(ask, say, q.label_columns(scan.objects, [*features, *taken]))
    fingerprint = aim not in ("population", "label")
    tolerance = q.ask_tolerance(ask, say, features) if fingerprint else None
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
        control_parents=parents,
        master_control=master,
        rare_label=rare_label,
        scales=scales or None,
        experiment=experiment,
    ), scan


def _table_columns(tables: Mapping[str, Sequence[str]]) -> list[str]:
    """Every other table's columns: never a phenotype label."""
    return [c for columns in tables.values() for c in columns]


def guide(
    objects: pd.DataFrame | str | Path | None = None,
    fovs: pd.DataFrame | str | Path | None = None,
    *,
    folder: str | Path | None = None,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
    run_report: bool = True,
    feature_tables: Mapping[str, Any] | None = None,
    **setup_fields: Any,
) -> GuidedResult:
    """Run the guided session (see the module) and return what it produced.

    ``objects`` and ``fovs`` are DataFrames or CSV paths (asked for when ``objects`` is
    ``None``); ``folder`` is the results folder (asked for when ``None``). ``setup_fields``
    go to the ``Setup`` (``n_jobs=4``, ``normalize="control_mean"``). With
    ``run_report=False`` the session stops after saving the answers. ``feature_tables`` are
    other feature tables by name (DataFrames or CSV paths, non-explainable, or either with its
    tag: ``(table, "explainable")``), asked for with their tags when the tables are paths.
    """
    objects, fovs, tables, dataset = _load(ask, say, objects, fovs, feature_tables)
    others = tagged(tables) or None
    if folder is None:
        folder = q.text(ask, "Folder for the results:", DEFAULT_FOLDER)
    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    answers, stop = _saved(ask, say, folder, dataset)
    if stop:
        say(f"Stopped: {stop}.")
        return GuidedResult(folder, None, None, None, None, stop)
    observation_unit = answers.observation_unit if answers else None
    scan = scan_design(objects, fovs, observation_unit=observation_unit, feature_tables=others)
    (folder / "scan.txt").write_text(scan.summary() + "\n")
    if answers is None:
        say(scan.summary(questions=False))

        def rescan(unit: str) -> DesignScan:
            again = scan_design(objects, fovs, observation_unit=unit, feature_tables=others)
            (folder / "scan.txt").write_text(again.summary() + "\n")
            return again

        answers, scan = _questions(ask, say, scan, dataset, rescan)
        if answers is None:
            return GuidedResult(folder, scan, None, None, None, "the dataset did not load")
        path = save_answers(folder, answers)
        say(f"Answers saved to {path}; a later session in this folder can continue from them.")
    setup = answers.setup(scan, **setup_fields)
    # Run now (the default), or stop with the answers saved, as run_report=False does.
    if run_report and not q.ask_run(ask, say):
        say(f"Stopped with the answers saved; guide() in {folder} continues from them.")
        run_report = False
    if not run_report:
        return GuidedResult(folder, scan, answers, setup, None)
    result = _run_report(folder, scan, setup, say)
    return GuidedResult(folder, scan, answers, setup, result)


def _run_report(folder: Path, scan: DesignScan, setup: Setup, say: Callable[[str], None]) -> Report:
    """Run the report of the session, save its text, page and archive in ``folder``, and say
    where."""
    fields = scan.fovs
    conditions = int(fields[CONDITION_COLUMN].nunique()) if CONDITION_COLUMN in fields else 1
    cores = setup.n_jobs if setup.n_jobs > 0 else os.cpu_count() or 1
    tables = [len(c) for c in (setup.feature_tables or {}).values()]
    estimate = rough_runtime(len(setup.columns), conditions, setup.n_jobs, tables)
    say(
        f"Running the report for {features_words(setup)} and "
        f"{count(conditions, 'condition')}; on {count(cores, 'core')} this takes roughly "
        f"{duration(estimate)}..."
    )
    result = report(scan.objects, scan.fovs, setup, integrity=True)
    (folder / "report_short.txt").write_text(result.summary() + "\n")
    (folder / "report_full.txt").write_text(result.summary(detail="full") + "\n")
    say(result.summary())
    # The figures an earlier session left in figures/ are not this report's.
    started = time.time() - 1
    page = html_report(
        result, scan.objects, scan.fovs, folder, attach=[folder / "scan.txt", folder / ANSWERS_FILE]
    )
    archive = zip_results(folder, since=started)
    say(
        f"Saved in {folder}: {page.name}, one file to open with a double click (it carries "
        f"its figures and files up to {EMBED_BUDGET // 1024**2} MB, to share as it is; past "
        "that the largest open from figures/); scan.txt, report_short.txt, "
        f"report_full.txt, {ANSWERS_FILE}, the figures in figures/ as PNG and PDF; and all of "
        f"it in {archive.name}, to archive or send as one file."
    )
    return result


__all__ = [
    "ANSWERS_FILE",
    "Answers",
    "GuidedResult",
    "guide",
    "load_answers",
    "save_answers",
    "table_checksum",
]
