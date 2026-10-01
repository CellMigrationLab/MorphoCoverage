"""Run a new dataset through the four modes with a few lines of code.

Start with ``scan_design(objects, fovs)``: what the dataset is, whether it was
loaded correctly (data check, copies, balance over the biological repeats),
which columns would say more, and the questions to answer before the analysis;
``scan.setup(readout=...)`` turns the answers into a ``Setup``. ``guide()`` (or
``python -m morphocoverage.workflows``) asks those questions interactively,
saves the answers in the results folder, and runs the report.

Declare the dataset once with a ``Setup``, then:

- ``report(objects, fovs, setup)``: REPORT, what the finished acquisition can
  be trusted for (the Foundation and R1 to R7 of ``docs/questions.md``);
- ``optimise(objects, fovs, setup, fov_counts=..., reservoir_fraction=...)``:
  OPTIMISE / PLAN, what the next experiment should do;
- ``live_session(setup)`` with ``LiveFolder(folder, session).watch()`` (files
  on disk) or ``LiveTables(session).update(objects, fovs)`` (tables in
  memory): ONLINE, continue or stop while the microscope acquires, one field
  at a time;
- ``benchmark(objects, fovs, setup, reservoir_fraction=...)``: BENCHMARK,
  which rule needs the fewest fields for the same risk;
- ``plots(objects, fovs, setup, ranking=result.discovery, folder=...)``: PlotsOfData-style
  figures of the readout, the biological repeats side by side, every object a
  point coloured by technical repeat, the median with its interval from
  resampling whole technical repeats with their fields as acquired (not
  reliable with only 2-3 technical repeats, which the title says);
  ``embedding_plots(objects, fovs, setup, discovery=result.discovery)``: the PCA
  biplot and the UMAP of the declared features, coloured by condition.

Each returns its results with a ``summary()`` in sentences. Their scientific
choices are the ``Setup``'s fields and explicit arguments; the few presets they
default (``Setup.minimum_detection``, the default stopping rules, the
positive control's pairing, ``protocol_threshold``, ``alphas``) are printed in
the declaration and the summaries, with the computational settings (orders,
resamples, grids, the fields held back), and every random procedure is seeded
from ``Setup.random_state``. See ``docs/workflows.md``.
"""

from .figures import embedding_plots, plots
from .guided import Answers, GuidedResult, guide, load_answers, save_answers
from .live import LiveFolder, LiveTables, LiveUpdate, live_session, write_fov
from .retrospective import (
    Benchmark,
    Family,
    Plan,
    Report,
    benchmark,
    default_families,
    document,
    optimise,
    report,
)
from .scan import DesignScan, scan_design
from .setup import PositiveControl, Setup, default_novelty

__all__ = [
    "Answers",
    "Benchmark",
    "DesignScan",
    "Family",
    "GuidedResult",
    "LiveFolder",
    "LiveTables",
    "LiveUpdate",
    "Plan",
    "PositiveControl",
    "Report",
    "Setup",
    "benchmark",
    "default_families",
    "default_novelty",
    "document",
    "embedding_plots",
    "guide",
    "live_session",
    "load_answers",
    "optimise",
    "plots",
    "report",
    "save_answers",
    "scan_design",
    "write_fov",
]
