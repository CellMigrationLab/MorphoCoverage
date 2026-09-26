"""Run a new dataset through the four modes with a few lines of code.

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
- ``plots(objects, fovs, setup, ranking=result.ranking, folder=...)``: PlotsOfData-style
  figures of the readout, the biological repeats side by side, every object a
  point coloured by technical repeat, the median with its interval from
  resampling whole technical repeats with their fields as acquired (not
  reliable with only 2-3 technical repeats, which the title says).

Each returns its results with a ``summary()`` in sentences. Their scientific
choices are the ``Setup``'s fields and explicit arguments; the few presets they
default (``protocol_threshold``, ``alphas``) are printed in the summaries. See
``docs/workflows.md``.
"""

from .figures import plots
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
from .setup import PositiveControl, Setup, default_novelty

__all__ = [
    "Benchmark",
    "Family",
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
    "live_session",
    "optimise",
    "plots",
    "report",
    "write_fov",
]
