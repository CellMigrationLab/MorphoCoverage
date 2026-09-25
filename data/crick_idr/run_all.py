"""Run every analysis on the stored datasets, in order, and record how long each took.

Each script writes its own tables to ``results/`` (see ``README.md``); this
runner only calls them in dependency order (``report_steps_idr0028.py`` reads
the output of ``reliability_idr0028.py``) and stops at the first failure.
The benchmark runs under its locked protocol. Usage, from the repository
root::

    python data/crick_idr/run_all.py

Standard output of each script goes to ``results/logs/<script>.log``; the run
times are written to ``results/logs/run_all.txt``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "results" / "logs"
SCRIPTS = (
    "fov_vs_objects.py",
    "positive_control.py",
    "frequency_control.py",
    "reliability_idr0028.py",
    "stopping_idr0028.py",
    "report_steps_idr0028.py",
    "heldout_idr0139.py",
    "benchmark.py",
    "objects_vs_fields.py",
    "jcs_effect_fovs.py",
)


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    times = []
    for script in SCRIPTS:
        start = time.perf_counter()
        with open(LOGS / f"{Path(script).stem}.log", "w") as log:
            done = subprocess.run(
                [sys.executable, str(HERE / script)],
                cwd=HERE.parents[1],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,  # the return code is reported below
            )
        seconds = time.perf_counter() - start
        times.append(f"{script}: {seconds:.0f} s, exit {done.returncode}")
        print(times[-1], flush=True)
        if done.returncode:
            break
    (LOGS / "run_all.txt").write_text("\n".join(times) + "\n")
    if done.returncode:
        sys.exit(f"{script} failed; see {LOGS / Path(script).stem}.log")


if __name__ == "__main__":
    main()
