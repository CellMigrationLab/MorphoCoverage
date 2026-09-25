"""Test-session setup: one linear-algebra thread per test process.

The suite runs one test process per CPU (pytest-xdist, see pyproject.toml).
Test matrices are small, and a multithreaded BLAS in every process
oversubscribes the CPUs: on 4 CPUs the suite took 179 s with BLAS threads and
114 s with one thread each (288 s serially). This file is loaded before the
test processes start and before numpy is imported, so they inherit the
setting. A thread count set in the environment is kept.

One thread here also makes the serial reference of ``tests/test_parallel.py``
bit for bit comparable with the single-threaded workers; with a multithreaded
BLAS the two can differ in the last digit (see ``morphocoverage._parallel``).
"""

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")
