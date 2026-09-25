"""Run independent units of work in worker processes (private).

A unit is anything whose result does not depend on another unit: one analysis
group, one strategy within a group, one replayed acquisition. Every random
choice is either seeded from the unit's identity or drawn by the caller before
the work is handed out, so a unit's result never depends on which process
computed it or in which order; results are returned in input order.
``n_jobs=1`` runs in this process, which is the reference.

Workers start from a clean process ("forkserver", or "spawn" where that is
unavailable), never as a copy of the caller, so strategies, estimators and
criteria must be picklable from an importable module (a class defined only
in a notebook cell cannot be sent to a worker; run it with ``n_jobs=1``).

Each worker limits its linear-algebra library to one thread. The matrices of
one unit are small, and a multithreaded BLAS in every worker oversubscribes
the CPUs: on idr0028 the frequency control took 6.8 minutes with BLAS threads
in four workers and 1.8 minutes with one thread each, with identical tables.
A thread count the caller has set in the environment is kept.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from typing import Any

# Units handed to a worker at once, as a share of the units per worker: small
# enough to balance units of unequal cost, large enough that sending the data
# does not dominate many cheap units.
UNITS_PER_WORKER = 4
# Thread-count variables read by the common BLAS/OpenMP builds when they load.
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def check_n_jobs(n_jobs: int) -> int:
    """The number of worker processes: a positive integer, or -1 for every CPU."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or (n_jobs < 1 and n_jobs != -1):
        raise ValueError("n_jobs must be a positive integer, or -1 for every CPU")
    return (os.cpu_count() or 1) if n_jobs == -1 else n_jobs


@contextmanager
def worker_pool(n_jobs: int) -> Iterator[Executor | None]:
    """A pool of ``n_jobs`` single-threaded worker processes, or None for ``n_jobs=1``.

    One pool can serve several ``map_units`` calls, so workers start once.
    """
    workers = check_n_jobs(n_jobs)
    if workers <= 1:
        yield None
        return
    unset = [name for name in _THREAD_VARIABLES if name not in os.environ]
    # Workers read the environment when they start, which is when work is first
    # submitted, so the variables stay set while the pool is open. This process
    # has already loaded its libraries and keeps its own thread count.
    os.environ.update(dict.fromkeys(unset, "1"))
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=_context()) as pool:
            yield pool
    finally:
        for name in unset:
            os.environ.pop(name, None)


def _context() -> multiprocessing.context.BaseContext:
    """Workers forked from a clean server that has imported this package once, else spawned.

    The server starts, with the one-thread variables set, the first time a
    pool needs it and then serves every later pool of this process, so workers
    neither inherit this process's threads nor import numpy and pandas anew.
    """
    if "forkserver" not in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("spawn")
    context = multiprocessing.get_context("forkserver")
    context.set_forkserver_preload(["morphocoverage"])
    return context


def map_units(
    function: Callable[..., Any],
    jobs: Sequence[tuple],
    n_jobs: int,
    pool: Executor | None = None,
) -> list[Any]:
    """``[function(*job) for job in jobs]``, in worker processes when ``n_jobs > 1``.

    ``pool`` (from ``worker_pool``) reuses running workers; otherwise a pool
    is opened for this call when there is more than one job.
    """
    jobs = list(jobs)
    workers = min(check_n_jobs(n_jobs), len(jobs))
    if workers <= 1:
        return [function(*job) for job in jobs]
    chunksize = max(1, len(jobs) // (workers * UNITS_PER_WORKER))
    tasks = [(function, job) for job in jobs]
    if pool is not None:
        return list(pool.map(_call, tasks, chunksize=chunksize))
    with worker_pool(workers) as own:
        return list(own.map(_call, tasks, chunksize=chunksize))


def _call(task: tuple[Callable[..., Any], tuple]) -> Any:
    function, job = task
    return function(*job)
