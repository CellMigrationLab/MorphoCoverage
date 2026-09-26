"""Run independent units of work in worker processes (private).

A unit is anything whose result does not depend on another unit: one analysis
group, one strategy within a group, one replayed acquisition. Every random
choice is either seeded from the unit's identity or drawn by the caller before
the work is handed out, so a unit's result never depends on which process
computed it or in which order; results are returned in input order.
``n_jobs=1`` runs in this process, which is the reference.

What "identical for every ``n_jobs``" means: the same computations on the same
data in the same order. The one difference left is the BLAS thread count.
Workers run BLAS with one thread; this process may run it with several, and a
multithreaded BLAS splits long sums differently, which can change the last
digit of a floating-point result (about 1e-16 relative). With BLAS limited to
one thread in this process too (``OMP_NUM_THREADS=1`` and
``OPENBLAS_NUM_THREADS=1`` before numpy is imported, as the test suite does),
results are bit for bit identical for every ``n_jobs``.

Workers start from a clean process ("forkserver", or "spawn" where that is
unavailable), never as a copy of the caller, so strategies, estimators and
criteria must be picklable from an importable module. A class defined only in
a notebook object or a script's ``__main__`` cannot be rebuilt in a worker; run
it with ``n_jobs=1``, or move it to a module. A script that uses ``n_jobs``
must guard its entry point with ``if __name__ == "__main__":``.

Each worker limits its linear-algebra library to one thread. The matrices of
one unit are small, and a multithreaded BLAS in every worker oversubscribes
the CPUs: on idr0028 the frequency control took 6.8 minutes with BLAS threads
in four workers and 1.8 minutes with one thread each, with identical tables.
A thread count the caller has set in the environment is kept. The worker
server reads these settings once, when the first pool of this process starts
it; later changes to the environment do not reach it, and a server that other
code started first keeps that code's settings.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.forkserver
import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from typing import Any

from ._checks import is_integer

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
_BROKEN = (
    "a worker process stopped before returning a result. The usual cause is a "
    "strategy, estimator or criterion whose class is defined in a notebook cell "
    "or a script's __main__, which a worker cannot import, or a script without an "
    "'if __name__ == \"__main__\":' guard. Move the class to a module or use n_jobs=1."
)


def check_n_jobs(n_jobs: int) -> int:
    """The number of worker processes: a positive integer, or -1 for every CPU.

    Python and numpy integers are accepted alike; booleans are not.
    """
    if not is_integer(n_jobs) or (n_jobs < 1 and n_jobs != -1):
        raise ValueError("n_jobs must be a positive integer, or -1 for every CPU")
    return (os.cpu_count() or 1) if n_jobs == -1 else int(n_jobs)


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
    # The worker server reads the environment when it starts, which is when
    # work is first submitted, so the variables stay set while the pool is
    # open. This process has already loaded its libraries and keeps its own
    # thread count.
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
    This package is added to the server's preload list, keeping any modules
    other code has put there.
    """
    if "forkserver" not in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("spawn")
    context = multiprocessing.get_context("forkserver")
    server = getattr(multiprocessing.forkserver, "_forkserver", None)
    preload = list(getattr(server, "_preload_modules", None) or [])
    if "morphocoverage" not in preload:
        context.set_forkserver_preload([*preload, "morphocoverage"])
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
    try:
        if pool is not None:
            return list(pool.map(_call, tasks, chunksize=chunksize))
        with worker_pool(workers) as own:
            return list(own.map(_call, tasks, chunksize=chunksize))
    except BrokenProcessPool as error:
        raise RuntimeError(_BROKEN) from error


def _call(task: tuple[Callable[..., Any], tuple]) -> Any:
    function, job = task
    return function(*job)
