"""Run independent analysis groups in worker processes (private).

Every retrospective analysis seeds each group from the random state and the
group's identity, so a group's result never depends on which process computed
it or in which order; results are returned in input order. ``n_jobs=1`` runs
in this process, which is the reference.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any


def check_n_jobs(n_jobs: int) -> int:
    """The number of worker processes: a positive integer, or -1 for every CPU."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or (n_jobs < 1 and n_jobs != -1):
        raise ValueError("n_jobs must be a positive integer, or -1 for every CPU")
    return (os.cpu_count() or 1) if n_jobs == -1 else n_jobs


def map_groups(function: Callable[..., Any], jobs: Sequence[tuple], n_jobs: int) -> list[Any]:
    """``[function(*job) for job in jobs]``, in worker processes when ``n_jobs > 1``."""
    workers = min(check_n_jobs(n_jobs), len(jobs))
    if workers <= 1:
        return [function(*job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(function, *job) for job in jobs]
        return [future.result() for future in futures]
