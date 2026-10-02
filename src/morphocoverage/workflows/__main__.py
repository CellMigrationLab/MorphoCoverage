"""``python -m morphocoverage.workflows``: the guided session from a terminal (``guided.guide``).

Usage::

    python -m morphocoverage.workflows [objects.csv [fovs.csv]] [--folder results] [--n-jobs 4]

Without paths, the session asks for them; without ``--folder``, for the results folder.
"""

from __future__ import annotations

import argparse

from .guided import guide


def main(argv: list[str] | None = None) -> None:
    """Run the guided session on the tables named in ``argv`` (asked for when absent)."""
    parser = argparse.ArgumentParser(
        prog="python -m morphocoverage.workflows",
        description="Scan a dataset, answer a few questions, and get the report.",
    )
    parser.add_argument("objects", nargs="?", help="the observation table (CSV)")
    parser.add_argument("fovs", nargs="?", help="the FOV table (CSV), when there is one")
    parser.add_argument("--folder", help="the results folder (asked for when not given)")
    parser.add_argument("--n-jobs", type=int, default=1, help="processes for the report")
    arguments = parser.parse_args(argv)
    guide(arguments.objects, arguments.fovs, folder=arguments.folder, n_jobs=arguments.n_jobs)


if __name__ == "__main__":
    main()
