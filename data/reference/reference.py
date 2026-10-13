"""The reference check: run every committed analysis, then compare two runs file for file.

Run the jobs on a commit (exported with ``git archive``) or on the working tree (a copy of
its tracked and untracked, not ignored, files), each job in its own subprocess, the outputs
outside the repository::

    python data/reference/reference.py run OUT [--commit REV | --working-tree]
        [--jobs FILE] [--only SELECTOR ...] [--skip SELECTOR ...]
        [--owner-jobs FILE] [--parallel N]
    python data/reference/reference.py compare BEFORE AFTER

A job is a line ``<kind> <name>`` of ``jobs.txt``; it writes ``OUT/<kind>__<name>/`` and its
log ``OUT/<kind>__<name>.log``. A selector is a kind (``idr``) or a ``kind/name`` pattern
(``scenario/e0*``). ``compare`` removes the run's stamps only (``reference_compare.py``) and
exits with 1 when any file differs. See README.md.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
JOB_SCRIPT = HERE / "reference_job.py"
OWNER_JOBS_VARIABLE = "MORPHOCOVERAGE_OWNER_JOBS"
sys.path.insert(0, str(HERE))
from reference_compare import compare


def _git(*arguments: str) -> bytes:
    """The output of a git command in the repository."""
    return subprocess.run(
        ["git", "-C", str(REPOSITORY), *arguments], check=True, capture_output=True
    ).stdout


def export_commit(commit: str, tree: Path) -> None:
    """Write the files of ``commit`` into ``tree``, as ``git archive`` gives them."""
    tree.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-x", "-C", str(tree)], input=_git("archive", commit), check=True)


def export_working_tree(tree: Path) -> None:
    """Copy the working tree's tracked and untracked (not ignored) files into ``tree``."""
    listed = _git("ls-files", "-z", "--cached", "--others", "--exclude-standard")
    for name in filter(None, listed.decode().split("\0")):
        source = REPOSITORY / name
        if source.is_file():  # a tracked file deleted in the working tree is not copied
            target = tree / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def read_jobs(path: Path) -> list[str]:
    """The jobs of a job list: ``<kind> <name>`` lines, ``#`` comments and blanks skipped."""
    jobs = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            if len(line.split()) != 2:
                raise ValueError(f"{path}: {line!r} is not '<kind> <name>'")
            jobs.append(" ".join(line.split()))
    return jobs


def owner_job_lines(jobs_file: Path, tree: Path) -> list[str]:
    """One ``owner <name>`` line per entry of the owner's ``JOBS``, read in the tree."""
    command = [
        sys.executable,
        "-c",
        (
            "import sys, pathlib; sys.path.insert(0, sys.argv[1]); import reference_job; "
            "print('\\n'.join(reference_job.owner_jobs(pathlib.Path(sys.argv[2]))))"
        ),
        str(HERE),
        str(jobs_file),
    ]
    done = subprocess.run(
        command, cwd=tree, env=job_environment(tree), capture_output=True, text=True, check=False
    )
    if done.returncode:
        raise RuntimeError(f"the owner's jobs file {jobs_file} cannot be read:\n{done.stderr}")
    names = done.stdout.splitlines()
    for name in names:
        if name.split() != [name] or "/" in name:
            raise ValueError(f"{jobs_file}: the job name {name!r} is not one word")
    return [f"owner {name}" for name in names]


def selected(job: str, selectors: list[str]) -> bool:
    """Whether a selector names the job's kind or matches ``kind/name``."""
    kind, name = job.split()
    return any(item == kind or fnmatch.fnmatchcase(f"{kind}/{name}", item) for item in selectors)


def job_environment(tree: Path) -> dict[str, str]:
    """The jobs' environment: the tree's package, no display, one BLAS thread, fixed hashing."""
    return {
        **os.environ,
        "PYTHONPATH": str(tree / "src"),
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    }


def _idr_scripts(tree: Path) -> list[str]:
    """``data/crick_idr/run_all.py``'s scripts, in its order, as the tree defines them."""
    command = [
        sys.executable,
        "-c",
        (
            "import sys; sys.path.insert(0, 'data/crick_idr'); import run_all; "
            "print(' '.join(run_all.SCRIPTS))"
        ),
    ]
    done = subprocess.run(
        command, cwd=tree, env=job_environment(tree), check=True, capture_output=True, text=True
    )
    return done.stdout.split()


def run_idr(tree: Path, target: Path, log_file: Path) -> int:
    """Every ``data/crick_idr`` script in order, stopping at the first failure, into a copy.

    The scripts write to the tree's ``data/crick_idr/results``, so the tree must be an export,
    never the repository; that folder is emptied first and copied to ``target`` after.
    """
    if (tree / ".git").exists():
        raise ValueError(f"{tree} is a git checkout: the idr job runs on an export only")
    results = tree / "data" / "crick_idr" / "results"
    shutil.rmtree(results, ignore_errors=True)  # fallback: a fresh export may have none
    code = 0
    with open(log_file, "w") as log:
        for script in _idr_scripts(tree):
            code = subprocess.run(
                [sys.executable, f"data/crick_idr/{script}"],
                cwd=tree,
                env=job_environment(tree),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,  # the exit code is the job's result
            ).returncode
            if code:
                break
    if results.is_dir():
        shutil.copytree(results, target, dirs_exist_ok=True)
    shutil.rmtree(target / "logs", ignore_errors=True)  # fallback: the logs hold run times
    return code


def run_job(job: str, tree: Path, out: Path, owner_jobs: Path | None) -> tuple[str, int]:
    """One job in a subprocess, into ``out/<kind>__<name>``: its line for the log, exit code."""
    kind, name = job.split()
    folder = f"{kind}__{name}"
    target = out / folder
    shutil.rmtree(target, ignore_errors=True)  # fallback: a first run has no earlier output
    target.mkdir(parents=True)
    started = time.perf_counter()
    if kind == "idr":
        code = run_idr(tree, target, out / f"{folder}.log")
    else:
        command = [sys.executable, str(JOB_SCRIPT), kind, name, str(target)]
        command += [str(owner_jobs)] if owner_jobs is not None else []
        with open(out / f"{folder}.log", "w") as log:
            code = subprocess.run(
                command,
                cwd=tree,
                env=job_environment(tree),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,  # the exit code is the job's result
            ).returncode
    return f"{kind}/{name}: exit {code}, {time.perf_counter() - started:.0f} s", code


def run_jobs(
    tree: Path, out: Path, jobs: list[str], parallel: int, owner_jobs: Path | None = None
) -> int:
    """Run the jobs on ``tree``, ``parallel`` at a time; the number that exited non-zero."""
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    lines, failed = [], 0
    with ThreadPoolExecutor(parallel) as pool:
        for line, code in pool.map(lambda job: run_job(job, tree, out, owner_jobs), jobs):
            print(line, flush=True)
            lines.append(line)
            failed += code != 0
    summary = (
        f"{len(jobs)} jobs, {failed} exited non-zero, {time.perf_counter() - started:.0f} s "
        f"wall clock, --parallel {parallel}, {os.cpu_count()} CPUs"
    )
    print(summary, flush=True)
    (out / "reference_run.log").write_text("\n".join([*lines, summary]) + "\n")
    return failed


def _outside_repository(path: Path) -> Path:
    """``path`` made absolute, refused inside the repository."""
    path = path.resolve()
    if path.is_relative_to(REPOSITORY):
        raise ValueError(f"{path} is inside the repository; write the outputs elsewhere")
    return path


def _jobs(arguments: argparse.Namespace, tree: Path, owner_jobs: Path | None) -> list[str]:
    """The job list, the owner's jobs appended, then ``--only`` and ``--skip`` applied."""
    jobs = read_jobs(arguments.jobs)
    if owner_jobs is not None:
        jobs += owner_job_lines(owner_jobs, tree)
    if arguments.only:
        jobs = [job for job in jobs if selected(job, arguments.only)]
    return [job for job in jobs if not selected(job, arguments.skip)]


def command_run(arguments: argparse.Namespace) -> int:
    """Export the commit (or the working tree) to a temporary folder and run the jobs."""
    out = _outside_repository(arguments.out)
    owner_file = arguments.owner_jobs or os.environ.get(OWNER_JOBS_VARIABLE)
    owner_jobs = Path(owner_file).resolve() if owner_file else None
    with tempfile.TemporaryDirectory(prefix="morphocoverage-reference-") as folder:
        tree = Path(folder) / "tree"
        if arguments.working_tree:
            export_working_tree(tree)
        else:
            export_commit(arguments.commit, tree)
        jobs = _jobs(arguments, tree, owner_jobs)
        source = "working tree" if arguments.working_tree else arguments.commit
        print(f"{len(jobs)} jobs on {source}, into {out}", flush=True)
        failed = run_jobs(tree, out, jobs, arguments.parallel, owner_jobs)
    return 1 if failed else 0


def parser() -> argparse.ArgumentParser:
    """The command line of ``run`` and ``compare``."""
    main_parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = main_parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the jobs into OUT (outside the repository)")
    run.add_argument("out", type=Path)
    source = run.add_mutually_exclusive_group()
    source.add_argument("--commit", default="HEAD", help="the commit to export (HEAD)")
    source.add_argument("--working-tree", action="store_true", help="copy the working tree")
    run.add_argument("--jobs", type=Path, default=HERE / "jobs.txt", help="the job list")
    run.add_argument("--only", action="append", default=[], help="keep these jobs only")
    run.add_argument("--skip", action="append", default=[], help="leave these jobs out")
    run.add_argument(
        "--owner-jobs",
        type=Path,
        help=f"the owner's jobs file (README.md); else ${OWNER_JOBS_VARIABLE}",
    )
    run.add_argument("--parallel", type=int, default=1, help="jobs at a time (1)")
    compared = commands.add_parser("compare", help="compare two output folders")
    compared.add_argument("before", type=Path)
    compared.add_argument("after", type=Path)
    return main_parser


def main(argv: list[str] | None = None) -> int:
    """Run the command; its exit code."""
    arguments = parser().parse_args(argv)
    if arguments.command == "compare":
        return compare(arguments.before, arguments.after)
    if arguments.parallel < 1:
        raise SystemExit("--parallel must be 1 or more")
    return command_run(arguments)


if __name__ == "__main__":
    sys.exit(main())
