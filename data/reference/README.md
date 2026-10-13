# The reference check

Runs every analysis the repository's data hold, on a commit or on the working tree, and compares
two such runs file for file. Two runs of the same code agree exactly once the run's stamps are
removed, so any other difference is a change of behaviour. Use it for any change that must not
change behaviour (a refactor, a clean-up, a move) and in audits, and to list every output a
change does alter (roadmap ENG-5; first used for PR #405).

```bash
# the reference, from main, then the change, from the working tree
python data/reference/reference.py run ../reference-main --commit main --parallel 2
python data/reference/reference.py run ../reference-change --working-tree --parallel 2
python data/reference/reference.py compare ../reference-main ../reference-change
```

`compare` prints each file that differs, or is found in one run only, then the count, and exits
with 1 when there is any difference.

## What runs

`run` exports the commit with `git archive` (or copies the working tree's tracked and untracked,
not ignored, files) to a temporary folder, and runs each job there in its own subprocess, with
that tree's package (`PYTHONPATH=<tree>/src`, checked), one BLAS thread and `PYTHONHASHSEED=0`.
The outputs go to `OUT`, which must be outside the repository: `OUT/<kind>__<name>/` per job,
its log beside it, and `reference_run.log` with each job's exit code and time.

The committed tier, [`jobs.txt`](jobs.txt), one job per line:

| job | what it writes |
|---|---|
| `idr all` | every script of `data/crick_idr/run_all.py`, in its order, stopping at the first failure; their `results/` folder, from the export, never the repository's |
| `lab <dataset>` | the 3 datasets of `data/lab`: the report, short and full, its warnings and its page |
| `synthetic synthetic` | the same for `data/synthetic`, with stopping rules and a rare target |
| `morphobricks fiberfeatures` | the same for `data/morphobricks/fiberfeatures_patches.csv` |
| `scenario <name>` | the same for each of the 50 report scenarios of `data/scenarios`, the user mistakes failing as they should |

A failure is an output like any other: `record.json` holds the stage, the error's type and
message. `tests/test_reference.py` checks that `jobs.txt` lists every scenario and lab dataset.

Subsets: `--only` and `--skip` take a kind (`--skip idr`) or a `kind/name` pattern (`--only
'scenario/e*'`), and may be repeated; `--jobs FILE` reads another job list. `--parallel N` runs
N jobs at a time; the `idr` job comes first in `jobs.txt` because it takes longest, and its
scripts use every CPU in places (`n_jobs=-1`). Prefix the run with `nice` on a shared machine.

## What is compared

Every file of both runs, byte for byte, after removing these stamps and nothing else
(`reference_compare.py`):

- the page's `Generated <date> UTC` line;
- each PDF's `/CreationDate`, also in the PDFs the page carries;
- the `run time:` and `morphocoverage:` lines of a `run.txt`;
- the replay seconds `benchmark_summary.txt` prints (`replay 870 s)`).

Logs (`*.log`) and `traceback.txt` (its paths and line numbers move with any refactor) are not
compared.

## The owner's tier

Data the repository does not hold run as extra jobs from a Python file given with `--owner-jobs
FILE`, or in the variable `MORPHOCOVERAGE_OWNER_JOBS`. The file stays outside the repository; it
defines `JOBS`, a dictionary from a job name (one word) to a function without arguments returning the job's
fields:

```python
from pathlib import Path

import pandas as pd
from morphocoverage.workflows import Setup, default_novelty

DATA = Path("/path/to/the/tables")  # outside the repository


def explainable():
    objects = pd.read_csv(DATA / "objects.csv")
    fovs = pd.read_csv(DATA / "fovs.csv")
    features = ("area", "speed")
    setup = Setup(
        readout="area",
        features=features,
        control={"condition": "control"},
        novelty=default_novelty(features),
        random_state=0,
        n_jobs=1,
    )
    return {
        "objects": objects,
        "fovs": fovs,
        "setup": setup,
        "title": "MorphoCoverage report: explainable features",
        # optional:
        "feature_tables": {"embedding": pd.read_csv(DATA / "embedding.csv")},
        "extra_files": {"contrasts_used.csv": setup.contrast_table},  # written and attached
        "scan": {"fov_col": "image_file"},  # scan_design's arguments: scan.txt is written
    }


JOBS = {"explainable": explainable}
```

Each entry becomes the job `owner <name>`, run and compared like the others (`--skip owner`
leaves them out). The file is imported in the tree under test, so it uses that commit's package.

## Notes

A `TMPDIR` path longer than about 70 characters breaks the worker pools of the `idr` scripts
(`OSError: AF_UNIX path too long`); keep the default or a short one.
