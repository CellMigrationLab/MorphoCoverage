# The pre-registered benchmark (P0-P5)

The code behind [the validation protocol](../../../docs/common/validation.md): it chooses,
by a fair benchmark fixed in advance, the rules MorphoCoverage ships. Nothing here is part
of the package, and nothing here is wired into the reports. The protocol holds the
selection rule, the designs and the modules; this page says how the code is laid out and
run. No result is stored here until the owner's confirmatory runs.

## Layout

| File | What it holds |
|---|---|
| `design.py` | the common skeleton: repeats, conditions, wells (recorded or not), fields, cells, missingness, sessions, acquisition order |
| `scalar.py` | scalar features, with the truth of every estimand by quadrature (P1, P2, P4) |
| `embedding.py` | embeddings: spectrum, global basis, mean or covariance shifts, batches; `rotate`; `semi_synthetic` (P3) |
| `populations.py` | groups of cells: none, clear, weak, rare, split, batch-only (P6, P7, later) |
| `sequential.py` | one sample acquired field by field, the looks (P4) |
| `runner.py` | seeds, the same dataset for every method, chunks, resume, workers, the command line |
| `summary.py` | rates and Monte Carlo SEs, the selection rule, `summary.csv` and `summary.md` |
| `p1_repeat_inference.py` ... `p5_multiple_testing.py` | one module each: its points (core, stress, screening), methods and rules, as data |
| `candidates/` | methods the package does not hold yet: the fixed-block engine and estimability, repeat means, high-dimensional tests, centrings, sequential rules, multiplicity |
| `real_parameters.py` | summary statistics of real tables for the provenance table; `results/real_parameters.csv` |
| `colab.py`, `benchmark_colab.ipynb` | what the owner runs the benchmark with |

P0 is a test file, `tests/test_validation_p0.py`; `tests/test_validation_smoke.py` runs
everything here at a tiny size.

## Running

From the repository root, with the extra installed (`pip install -e ".[validation]"`:
statsmodels and harmonypy, used only here; harmonypy is GPL-3.0):

    PYTHONPATH=src:data/simulations python -m decisions.p1_repeat_inference --smoke
    PYTHONPATH=src:data/simulations python -m decisions.p1_repeat_inference \
        --out results --size full --workers 8 --chunk 100
    PYTHONPATH=src:data/simulations python -m decisions.p1_repeat_inference summarise --out results

- `--size`: `smoke` (one replicate of a few points: a test of the code), `pilot` (1% of
  the replicates: timing, memory and convergence only), `full` (the pre-registered counts
  and the confirmatory seed, `runner.CONFIRMATORY_SEED`). `--smoke` is `--size smoke`.
- `--workers`: processes (default every CPU); `--chunk`: replicates per file; `--kinds`:
  the layers to run (`core,stress,screening`); `--seed`: a seed other than the default.
- Output: `<out>/<module>/chunks/*.csv` (one row per replicate and method),
  `points.csv`, then `summary.csv`, `summary.md` and `selection_<question>.csv`. A run
  resumes from the chunks on disk: stopped at any time, it loses at most the chunks in
  progress.

On Google Colab, open `benchmark_colab.ipynb`: it gets the code (a GitHub token in Colab's
secrets, or an uploaded zip), mounts Google Drive for the output, runs the chosen modules
at the chosen size with a progress line and the time left, summarises and zips the
results to send back. Its cells only call `colab.py`.

The real-data summaries:

    PYTHONPATH=src:data/simulations python -m decisions.real_parameters
    PYTHONPATH=src:data/simulations python -m decisions.real_parameters \
        --table ECM objects.csv fovs.csv --control VEC_SiNC --embedding-prefix dino_ \
        --out ecm_summary.csv

The first writes `results/real_parameters.csv` from the lab's datasets and idr0028
(idr0139 is held out and not read); the second reads a table outside the repository and
writes summaries only.

## Adding a method or a design

A method is a function `(data, context) -> dict of outputs` declared with
`runner.Method` in its module's `METHODS` (its role, question and simplicity rank); a
design is a `runner.Point` (its layer, whether the calibration gate reads it, its
alternative class). Changing either after the protocol's commit is a deviation, recorded in
the protocol with its date and reason.
