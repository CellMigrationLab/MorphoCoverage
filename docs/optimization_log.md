# Speed optimization log (2026-09-24 to 2026-09-25)

What was asked, everything that was tried, what was kept, how much faster it is,
and how each kept change was shown to leave every result unchanged.

## The requests

As given by the project owner:

1. "When this is done, work on code optimization. Try to make slow computation as fast as possible. Identify bottleneck and optimization"
2. "Any part of the code that we should parallelizes ?"
3. "Go ahead with items 1-3. But I also wonder if when we compare different strategies, if they should be sent to different cores when possible. Especially if multiple are required for some higher functions"
4. "Once you are done, please rewrite the prompts I gave you for speed optimization, everything you have tried and done, and the speed improvements. For each successful trial, check and guarantee that the results are identical"
5. "Please add pytest-xdist"

The same requests as one prompt:

> Profile the package and the analysis scripts, find the bottlenecks, and make the slow computations as fast as possible. Parallelize across CPUs where work is independent, including sending different strategies to different cores when a higher-level analysis compares several. Run the test suite in parallel. For every change kept, prove that the results are byte-identical to before. Report every attempt, including rejected ones, with measured speedups.

## Speed

All five idr0028 scripts were run on one 4-CPU machine, one after another, at four stages of the code:

- before any optimization (`c835b0f`);
- after round 1 (`5a463dc`, PR #28);
- after round 2 (`ce99179`, PR #29);
- after round 3 (this change).

| Script | Before | Round 1 | Round 2 | Round 3 | Overall |
|---|---|---|---|---|---|
| `frequency_control.py` | 1408 s | 758 s | 429 s | 121 s | 11.6x |
| `positive_control.py` | 362 s | 186 s | 99 s | 31 s | 11.7x |
| `reliability_idr0028.py` | 84 s | 18 s | 19 s | 19 s | 4.4x |
| `stopping_idr0028.py` | 52 s | 19 s | 17 s | 20 s | 2.6x |
| `report_steps_idr0028.py` | 36 s | 15 s | 15 s | 15 s | 2.4x |

Timings vary with machine load. The same scripts were also timed earlier, when each round was made, and those runs gave lower "before" times:

- frequency control: 696 s;
- positive control: 203 s;
- reliability: 85 s;
- stopping replay: 45 s;
- Steps 4–8: 35 s.

Round 3 adds about one second of worker start-up to each script (see below). That is why the three short scripts did not get faster in round 3.

Test suite:

| Stage | Time |
|---|---|
| Before any optimization | 378 s |
| Round 1 | 305 s |
| Round 2 | 271 s |
| Round 3, serial (`pytest -n 0`; 1974 tests, 13 more than before) | 288 s |
| Round 3, parallel (`pytest`, pytest-xdist) | 114 s |

## Identity of the results

**Tables.** At each of the four stages, every table the five scripts write was deleted and regenerated. The regenerated tables were compared byte for byte (`diff -r`) with the committed ones. All four stages reproduce all 16 tables exactly. Deleting first means a script that failed to write would show up as a missing file rather than pass on a stale one.

**Tests.** They check that every alternative code path gives the result of the path it replaces:

- each support-novelty counting method against the original per-FOV count, on random data;
- `n_jobs > 1` against `n_jobs = 1` in `tests/test_parallel.py`, which covers:
  - `run_strategies` for every packaged strategy;
  - prefix replay, full and incremental;
  - `replay_acquisitions`;
  - `run_benchmark`;
- `n_jobs > 1` against `n_jobs = 1` in the per-function tests for `sampling_reliability` and `representation_control`;
- every granularity of parallel work: whole groups, single runs or splits, and strategies apart;
- the input cache, and that validation errors are raised in the same order.

## Round 1 (PR #28): algorithms

Profiling found the time in three places: taking many subsets of whole FOVs, support-novelty neighbour counting, and building replay prefixes.

Kept:

- `evaluator.FovSubsets` looks up each observation's FOV once; a subset of FOVs is then one vectorized mask instead of a key match.
- Outcome prefixes in replay, and each newly revealed FOV in sequential replay, are slices of observations sorted by acquisition rank.
- `_supporting_fovs` picks the fastest exact counting method for each problem:
  - pairwise distances, with a per-FOV count by matrix product, for small problems or 8 or more dimensions, where trees do not prune;
  - otherwise one tree over all reference observations;
  - one tree per reference FOV when neighbourhoods are dense.
- New-state clusters of all FOVs are found in one graph.
- `n_jobs` for `sampling_reliability` and `replay_acquisitions`: one analysis group per worker process.
- Effort bootstrap: faster flattening and code renumbering.

Tried and rejected:

- One tree with the neighbours listed and counted per FOV, for every problem. In dense neighbourhoods it was about 3x slower; the calibration test went from 8.5 s to 24 s. The adaptive choice above replaced it.

Note: the counting methods can differ only for a point at exactly `epsilon` on a rounded grid, by floating-point rounding. This did not occur on idr0028.

## Round 2 (PR #29): repeated work

Kept:

- `ValidatedInput` caches its feature matrix and the FOV positions of its own FOV table. Each read returns a copy, and every estimator call gets a fresh copy with an empty cache.
- Validation checks for infinity in one array, with the same error order as before, and matches FOV keys with Python sets.
- Pair indices for clustering are cached per FOV size.
- The control scripts build each table in one frame instead of one per FOV.

Tried and rejected:

- Vectorized `fov_positions` through a pandas MultiIndex: slower; caching replaced it.
- A BLAS distance expansion (`|x|^2 + |y|^2 - 2xy`): slower once the exact recheck near `epsilon` is included, and without the recheck it resolved ties differently.

## Round 3 (this change): parallelization

**Finding.** OpenBLAS is multithreaded on this machine. On the small matrices of these analyses its threads mostly spin, using 4x the CPU time for no speedup: 0.45 s with one thread against 0.43 s with four, on a serial novelty benchmark. Worker processes that each inherit four BLAS threads oversubscribe the CPUs. The first parallel version of the frequency control therefore took 6m51s, against 6m57s serially. With one BLAS thread per worker it took 1m51s, with identical tables.

Kept:

- `morphocoverage._parallel` (private) provides `map_units` and `worker_pool`.
  - Workers start from a clean process: a `forkserver` that imports the package once, or `spawn` where that is unavailable.
  - Each worker limits BLAS, OpenMP and MKL to one thread, unless the caller has set a thread count.
  - One pool can serve several calls, so a script starts its workers once.
- `n_jobs` on `run_strategies`, `evaluate_acquisition_prefixes`, `run_benchmark` and `representation_control`. `replay_acquisitions` and `sampling_reliability` already had it.
- Strategies on separate cores, when that helps. Each unit of work sends its data to a worker once, so units are only as fine as it takes to keep every worker busy:
  - whole analysis groups, when there are at least as many groups as workers;
  - otherwise single replayed acquisitions (group, split, order), or single reliability splits;
  - and every strategy apart when even those are fewer than the workers, for example one sample compared across several strategies.
  - Decisions of strategies replayed apart are put back in the order of one joint replay.
- Deterministic parallel control scripts. Every random choice is drawn in the main process, in the same order as a serial run; only the novelty checks, which draw nothing, go to workers. `representation_control` works the same way.
- Tests run in parallel with pytest-xdist: `-n auto --dist loadfile` in `pyproject.toml`, and `pytest -n 0` runs serially. `tests/conftest.py` gives each test process one BLAS thread: 179 s with BLAS threads, 114 s with one each.

Tried and rejected (all measured):

| Tried | Result |
|---|---|
| Workers forked from the caller (inheriting its BLAS threads) | no speedup on the controls |
| A spawned pool for every call | 3–4 s slower on the short scripts; replaced by the forkserver, which is reused |
| Every strategy of every run sent apart | stopping replay 16.6 s, against 12.4 s with strategies together |
| Reliability always split per split | 18.3 s, against 17 s per group |
| Single runs even when groups suffice | stopping replay 12 s, against 11.3 s per group |
| `pytest-xdist --dist load` (tests spread individually) | 192 s, against 179 s for `loadfile` |

Remaining cost: starting the worker server takes about 1.2 s, once per process. The old fork-based workers needed no start-up, but they inherited the caller's BLAS threads. So the short scripts are 1–3 s slower than after round 2, and the heavy controls are 3–4x faster.

## Limits

- Strategies, estimators and criteria sent to workers must be picklable from an importable module. A class defined only in a notebook cell must run with `n_jobs=1`.
- The analyses themselves are unchanged. Every speedup comes from how the work is computed, never from computing less of it.
