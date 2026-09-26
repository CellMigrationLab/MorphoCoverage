"""Run the locked benchmark (``benchmark_protocol.py``): fields saved at matched risk.

Two questions of ``docs/questions.md`` are scored with
``morphocoverage.retrospective.frontier``:

- **mean**: on idr0028 (development) and idr0139 (held out), the JCB rule and
  the JCS descriptor spread, each per object and per field, and the FOV
  bootstrap of the mean, against a fixed number of fields. Per object against
  per field is stage 1; the rest is stage 2. Each rule's setting chosen on
  idr0028 to meet a risk level is then scored on idr0139 (``transfer``).
- **rare**: on planted states in idr0028 MOCK wells, the support-novelty
  bound and the label-based rare-incidence rule against a fixed number of
  fields.

Writes ``results/benchmark_<question>_<dataset>_curve.csv`` and
``..._scores.csv``, ``results/benchmark_transfer.csv`` and the sentences in
``results/benchmark_summary.txt``.

Usage::

    python data/crick_idr/benchmark.py --lock   # once, before the first run
    python data/crick_idr/benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_protocol import PROTOCOL, digest
from datasets import FASCIN_RATIO, RESULTS, YAPTAZ_RATIO, load, registry
from frequency_control import _planted_fovs

from morphocoverage import (
    ObjectiveConfig,
    RareTarget,
    SamplingObjective,
    ScalarPrecisionStrategy,
)
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.outcomes import RareStateDetected, RelativeMeanError
from morphocoverage.retrospective import OutcomeCriterion, replay_acquisitions
from morphocoverage.retrospective.frontier import frontier, sweep, transfer
from morphocoverage.strategies import (
    DescriptorSpreadStrategy,
    FOVBootstrapStrategy,
    RareIncidenceStrategy,
    SupportNoveltyStrategy,
)
from morphocoverage.transforms import Standardize

LOCK = Path(__file__).resolve().parent / "benchmark_protocol.lock"
GROUPS = ["condition", "repeat", "technical_repeat"]
POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
READOUT = {"idr0028_screenB": YAPTAZ_RATIO, "idr0139": FASCIN_RATIO}


def _locked() -> dict[int, str]:
    """The locked digest of every protocol version, one ``version digest`` line each."""
    if not LOCK.exists():
        return {}
    return {int(v): d for v, d in (line.split() for line in LOCK.read_text().splitlines())}


def check_lock(write: bool) -> None:
    """Refuse to run unless this protocol version matches the digest locked before its runs."""
    current, version = digest(), PROTOCOL["version"]
    locked = _locked()
    if write:
        if version in locked:
            raise SystemExit(f"version {version} is locked; a changed protocol needs a new version")
        with LOCK.open("a") as lock:
            lock.write(f"{version} {current}\n")
        print(f"locked protocol version {version}: {current}")
        return
    if version not in locked:
        raise SystemExit(f"version {version} is not locked; run with --lock before its first run")
    if locked[version] != current:
        raise SystemExit("the protocol changed after it was locked; make a new version")


def _orders(settings: dict) -> list[AcquisitionOrder]:
    recorded = [AcquisitionOrder.recorded()] if settings["recorded_order"] else []
    return recorded + [AcquisitionOrder.permuted(s) for s in range(settings["permuted_orders"])]


def _strategy(spec: dict, readout: str, question: dict):
    """One strategy of a family, at its first threshold; the sweep re-decides the rest."""
    first = spec["thresholds"][0]
    kind = spec["kind"]
    if kind == "scalar_precision":
        return ScalarPrecisionStrategy(readout, first, spec["resampling_unit"], "upper_95")
    if kind == "descriptor_spread":
        n = question["n_resamples"]
        return DescriptorSpreadStrategy(readout, first, spec["resampling_unit"], n, 0)
    if kind == "fov_bootstrap_centroid":
        return FOVBootstrapStrategy("centroid", first, None, question["n_resamples"], 0, 2)
    if kind == "support_novelty":
        return SupportNoveltyStrategy(
            np.sqrt(30), first, 1, Standardize(), 2, 3, "lofo_novel_fov_upper"
        )
    if kind == "rare_incidence":
        return RareIncidenceStrategy("planted", 1.0, 1.0, first)
    raise ValueError(f"unknown family kind {kind!r}")


def _families(replay, strategies, specs) -> list[pd.DataFrame]:
    return [
        sweep(
            replay,
            config_id=strategies[name].config_id,
            quantity=spec["quantity"],
            thresholds=spec["thresholds"],
            name=name,
        )
        for name, spec in specs.items()
        if name in strategies
    ]


def _write(result, question: str, dataset: str, criterion_index: int) -> None:
    stem = f"benchmark_{question}_{dataset}_{criterion_index}"
    result.curve.to_csv(RESULTS / f"{stem}_curve.csv", index=False, float_format="%.4f")
    result.scores.to_csv(RESULTS / f"{stem}_scores.csv", index=False, float_format="%.4f")


def run_mean() -> tuple[list[str], list[pd.DataFrame]]:
    question = PROTOCOL["questions"]["mean"]
    lines, results = [], {}
    for dataset in question["datasets"]:
        settings = PROTOCOL["datasets"][dataset]
        readout = READOUT[dataset]
        objects, fovs = load(dataset).tables
        criteria = [
            OutcomeCriterion(
                RelativeMeanError(readout), c["tolerance"], "lower_is_better", c["label"]
            )
            for c in question["criteria"]
        ]
        strategies = {
            name: _strategy(spec, readout, question) for name, spec in question["families"].items()
        }
        started = time.perf_counter()
        replay = replay_acquisitions(
            objects,
            fovs,
            feature_cols=[readout],
            strategies=list(strategies.values()),
            criteria=criteria,
            objective=POPULATION,
            reservoir_fraction=settings["reservoir_fraction"],
            orders=_orders(settings),
            acquisition_order_col="acquisition_order",
            n_splits=settings["n_splits"],
            random_state=PROTOCOL["random_state"],
            min_fovs=question["min_fovs"],
            keep_results=True,
            group_cols=GROUPS,
            observation_unit="nucleus",
            n_jobs=-1,
        )
        families = _families(replay, strategies, question["families"])
        lines.append(
            f"== mean, {dataset} ({settings['role']}; replay "
            f"{time.perf_counter() - started:.0f} s) =="
        )
        for index, criterion in enumerate(criteria):
            result = frontier(
                replay,
                families,
                criterion=criterion.name,
                alphas=PROTOCOL["alphas"],
                unit_cols=settings["unit_cols"],
                n_boot=PROTOCOL["n_boot"],
                random_state=PROTOCOL["random_state"],
                label=criterion.label,
            )
            _write(result, "mean", dataset, index)
            results[(dataset, index)] = result
            lines.append(result.summary())
    transfers = []
    for index in range(len(question["criteria"])):
        for alpha in PROTOCOL["alphas"]:
            table = transfer(
                results[("idr0028_screenB", index)], results[("idr0139", index)], alpha
            )
            transfers.append(table.assign(criterion=question["criteria"][index]["label"]))
    return lines, transfers


def _planted_tables(question: dict):
    """Planted acquisitions: MOCK host wells with donor siYAP nuclei in a share of fields."""
    dataset = load("idr0028_screenB")
    objects, fovs, features = dataset.objects, dataset.fovs, list(dataset.features)
    rng = np.random.default_rng(PROTOCOL["random_state"])
    object_parts, fov_parts = [], []
    for plate, host, donor in question["design"]:
        in_host = (fovs["repeat"] == plate) & (fovs["technical_repeat"] == host)
        host_fovs = fovs[in_host].reset_index(drop=True)
        donor_cells = objects[(objects["repeat"] == plate) & (objects["technical_repeat"] == donor)]
        donor_blocks = [block for _, block in donor_cells.groupby("image_file")]
        usable = [b for b in donor_blocks if len(b) >= question["planted_objects"]]
        xy = host_fovs[["stage_x_um", "stage_y_um"]].to_numpy(float)
        for frequency in question["frequencies"]:
            for layout in question["layouts"]:
                for replicate in range(question["replicates"]):
                    planted = _planted_fovs(frequency, layout, xy, rng)
                    group = {
                        "plate": plate,
                        "frequency": frequency,
                        "layout": layout,
                        "replicate": replicate,
                    }
                    fov_parts.append(host_fovs.assign(**group, planted_fov=planted))
                    for fov, plant in zip(host_fovs["image_file"], planted, strict=True):
                        cells = objects[
                            (objects["repeat"] == plate) & (objects["image_file"] == fov)
                        ]
                        object_parts.append(cells.assign(**group, planted=False))
                        if plant:
                            source = usable[rng.integers(len(usable))]
                            pick = rng.choice(
                                len(source), question["planted_objects"], replace=False
                            )
                            added = source.iloc[np.sort(pick)].assign(
                                **group, image_file=fov, planted=True
                            )
                            object_parts.append(added)
    keep = ["plate", "frequency", "layout", "replicate", "image_file", "planted", *features]
    planted_objects = pd.concat(object_parts, ignore_index=True)[keep]
    columns = ["plate", "frequency", "layout", "replicate", "image_file"]
    planted_fovs = pd.concat(fov_parts, ignore_index=True)[
        [*columns, "acquisition_order", "stage_x_um", "stage_y_um", "planted_fov"]
    ]
    return planted_objects, planted_fovs, features


def run_rare() -> list[str]:
    question = PROTOCOL["questions"]["rare"]
    objects, fovs, features = _planted_tables(question)
    groups = ["plate", "frequency", "layout", "replicate"]
    criterion = OutcomeCriterion(
        RareStateDetected("planted", 1), 1, "higher_is_better", question["criterion"]["label"]
    )
    specs = question["families"]
    target = specs["rare incidence (labels)"]["target_incidence"]
    objectives = {
        # The novelty bound is distribution evidence (the phenotypic fingerprint);
        # the frontier reads each rule's own verdict, so the scores do not depend on it.
        "novelty bound": ObjectiveConfig(SamplingObjective.DISTRIBUTION),
        "rare incidence (labels)": ObjectiveConfig(
            SamplingObjective.RARE_DISCOVERY,
            rare=RareTarget(
                detection_probability=0.95,
                min_fov_occurrences=1,
                phenotype="planted label",
                fov_incidence=target,
            ),
        ),
    }
    common = {
        "feature_cols": features,
        "criteria": [criterion],
        "reservoir_fraction": question["reservoir_fraction"],
        "orders": _orders(question),
        "acquisition_order_col": "acquisition_order",
        "n_splits": question["n_splits"],
        "random_state": PROTOCOL["random_state"],
        "min_fovs": question["min_fovs"],
        "keep_results": True,
        "fov_col": "image_file",
        "group_cols": groups,
        "observation_unit": "nucleus",
        "n_jobs": -1,
    }
    replays, families = {}, []
    started = time.perf_counter()
    for name, spec in specs.items():
        strategy = _strategy(spec, None, question)
        replay = replay_acquisitions(
            objects, fovs, strategies=[strategy], objective=objectives[name], **common
        )
        replays[name] = replay
        families += _families(replay, {name: strategy}, {name: spec})
    first = next(iter(replays.values()))
    result = frontier(
        first,
        families,
        criterion=criterion.name,
        alphas=PROTOCOL["alphas"],
        unit_cols=question["unit_cols"],
        n_boot=PROTOCOL["n_boot"],
        random_state=PROTOCOL["random_state"],
        label=criterion.label,
    )
    _write(result, "rare", "idr0028_planted", 0)
    heading = (
        "== rare, planted states in idr0028 (semi-synthetic; replays "
        f"{time.perf_counter() - started:.0f} s) =="
    )
    return [heading, result.summary()]


def main(argv: list[str]) -> None:
    check_lock("--lock" in argv)
    if "--lock" in argv:
        return
    datasets = registry()
    for name, settings in PROTOCOL["datasets"].items():
        assert datasets[name].role.value == settings["role"], name
    # No setting of this protocol was chosen on the held-out dataset.
    datasets.check_tuning(["idr0028_screenB"])
    RESULTS.mkdir(exist_ok=True)
    lines = [f"Benchmark, protocol version {PROTOCOL['version']} ({digest()[:12]})"]
    mean_lines, transfers = run_mean()
    lines += mean_lines
    table = pd.concat(transfers, ignore_index=True)
    table.to_csv(RESULTS / "benchmark_transfer.csv", index=False, float_format="%.4f")
    lines += ["== settings chosen on idr0028, scored on idr0139 ==", table.to_string(index=False)]
    lines += run_rare()
    text = "\n\n".join(lines) + "\n"
    (RESULTS / "benchmark_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
