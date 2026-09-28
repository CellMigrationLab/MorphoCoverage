"""``benchmark``: which stopping rule needs the fewest fields for the same risk.

Each rule ``Family`` (default: ``default_families``, the locked benchmark's
grids) is replayed once against held-back fields and swept over its grid
(``retrospective.frontier``); ``Benchmark.summary`` prints the scores and
every preset and setting of the run (the risk levels, the fields held back,
the orders, the resamples, the grids and their seed). Also re-exported by
``workflows.retrospective``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .._checks import check_seed
from .._wording import fields as fields_text
from .._wording import listing, percent
from ..retrospective.frontier import FrontierResult, frontier, sweep
from ..strategies import DescriptorSpreadStrategy, FOVBootstrapStrategy, ScalarPrecisionStrategy
from ..strategies.base import SamplingStrategy
from ._document import held_back_text
from ._shared import check_workflow_inputs, replay_rules, reservoir_sizes
from ._shared import orders as _orders
from .setup import Setup


@dataclass(frozen=True)
class Family:
    """A rule family for the benchmark: one strategy, the quantity it thresholds, a grid.

    ``strategy`` is built at any threshold of the grid; the sweep re-decides
    every threshold from its stored evidence (``redecide``), so the replay runs
    once per family.
    """

    name: str
    strategy: SamplingStrategy
    quantity: str
    thresholds: tuple[float, ...]


def default_families(readout: str, n_resamples: int = 200, random_state: int = 0) -> list[Family]:
    """The rule families and grids of the locked benchmark (protocol version 4).

    The JCB rule and the JCS descriptor spread, each per object and per FOV,
    and the FOV bootstrap of the mean, on ``readout``. The grids reach from a
    setting that never stops to one that stops at the first field on the
    committed data; they are absolute (the JCS spread and the bootstrap's
    standard error are in the readout's units), so widen them if your data
    need it: ``Benchmark.summary`` warns for every grid that does not bracket
    the data it was run on. ``random_state`` seeds
    the resampling rules (the JCS spread and the bootstrap); ``benchmark``
    passes ``Setup.random_state``, and the locked protocol uses 0.
    """
    random_state = check_seed(random_state)
    jcb = (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.4)
    jcs = (0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
    boot = (0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064)
    return [
        Family(
            "JCB per object",
            ScalarPrecisionStrategy(readout, jcb[0], "object", "upper_95"),
            "averaging_cv",
            jcb,
        ),
        Family(
            "JCB per field",
            ScalarPrecisionStrategy(readout, jcb[0], "fov", "upper_95"),
            "averaging_cv",
            jcb,
        ),
        Family(
            "JCS spread per object",
            DescriptorSpreadStrategy(readout, jcs[0], "object", n_resamples, random_state),
            "max_descriptor_spread",
            jcs,
        ),
        Family(
            "JCS spread per field",
            DescriptorSpreadStrategy(readout, jcs[0], "fov", n_resamples, random_state),
            "max_descriptor_spread",
            jcs,
        ),
        Family(
            "FOV bootstrap of the mean",
            # On the readout only, whatever other features the setup declares.
            FOVBootstrapStrategy(
                "centroid", boot[0], None, n_resamples, random_state, 2, features=(readout,)
            ),
            "fov_bootstrap_centroid_max_se",
            boot,
        ),
    ]


@dataclass(frozen=True)
class Benchmark:
    """Fields saved at matched risk, one ``FrontierResult`` per criterion."""

    frontiers: dict[str, FrontierResult]
    families: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Every criterion's scores, in sentences, a warning for each grid that does not
        bracket the data, then the risk levels and settings."""
        blocks = [result.summary() for result in self.frontiers.values()]
        warnings = grid_warnings(self)
        if warnings:
            blocks.append("\n".join(warnings))
        if self.settings:
            blocks.append(_benchmark_settings(self.settings))
        return "\n\n".join(blocks)


# A grid brackets the data when its strictest setting rarely stops early and its loosest
# setting stops: here, when not every setting leaves most acquisitions unstopped, and not
# every setting stops most of them at the first field the rule may decide.
_MOST = 0.5


def _grid_problem(runs: pd.DataFrame, first_field: int) -> str | None:
    """Why one family's grid does not bracket these data, or None when it does."""
    stopped = runs["stopped"].astype(bool)
    shares = (
        pd.DataFrame(
            {
                "setting": runs["setting"],
                "never": ~stopped,
                "first": stopped & (runs["effort"] <= first_field),
            }
        )
        .groupby("setting", sort=False)[["never", "first"]]
        .mean()
    )
    never, at_first = shares["never"], shares["first"]
    if len(never) and never.min() > _MOST:
        if never.min() == 1:
            return "it never stopped at any setting tried"
        return (
            f"it never stopped in {percent(never.min())} or more of the acquisitions at "
            "every setting tried"
        )
    if len(at_first) and at_first.min() > _MOST:
        if at_first.min() == 1:
            return "it stopped at the first field it could decide at every setting tried"
        return (
            f"it stopped at the first field it could decide in {percent(at_first.min())} or "
            "more of the acquisitions at every setting tried"
        )
    return None


def grid_warnings(result: Benchmark) -> list[str]:
    """One warning per family whose threshold grid does not bracket these data.

    A family that never stops at any setting reads as "never got there", as if
    the rule were worse, when the grid only failed to reach these data: the
    thresholds are absolute (a spread or a standard error in the readout's
    units), so a readout in other units (an area in µm², a ratio times 1000)
    needs another grid. The stops do not depend on the criterion, so the
    first frontier's runs decide.
    """
    if not result.frontiers:
        return []
    first = next(iter(result.frontiers.values()))
    runs = first.runs
    first_field = int(first.settings.get("min_fovs", 1) or 1)
    grids = result.settings.get("families", {})
    readout = result.settings.get("readout")
    lines = []
    for name in result.families:
        problem = _grid_problem(runs[runs["family"] == name], first_field)
        if problem is None:
            continue
        grid = grids.get(name, {})
        thresholds = grid.get("thresholds") or []
        span = (
            f" ({grid.get('quantity')} {min(thresholds):g} to {max(thresholds):g})"
            if thresholds
            else ""
        )
        units = f" (here in the units of {readout})" if readout else ""
        lines.append(
            f"Warning: the grid of {name}{span} does not bracket these data: {problem}. "
            f"Its thresholds are absolute{units} where the rule thresholds a spread or a "
            "standard error, so this says nothing about the rule; widen the grid "
            "(Family(..., thresholds=...)) and run the benchmark again."
        )
    return lines


def _benchmark_settings(settings: dict[str, Any]) -> str:
    """The benchmark's presets and computational settings in sentences."""
    levels = " and ".join(percent(alpha) for alpha in settings["alphas"])
    units = ", ".join(settings["unit_cols"]) or "one pooled unit"
    orders = f"{settings['permuted_orders']} random orders"
    if settings.get("recorded_order"):
        orders = f"the recorded order and {orders}"
    grids = listing(
        [
            f"{name} ({family['quantity']} {min(family['thresholds']):g} to "
            f"{max(family['thresholds']):g}, {len(family['thresholds'])} settings)"
            for name, family in settings["families"].items()
        ],
        limit=10,
    )
    seed = settings.get("family_seed")
    seeded = (
        f"; the default families' resampling rules seeded with random_state {seed}"
        if seed is not None
        else ""
    )
    return (
        f"Settings: risk levels {levels} of stopping too early (alphas, a preset); "
        f"{held_back_text(settings)}, {settings['n_splits']} hold-outs, {orders}; rules "
        f"evaluated from {fields_text(settings['min_fovs'])} on (min_fovs); intervals over "
        f"{units} from {settings['n_boot']} bootstrap resamples (n_boot); families and grids: "
        f"{grids}{seeded}; random_state {settings['random_state']}."
    )


def benchmark(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    *,
    reservoir_fraction: float,
    families: Sequence[Family] | None = None,
    alphas: Sequence[float] = (0.05, 0.10),
    unit_cols: Sequence[str] | None = None,
    n_splits: int = 5,
    permuted_orders: int = 10,
    min_fovs: int = 1,
    n_boot: int = 2000,
) -> Benchmark:
    """BENCHMARK: which rule needs the fewest fields for the same risk of stopping too early.

    Each family (default: ``default_families(setup.readout)``, its resampling
    rules seeded with ``Setup.random_state``) is replayed
    once against held-back fields and swept over its grid; at each risk level
    in ``alphas`` its cheapest setting from which the risk stays at or below
    the level is scored against a fixed field count, with intervals over the
    independent units ``unit_cols`` (default: the analysis groups). ``alphas``
    (default 5% and 10%, a preset) are printed in the summary, with the fields
    held back, the orders, ``min_fovs``, ``n_boot`` and the grids. The rules run on
    the raw values, as live; the criteria judge the normalised values when
    ``normalize`` is set. Settings chosen here describe these data; score them
    on data they were not chosen on before using them
    (``retrospective.frontier.transfer``).
    """
    check_workflow_inputs(objects, fovs, setup)
    default = families is None
    families = list(
        default_families(setup.readout, random_state=setup.random_state)
        if families is None
        else families
    )
    if not families or any(not isinstance(f, Family) for f in families):
        raise TypeError("families must be a non-empty sequence of Family")
    criteria = setup.criteria()
    # The rules on the raw values, as live; the criteria on the normalised ones.
    replay = replay_rules(
        objects,
        setup.prepared(objects, fovs),
        fovs,
        setup,
        strategies=[f.strategy for f in families],
        criteria=criteria,
        reservoir_fraction=reservoir_fraction,
        orders=_orders(setup, permuted_orders, dispersed=0),
        acquisition_order_col=setup.acquisition_order_col,
        n_splits=n_splits,
        random_state=setup.random_state,
        min_fovs=min_fovs,
        keep_results=True,
        n_jobs=setup.n_jobs,
    )
    swept = [
        sweep(
            replay,
            config_id=f.strategy.config_id,
            quantity=f.quantity,
            thresholds=list(f.thresholds),
            name=f.name,
        )
        for f in families
    ]
    units = list(replay.group_cols) if unit_cols is None else list(unit_cols)
    labels = [criterion.label for criterion in criteria]
    # Frontiers are keyed by label; two criteria with one label (tolerances 5% and
    # 5.02% both print as 5.0%) keep both, the second named by its criterion.
    keys = [
        label if labels.count(label) == 1 or labels.index(label) == i else f"{label} ({c.name})"
        for i, (label, c) in enumerate(zip(labels, criteria, strict=True))
    ]
    frontiers = {
        key: frontier(
            replay,
            swept,
            criterion=criterion.name,
            alphas=list(alphas),
            unit_cols=units,
            n_boot=n_boot,
            random_state=setup.random_state,
            label=criterion.label,
        )
        for key, criterion in zip(keys, criteria, strict=True)
    }
    settings = {
        "readout": setup.readout,
        "reservoir_fraction": reservoir_fraction,
        "reservoir_fovs": reservoir_sizes(replay),
        "alphas": list(alphas),
        "unit_cols": units,
        "n_splits": n_splits,
        "recorded_order": setup.acquisition_order_col is not None,
        "permuted_orders": permuted_orders,
        "min_fovs": min_fovs,
        "n_boot": n_boot,
        "random_state": setup.random_state,
        "family_seed": setup.random_state if default else None,
        "families": {
            f.name: {"quantity": f.quantity, "thresholds": list(f.thresholds)} for f in families
        },
    }
    return Benchmark(frontiers, tuple(f.name for f in families), settings)


__all__ = ["Benchmark", "Family", "benchmark", "default_families", "grid_warnings"]
