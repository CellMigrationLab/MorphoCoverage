"""Guards against the bug classes the three audits kept finding (docs/audit/README.md).

Each audit found some of the same kinds of error again, in a place the last
fix had not reached: the live session and the replay resolving a field
differently, the two rule sets of the validator drifting apart, a numpy
integer refused by a hand-written check, a percentage written inline as "0%",
a 32-bit hash used as a group's identity, an interval whose coverage was never
measured. Every test here checks a whole class, package-wide, so the next
instance fails the suite instead of waiting for an audit.
"""

# ruff: noqa: E501  (the table of mutated inputs reads best one per line)

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from contract_registry import POPULATION, STRATEGIES_UNDER_CONTRACT, replay
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS
from scipy import stats

from morphocoverage import OnlineSession, check_tables, validate_input_tables
from morphocoverage.bounds import interval
from morphocoverage.statistics import bootstrap_fields, bootstrap_nested

SOURCE = Path(__file__).resolve().parents[1] / "src" / "morphocoverage"
MODULES = sorted(SOURCE.rglob("*.py"))


def _relative(path: Path) -> str:
    return str(path.relative_to(SOURCE))


# --- Live equals replay ------------------------------------------------------


def _live_results(scenario, strategies):
    """Feed a scenario to a live session field by field, in acquisition order."""
    session = OnlineSession(
        strategies=strategies, feature_cols=list(FEATURE_COLS), objective=POPULATION
    )
    results = []
    ordered = scenario.fovs.sort_values(
        ["condition", "biological_repeat", ORDER_COL], kind="stable"
    )
    for _, fov in ordered.iterrows():
        mask = pd.Series(True, index=scenario.objects.index)
        for column in ("condition", "biological_repeat", "image_file"):
            mask &= scenario.objects[column].eq(fov[column])
        results += session.add_fov(scenario.objects.loc[mask].copy(), fov.to_dict())
    return results


def _keyed(results):
    return {
        (tuple(sorted(r.group.items())), r.config_id, r.n_fovs): r.without_timing() for r in results
    }


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_live_session_gives_the_replays_result_at_every_step(name):
    scenario = SCENARIOS[name]()
    strategies = list(STRATEGIES_UNDER_CONTRACT)
    live = _keyed(_live_results(scenario, strategies))
    replayed = _keyed(item.result for item in replay(scenario, strategies, incremental=True))
    assert live.keys() == replayed.keys()
    for key, result in live.items():
        assert result == replayed[key], key


# --- The two rule sets of the validator agree --------------------------------


def _base():
    scenario = SCENARIOS["condition_repeat_groups"]()
    return scenario.objects.drop(columns=["true_state"]), scenario.fovs


def _mutations():
    """Tables a person could hand over, each with one mistake (or none)."""
    objects, fovs = _base()
    first = objects.index[0]
    # fmt: off
    return {
        "as given": (objects, fovs),
        "blank condition": (objects.assign(condition=objects["condition"].where(objects.index != first, " ")), fovs),
        "missing repeat": (objects.assign(biological_repeat=objects["biological_repeat"].where(objects.index != first)), fovs),
        "inf feature": (objects.assign(feature_a=objects["feature_a"].where(objects.index != first, np.inf)), fovs),
        "nan feature": (objects.assign(feature_a=objects["feature_a"].where(objects.index != first)), fovs),
        "text feature": (objects.assign(feature_a=objects["feature_a"].astype(str)), fovs),
        "Condition spelled": (objects.rename(columns={"condition": "Condition"}), fovs.rename(columns={"condition": "Condition"})),
        "plural repeats": (objects.rename(columns={"biological_repeat": "repeats"}), fovs.rename(columns={"biological_repeat": "repeats"})),
        "dotted repeat": (objects.rename(columns={"biological_repeat": "Repeat.1"}), fovs),
        "repeat only in objects": (objects, fovs.drop(columns=["biological_repeat"])),
        "former name repeat": (objects.rename(columns={"biological_repeat": "repeat"}), fovs.rename(columns={"biological_repeat": "repeat"})),
        "former name beside the new one": (objects.assign(repeat="other"), fovs),
        "bio_repeat spelled": (objects.rename(columns={"biological_repeat": "bio_repeat"}), fovs.rename(columns={"biological_repeat": "bio_repeat"})),
        "tec_repeat spelled": (objects.assign(tec_repeat="w1"), fovs.assign(tec_repeat="w1")),
        "condition only in fovs": (objects.drop(columns=["condition"]), fovs),
        "field listed twice": (objects, pd.concat([fovs, fovs.iloc[:1]], ignore_index=True)),
        "field missing from fovs": (objects, fovs.iloc[1:]),
        "column twice": (pd.concat([objects, objects[["feature_a"]]], axis=1), fovs),
        "z plane feature": (objects.assign(z=0.0), fovs),
        "upper-case Z": (objects.assign(Z=np.arange(len(objects)) % 3), fovs),
        "complex feature": (objects.assign(feature_a=objects["feature_a"] + 1j), fovs),
        "categorical blank": (objects.assign(condition=objects["condition"].astype("category").cat.add_categories([" "]).where(objects.index != first, " ")), fovs),
    }
    # fmt: on


@pytest.mark.parametrize("case", sorted(_mutations()))
@pytest.mark.filterwarnings("ignore:'repeat' is the former name:DeprecationWarning")
def test_check_tables_accepts_exactly_what_the_validator_accepts(case):
    objects, fovs = _mutations()[case]
    features = [c for c in FEATURE_COLS if c in objects.columns]
    if "z" in objects.columns and case == "z plane feature":
        features = [*features, "z"]
    try:
        validate_input_tables(objects, fovs=fovs, feature_cols=features)
        accepted = True
    except (ValueError, TypeError):
        accepted = False
    report = check_tables(objects, fovs, feature_cols=features)
    assert report.ok == accepted, report.summary()


# --- Source-level guards -----------------------------------------------------

# Checks of a count or a number are written once, in ``_checks`` (numpy integers
# and reals are counts and numbers there; booleans are not).
_BARE_TYPE_CHECK = re.compile(r"not isinstance\([^()]*,\s*(\(?\s*int\b|float\b|int \|)")


def test_counts_and_numbers_are_checked_only_through_checks():
    offenders = [
        f"{_relative(path)}:{number}"
        for path in MODULES
        if path.name != "_checks.py"
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if _BARE_TYPE_CHECK.search(line)
    ]
    assert not offenders, f"use morphocoverage._checks instead: {offenders}"


# A percentage is written through ``_wording.percent``: an inline format writes
# 0.3% as "0%" and 99.7% as "100%".
_INLINE_PERCENT = re.compile(r":\.\d+%\}|100 \* [^}]*:\.\d+f\}%|\{100 \*")


def test_percentages_are_written_only_through_wording():
    offenders = [
        f"{_relative(path)}:{number}: {line.strip()}"
        for path in MODULES
        if path.name != "_wording.py"
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if _INLINE_PERCENT.search(line)
    ]
    assert not offenders, "use morphocoverage._wording.percent:\n" + "\n".join(offenders)


# ``group_identity`` is a 32-bit hash for seeding: two groups can share it, so
# it never identifies a group in a lookup (``_seeds.group_key`` does).
_IDENTITY_AS_KEY = re.compile(
    r"\[group_identity\(|\.get\(group_identity\("
    r"|\{group_identity\([^)]*\)\s*:|in \w+\[group_identity"
)


def test_the_seeding_hash_is_never_a_lookup_key():
    offenders = [
        f"{_relative(path)}:{number}"
        for path in MODULES
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if _IDENTITY_AS_KEY.search(line)
    ]
    assert not offenders, f"look groups up with _seeds.group_key: {offenders}"


def test_every_module_stays_under_600_lines():
    long = {_relative(p): n for p in MODULES if (n := len(p.read_text().splitlines())) > 600}
    assert not long, f"split these modules (AGENTS.md): {long}"


def test_field_identity_is_resolved_only_by_metadata():
    # The live session, LiveTables and the validator each once resolved the field
    # their own way; only metadata may define how a FOV column is chosen. The one
    # exception is check_tables' reporting twin in _data_rules (it records every
    # problem instead of raising at the first), held to the validator by
    # test_check_tables_accepts_exactly_what_the_validator_accepts.
    definers = []
    for path in MODULES:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and re.fullmatch(
                r"_?(resolve|detect|default)_(fov_col|group_cols|identity_cols)", node.name
            ):
                definers.append(f"{_relative(path)}:{node.name}")
    allowed = ("metadata.py:", "_data_rules.py:")
    assert all(d.startswith(allowed) for d in definers), definers


# --- A readout constant so far is not yet enough data --------------------------

# The precision and stabilization rules judge a spread; with no spread so far (a count of 0
# in every object, a saturated intensity) theirs is 0, which says nothing (audit 3 C-1 for the
# JCB rule, audit 4 P4-1 for the JCS rules and the FOV bootstrap).
_READOUT_RULES = (
    "ScalarPrecisionStrategy",
    "DescriptorPlateauStrategy",
    "DescriptorSpreadStrategy",
    "FOVBootstrapStrategy",
)


def test_no_readout_rule_stops_on_a_readout_constant_so_far():
    from scenarios import homogeneous

    rules = [s for s in STRATEGIES_UNDER_CONTRACT if type(s).__name__ in _READOUT_RULES]
    assert {type(s).__name__ for s in rules} == set(_READOUT_RULES)
    scenario = homogeneous()
    constant = scenario.objects.assign(**dict.fromkeys(FEATURE_COLS, 255.0))
    results = replay(type(scenario)("constant", constant, scenario.fovs), rules)
    stopped = {r.result.strategy for r in results if r.result.rule_decision == "stop"}
    assert not stopped, f"stopped on a constant readout: {sorted(stopped)}"
    assert all(
        q.availability in ("insufficient_data", "not_applicable")
        for r in results
        for q in r.result.evidence.quantities.values()
    )


# --- Every interval covers what it says --------------------------------------


def test_clopper_pearson_intervals_cover_at_least_their_level():
    for trials in (5, 12, 40):
        for p in np.linspace(0.02, 0.98, 25):
            covered = 0.0
            for successes in range(trials + 1):
                low, high = interval(successes, trials, 0.95)
                if low <= p <= high:
                    covered += stats.binom.pmf(successes, trials, p)
            assert covered >= 0.95 - 1e-9, (trials, p, covered)


def _coverage(units, per_unit_fields, reps, rng):
    hit = 0
    for _ in range(reps):
        wells = []
        for _ in range(units):
            shift = rng.normal(0, 1.0)
            wells.append(
                [shift + rng.normal(0, 0.5) + rng.normal(0, 1, 15) for _ in range(per_unit_fields)]
            )
        _, low, high = bootstrap_nested(wells, np.mean, n_boot=300, rng=rng)
        hit += low <= 0 <= high
    return hit / reps


@pytest.mark.parametrize("units", [3, 6, 12])
def test_nested_bootstrap_intervals_cover_near_their_level(units):
    rng = np.random.default_rng(units)
    assert _coverage(units, 3, 250, rng) >= 0.89


def test_field_bootstrap_intervals_cover_near_their_level():
    rng = np.random.default_rng(1)
    hit = 0
    for _ in range(250):
        fields = [rng.normal(0, 1.0) + rng.normal(0, 1, 15) for _ in range(8)]
        _, low, high = bootstrap_fields(fields, np.mean, n_boot=300, rng=rng)
        hit += low <= 0 <= high
    assert hit / 250 >= 0.89
