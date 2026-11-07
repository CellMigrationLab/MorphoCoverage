"""Builders shared by the sampling-unit tests: conditions, technical repeats, repeat designs."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from _helpers_retrospective import _efficiency, _gradient

from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.retrospective import RepresentationControl, representation_control
from morphocoverage.sampling_unit import (
    condition_contrast,
    contrast_cells,
    descriptor_spread_check,
    feature_discovery,
    fov_clustering,
    fov_variability,
    object_rule_check,
    unusual_objects,
)
from morphocoverage.strategies import SupportNoveltyEstimator


def _tables(shift=1.0, n_repeats=3, cells=30, seed=0, unbalanced=False):
    """One well per condition and repeat: 'area' differs in every repeat, 'noise' does not,
    'flat' holds one value."""
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(n_repeats):
        for condition, effect in (("ctrl", 0.0), ("drug", shift)):
            n = cells * (4 if unbalanced and repeat == 0 else 1)
            for field in range(n):
                rows.append(
                    {
                        "condition": condition,
                        "biological_repeat": f"r{repeat}",
                        "position_id": f"{condition}-{repeat}-{field}",
                        "area": rng.normal(10 + effect, 1),
                        "noise": rng.normal(5, 1),
                        "aspect": rng.normal(2, 0.3),
                        "flat": 1.0,
                    }
                )
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "position_id"]].drop_duplicates()
    return objects, fovs


def _repeat_tables(conditions, repeats, fields=6, cells=20, effect=None, seed=0, shift=None):
    """``conditions`` per repeat, ``fields`` fields of ``cells`` cells each; ``effect`` the
    mean of each condition, ``shift`` a multiplier of each repeat's values."""
    rng = np.random.default_rng(seed)
    rows, fovs = [], []
    for repeat in repeats:
        for condition in conditions[repeat]:
            for field in range(fields):
                key = {
                    "condition": condition,
                    "biological_repeat": repeat,
                    "position_id": f"{condition}_{repeat}_{field}",
                }
                fovs.append(key)
                mean = (effect or {}).get(condition, 10.0) * (shift or {}).get(repeat, 1.0)
                rows += [{**key, "x": v} for v in rng.normal(mean, 1.0, cells)]
    return pd.DataFrame(rows), pd.DataFrame(fovs)


def _three(effect=None, seed=0):
    repeats = ["r1", "r2", "r3"]
    return _repeat_tables({r: ["ctrl", "drug"] for r in repeats}, repeats, effect=effect, seed=seed)


def _two_conditions(
    shift: float, field_sd: float, seed: int = 3, n_fields: int = 12, n_wells: int = 2
):
    """Control and treated wells in two repeats; ``shift`` moves the treated condition's ``x``.

    ``y`` never differs between the conditions.
    """
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for repeat in ("R1", "R2"):
        for condition, well_shift in (("control", 0.0), ("treated", shift)):
            for well in [f"w{i + 1}" for i in range(n_wells)]:
                for j in range(n_fields):
                    name = f"{repeat}_{condition}_{well}_f{j}"
                    fovs.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "well": well,
                            "image_file": name,
                        }
                    )
                    offset = well_shift + rng.normal(0, field_sd)
                    for _ in range(25):
                        objects.append(
                            {
                                "condition": condition,
                                "biological_repeat": repeat,
                                "well": well,
                                "image_file": name,
                                "x": 10 + offset + rng.normal(),
                                "y": rng.normal(),
                            }
                        )
    return pd.DataFrame(objects), pd.DataFrame(fovs)


CONTRAST_GROUPS = ["condition", "biological_repeat", "well"]


def _technical(objects, fovs):
    """The same tables with the well named as the MorphoBricks technical repeat."""
    rename = {"well": "technical_repeat"}
    return objects.rename(columns=rename), fovs.rename(columns=rename)


CONTROL = {"condition": "ctrl"}


def _well_tables(spec, seed=0, per_field=20, obs_sd=1.0):
    """``spec``: (condition, repeat, technical repeat, fields, well mean, field SD) per well."""
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for condition, repeat, well, n_fields, mean, field_sd in spec:
        for j in range(n_fields):
            key = {
                "condition": condition,
                "biological_repeat": repeat,
                "technical_repeat": well,
                "position_id": f"{repeat}_{condition}_{well}_f{j}",
            }
            fovs.append(key)
            offset = mean + rng.normal(0, field_sd)
            n = per_field if np.isscalar(per_field) else per_field[j % len(per_field)]
            objects += [{**key, "x": v} for v in offset + rng.normal(0, obs_sd, n)]
    return pd.DataFrame(objects), pd.DataFrame(fovs)


def _two_repeats():
    spec = [
        (c, r, f"{c}{r}{i}", 4, (1.0 if c == "drug" else 0) + (2.0 if r == "r2" else 0), 0.2)
        for r in ("r1", "r2")
        for c in ("ctrl", "drug")
        for i in range(2)
    ]
    return _well_tables(spec)


def _well_tables_with_empty_fields(spec, seed=0, per_field=20, obs_sd=1.0, empty_every=0):
    """``spec``: (condition, repeat, technical repeat, fields, well mean, field SD) per well."""
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for condition, repeat, well, n_fields, mean, field_sd in spec:
        for j in range(n_fields):
            key = {
                "condition": condition,
                "biological_repeat": repeat,
                "technical_repeat": well,
                "position_id": f"{repeat}_{condition}_{well}_f{j}",
            }
            fovs.append(key)
            if empty_every and j % empty_every == empty_every - 1:
                continue
            offset = mean + rng.normal(0, field_sd)
            objects += [{**key, "x": v} for v in offset + rng.normal(0, obs_sd, per_field)]
    return pd.DataFrame(objects), pd.DataFrame(fovs)


def _without_technical(objects, fovs):
    return objects.drop(columns="technical_repeat"), fovs.drop(columns="technical_repeat")


def _well_column_tables(spec, seed=0, per_field=20, well_col="technical_repeat"):
    """``spec``: (condition, repeat, well, fields, well mean) per well; field SD 0.3."""
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for condition, repeat, well, n_fields, mean in spec:
        for j in range(n_fields):
            key = {
                "condition": condition,
                "biological_repeat": repeat,
                well_col: well,
                "position_id": f"{repeat}_{condition}_{well}_f{j}",
            }
            fovs.append(key)
            level = mean + rng.normal(0, 0.3)
            objects += [
                {**key, "x": v, "y": 2 * v + 1} for v in level + rng.normal(0, 1, per_field)
            ]
    return pd.DataFrame(objects), pd.DataFrame(fovs)


def _wells(n, *, repeats=("r1",), shift=0.0, fields=4, well_col="technical_repeat", seed=0):
    spec = [
        (condition, repeat, f"{condition}{i}", fields, 10 + (shift if condition == "drug" else 0))
        for repeat in repeats
        for condition in ("ctrl", "drug")
        for i in range(n)
    ]
    return _well_column_tables(spec, seed=seed, well_col=well_col)


def _signed(offset_b: float, seed: int = 0):
    """ctrl, a and b, 4 repeats, 3 fields of 5 cells; ``area`` near 2 for ctrl and a, near
    ``offset_b`` for b; ``positive`` the same plus 10."""
    rng = np.random.default_rng(seed)
    rows = []
    for condition, centre in (("ctrl", 2.0), ("a", 2.5), ("b", offset_b)):
        for r in range(4):
            for f in range(3):
                for _ in range(5):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": f"R{r}",
                            "image_file": f"{condition}{r}_{f}.tif",
                            "area": centre + rng.normal(scale=0.3),
                        }
                    )
    objects = pd.DataFrame(rows).assign(positive=lambda t: t["area"] + 10)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs.reset_index(drop=True)


def _constant_field_means():
    rows, fields = [], []
    for condition, mean in (("control", 10.0), ("treated", 12.0)):
        for j in range(6):
            name = f"{condition}_{j}"
            fields.append({"condition": condition, "image_file": name})
            rows += [
                {"condition": condition, "image_file": name, "x": v} for v in (mean - 1, mean + 1)
            ]
    return pd.DataFrame(rows), pd.DataFrame(fields)


def _screen():
    """Two sets, each its own control and repeats; set B's whole repeats are twice as bright."""
    conditions = {r: ["ctrl", "drugA"] for r in ("a1", "a2", "a3")}
    conditions.update({r: ["ctrl", "drugB"] for r in ("b1", "b2", "b3")})
    shift = {"b1": 2.0, "b2": 2.0, "b3": 2.0}
    return _repeat_tables(conditions, list(conditions), fields=4, cells=15, shift=shift)


def _field_rows(condition, repeat, n_fields, per_field, mean, seed, sd=1.0, prefix="f"):
    rng = np.random.default_rng(seed)
    rows = [
        {
            "condition": condition,
            "biological_repeat": repeat,
            "image_file": f"{prefix}{j}",
            "x": float(v),
        }
        for j in range(n_fields)
        for v in mean + rng.normal(0, sd, per_field)
    ]
    fovs = pd.DataFrame(
        {
            "condition": condition,
            "biological_repeat": repeat,
            "image_file": [f"{prefix}{j}" for j in range(n_fields)],
        }
    )
    return pd.DataFrame(rows), fovs


def _join(*parts):
    return (
        pd.concat([p[0] for p in parts], ignore_index=True),
        pd.concat([p[1] for p in parts], ignore_index=True),
    )


def _plain(frame):
    """The rows of ``frame`` with missing values as ``None``, so they compare equal."""
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def _error(call):
    """The type and message of the exception ``call`` raises."""
    with pytest.raises(Exception) as caught:
        call()
    return type(caught.value).__name__, str(caught.value)


R = ["2026_01_05", "2026_01_12", "2026_01_19", "2026_01_26"]


OTHER = ["2026_02_01", "2026_02_08", "2026_02_15"]


SET_B = ["2026_03_02", "2026_03_09", "2026_03_16"]


CONTROL_ONLY = "2026_04_06"


DESIGNS = {
    "E1": {"ctrl": R[:1], "cond1": R[:1], "cond2": R[:1]},
    "E2": {"ctrl": R, "cond1": R[:3], "cond2": R[:2], "cond3": R, "cond4": R[:1]},
    "E4": {"ctrl": R[:3] + [CONTROL_ONLY], "cond1": R[:3], "cond2": R[:3]},
    "E5": {"ctrl": R[:3], "cond1": R, "cond2": R[:3]},
    "E6": {
        "ctrl": R[:3] + SET_B,
        "condA": R[:3],
        "condB": SET_B,
        "bridge": [R[0], SET_B[0], R[1], SET_B[1]],
    },
}


# Not in DESIGNS: cond3 was imaged in repeats under labels the control does not have, which a
# contrast refuses (#617), so it has no contrast to build.
UNSHARED_LABELS = {"ctrl": R[:3], "cond1": R[:3], "cond3": OTHER}


EFFECTS = {"cond1": 2.0, "condA": 2.0}


def _make(design, fields=3, cells=15, seed=0):
    """Two features (area, the effect; intensity, none), a batch shift per repeat, one well
    per condition and repeat."""
    rng = np.random.default_rng(seed)
    plan = DESIGNS.get(design, UNSHARED_LABELS)
    repeats = sorted({r for held in plan.values() for r in held})
    batch = {r: rng.normal(0, 0.3, 2) for r in repeats}
    rows, fovs = [], []
    for index, (condition, held) in enumerate(plan.items()):
        for repeat in held:
            well = f"{'ABCDEFG'[index]}1"
            for field in range(fields):
                image = f"{condition}_{repeat}_{field}.tif"
                z = rng.normal(0, 1, (cells, 2)) + batch[repeat]
                z[:, 0] += EFFECTS.get(condition, 0.0)
                for i in range(cells):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "technical_repeat": well,
                            "image_file": image,
                            "label_id": i + 1,
                            "area": float(np.exp(5 + 0.3 * z[i, 0])),
                            "intensity": float(np.exp(7 + 0.25 * z[i, 1])),
                        }
                    )
                fovs.append(
                    {
                        "condition": condition,
                        "biological_repeat": repeat,
                        "technical_repeat": well,
                        "image_file": image,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(fovs)


@pytest.fixture(scope="module")
def designs():
    """Each design's contrast on area and its feature discovery, at few draws and shuffles."""
    found = {}
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch.setattr(contrast_cells, "BALANCED_DRAWS", 3)
        for name in DESIGNS:
            objects, fovs = _make(name)
            contrast = condition_contrast(
                objects,
                fovs,
                feature="area",
                control=CONTROL,
                n_permutations=99,
                n_draws=2,
                object_counts=(10,),
            )
            discovery = feature_discovery(
                objects, fovs, features=["area", "intensity"], control=CONTROL, n_permutations=99
            )
            found[name] = contrast, discovery
    return found


def _flat(lines):
    return " ".join(" ".join(lines).split())


CONTROL_CONDITION = {"condition": "control"}


def _flat_text(text: str) -> str:
    return " ".join(text.split())


def _widths(text: str) -> list[int]:
    return [len(line) for line in text.splitlines()]


def _orders():
    orders = [AcquisitionOrder.recorded(), AcquisitionOrder.permuted(1)]
    return _efficiency(*_gradient(), [*orders, AcquisitionOrder.dispersed(0)])


def _five_wells():
    return _technical(*_two_conditions(shift=2.0, field_sd=0.3, seed=7, n_wells=5, n_fields=6))


def _group(condition, repeat, *, n_fields=24, per_field=30, field_sd=0.0, seed=0, mean=10.0):
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0, field_sd, n_fields)
    objects = pd.DataFrame(
        [
            {
                "condition": condition,
                "biological_repeat": repeat,
                "image_file": f"{condition}{repeat}{j}",
                "x": v,
            }
            for j in range(n_fields)
            for v in mean + offsets[j] + rng.normal(0, 1.0, per_field)
        ]
    )
    fovs = (
        objects[["condition", "biological_repeat", "image_file"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return objects, fovs


def _mock_kd_tables(field_sd=0.0):
    parts = [
        _group("MOCK", "r1", field_sd=field_sd, seed=1),
        _group("KD", "r1", field_sd=field_sd, seed=2, mean=12.0),
    ]
    return (
        pd.concat([p[0] for p in parts], ignore_index=True),
        pd.concat([p[1] for p in parts], ignore_index=True),
    )


@pytest.fixture(scope="module")
def foundation_checks():
    objects, fovs = _mock_kd_tables(field_sd=1.5)
    return {
        "variability": fov_variability(objects, fovs, feature="x", observation_unit="cell"),
        "object_rules": object_rule_check(
            objects, fovs, feature="x", control={"condition": "MOCK"}, n_draws=200
        ),
        "descriptor_spread": descriptor_spread_check(
            objects, fovs, feature="x", threshold=100, n_draws=20
        ),
    }


def _plate(shift=6.0, plates=("P1", "P2")):
    rng = np.random.default_rng(0)
    fov_rows, object_rows = [], []
    for plate in plates:
        for condition, offset in (("MOCK", 0.0), ("KD", shift)):
            for i in range(12):
                name = f"{plate}_{condition}_{i}"
                fov_rows.append(
                    {"condition": condition, "biological_repeat": plate, "image_file": name}
                )
                values = rng.normal(0.0, 1.0, size=(25, 2))
                values[:, 0] += offset
                object_rows.append(
                    pd.DataFrame(values, columns=["a", "b"]).assign(
                        condition=condition, biological_repeat=plate, image_file=name
                    )
                )
    return pd.concat(object_rows, ignore_index=True), pd.DataFrame(fov_rows)


KD_CONTROL = RepresentationControl(
    positive={"condition": "KD"}, reference={"condition": "MOCK"}, planted_objects=5
)


def _control(objects, fovs):
    novelty = SupportNoveltyEstimator(1.0, 1, None, 3, 3)
    return representation_control(
        objects,
        fovs,
        feature_cols=["a", "b"],
        novelty=novelty,
        control=KD_CONTROL,
        pair_cols=["biological_repeat"],
        fovs_seen=6,
        draws=10,
    )


@pytest.fixture(scope="module")
def controls():
    return _control(*_plate())


@pytest.fixture(scope="module")
def states():
    objects, fovs = _mock_kd_tables()
    labels = pd.Series(False, index=objects.index)
    # MOCK: the state in 2 fields only (clustered); KD: one object in every field (spread).
    mock = objects["condition"].eq("MOCK") & objects["image_file"].isin(["MOCKr10", "MOCKr11"])
    labels[mock] = True
    kd = objects[objects["condition"].eq("KD")].groupby("image_file").head(1).index
    labels[kd] = True
    clustering = fov_clustering(
        objects.assign(state=labels), fovs, label_col="state", states=[True]
    )
    unusual = unusual_objects(
        objects, fovs, feature_cols=["x"], reference={"condition": "MOCK"}, seed_checks=2
    )
    return clustering, unusual
