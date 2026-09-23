"""Convert the Crick Enhancing-Reproducibility CellProfiler tables to MorphoBricks tables.

Source: https://github.com/FrancisCrickInstitute/Enhancing-Reproducibility
(``inputs/``, CC BY 4.0), CellProfiler outputs for IDR studies idr0028 (screen B)
and idr0139. See README.md in this directory for attribution and caveats.

Usage::

    python data/crick_idr/convert.py RAW_DIR OUT_DIR

RAW_DIR must contain the LFS-resolved files, named as downloaded by ``fetch()``.
Only the wells analysed in Marcotti et al. 2026 are kept (the same wells as
the Crick ``data_subsets``). The output is two linked tables per dataset: one
row per acquired field (including fields without nuclei) and one row per
nucleus. All nuclear and cytoplasmic CellProfiler intensity measurements are
kept, with their CellProfiler names prefixed by the compartment. Values are
written with 6 significant digits.

This is a data-preparation script, not part of the package.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

# Pinned source commit of FrancisCrickInstitute/Enhancing-Reproducibility, so the
# download cannot drift with their main branch.
SOURCE_COMMIT = "a81d20c9393485e57395cd4a1be9e7ff83c7f11d"
MEDIA = (
    "https://media.githubusercontent.com/media/FrancisCrickInstitute/"
    f"Enhancing-Reproducibility/{SOURCE_COMMIT}/inputs"
)
# SHA-256 of each source file (equal to its Git LFS object id at SOURCE_COMMIT).
SOURCE_SHA256 = {
    "idr0028-screenB-annotation.csv": "898914f2c72808fbe97f162d90ade0c7ea1013cdbcece47470aa93449708a802",
    "idr0028_1A_Cytoplasm.csv": "a212e43ef4196caee91fd27cd004abeff416b51b7746e2fca0c154f6ad2d2b6e",
    "idr0028_1A_Image.csv": "56762eee76dfbdfcf5cadb8c5e4aac2e18063ccff000a4490a0a3bdf9a99b4b5",
    "idr0028_1A_Nuclei.csv": "a8608f7bff4d480fd2313cee51d4ffb5f087b36dd7c88481738f152648cf2092",
    "idr0028_1A_index.tsv": "768affc7c6df557907715719c1000f60700443ccc17b746566546df93d221520",
    "idr0028_2A_Cytoplasm.csv": "627f64cbb07167dc0cb7f9f7b4de3760509ddf1c204999f70c2147fb1815cd5e",
    "idr0028_2A_Image.csv": "980ee507f3555d46e941dd27165e36a1e981f74f74532de0dbd27e79ea1e6cd3",
    "idr0028_2A_Nuclei.csv": "c4842b3c9f247f71f62b8003b3cba1d19cba9b93b58fc4d8645e348edc1d6bb6",
    "idr0028_2A_index.tsv": "08cbf3a38bc3f02f88ad870a6432daed098ae3574cbddae9d713e8fe9b8fab65",
    "idr0028_2B_Cytoplasm.csv": "dcee471db7ec57137d5ddf027426e303b867c580e0249395f6138c98fedfa39d",
    "idr0028_2B_Image.csv": "da8543f3e3b2b8a60716c77b6dda27d7df5f46701ffefbe8d1092f707694dac7",
    "idr0028_2B_Nuclei.csv": "9c72a10f11e1ee7649fd5d0283b9c7208c2911d0279dbfa68f0f0be193670265",
    "idr0028_2B_index.tsv": "0122f109773888e4abc1d9ee989b4f32d388bc667e781e5d45097b2ed1804c09",
    "idr0139-screenA-annotation.csv": "4ce347396467cb2948e5ec2857ebe42b2506ca76d00a3b5be3b3f9c69886d20a",
    "idr0139_Cytoplasm.csv": "83b1cf0d59ce4baa6645e1a5f8263f5e90badfbb453c583c2708a3aa08d7c2fa",
    "idr0139_Image.csv": "b7b33467e79c585e50c65e0957411115ba2b4d1ca211a78d6e011e01e3ea8480",
    "idr0139_Nuclei.csv": "2cba4c0d66385deea4062e8d56561795eb52b031a978c84ad75e0d919551c690",
}
IDR0028_PLATES = ("1A", "2A", "2B")
IDR0139_PLATE = 1093711385
FLOAT_FORMAT = "%.6g"

# The wells analysed in Marcotti et al. 2026, exactly as selected by the Crick
# scripts subset_cp_data.py (idr0028) and subset_idr0139_data.py (idr0139),
# which build the data_subsets.zip used by their notebooks
# (https://zenodo.org/records/15727751).
IDR0028_PAPER_WELLS = {
    "1A": ["I23", "J23", "M23", "N23", "B6"],
    "2A": ["C13", "J23", "N23", "K16"],
    "2B": ["C13", "C17", "P13"],
}
IDR0139_PAPER_WELLS = ["J05", "I19", "G15", "O02", "B02", "N12", "L08", "L18", "H13", "E22"]
# Control-type labels used by the Crick notebook for idr0139 (companion_notebook_idr0139).
IDR0139_CONTROLS = {
    "Negative Control": "Untreated",
    "Neutral Control": "DMSO",
    "Stimulator Control": "Leptomycin b",
}


def normalize_well(well: str) -> str:
    """Zero-pad the column of a well name ("J5" -> "J05"), as the Crick notebooks do."""
    return f"{well[0]}{int(well[1:]):02d}"


def fetch(raw_dir: Path) -> None:
    """Download the LFS-resolved source files into ``raw_dir``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "idr0028-screenB-annotation.csv": "idr/idr0028-screenB-annotation.csv",
        "idr0139-screenA-annotation.csv": "idr/idr0139-screenA-annotation.csv",
    }
    for plate in IDR0028_PLATES:
        base = f"cell_profiler_outputs/idr0028/screenB/LM2_GEFGAP_ONTARGETPlus_{plate}"
        files[f"idr0028_{plate}_index.tsv"] = (
            f"idr/LM2_GEFGAP_ONTARGETPlus_{plate}_ImageIndex.ColumbusIDX.csv"
        )
        for table in ("Image", "Nuclei", "Cytoplasm"):
            files[f"idr0028_{plate}_{table}.csv"] = f"{base}/{table}.csv"
    for table in ("Image", "Nuclei", "Cytoplasm"):
        files[f"idr0139_{table}.csv"] = f"cell_profiler_outputs/idr0139/{table}.csv"
    for name, path in files.items():
        target = raw_dir / name
        if not target.exists():
            urllib.request.urlretrieve(f"{MEDIA}/{path}", target)
        verify(target)


def verify(path: Path) -> None:
    """Raise if a source file does not match the pinned checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != SOURCE_SHA256[path.name]:
        raise ValueError(
            f"{path} does not match the pinned source (SHA-256 {digest.hexdigest()}); "
            "delete it and download again"
        )


def _objects(raw_dir: Path, prefix: str, image_numbers) -> pd.DataFrame:
    """Nuclei joined one-to-one with their cytoplasm; all intensity features, prefixed."""
    nuclei = pd.read_csv(raw_dir / f"{prefix}_Nuclei.csv")
    nuclei = nuclei[nuclei["ImageNumber"].isin(image_numbers)]
    intensity = [c for c in nuclei.columns if c.startswith("Intensity_")]
    nuclei = nuclei[["ImageNumber", "ObjectNumber", "Location_Center_X", "Location_Center_Y",
                     *intensity]].rename(columns={c: f"Nuclei_{c}" for c in intensity})
    cytoplasm = pd.read_csv(raw_dir / f"{prefix}_Cytoplasm.csv")
    cytoplasm = cytoplasm[cytoplasm["ImageNumber"].isin(image_numbers)]
    intensity = [c for c in cytoplasm.columns if c.startswith("Intensity_")]
    cytoplasm = cytoplasm[["ImageNumber", "Parent_Nuclei", *intensity]].rename(
        columns={"Parent_Nuclei": "ObjectNumber", **{c: f"Cytoplasm_{c}" for c in intensity}}
    )
    return nuclei.merge(
        cytoplasm, on=["ImageNumber", "ObjectNumber"], how="left", validate="one_to_one"
    )


def convert_idr0028(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotation = pd.read_csv(raw_dir / "idr0028-screenB-annotation.csv")
    fov_tables, object_tables = [], []
    for plate in IDR0028_PLATES:
        index = pd.read_csv(raw_dir / f"idr0028_{plate}_index.tsv", sep="\t")
        index = index[index["Channel"] == index["Channel"].min()]
        fovs = pd.DataFrame(
            {
                "repeat": plate,
                "well": index["WellName"].to_numpy(),
                "image_file": index["sourcefilename"].to_numpy(),
                "field": index["Field"].to_numpy(),
                "acquisition_time_s": index["AbsoluteTime@s"].to_numpy(),
                "stage_x_um": index["PositionX@um"].to_numpy(),
                "stage_y_um": index["PositionY@um"].to_numpy(),
                "area_um2": (
                    index["ImageWidth"] * index["ImageResolutionX@um"]
                    * index["ImageHeight"] * index["ImageResolutionY@um"]
                ).to_numpy(),
            }
        )
        wells = annotation[annotation["Plate"] == f"LM2_ONTARGETPlus_{plate}"].rename(
            columns={"Well": "well"}
        )
        # Condition labels as in the Crick idr0028 notebook: gene symbol, with
        # untargeted (negative-control) wells labelled MOCK.
        wells = wells.assign(
            condition=wells["Gene Symbol"].fillna("MOCK"),
            control_type=wells["Control Type"],
            quality_control=wells["Quality Control"],
        )[["well", "condition", "control_type", "quality_control"]]
        fovs = fovs[fovs["well"].isin(IDR0028_PAPER_WELLS[plate])]
        fovs = fovs.merge(wells, on="well", how="left", validate="many_to_one")
        # Acquisition order within a well: timestamp, ties broken by field number.
        fovs = fovs.sort_values(["well", "acquisition_time_s", "field"], kind="stable")
        fovs["acquisition_order"] = fovs.groupby("well").cumcount() + 1

        images = pd.read_csv(
            raw_dir / f"idr0028_{plate}_Image.csv", usecols=["ImageNumber", "Metadata_FileLocation"]
        )
        images["image_file"] = images["Metadata_FileLocation"].str.rsplit("/", n=1).str[-1]
        images = images[images["image_file"].isin(fovs["image_file"])]
        objects = _objects(raw_dir, f"idr0028_{plate}", images["ImageNumber"])
        objects = objects.merge(images[["ImageNumber", "image_file"]], on="ImageNumber",
                                validate="many_to_one")
        objects = objects.merge(fovs[["image_file", "condition", "repeat", "well"]],
                                on="image_file", validate="many_to_one")
        fov_tables.append(fovs)
        object_tables.append(objects)

    fovs = pd.concat(fov_tables, ignore_index=True)
    objects = pd.concat(object_tables, ignore_index=True).rename(
        columns={"ObjectNumber": "label_id", "Location_Center_X": "nucleus_x_px",
                 "Location_Center_Y": "nucleus_y_px"}
    )
    fov_cols = ["condition", "repeat", "well", "image_file", "field", "acquisition_order",
                "acquisition_time_s", "stage_x_um", "stage_y_um", "area_um2",
                "control_type", "quality_control"]
    feature_cols = [c for c in objects.columns if c.startswith(("Nuclei_", "Cytoplasm_"))]
    object_cols = ["condition", "repeat", "well", "image_file", "label_id",
                   "nucleus_x_px", "nucleus_y_px", *feature_cols]
    return fovs[fov_cols], objects[object_cols]


def convert_idr0139(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotation = pd.read_csv(raw_dir / "idr0139-screenA-annotation.csv")
    wells = annotation[annotation["Plate"] == IDR0139_PLATE].rename(columns={"Well": "well"})
    wells = wells.assign(well=wells["well"].map(normalize_well))
    control = wells["Control Type"].map(IDR0139_CONTROLS)
    wells = wells.assign(
        condition=control.fillna(wells["Proprietary Compound"]).fillna("unannotated"),
        control_type=wells["Control Type"],
        quality_control=wells["Quality Control"],
    )[["well", "condition", "control_type", "quality_control"]]

    images = pd.read_csv(
        raw_dir / "idr0139_Image.csv",
        usecols=["ImageNumber", "FileName_DNA", "Metadata_Well", "Metadata_Field"],
    )
    fovs = pd.DataFrame(
        {
            "repeat": str(IDR0139_PLATE),
            "well": images["Metadata_Well"],
            "image_file": images["FileName_DNA"],
            "field": images["Metadata_Field"],
            # No timestamps or stage positions in the source: field number is the
            # only available proxy for acquisition order.
            "acquisition_order": images["Metadata_Field"],
        }
    ).merge(wells, on="well", how="left", validate="many_to_one")
    keep = fovs["well"].isin(IDR0139_PAPER_WELLS).to_numpy()
    fovs, images = fovs[keep], images[keep]

    objects = _objects(raw_dir, "idr0139", images["ImageNumber"])
    objects = objects.merge(images[["ImageNumber", "FileName_DNA"]], on="ImageNumber",
                            validate="many_to_one").rename(columns={"FileName_DNA": "image_file"})
    objects = objects.merge(fovs[["image_file", "condition", "repeat", "well"]],
                            on="image_file", validate="many_to_one")
    objects = objects.rename(
        columns={"ObjectNumber": "label_id", "Location_Center_X": "nucleus_x_px",
                 "Location_Center_Y": "nucleus_y_px"}
    )
    feature_cols = [c for c in objects.columns if c.startswith(("Nuclei_", "Cytoplasm_"))]
    fov_cols = ["condition", "repeat", "well", "image_file", "field", "acquisition_order",
                "control_type", "quality_control"]
    object_cols = ["condition", "repeat", "well", "image_file", "label_id",
                   "nucleus_x_px", "nucleus_y_px", *feature_cols]
    return fovs[fov_cols], objects[object_cols]


def main(raw_dir: Path, out_dir: Path) -> None:
    fetch(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, convert in (("idr0028_screenB", convert_idr0028), ("idr0139", convert_idr0139)):
        fovs, objects = convert(raw_dir)
        fovs.to_csv(out_dir / f"{name}_fovs.csv.gz", index=False, float_format=FLOAT_FORMAT)
        objects.to_csv(out_dir / f"{name}_nuclei.csv.gz", index=False, float_format=FLOAT_FORMAT)
        print(f"{name}: {len(fovs)} fields, {len(objects)} nuclei")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
