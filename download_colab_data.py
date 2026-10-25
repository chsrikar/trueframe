"""Download the project datasets in Colab's temporary disk and prepare a capped sample.

The SFHQ-T2I archive is large (about 23 GB compressed). Only the capped, stratified
sample is extracted. CelebA-HQ is hosted by a third-party Hugging Face mirror whose
dataset card currently does not state a license; review its terms before enabling it.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


DATA_DIR = Path(os.environ.get("TRUEFRAME_DATA_DIR", "/content/trueframe-data"))
SYNTHETIC_REPO = "bitmind/SyntheticFacesHQ"
SYNTHETIC_ARCHIVE = "SyntheticFacesHighQuality-T2I.zip"
REAL_REPO = "saitsharipov/CelebA-HQ"
REAL_FILES = [
    "data/train-00000-of-00003-30fbdf94a5d8e722.parquet",
    "data/train-00001-of-00003-74b418aa7a721747.parquet",
    "data/train-00002-of-00003-4f9459a488896f53.parquet",
]


def _column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    return next((c for c in columns if any(k in c.lower() for k in candidates)), None)


def _prepare_synthetic(cap: int = 40_000) -> None:
    from build_manifest import _generator_from_name

    target = DATA_DIR / "sfhq_t2i"
    images_dir = target / "images" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    archive_path = Path("/content") / SYNTHETIC_ARCHIVE

    print(f"Downloading {SYNTHETIC_REPO}/{SYNTHETIC_ARCHIVE} (~22.9 GB).")
    hf_hub_download(
        repo_id=SYNTHETIC_REPO,
        repo_type="dataset",
        filename=SYNTHETIC_ARCHIVE,
        local_dir="/content",
        token=False,
    )

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        csv_members = [m for m in members if Path(m.filename).name.lower() == "sfhq_t2i_dataset.csv"]
        if not csv_members:
            csv_members = [m for m in members if m.filename.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError("The downloaded SFHQ-T2I archive contains no metadata CSV.")
        csv_info = max(csv_members, key=lambda m: m.file_size)
        metadata = pd.read_csv(archive.open(csv_info), low_memory=False)
        filename_col = _column(list(metadata.columns), ("filename", "image", "file", "path", "name"))
        model_col = next((c for c in metadata.columns if c.lower() in ("model_used", "model", "generator")), None)
        if filename_col is None or model_col is None:
            raise RuntimeError(f"SFHQ-T2I CSV must contain an image filename and model column; found {list(metadata.columns)}")

        image_members = {
            Path(m.filename).name: m
            for m in members
            if m.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        }
        metadata["image_filename"] = metadata[filename_col].map(lambda v: Path(str(v).strip()).name)
        metadata["_archive_member"] = metadata["image_filename"].map(image_members.get)
        metadata = metadata[metadata["_archive_member"].notna()].copy()
        metadata["_generator_family"] = metadata[model_col].map(_generator_from_name)
        families = sorted(metadata["_generator_family"].dropna().unique().tolist())
        if len(families) < 3:
            raise RuntimeError(f"Need at least 3 synthetic generator families; found {families}.")

        # Stratify the cap so small generator families such as DALL-E are retained.
        if len(metadata) > cap:
            selected = []
            grouped = metadata.groupby("_generator_family", sort=True, group_keys=False)
            for _, group in grouped:
                quota = max(1, round(cap * len(group) / len(metadata)))
                selected.append(group.sample(n=min(quota, len(group)), random_state=42))
            metadata = pd.concat(selected).sample(frac=1, random_state=42).head(cap)

        print(f"Extracting {len(metadata):,} images from {len(families)} generator families.")
        out_rows = []
        for _, row in metadata.iterrows():
            basename = row["image_filename"]
            member = row["_archive_member"]
            if not basename or member is None:
                continue
            destination = images_dir / basename
            with archive.open(member) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            out_rows.append({"image_filename": basename, "model_used": row[model_col]})
        pd.DataFrame(out_rows).to_csv(target / "SFHQ_T2I_dataset.csv", index=False)

    archive_path.unlink(missing_ok=True)
    print(f"Prepared {len(out_rows):,} synthetic face images in {images_dir}")


def _prepare_genuine() -> None:
    target = DATA_DIR / "genuine"
    target.mkdir(parents=True, exist_ok=True)
    for filename in REAL_FILES:
        local_path = Path(hf_hub_download(
            repo_id=REAL_REPO,
            repo_type="dataset",
            filename=filename,
            local_dir=target,
            token=False,
        ))
        print(f"Prepared {local_path.relative_to(DATA_DIR)}")


if __name__ == "__main__":
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        raise RuntimeError(f"{DATA_DIR} is not empty. Use a clean folder or clear its contents before downloading.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(DATA_DIR).free
    if free_bytes < 32 * 1024**3:
        raise RuntimeError(
            f"Need at least 32 GiB free on Colab's temporary disk for the SFHQ archive and extracted sample; "
            f"only {free_bytes / 1024**3:.1f} GiB is available."
        )
    _prepare_genuine()
    _prepare_synthetic()
    print(f"Dataset download/preparation complete: {DATA_DIR}")
