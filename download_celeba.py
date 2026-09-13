import os
import sys
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Project base and target data directory
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# Point all Hugging Face caches to C: drive data folder (avoid filling D: drive)
hf_cache = data_dir / ".hf_cache"
hf_cache.mkdir(parents=True, exist_ok=True)
hf_hub_cache = hf_cache / "hub"
hf_hub_cache.mkdir(parents=True, exist_ok=True)
hf_ds_cache = hf_cache / "datasets"
hf_ds_cache.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(hf_cache.resolve())
os.environ["HF_HUB_CACHE"] = str(hf_hub_cache.resolve())
os.environ["HF_DATASETS_CACHE"] = str(hf_ds_cache.resolve())
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Import datasets after setting cache paths
from datasets import load_dataset

def main():
    print("=" * 65)
    print("CelebA-HQ-256x256 Dataset Downloader")
    print("=" * 65)

    print(f"[*] Project data folder: {data_dir.resolve()} (Exists: {data_dir.exists()})")
    print(f"[*] Cache location:      {hf_cache.resolve()}")

    dest_dir = data_dir / "celeba-hq-256x256"
    parquet_dir = data_dir / "genuine" / "data"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load dataset from Hugging Face
    print("\n[*] Downloading dataset 'korexyz/celeba-hq-256x256' using load_dataset...")
    print("    (~2.8 GB total across 30,000 images: 28,000 train + 2,000 val)")
    
    ds = load_dataset("korexyz/celeba-hq-256x256")
    print("\n[+] Successfully loaded dataset:")
    print(ds)

    # 3. Save to datafolder (Hugging Face Dataset format)
    print(f"\n[*] Saving dataset to disk at: {dest_dir.resolve()} ...")
    ds.save_to_disk(str(dest_dir))
    print(f"[+] Saved Arrow format dataset to: {dest_dir}")

    # 4. Save Parquet shards to data/genuine/data for TrueFrame pipeline compatibility
    print(f"\n[*] Exporting Parquet shards to: {parquet_dir.resolve()} ...")
    train_parquet_path = parquet_dir / "train-00000-of-00001.parquet"
    ds["train"].to_parquet(str(train_parquet_path))
    print(f"[+] Saved train Parquet ({len(ds['train']):,} rows) to: {train_parquet_path.name}")

    if "validation" in ds:
        val_parquet_path = parquet_dir / "validation-00000-of-00001.parquet"
        ds["validation"].to_parquet(str(val_parquet_path))
        print(f"[+] Saved validation Parquet ({len(ds['validation']):,} rows) to: {val_parquet_path.name}")

    print("\n" + "=" * 65)
    print(" Download & Save Complete!")
    print(f" - Hugging Face Dataset: {dest_dir.resolve()}")
    print(f" - TrueFrame Parquet:    {parquet_dir.resolve()}")
    print("=" * 65)

if __name__ == "__main__":
    main()
