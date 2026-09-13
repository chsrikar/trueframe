import os
import sys
import glob
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ensure UTF-8 output encoding for terminal
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

# ---------------------------------------------------------------------------
# SFHQ-T2I subsampling cap
# CIFAKE has been REPLACED by SFHQ-T2I for the ai_generated class.
# Rationale: CIFAKE = low-res CIFAR-10 objects (not faces, not modern AI).
#            SFHQ-T2I = 1024×1024 photorealistic faces from Flux/SDXL/DALL-E 3.
#
# Genuine class size: ~24,843
# Target ratio ai_generated / genuine ≈ 1.5–2.0×  → cap = 40,000 ≈ 1.6×
# Raise to 50,000 if you want a higher ratio, lower to 25,000 for 1.0×.
# ---------------------------------------------------------------------------
SFHQ_T2I_CAP = 40_000

def is_valid_image_file(file_path):
    """Fast check: exists, supported extension, non-empty (>0 bytes)."""
    try:
        p = Path(file_path)
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            return p.stat().st_size > 0
    except Exception:
        pass
    return False

def scan_celebahq():
    """Scan CelebA-HQ-256 Parquet shards from data/genuine/data/*.parquet."""
    print("Scanning CelebA-HQ-256 Parquet Dataset...")
    import pyarrow.parquet as pq
    parquet_files = sorted(glob.glob(str(BASE_DATA_DIR / "genuine" / "data" / "train-*.parquet")))
    records = []

    for pf in parquet_files:
        pf_abs = str(Path(pf).resolve())
        num_rows = pq.ParquetFile(pf_abs).metadata.num_rows
        for idx in range(num_rows):
            records.append({
                'filepath': f"{pf_abs}#{idx}",
                'label': 'genuine',
                'source': 'celebahq'
            })

    print(f"CelebA-HQ valid parquet records: {len(records):,}\n")
    return records

def scan_cifake():
    print("Scanning CIFAKE Dataset...")
    src_dir = BASE_DATA_DIR / "ai_generated" / "cifake-real-and-ai-generated-synthetic-images"
    records = []
    
    for root, dirs, files in os.walk(src_dir):
        parts = [p.upper() for p in Path(root).parts]
        # Only include FAKE subfolders
        if "FAKE" in parts and "REAL" not in parts:
            for f in files:
                p = Path(root) / f
                if is_valid_image_file(p):
                    records.append({'filepath': str(p.resolve()), 'label': 'ai_generated', 'source': 'cifake'})
                    
    print(f"CIFAKE valid records: {len(records):,}\n")
    return records

def scan_casia():
    print("Scanning CASIA v2 Dataset (Authentic/Genuine only)...")
    records = []
    casia_candidates = [
        BASE_DATA_DIR / "casia",
        BASE_DATA_DIR / "casia" / "CASIA2",
        BASE_DATA_DIR / "manipulated" / "casia-20-image-tampering-detection-dataset"
    ]
    src_dir = None
    for cand in casia_candidates:
        if cand.exists():
            src_dir = cand
            break

    if src_dir is None:
        print("  ⚠️  CASIA directory not found — skipping.\n")
        return []

    for root, dirs, files in os.walk(src_dir):
        folder_name = Path(root).name
        if folder_name == "Au":
            for f in files:
                p = Path(root) / f
                if is_valid_image_file(p):
                    records.append({'filepath': str(p.resolve()), 'label': 'genuine', 'source': 'casia_au'})
                    
    print(f"CASIA v2 valid records (genuine only): {len(records):,}\n")
    return records

def scan_tiny_genimage():
    """Scan Tiny-GenImage Parquet — skipped gracefully if folder not present."""
    print("Scanning Tiny-GenImage Parquet Dataset...")
    parquet_files = sorted(glob.glob(str(BASE_DATA_DIR / "Tiny-GenImage" / "data" / "*.parquet")))
    if not parquet_files:
        print("  ⚠️  Tiny-GenImage folder not found — skipping.\n")
        return []
    records = []

    for pf in parquet_files:
        pf_abs = str(Path(pf).resolve())
        df = pd.read_parquet(pf_abs, columns=['label'])
        labels = df['label'].to_numpy()

        file_records = [
            {
                'filepath': f"{pf_abs}#{idx}",
                'label': 'genuine' if lbl == 0 else 'ai_generated',
                'source': 'tiny_genimage_real' if lbl == 0 else 'tiny_genimage_fake'
            }
            for idx, lbl in enumerate(labels)
        ]
        records.extend(file_records)

    print(f"Tiny-GenImage valid parquet records: {len(records):,}\n")
    return records


def scan_sfhq_t2i(cap: int = SFHQ_T2I_CAP, seed: int = 42):
    """
    Scan SFHQ-T2I (Synthetic Faces High Quality — Text2Image) dataset.

    Args:
        cap:  Maximum number of records to keep (subsampled randomly).
        seed: Random seed for reproducible subsampling.

    Returns:
        list[dict]  with keys: filepath, label, source
    """
    import random as _random

    # Locate the root directory and images directory
    possible_roots = [
        BASE_DATA_DIR / "shfq-t2i",
        BASE_DATA_DIR / "sfhq_t2i",
        BASE_DATA_DIR / "sfhq-t2i",
        BASE_DATA_DIR / "ai_generated" / "sfhq_t2i",
        BASE_DATA_DIR / "ai_generated"
    ]
    src_dir = None
    images_dir = None
    for cand in possible_roots:
        if cand.exists():
            nested = cand / "images" / "images"
            single = cand / "images"
            if nested.exists():
                src_dir = cand
                images_dir = nested
                break
            elif single.exists() and any(single.glob("*.jpg")):
                src_dir = cand
                images_dir = single
                break
            elif any(cand.glob("*.jpg")):
                src_dir = cand
                images_dir = cand
                break

    if images_dir is None:
        print(f"  ⚠️  SFHQ-T2I images not found in any standard path.\n")
        return []

    print(f"Scanning SFHQ-T2I dataset at `{images_dir}`...")

    # Fast inventory of valid images on disk using os.scandir
    print("  Indexing image directory...")
    disk_images = {}
    for entry in os.scandir(images_dir):
        if entry.is_file() and entry.name.lower().endswith(tuple(IMAGE_EXTENSIONS)):
            try:
                if entry.stat().st_size > 0:
                    disk_images[entry.name] = entry.path
            except Exception:
                pass
    print(f"  Indexed {len(disk_images):,} valid images on disk.")

    # Try to load the official CSV for metadata-driven ordering/validation
    csv_candidates = list(src_dir.glob("*.csv")) or list(images_dir.glob("*.csv"))
    records_from_csv = []

    if csv_candidates:
        csv_path = csv_candidates[0]
        print(f"  Found metadata CSV: {csv_path.name}")
        try:
            meta_df = pd.read_csv(csv_path, low_memory=False)
            fn_col = None
            for col in meta_df.columns:
                if any(kw in col.lower() for kw in ("filename", "image", "file", "path", "name")):
                    fn_col = col
                    break

            if fn_col is not None:
                print(f"  Using column `{fn_col}` as image filename.")
                for fname in meta_df[fn_col]:
                    base_fname = Path(str(fname).strip()).name
                    if base_fname in disk_images:
                        records_from_csv.append({
                            'filepath': disk_images[base_fname],
                            'label': 'ai_generated',
                            'source': 'sfhq_t2i'
                        })
                print(f"  CSV-validated images: {len(records_from_csv):,}")
        except Exception as e:
            print(f"  ⚠️  Failed to parse CSV ({e}). Falling back to directory index.")

    # Fallback to direct directory scan if CSV yielded no valid records
    if not records_from_csv:
        print("  Using direct disk image inventory...")
        for path_str in disk_images.values():
            records_from_csv.append({
                'filepath': path_str,
                'label': 'ai_generated',
                'source': 'sfhq_t2i'
            })
        print(f"  Directory index found: {len(records_from_csv):,} images")

    # --- Step 3: Stratified subsample by generator model ---
    # SFHQ-T2I images are named like:
    #   FLUX1_dev_image_000123.jpg
    #   SDXL_image_001234.jpg
    #   DALL-E_3_image_002345.jpg
    # We subsample proportionally per generator to preserve diversity.
    if len(records_from_csv) <= cap:
        final_records = records_from_csv
        print(f"  Total images ({len(final_records):,}) is at or below cap ({cap:,}) — keeping all.")
    else:
        # Group by generator prefix
        from collections import defaultdict
        buckets = defaultdict(list)
        for rec in records_from_csv:
            fname = Path(rec['filepath']).name
            # Extract prefix: everything before the first '_image_' or first digit run
            import re
            m = re.match(r'^([A-Za-z0-9\-\.]+?)[\_\-]?(?:image|img)?[\_\-]\d', fname)
            prefix = m.group(1) if m else "unknown"
            buckets[prefix].append(rec)

        print(f"  Generator buckets found: { {k: len(v) for k, v in buckets.items()} }")

        _random.seed(seed)
        final_records = []
        total = len(records_from_csv)
        for prefix, bucket in buckets.items():
            # Proportional quota per generator
            quota = max(1, round(cap * len(bucket) / total))
            sampled = _random.sample(bucket, min(quota, len(bucket)))
            final_records.extend(sampled)

        # Trim/top-up to exactly `cap` if rounding caused drift
        _random.shuffle(final_records)
        final_records = final_records[:cap]

        print(f"  Subsampled {len(final_records):,} images from {len(records_from_csv):,} "
              f"(cap={cap:,}, seed={seed}).")

    print(f"SFHQ-T2I valid records: {len(final_records):,}\n")
    return final_records

def build_and_save_manifests():
    all_records = []

    all_records.extend(scan_celebahq())      # CelebA-HQ-256 (Genuine faces)
    # scan_cifake() intentionally removed — replaced by SFHQ-T2I below.
    # CIFAKE is CIFAR-10 objects (low-res, non-face); SFHQ-T2I is a strict
    # upgrade: 1024×1024 photorealistic AI faces from Flux/SDXL/DALL-E 3.
    all_records.extend(scan_casia())         # CASIA v2 (Manipulated + Genuine)
    all_records.extend(scan_tiny_genimage()) # Tiny-GenImage (optional)
    all_records.extend(scan_sfhq_t2i())     # SFHQ-T2I (replaces CIFAKE)

    df_full = pd.DataFrame(all_records)
    print(f"Total verified dataset items: {len(df_full):,}\n")
    
    # Save full manifest
    full_path = OUTPUT_DIR / "manifest_full.csv"
    df_full.to_csv(full_path, index=False)
    print(f"Saved full manifest to `{full_path}`")
    
    # Stratified 70/15/15 split
    train_df, test_val_df = train_test_split(
        df_full, test_size=0.30, random_state=42, stratify=df_full['label']
    )
    val_df, test_df = train_test_split(
        test_val_df, test_size=0.50, random_state=42, stratify=test_val_df['label']
    )
    
    train_path = OUTPUT_DIR / "manifest_train.csv"
    val_path = OUTPUT_DIR / "manifest_val.csv"
    test_path = OUTPUT_DIR / "manifest_test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved train split ({len(train_df):,} rows) to `{train_path}`")
    print(f"Saved val split ({len(val_df):,} rows) to `{val_path}`")
    print(f"Saved test split ({len(test_df):,} rows) to `{test_path}`")
    
    # Summary Table
    print("\n" + "="*80)
    print("                    MANIFEST SUMMARY & CLASS BREAKDOWN")
    print("="*80)
    
    summary = df_full.groupby(['label', 'source']).size().reset_index(name='count')
    print(summary.to_string(index=False))
    print("-" * 80)
    
    class_totals = df_full['label'].value_counts()
    print("\nClass Totals:")
    for lbl, count in class_totals.items():
        pct = (count / len(df_full)) * 100
        print(f" - {lbl:<15}: {count:>8,} samples ({pct:.1f}%)")
    print("="*80)
    
    if 'manipulated' in class_totals:
        m_count = class_totals['manipulated']
        print(f"\n⚠️  CLASS IMBALANCE: 'manipulated' has only {m_count:,} samples "
              f"({m_count/len(df_full)*100:.1f}% of total).")
        print("  WeightedRandomSampler will compensate during training.\n")

    # --- Class ratio health check ---
    if 'ai_generated' in class_totals and 'genuine' in class_totals:
        ratio = class_totals['ai_generated'] / class_totals['genuine']
        if ratio > 3.0:
            print(f"⚠️  BALANCE WARNING: ai_generated is {ratio:.1f}× the genuine class.")
            print(f"   Consider lowering SFHQ_T2I_CAP (currently {SFHQ_T2I_CAP:,}) in build_manifest.py")
            print(f"   Recommended cap to reach 2× ratio: "
                  f"{max(0, 2 * class_totals['genuine'] - class_totals.get('cifake', class_totals['ai_generated'])):,}")
        elif ratio < 0.8:
            print(f"⚠️  BALANCE WARNING: ai_generated ({class_totals['ai_generated']:,}) "
                  f"is fewer than genuine ({class_totals['genuine']:,}). Consider adding more AI data.")
        else:
            print(f"✅  ai_generated / genuine ratio: {ratio:.2f}× (healthy)")

if __name__ == "__main__":
    build_and_save_manifests()
