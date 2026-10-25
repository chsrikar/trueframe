import os
import sys
import glob
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ensure UTF-8 output encoding for terminal
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DATA_DIR = Path(os.environ.get("TRUEFRAME_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("TRUEFRAME_MANIFEST_DIR", PROJECT_ROOT)).expanduser().resolve()

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


def _generator_from_name(name: str) -> str:
    """Return a stable generator family name from an SFHQ filename/model value."""
    import re
    value = Path(str(name)).stem.upper().replace("_IMAGE", "").replace("-IMAGE", "")
    value = re.sub(r"[_\-]?\d+$", "", value)
    value = re.sub(r"[^A-Z0-9]+", "_", value).strip("_")
    aliases = {
        "DALLE": "dalle3", "DALLE3": "dalle3", "DALL_E_3": "dalle3",
        "FLUX": "flux", "FLUX1": "flux", "FLUX1_DEV": "flux",
        "FLUX1_SCHNELL": "flux", "SDXL": "sdxl", "STABLE_DIFFUSION_XL": "sdxl",
    }
    for key, normalized in aliases.items():
        if value == key or value.startswith(key + "_"):
            return normalized
    return value.lower() or "unknown"

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
                'source': 'celebahq',
                'generator': 'camera_photo'
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
                    records.append({'filepath': str(p.resolve()), 'label': 'ai_generated', 'source': 'cifake', 'generator': 'cifake'})
                    
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
                    records.append({'filepath': str(p.resolve()), 'label': 'genuine', 'source': 'casia_au', 'generator': 'camera_photo'})
                    
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
                'source': 'tiny_genimage_real' if lbl == 0 else 'tiny_genimage_fake',
                'generator': 'camera_photo' if lbl == 0 else 'tiny_genimage'
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
                model_col = next((c for c in meta_df.columns if c.lower() in ("model_used", "model", "generator")), None)
                for _, meta_row in meta_df.iterrows():
                    fname = meta_row[fn_col]
                    base_fname = Path(str(fname).strip()).name
                    if base_fname in disk_images:
                        generator = _generator_from_name(meta_row[model_col] if model_col else base_fname)
                        records_from_csv.append({
                            'filepath': disk_images[base_fname],
                            'label': 'ai_generated',
                            'source': 'sfhq_t2i',
                            'generator': generator
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
                'source': 'sfhq_t2i',
                'generator': _generator_from_name(Path(path_str).name)
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


def split_records(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Create a generator-held-out test and validation split for synthetic images.

    Real images are stratified by source into the three partitions. With only one
    synthetic dataset, this is a generator generalization stress test, not a
    fully independent cross-dataset study; the model card states that limitation.
    """
    df = df.copy()
    if "generator" not in df:
        df["generator"] = df["source"]
    real = df[df.label == "genuine"].copy()
    synthetic = df[df.label == "ai_generated"].copy()
    if real.empty or synthetic.empty:
        raise ValueError("Both genuine and ai_generated examples are required.")

    real_train, real_holdout = train_test_split(
        real, test_size=0.30, random_state=seed, stratify=real.source
    )
    real_val, real_test = train_test_split(
        real_holdout, test_size=0.50, random_state=seed, stratify=real_holdout.source
    )

    generators = sorted(synthetic.generator.astype(str).unique())
    if len(generators) < 3:
        raise ValueError(
            f"Generator-held-out splitting requires at least 3 synthetic generator groups; found {generators}. "
            "Add generator metadata or more generators before making generalization claims."
        )
    def preferred_group(preference: str, available: list[str]) -> str:
        return next((g for g in available if preference in g.lower()), available[0])

    val_generator = preferred_group("sdxl", generators)
    remaining = [g for g in generators if g != val_generator]
    test_generator = preferred_group("dalle", remaining)
    if test_generator == val_generator:
        raise ValueError("Validation and test generator groups must be distinct.")
    train_generators = [g for g in generators if g not in (val_generator, test_generator)]
    if not train_generators:
        raise ValueError("At least one synthetic generator must remain for training.")

    syn_train = synthetic[synthetic.generator.isin(train_generators)]
    syn_val = synthetic[synthetic.generator == val_generator]
    syn_test = synthetic[synthetic.generator == test_generator]
    train_df = pd.concat([real_train, syn_train], ignore_index=True).sample(frac=1, random_state=seed)
    val_df = pd.concat([real_val, syn_val], ignore_index=True).sample(frac=1, random_state=seed)
    test_df = pd.concat([real_test, syn_test], ignore_index=True).sample(frac=1, random_state=seed)
    protocol = {
        "name": "generator_holdout_stress_test",
        "validation_generators": [val_generator],
        "test_generators": [test_generator],
        "training_generators": train_generators,
        "real_image_split": "stratified random split within source; identity-disjointness is not established",
        "limitation": "AI class has one source dataset; this is not a cross-dataset or universal detector evaluation."
    }
    return train_df, val_df, test_df, protocol

def build_and_save_manifests():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []

    all_records.extend(scan_celebahq())      # CelebA-HQ-256 (Genuine faces)
    # scan_cifake() intentionally removed — replaced by SFHQ-T2I below.
    # CIFAKE is CIFAR-10 objects (non-face); SFHQ-T2I supplies synthetic faces.
    all_records.extend(scan_casia())         # CASIA v2 authentic images only
    # Tiny-GenImage is intentionally excluded: it contains non-face categories,
    # which would change the problem from face-image detection to source recognition.
    all_records.extend(scan_sfhq_t2i())     # SFHQ-T2I (replaces CIFAKE)

    df_full = pd.DataFrame(all_records)
    print(f"Total manifest records: {len(df_full):,}\n")
    
    # Save full manifest
    full_path = OUTPUT_DIR / "manifest_full.csv"
    df_full.to_csv(full_path, index=False)
    print(f"Saved full manifest to `{full_path}`")
    
    # Keep synthetic generator families disjoint across partitions.
    train_df, val_df, test_df, protocol = split_records(df_full, seed=42)
    
    train_path = OUTPUT_DIR / "manifest_train.csv"
    val_path = OUTPUT_DIR / "manifest_val.csv"
    test_path = OUTPUT_DIR / "manifest_test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    (OUTPUT_DIR / "split_protocol.json").write_text(
        pd.Series(protocol).to_json(indent=2), encoding="utf-8"
    )
    
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
