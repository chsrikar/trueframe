import io
import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pyarrow.parquet as pq

import collections

# Label dictionary (2-class binary classification)
LABEL_TO_IDX = {
    "genuine": 0,
    "ai_generated": 1
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

# -----------------------------------------------------------------
# Row-group-level LRU parquet cache.
# Each parquet shard has ~5 row groups of ~426 rows × ~96 MB.
# We cache at most MAX_CACHED_GROUPS groups to cap RAM usage.
# -----------------------------------------------------------------
MAX_CACHED_GROUPS = 32   # Cache all ~30 FFHQ row groups (~2.7 GB RAM) — eliminates per-epoch disk re-reads
_PARQUET_METADATA: dict = {}          # path -> pyarrow ParquetFile metadata
_ROW_GROUP_CACHE: "collections.OrderedDict" = collections.OrderedDict()

def _get_parquet_metadata(parquet_path: str):
    """Cache the lightweight ParquetFile metadata (no data loaded)."""
    if parquet_path not in _PARQUET_METADATA:
        _PARQUET_METADATA[parquet_path] = pq.ParquetFile(parquet_path)
    return _PARQUET_METADATA[parquet_path]

def get_parquet_image_bytes(parquet_path: str, row_idx: int) -> bytes:
    """Fetch image bytes for a single row using row-group-level caching."""
    pf = _get_parquet_metadata(parquet_path)
    meta = pf.metadata

    # Locate the row group containing row_idx
    cumulative = 0
    rg_index = 0
    local_row = row_idx
    for rg in range(meta.num_row_groups):
        rg_rows = meta.row_group(rg).num_rows
        if cumulative + rg_rows > row_idx:
            rg_index = rg
            local_row = row_idx - cumulative
            break
        cumulative += rg_rows

    cache_key = (parquet_path, rg_index)

    if cache_key not in _ROW_GROUP_CACHE:
        # Evict oldest entry if at capacity
        if len(_ROW_GROUP_CACHE) >= MAX_CACHED_GROUPS:
            _ROW_GROUP_CACHE.popitem(last=False)
        # Load only this row group, only the 'image' column
        table = pf.read_row_group(rg_index, columns=['image'])
        _ROW_GROUP_CACHE[cache_key] = table
    else:
        # Move to most-recently-used end
        _ROW_GROUP_CACHE.move_to_end(cache_key)

    table = _ROW_GROUP_CACHE[cache_key]
    img_data = table['image'][local_row].as_py()

    if isinstance(img_data, dict) and 'bytes' in img_data:
        return img_data['bytes']
    elif isinstance(img_data, bytes):
        return img_data
    else:
        raise ValueError(f"Unexpected image structure in parquet at row {row_idx}")

class InMemJPEGCompress:
    """Augmentation: random JPEG compression quality in memory."""
    def __init__(self, min_quality=60, max_quality=100, prob=0.5):
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            quality = random.randint(self.min_quality, self.max_quality)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")
        return img

def get_transforms(is_train=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop((256, 256), scale=(0.75, 1.0)),
            InMemJPEGCompress(min_quality=50, max_quality=95, prob=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomAutocontrast(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))], p=0.35),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.35, scale=(0.02, 0.15))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

class TrueframeDataset(Dataset):
    def __init__(self, manifest_input, is_train=True):
        if isinstance(manifest_input, (str, os.PathLike)):
            self.df = pd.read_csv(manifest_input)
        elif isinstance(manifest_input, pd.DataFrame):
            self.df = manifest_input.reset_index(drop=True)
        else:
            raise ValueError("manifest_input must be a CSV file path or pandas DataFrame")
            
        self.is_train = is_train
        self.transform = get_transforms(is_train=is_train)
        
    def __len__(self):
        return len(self.df)

    def get_label(self, idx: int) -> tuple[str, int]:
        """Return (label_str, label_idx) for index *idx* without loading any image.

        Used by the stratified sanity-check to build per-class index buckets
        cheaply (pure DataFrame lookup, no I/O).
        """
        label_str = self.df.iloc[idx]['label']
        return label_str, LABEL_TO_IDX[label_str]

    def _load_image(self, filepath_str):
        if '#' in filepath_str:
            parquet_path, idx_str = filepath_str.split('#')
            row_idx = int(idx_str)
            img_bytes = get_parquet_image_bytes(parquet_path, row_idx)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            img = Image.open(filepath_str).convert("RGB")
        return img

    def __getitem__(self, idx, retries=0):
        if retries > 5:
            raise RuntimeError(f"Failed to load image after 5 retries. Check if dataset file paths exist on disk!")

        row = self.df.iloc[idx]
        filepath_str = row['filepath']
        label_str = row['label']
        label_idx = LABEL_TO_IDX[label_str]
        
        try:
            img = self._load_image(filepath_str)
            tensor_img = self.transform(img)
            return tensor_img, label_idx
        except Exception as e:
            fallback_idx = random.randint(0, len(self.df) - 1)
            print(f"⚠️ Warning: Failed to load index {idx} ({filepath_str}): {e}. Using fallback index {fallback_idx}.")
            return self.__getitem__(fallback_idx, retries=retries + 1)


def get_weighted_sampler(manifest_df_or_path):
    if isinstance(manifest_df_or_path, (str, os.PathLike)):
        df = pd.read_csv(manifest_df_or_path)
    else:
        df = manifest_df_or_path
        
    class_counts = df['label'].value_counts().to_dict()
    print(f"Dataset class distribution for sampler: {class_counts}")
    
    class_weights = {}
    for label_str, count in class_counts.items():
        class_weights[LABEL_TO_IDX[label_str]] = 1.0 / count
        
    sample_weights = [class_weights[LABEL_TO_IDX[label_str]] for label_str in df['label']]
    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    return sampler, class_weights

if __name__ == "__main__":
    # Test script locally
    train_manifest = r"D:\demo\manifest_train.csv"
    if os.path.exists(train_manifest):
        print("Testing TrueframeDataset on manifest_train.csv...")
        dataset = TrueframeDataset(train_manifest, is_train=True)
        print(f"Dataset length: {len(dataset):,}")
        img, lbl = dataset[0]
        print(f"Sample 0 shape: {img.shape}, label: {lbl} ({IDX_TO_LABEL[lbl]})")
        sampler, weights = get_weighted_sampler(train_manifest)
        print(f"Sampler initialized successfully with class weights: {weights}")
