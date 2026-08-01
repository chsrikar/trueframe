"""Quick verification of TrueframeDataset after parquet LRU cache fix."""
import pandas as pd
from dataset import TrueframeDataset, IDX_TO_LABEL

print("=== Re-testing TrueframeDataset after parquet LRU cache fix ===")
df = pd.read_csv(r"D:\demo\manifest_train.csv")

# 5 parquet + 5 file-based samples
parquet_rows = df[df["filepath"].str.contains("#")].sample(5, random_state=1)
non_parquet_rows = df[~df["filepath"].str.contains("#")].sample(5, random_state=1)
test_df = pd.concat([parquet_rows, non_parquet_rows]).reset_index(drop=True)

dataset = TrueframeDataset(test_df, is_train=True)
print(f"Mini-dataset: {len(dataset)} samples\n")

all_ok = True
for i in range(len(dataset)):
    try:
        img, lbl = dataset[i]
        fp = test_df.iloc[i]["filepath"]
        kind = "parquet" if "#" in fp else "file   "
        assert img.shape == (3, 256, 256), f"Bad shape: {img.shape}"
        print(f"  [{i+1:02d}] {kind} | {IDX_TO_LABEL[lbl]:<15} | {tuple(img.shape)}")
    except Exception as e:
        print(f"  [{i+1:02d}] ERROR: {e}")
        all_ok = False

print()
print("RESULT:", "PASSED" if all_ok else "FAILED")
