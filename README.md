# TRUEFRAME

TRUEFRAME is a research prototype that classifies face images as `genuine` or `ai_generated`. It does not detect faces before classification, so non-face uploads are out of distribution. Its current datasets do **not** include manipulated/deepfake examples, so it does not support general deepfake, image-editing, or authenticity claims.

## Important evaluation limits

- The synthetic class currently comes from the SFHQ-T2I dataset. Synthetic generator families are separated across training, validation, and test by `build_manifest.py`.
- Genuine photos are randomly split within source. Identity disjointness is not established.
- Since AI examples come from one source dataset family, these results do not establish cross-dataset or universal generalization. Add independently collected datasets for both classes before making those claims.
- `evaluate.py` reports a source-only baseline. If that baseline is strong, the data itself makes the label predictable from source and the classifier score must be interpreted cautiously; the present dataset mix is known to have this risk.
- Softmax confidence is uncalibrated. The optional metadata/artifact `trust_score` is a hand-weighted heuristic, not a probability or validated forensic conclusion.
- Existing files in `outputs/` are historical local artifacts and must not be quoted as results for the new generator-held-out protocol. Regenerate manifests, train a new checkpoint, and run test evaluation.

## Run on Google Colab with a T4

1. Open `colab_train.ipynb` in Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Keep the GitHub clone cell pointed at `https://github.com/chsrikar/trueframe.git`.
3. Put the data folder in Google Drive with this structure:

   ```text
   trueframe-data/
   ├── genuine/data/*.parquet       # CelebA-HQ rows (image column)
   ├── casia/CASIA2/Au/              # optional genuine CASIA images
   └── shfq-t2i/
       ├── images/images/*.jpg
       └── SFHQ_T2I_dataset.csv      # includes image_filename and model_used
   ```

   Obtain each dataset from its official or authorized source and follow its license. The repository does not contain the datasets or trained weights.
4. Run the cells in order. The notebook builds manifests, trains on the T4, evaluates the held-out test manifest, and stores manifests, checkpoints, and reports in Drive so they survive a runtime reset.

The manifest builder requires at least three synthetic generator families. It reserves distinct families for validation and test and raises an error if that cannot be done. Do not replace this failure with a random split and then describe the result as generator generalization.

## Local setup

Use Python 3.10–3.12. Install a PyTorch and torchvision pair compatible with your hardware from the official PyTorch install selector, then install the remaining packages:

```bash
pip install -r requirements.txt
python build_manifest.py
python train.py --train-manifest manifest_train.csv --val-manifest manifest_val.csv
python evaluate.py
```

Set `TRUEFRAME_DATA_DIR`, `TRUEFRAME_MANIFEST_DIR`, `TRUEFRAME_CHECKPOINT_DIR`, and `TRUEFRAME_OUTPUT_DIR` to use data and run artifacts outside the repository. Manifest paths are absolute, so rebuild manifests on the machine that will train.

## Web demo

After training a checkpoint, install the dependencies, launch `python api_server.py`, then launch the Next.js app under `kj/kji-saa-s-landing-page/brutalist-ai-saa-s-landing-page`. The interface is a demonstration; sample cards are illustrative and not model evaluation results.
