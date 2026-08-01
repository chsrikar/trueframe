# TRUEFRAME — Error Analysis (Worst Misclassifications)

Total misclassified: 4 / 11,324 (0.0%)

## Top Worst Errors

| # | True Label | Predicted | Confidence | Source | File |
|---|-----------|-----------|------------|--------|------|
| 1 | ai_generated | genuine | 93.8% | sfhq_t2i | ...\ai_generated\images\images\SDXL_image_0009401.jpg |
| 2 | ai_generated | genuine | 71.7% | sfhq_t2i | ...ated\images\images\FLUX1_schnell_image_0010937.jpg |
| 3 | genuine | ai_generated | 71.0% | celebahq | ...ata\genuine\data\train-00001-of-00006.parquet#1362 |
| 4 | ai_generated | genuine | 69.5% | sfhq_t2i | ...i_generated\images\images\DALLE3_image_0000081.jpg |

## Class Confusion Patterns

- **ai_generated → genuine**: 3 cases
- **genuine → ai_generated**: 1 cases

## Observations

- Overall model accuracy on unseen test set is 99.96% (only 4 misclassifications out of 11,324 samples).
- `genuine ↔ ai_generated` confusions are extremely rare and occur only on subtle edge-case photorealistic faces.
- High-confidence misclassifications suggest domain gaps or subtle lighting/style variations in synthetic face generators.
- Consider cross-referencing these cases with metadata/artifact signals to check whether the forensic modules disagree with the classifier.
