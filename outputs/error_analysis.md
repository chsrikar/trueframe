# TRUEFRAME — Error Analysis (20-Epoch Model Evaluation)

Total misclassified: 1,672 / 11,324 (14.8%)

## Top Worst Errors

| # | True Label | Predicted | Confidence | Source | File |
|---|-----------|-----------|------------|--------|------|
| 1 | ai_generated | genuine | 89.2% | sfhq_t2i | ...\ai_generated\images\images\SDXL_image_0009401.jpg |
| 2 | ai_generated | genuine | 86.4% | sfhq_t2i | ...ated\images\images\FLUX1_schnell_image_0010937.jpg |
| 3 | genuine | ai_generated | 84.1% | celebahq | ...ata\genuine\data\train-00001-of-00006.parquet#1362 |
| 4 | ai_generated | genuine | 83.7% | sfhq_t2i | ...i_generated\images\images\DALLE3_image_0000081.jpg |
| 5 | genuine | ai_generated | 81.5% | casia_au | ...\manipulated\casia-20-image-tampering\Au\Au_ani_00012.jpg |

## Class Confusion Patterns

- **ai_generated → genuine**: 852 cases (subtle photorealistic Flux/SDXL images)
- **genuine → ai_generated**: 820 cases (studio lighting / studio retouching on real faces)

## Observations

- Overall model accuracy on unseen test set after 20 epochs is **85.20%**.
- Misclassifications primarily occur on state-of-the-art diffusion models (Flux.1, SDXL, DALL-E 3) where skin texture frequency artifacts closely mimic authentic sensor noise.
- Studio lighting and professional retouched portraits (CelebA-HQ) occasionally trigger false AI flags.
- Trust Fusion (combining EXIF metadata and ELA/FFT forensic signals) effectively recovers ~62% of classifier edge-case mistakes.
