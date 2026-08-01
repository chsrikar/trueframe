# TRUEFRAME Model Card — EfficientNet-B0 Classifier

## Model Overview
- **Architecture**: EfficientNet-B0 (Pretrained ImageNet-1K backbone + 2-class Linear Head)
- **Target Classes**: `genuine` (0), `ai_generated` (1)
- **Task**: Deepfake & AI Image Detection (Binary Classification)

## Dataset & Manifest Breakdown
- **Source Breakdown**:
  - `celebahq` (Genuine): 19,646
  - `cifake` (AI-Generated): 0
  - `casia_tp` (Manipulated): 0
  - `casia_au` (Genuine): 5,197
- **Total Training Samples**: 52,843
- **Total Validation Samples**: 11,324

## Training Hyperparameters
- **Optimizer**: AdamW
- **Phase 1 LR**: 0.001 (Frozen backbone, 3 epochs)
- **Phase 2 LR**: 0.0001 (Fine-tuning top 2 blocks, Cosine Scheduler)
- **Batch Size**: 16
- **Sampler**: WeightedRandomSampler (inverse class frequency)
- **Loss Function**: CrossEntropyLoss (with optional class weights)

## Validation Performance Results
- **Best Validation Loss**: 0.2003
- **Best Validation Accuracy**: 99.97%
- **Best Validation Macro F1**: 0.9997

### Per-Class Precision & Recall:
- **Genuine**: Precision = 0.9996, Recall = 0.9998
- **Ai_generated**: Precision = 0.9998, Recall = 0.9997
