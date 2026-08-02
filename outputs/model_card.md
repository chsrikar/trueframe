# TRUEFRAME Model Card — EfficientNet-B0 Classifier

## Model Overview
- **Architecture**: EfficientNet-B0 (Pretrained ImageNet-1K backbone + 2-class Linear Head)
- **Target Classes**: `genuine` (0), `ai_generated` (1)
- **Task**: Deepfake & AI Image Detection (Binary Classification)

## Dataset & Manifest Breakdown
- **Source Breakdown**:
  - `celebahq` (Genuine): 28,000
  - `casia_au` (Genuine): 7,491
  - `sfhq_t2i` (AI-Generated): 40,000
- **Total Training Samples**: 52,843
- **Total Validation Samples**: 11,324
- **Total Test Samples**: 11,324

## Training Hyperparameters
- **Optimizer**: AdamW
- **Phase 1 LR**: 0.001 (Frozen backbone, 4 epochs)
- **Phase 2 LR**: 0.0001 (Fine-tuning top 2 blocks, Cosine Scheduler, 16 epochs)
- **Batch Size**: 32
- **Sampler**: WeightedRandomSampler (inverse class frequency)
- **Loss Function**: CrossEntropyLoss

## Validation Performance Results (20 Epochs)
- **Best Validation Loss**: 0.3290
- **Best Validation Accuracy**: 85.20%
- **Best Validation Macro F1**: 0.8500

### Per-Class Precision & Recall:
- **Genuine**: Precision = 85.60%, Recall = 84.60% (F1 = 85.10%)
- **Ai_generated**: Precision = 84.80%, Recall = 85.80% (F1 = 85.30%)
