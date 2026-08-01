import os
import sys
import random
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from PIL import Image

from dataset import TrueframeDataset, LABEL_TO_IDX, IDX_TO_LABEL
from train import build_model

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

TEST_MANIFEST_PATH = Path(r"D:\demo\manifest_test.csv")
MODEL_PATH = Path(r"D:\demo\checkpoints\best_model.pth")

def verify_test_samples(num_samples=5):
    print("="*80)
    print("                PART D: TEST MANIFEST INFERENCE VERIFICATION")
    print("="*80)
    
    if not TEST_MANIFEST_PATH.exists():
        print(f"❌ Error: Test manifest `{TEST_MANIFEST_PATH}` not found!")
        return

    if not MODEL_PATH.exists():
        print(f"⚠️ Checkpoint `{MODEL_PATH}` not found. Attempting fallback to latest checkpoint...")
        fallback_path = Path(r"D:\demo\checkpoints\latest_checkpoint.pth")
        if fallback_path.exists():
            print(f"Found latest checkpoint `{fallback_path}`.")
            ckpt = torch.load(fallback_path, map_location='cpu')
            model_weights = ckpt['model_state_dict']
        else:
            print("❌ No model checkpoint found for verification. Please run train.py first.")
            return
    else:
        model_weights = torch.load(MODEL_PATH, map_location='cpu')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_df = pd.read_csv(TEST_MANIFEST_PATH)
    model = build_model(num_classes=2)
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()
    
    # Stratified or random sample
    sampled_df = test_df.sample(n=min(num_samples, len(test_df)), random_state=42).reset_index(drop=True)
    test_dataset = TrueframeDataset(sampled_df, is_train=False)
    
    print(f"\nEvaluating {len(sampled_df)} test samples...\n")
    print(f"{'Sample #':<10} | {'True Label':<15} | {'Predicted Label':<15} | {'Confidence':<12} | {'Probabilities (Genuine / AI)':<45}")
    print("-" * 105)
    
    correct_count = 0
    with torch.no_grad():
        for i in range(len(test_dataset)):
            tensor_img, true_idx = test_dataset[i]
            input_tensor = tensor_img.unsqueeze(0).to(device)
            
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            pred_idx = int(torch.argmax(logits, dim=1).item())
            pred_label = IDX_TO_LABEL[pred_idx]
            true_label = IDX_TO_LABEL[true_idx]
            
            conf = probs[pred_idx] * 100
            prob_str = f"[{probs[0]*100:.1f}%, {probs[1]*100:.1f}%, {probs[2]*100:.1f}%]"
            
            is_correct = "✅" if pred_idx == true_idx else "❌"
            if pred_idx == true_idx:
                correct_count += 1
                
            print(f"{i+1:<10} | {true_label:<15} | {pred_label:<15} | {conf:.1f}% {is_correct:<5} | {prob_str:<45}")

    print("="*80)
    print(f"Sample Accuracy: {correct_count}/{len(sampled_df)} ({correct_count/len(sampled_df)*100:.1f}%)\n")

if __name__ == "__main__":
    verify_test_samples()
