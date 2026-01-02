
import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pathlib import Path
import json
import numpy as np
import os
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from source
from src.utils import set_seed_deterministic, get_device, evaluate, ensure_dir
from src.models import create_model
from src.augmentations import build_transform_with_op, get_val_transform

def evaluate_on_test_set(policy_json: str, output_csv: str):
    """
    Evaluates the final policy on the OFFICIAL CIFAR-100 Test Set.
    This provides the true generalization performance for Table 1.
    """
    device = get_device()
    print(f"Evaluating Final Policy on {device}")
    
    # Load Policy
    with open(policy_json, 'r') as f:
        policy = json.load(f)
    
    ops = policy['ops']
    print(f"Loaded Policy: {ops}")
    
    # We only support single op policies for now based on the json structure seen
    # If multiple, we need to handle that. The current JSON has 1 op.
    op_info = ops[0]
    op_name = op_info['name']
    magnitude = op_info['magnitude']
    probability = op_info['probability']
    
    print(f"Testing: {op_name} (m={magnitude}, p={probability})")
    
    # Build Transforms
    # Note: We must train on FULL TRAIN SET (50k) or the Subsampled Train Set?
    # The paper says "we train on small subset". 
    # To compare fairly with "Validation Accuracy" in Table 1 reported so far,
    # we should likely use the SAME Training set size (40k or 45k or 10k?).
    # The prompt implies: "You search on Validation set... Table 1 must be on Test set."
    # Usually this means: Train on Train (or Train+Val) -> Eval on Test.
    # Since we are in "Small Sample Regime" (few-shot), we should stick to the same training data size.
    # The `CIFAR100Subsampled(train=True)` returns 9000 images (Fold 0).
    # We should train on that and eval on official Test.
    
    train_transform = build_transform_with_op(
        op_name=op_name,
        magnitude=magnitude,
        probability=probability,
        include_baseline=True,
        include_normalize=False 
    )
    test_transform = get_val_transform(include_normalize=False)
    
    # Datasets
    # Train: Subsampled (same as search)
    # Using Fold 0 for consistency with search, or maybe train on all 10k?
    # Let's stick to Fold 0 Train (9000) to be safe, or 10000 (Fold 0 full).
    # The search used 9000 train / 1000 val.
    # For final result, maybe we combine them? 
    # Let's stick to the standard 9000 to be perfectly comparable to "Validation" numbers derived fro 9k training.
    
    from src.dataset import CIFAR100Subsampled
    train_dataset = CIFAR100Subsampled(
        root="./data", train=True, fold_idx=0, transform=train_transform
    )
    
    # Test: OFFICIAL CIFAR-100 TEST SET (10,000 images)
    # We do NOT use CIFAR100Subsampled(train=False) as that's just the Val set (1000)!
    test_dataset = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform
    )
    
    print(f"Train Set: {len(train_dataset)} samples (Subsampled)")
    print(f"Test Set:  {len(test_dataset)} samples (Official)")
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    
    # Model
    model = create_model(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # Train
    print("Training for 200 epochs...")
    from src.utils import train_one_epoch
    
    best_acc = 0.0
    
    # For speed, let's run a slightly shorter training if 200 is too long?
    # 200 is standard. 9000 images -> 70 batches. 70 * 200 = 14000 steps. Very fast.
    
    # We'll run 5 different seeds to get Mean/Std Test Acc
    seeds = [42, 100, 2024, 7, 99]
    results = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        set_seed_deterministic(seed)
        
        # Reset model/opt
        model = create_model(num_classes=100).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        
        best_test_acc = 0.0
        
        pbar = tqdm(range(200), desc=f"Seed {seed}", unit="ep")
        for epoch in pbar:
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # We evaluate on Test Set every epoch? Or just last?
            # Standard: Eval on Val, pick best, measure on Test.
            # But here we don't have a Val set separate (we merged current val into training? Or just ignore Val selection?)
            # Valid approach: Train for fixed N epochs, report last or average of last 10.
            # OR: Split Train into Train/Val (9000/1000) again, select best on Val, eval on Test.
            # Since `CIFAR100Subsampled(train=True)` IS 9000, we can use `train=False` (1000) as Val.
            
            pbar.set_postfix({"status": "training"})
            scheduler.step()
            
        # Final evaluation on Test Set
        val_loss, test_acc, _ = evaluate(model, test_loader, criterion, device)
        print(f"Seed {seed} Test Acc: {test_acc:.2f}%")
        results.append(test_acc)
        
    mean = np.mean(results)
    std = np.std(results)
    
    print(f"\nFinal Test Set Results (5 Seeds): {mean:.2f} ± {std:.2f}%")
    
    # Save results
    df = pd.DataFrame({'seed': seeds, 'test_acc': results})
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="outputs/phase_c_final_policy.json")
    parser.add_argument("--output", type=str, default="outputs/test_set_results.csv")
    args = parser.parse_args()
    
    evaluate_on_test_set(args.policy, args.output)
