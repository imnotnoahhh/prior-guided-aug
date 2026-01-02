
import sys
import argparse
import random
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Project imports
from src.dataset import CIFAR100Subsampled
from src.models import create_model
from src.augmentations import get_randaugment_transform
from src.utils import (
    set_seed_deterministic,
    get_device,
    get_optimizer_and_scheduler,
    train_one_epoch,
    evaluate
)

def run_trial(trial_id, n, m, epochs, device):
    """Run a single training trial with specific RandAugment params."""
    print(f"Trial {trial_id}: RandAugment(N={n}, M={m})")
    
    # 1. Setup Data
    # Use Fold 0 for search
    train_tf = get_randaugment_transform(n=n, m=m, include_baseline=True, include_normalize=True)
    val_tf = get_randaugment_transform(n=0, m=0, include_baseline=False, include_normalize=True) # Just normalization
    
    train_ds = CIFAR100Subsampled(root="./data", train=True, fold_idx=0, transform=train_tf)
    val_ds = CIFAR100Subsampled(root="./data", train=False, fold_idx=0, transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    
    # 2. Setup Model
    model = create_model(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(model, total_epochs=epochs)
    
    # 3. Training Loop
    best_acc = 0.0
    
    # Use tqdm for epoch progress
    pbar = tqdm(range(epochs), desc=f"Trial {trial_id} [N={n}, M={m}]", leave=False)
    for epoch in pbar:
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            
        pbar.set_postfix({"train_acc": f"{train_acc:.1f}%", "val_acc": f"{val_acc:.1f}%", "best": f"{best_acc:.1f}%"})
        
    return best_acc

def main():
    # Hardcoded parameters as requested
    trials = 10
    epochs = 40
    seed = 42
    
    set_seed_deterministic(seed)
    device = get_device()
    print(f"Running Tuned RandAugment Search on {device}")
    
    results = []
    
    # Search Space
    # N in [1, 2, 3]
    # M in [1, ..., 15]
    
    print(f"Starting {trials} trials...")
    
    for i in range(trials):
        # Sample N and M
        n = random.randint(1, 3)
        m = random.randint(1, 14)
        
        # Check if already sampled
        duplicate = any(r['n'] == n and r['m'] == m for r in results)
        if duplicate:
            # Simple retry logic or skip
            m = random.randint(1, 14) 
        
        start_time = time.time()
        final_acc = run_trial(i, n, m, epochs, device)
        duration = time.time() - start_time
        
        results.append({
            "trial_id": i,
            "n": n,
            "m": m,
            "acc": final_acc,
            "duration": duration
        })
        
        print(f"--> Result: N={n}, M={m}, Acc={final_acc:.2f}% (Time: {duration:.1f}s)")
        
        # Save intermediate
        pd.DataFrame(results).to_csv("outputs/tuned_randaugment_results.csv", index=False)

    print("\nSearch Complete.")
    print("Top 3 Configurations:")
    df = pd.DataFrame(results)
    df = df.sort_values(by="acc", ascending=False)
    print(df.head(3))
    print(f"Results saved to outputs/tuned_randaugment_results.csv")

if __name__ == "__main__":
    main()
