# CIFAR-10 (50 samples/class) Experiment Script
"""
CIFAR-10 50-shot Experiment.

Runs 5-fold cross-validation on CIFAR-10 subsampled to 50 samples per class.
This demonstrates the generalization of our method to:
1. A different dataset (CIFAR-10 vs CIFAR-100)
2. An even smaller sample regime (50 vs 100 samples/class)

Usage:
    python scripts/run_cifar10_50shot.py
"""

import sys
import time
import argparse
import csv
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Callable, Dict, Union

from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold, train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import existing utilities where possible
from src.utils import (
    set_seed_deterministic,
    get_device,
    ensure_dir,
    get_optimizer_and_scheduler,
    train_one_epoch,
    evaluate,
    EarlyStopping
)
from src.models import create_model
from src.augmentations import (
    get_baseline_transform,
    get_randaugment_transform,
    build_transform_with_op
)

# =============================================================================
# CIFAR-10 Subsampled Dataset (Local Definition)
# =============================================================================

class CIFAR10Subsampled(Dataset):
    """CIFAR-10 dataset with stratified subsampling (50 samples/class)."""
    
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        fold_idx: int = 0,
        transform: Optional[Callable] = None,
        download: bool = True,
        n_splits: int = 5,
        samples_per_class: int = 50, # 50 samples * 10 classes = 500 total
        random_state: int = 42,
        val_size: float = 0.1,
    ) -> None:
        if not 0 <= fold_idx < n_splits:
            raise ValueError(f"fold_idx must be in [0, {n_splits-1}]")
        
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Load CIFAR-10
        try:
            self.full_dataset = torchvision.datasets.CIFAR10(
                root=str(self.root), train=True, download=download
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load CIFAR-10: {e}")
            
        targets = np.array(self.full_dataset.targets)
        
        # We want exactly 50 samples per class * 10 classes = 500 samples total per fold?
        # No, usually we take a subset of the whole dataset.
        # But to match '50 samples/class', we should select that many.
        # Strategy:
        # 1. Select 500 samples total (50/class) from the 50k.
        # 2. But we want 5 folds.
        # To be statistically robust, we should probably pick 5 disjoint sets of 500 samples?
        # Or just subsample ONCE to 500 samples, and do 5-fold CV on that?
        # Standard "Few-Shot" often implies the latter, but "Cross Validation" implies the former.
        # Let's stick to the protocol: Use StratifiedKFold to split the WHOLE dataset, 
        # but that would be 10k samples/fold (like CIFAR-100).
        # We need to SUBSAMPLE first.
        
        # Consistent Subsampling Strategy:
        # 1. For each class, select 50 * 5 = 250 samples first (for 5 folds).
        # 2. Then split those 250 into 5 folds of 50.
        # This ensures we are testing on valid '50-shot' scenarios.
        
        # Actually, simpler: StratifiedKFold with n_splits=100 would give 1% data (~500 samples).
        # Let's verify: 50,000 / 100 = 500 samples. 
        # So we can use fold_idx to pick different chunks.
        
        skf = StratifiedKFold(n_splits=100, shuffle=True, random_state=random_state)
        all_indices = np.arange(len(self.full_dataset))
        folds = list(skf.split(all_indices, targets))
        
        # We use fold_idx (0-4) to select one of the first 5 subsets
        # Each subset has 500 images (50 per class)
        subset_indices = folds[fold_idx][1] # test indices are the small chunk
        
        # Now split this fold's 500 images into Train/Val (90/10)
        subset_targets = targets[subset_indices]
        local_indices = np.arange(len(subset_indices))
        
        train_local, val_local = train_test_split(
            local_indices, test_size=val_size,
            stratify=subset_targets, random_state=random_state
        )
        
        if train:
            self.indices = subset_indices[train_local]
        else:
            self.indices = subset_indices[val_local]

    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        image, label = self.full_dataset[real_idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# =============================================================================
# Helper Functions
# =============================================================================

def get_cifar10_transforms(method: str, m: float = 0.5, p: float = 0.5):
    # CIFAR-10 Mean/Std
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    normalize = transforms.Normalize(mean, std)
    
    if method == "Baseline":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    elif method == "RandAugment":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize
        ])
    elif method == "Ours":
        # Hardcoded Best Single Op from CIFAR-100 (ColorJitter)
        # Assuming transferability (a strong claim!)
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # ColorJitter with m=0.5, p=0.8 (robust setting)
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0)
            ], p=0.8),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Validation
        return transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

# =============================================================================
# Main Experiment Loop
# =============================================================================

def run_experiment():
    device = get_device()
    print(f"Running CIFAR-10 (50 samples/class) on {device}")
    
    methods = ["Baseline", "RandAugment", "Ours"]
    results = []
    
    for method in methods:
        print(f"\nTraining {method}...")
        fold_accs = []
        
        for fold in range(5):
            print(f"  Fold {fold}...", end=" ", flush=True)
            
            # Setup
            train_tf = get_cifar10_transforms(method)
            val_tf = get_cifar10_transforms("Val")
            
            train_ds = CIFAR10Subsampled(root="./data", train=True, fold_idx=fold, transform=train_tf)
            val_ds = CIFAR10Subsampled(root="./data", train=False, fold_idx=fold, transform=val_tf)
            
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
            
            # Model & Training
            # CIFAR-10 is 10 classes
            model = create_model(num_classes=10, pretrained=False).to(device)
            optimizer, scheduler = get_optimizer_and_scheduler(model, total_epochs=200) # Faster 200 epochs
            criterion = nn.CrossEntropyLoss()
            
            best_acc = 0.0
            for epoch in tqdm(range(200), desc=f"Fold {fold}", leave=False):
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
                best_acc = max(best_acc, val_acc)
                scheduler.step()
                
            print(f"Best: {best_acc:.2f}%")
            fold_accs.append(best_acc)
            
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        results.append({"Method": method, "Mean": mean_acc, "Std": std_acc})
        # Save Intermediate Results
        print(f"--> {method}: {mean_acc:.2f} ± {std_acc:.2f}%")
        pd.DataFrame(results).to_csv("outputs/cifar10_50shot_results.csv", index=False)
        print("Updated outputs/cifar10_50shot_results.csv")

    # Final Save
    print("\nFinal Results (CIFAR-10 50-shot):")
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("outputs/cifar10_50shot_results.csv", index=False)
    print("Saved to outputs/cifar10_50shot_results.csv")

if __name__ == "__main__":
    run_experiment()
