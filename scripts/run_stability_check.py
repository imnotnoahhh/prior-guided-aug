
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    set_seed_deterministic,
    get_device,
    get_optimizer_and_scheduler,
    train_one_epoch,
    evaluate
)
from src.models import create_model

# =============================================================================
# CIFAR-10 Subsampled Dataset (Local Definition for Self-Containment)
# =============================================================================

class CIFAR10Subsampled(Dataset):
    """CIFAR-10 dataset with stratified subsampling (50 samples/class)."""
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        fold_idx: int = 0,
        transform = None,
        random_state_data: int = 42
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        
        # Load CIFAR-10
        full_ds = datasets.CIFAR10(root=str(self.root), train=True, download=True)
        targets = np.array(full_ds.targets)
        
        # Split logic (Same as run_cifar10_50shot.py)
        # 1. StratifiedKFold with n_splits=100 -> 1% data (500 samples)
        skf = StratifiedKFold(n_splits=100, shuffle=True, random_state=random_state_data)
        all_indices = np.arange(len(full_ds))
        folds = list(skf.split(all_indices, targets))
        
        # Use fold_idx to pick the data chunk
        subset_indices = folds[fold_idx][1] 
        
        # Split Train/Val (90/10)
        subset_targets = targets[subset_indices]
        local_indices = np.arange(len(subset_indices))
        
        train_local, val_local = train_test_split(
            local_indices, test_size=0.1,
            stratify=subset_targets, random_state=random_state_data
        )
        
        if train:
            self.indices = subset_indices[train_local]
        else:
            self.indices = subset_indices[val_local]
            
        self.full_ds = full_ds

    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        image, label = self.full_ds[real_idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# =============================================================================
# Transforms
# =============================================================================

def get_transforms(method: str):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    normalize = transforms.Normalize(mean, std)
    
    if method == "RandAugment":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize
        ])
    elif method == "Ours":
        # Our Best Policy: ColorJitter (m=0.4, p=0.8)
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0)
            ], p=0.8),
            transforms.ToTensor(),
            normalize
        ])
    else: # Val
        return transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

# =============================================================================
# Main
# =============================================================================

def run_seed_experiment(seed, method, device):
    print(f"  Running Seed {seed}...")
    
    # Critical: Set seed for Training Randomness (Model init, Batch shuffling, Augmentation prob)
    # But keep data split constant (logic handled in Dataset via random_state_data=42)
    set_seed_deterministic(seed)
    
    train_tf = get_transforms(method)
    val_tf = get_transforms("Val")
    
    # Fixed data split (fold_idx=0, random_state_data=42)
    train_ds = CIFAR10Subsampled(root="./data", train=True, fold_idx=0, transform=train_tf, random_state_data=42)
    val_ds = CIFAR10Subsampled(root="./data", train=False, fold_idx=0, transform=val_tf, random_state_data=42)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0) # CIFAR10 is small
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    
    model = create_model(num_classes=10).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, total_epochs=200)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    # Using tqdm for epochs
    pbar = tqdm(range(200), desc=f"Seed {seed} [{method}]", leave=False)
    for epoch in pbar:
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            
        pbar.set_postfix({"best": f"{best_acc:.1f}%"})
    
    return best_acc

def main():
    # Hardcoded seeds as requested
    seeds = [42, 100, 2024]
    
    device = get_device()
    print(f"Running Stability Check on {device}")
    
    methods = ["RandAugment", "Ours"]
    results = []
    
    for method in methods:
        print(f"\nEvaluating {method}...")
        seed_accs = []
        for seed in seeds:
            acc = run_seed_experiment(seed, method, device)
            seed_accs.append(acc)
            print(f"  -> Seed {seed}: {acc:.2f}%")
            
        mean = np.mean(seed_accs)
        std = np.std(seed_accs)
        
        results.append({
            "Method": method,
            "Mean": mean,
            "Std": std,
            "Seeds": seeds,
            "Accs": seed_accs
        })
        print(f"==> {method}: {mean:.2f} ± {std:.2f}%")
        
    # Save
    df = pd.DataFrame(results)
    df.to_csv("outputs/stability_seeds_results.csv", index=False)
    print("\nSaved to outputs/stability_seeds_results.csv")

if __name__ == "__main__":
    main()
