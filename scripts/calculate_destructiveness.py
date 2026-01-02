
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

import warnings
import torchvision.models as models

# FIX: Monkey-patch torchvision.models.alexnet to squash warnings from lpips
# lpips (v0.1.4) calls alexnet(pretrained=True), which is deprecated.
# We intercept this call and translate it to the new weights=... API.
_original_alexnet = models.alexnet

def _patched_alexnet(*args, **kwargs):
    # If legacy 'pretrained' arg is used, convert to new 'weights' arg
    if kwargs.pop('pretrained', False):
        try:
            # Try to get the default weights (IMAGENET1K_V1 usually)
            kwargs['weights'] = models.AlexNet_Weights.DEFAULT
        except AttributeError:
            # Fallback for older torchvision if AlexNet_Weights not found (unlikely in this env)
            kwargs['pretrained'] = True # Revert if we can't fix it
    return _original_alexnet(*args, **kwargs)

models.alexnet = _patched_alexnet

# Add project root to path (Restored)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CIFAR100Subsampled
from src.augmentations import (
    get_baseline_transform,
    get_randaugment_transform,
    build_transform_with_op
)

# Try imports
try:
    import lpips
except ImportError:
    print("Error: 'lpips' library not found. Please install it using: pip install lpips")
    sys.exit(1)

try:
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:
    print("Error: 'scikit-image' not found. Please install it using: pip install scikit-image")
    sys.exit(1)

def tensor_to_np(img_tensor):
    """Convert (3, H, W) tensor [-1, 1] or [0, 1] to (H, W, 3) numpy [0, 255]."""
    img = img_tensor.detach().cpu()
    # Assume [0, 1] for now as most augs behave that way before normalization
    # If normalized, we'd need to un-normalize. 
    # Our augmentations usually return [0, 1] (except Normalize).
    # We will run this script WITHOUT normalization in the transform for visualization/metric purpose.
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return img

def calc_ssim(img1, img2):
    """Calculate SSIM between two numpy images (H, W, C)."""
    return ssim_func(img1, img2, channel_axis=2, data_range=255)

def main():
    parser = argparse.ArgumentParser(description="Calculate Destructiveness Metrics (LPIPS, SSIM)")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for LPIPS")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load LPIPS model (AlexNet backbone is standard/fast)
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    # Dataset (Use Val split of Fold 0 to avoid training data)
    # Note: We want raw images, so we use a basic ToTensor transform initially
    base_ds = CIFAR100Subsampled(
        root="./data", train=False, fold_idx=0, transform=transforms.ToTensor()
    )
    
    # We will manually apply augmentations
    # Strategies to compare
    strategies = {
        "Baseline (Flip/Crop)": get_baseline_transform(include_normalize=False, include_totensor=True),
        "RandAugment (N=2, M=9)": get_randaugment_transform(n=2, m=9, include_baseline=True, include_normalize=False),
        # Ours: ColorJitter (m=0.4, p=0.8) + Baseline
        "Ours (ColorJitter)": build_transform_with_op("ColorJitter", magnitude=0.4, probability=0.8, include_baseline=True, include_normalize=False)
    }

    # Prepare results container
    results = {k: {"ssim": [], "lpips": []} for k in strategies.keys()}

    # Select indices
    indices = np.random.choice(len(base_ds), min(args.samples, len(base_ds)), replace=False)

    print(f"Evaluating {len(indices)} samples across {len(strategies)} strategies...")
    
    # Iterate
    # To batch LPIPS, we'll collect tensors
    
    for name, transform in strategies.items():
        print(f"\nProcessing {name}...")
        
        ssim_scores = []
        lpips_scores = []
        
        # We process manually to pair original vs augmented
        # Note: LPIPS expects inputs in [-1, 1]
        
        batch_orig = []
        batch_aug = []
        
        for i in tqdm(indices, desc=f"Calculating metrics"):
            # Get raw PIL image (to ensure fair augmentation application from scratch)
            # Accessing .full_dataset[idx] returns (PIL, label) usually, but CIFAR100Subsampled wrapper returns transformed.
            # We bypass the wrapper to get the raw underlying dataset item if possible, 
            # Or just use the base_ds (which is ToTensor) and Convert back to PIL for consistency if transforms expect PIL.
            
            # The dataset wrapper returns (image, label). 
            # We defined base_ds with ToTensor, so `orig_tensor` is [0, 1].
            orig_tensor, _ = base_ds[i] 
            orig_pil = transforms.ToPILImage()(orig_tensor)
            
            # Apply transform
            # The transforms defined above expect PIL usually (except the Tensor ones).
            # Our `get_xxx_transform` usually start with PIL ops.
            # But wait, `get_baseline_transform` includes `ToTensor` at the end.
            # So input should be PIL.
            
            # Apply specific strategy
            aug_tensor = transform(orig_pil)
            
            # 1. SSIM (H, W, C) [0, 255]
            img_orig_np = tensor_to_np(orig_tensor)
            img_aug_np = tensor_to_np(aug_tensor)
            
            s = calc_ssim(img_orig_np, img_aug_np)
            ssim_scores.append(s)
            
            # 2. LPIPS (N, C, H, W) in [-1, 1]
            # Convert [0, 1] -> [-1, 1]
            l_orig = (orig_tensor * 2 - 1).unsqueeze(0)
            l_aug = (aug_tensor * 2 - 1).unsqueeze(0)
            
            batch_orig.append(l_orig)
            batch_aug.append(l_aug)
            
            if len(batch_orig) >= args.batch_size:
                # Flush batch
                b_o = torch.cat(batch_orig).to(device)
                b_a = torch.cat(batch_aug).to(device)
                with torch.no_grad():
                    d = loss_fn_alex(b_o, b_a) # returns (N, 1, 1, 1)
                lpips_scores.extend(d.flatten().cpu().numpy().tolist())
                batch_orig = []
                batch_aug = []

        # Flush remaining
        if batch_orig:
            b_o = torch.cat(batch_orig).to(device)
            b_a = torch.cat(batch_aug).to(device)
            with torch.no_grad():
                d = loss_fn_alex(b_o, b_a)
            lpips_scores.extend(d.flatten().cpu().numpy().tolist())

        results[name]["ssim"] = (np.mean(ssim_scores), np.std(ssim_scores))
        results[name]["lpips"] = (np.mean(lpips_scores), np.std(lpips_scores))

    print("\n" + "="*60)
    print(f"{'Strategy':<25} | {'SSIM (Higher=Better)':<20} | {'LPIPS (Lower=Better)':<20}")
    print("-" * 70)
    for name, metrics in results.items():
        ssim_mean, ssim_std = metrics["ssim"]
        lpips_mean, lpips_std = metrics["lpips"]
        print(f"{name:<25} | {ssim_mean:.4f} ± {ssim_std:.4f}   | {lpips_mean:.4f} ± {lpips_std:.4f}")
    print("="*60)
    
    # Save to CSV
    import csv
    with open("outputs/destructiveness_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Strategy", "SSIM_Mean", "SSIM_Std", "LPIPS_Mean", "LPIPS_Std"])
        for name, metrics in results.items():
            writer.writerow([name, metrics["ssim"][0], metrics["ssim"][1], metrics["lpips"][0], metrics["lpips"][1]])
    print("Saved results to outputs/destructiveness_metrics.csv")

if __name__ == "__main__":
    main()
