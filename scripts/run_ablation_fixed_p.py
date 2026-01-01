# Ablation Study: Fixed Probability Search
"""
Ablation Study: Fixed Probability (Search Magnitude Only).

This script isolates the effect of searching for probability by fixing it to a constant
(default p=0.5) and only searching for the optimal magnitude.

This verifies whether the sophisticated 2D search (m, p) is actually necessary,
or if a simpler 1D search (m) would suffice.

Usage:
    python scripts/run_ablation_fixed_p.py --fixed_p 0.5
"""

import sys
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from existing modules
from src.utils import (
    set_seed_deterministic,
    get_device,
    ensure_dir,
    get_optimizer_and_scheduler
)
from src.augmentations import OP_SEARCH_SPACE
from main_phase_b import (
    load_phase_a_results,
    load_baseline_result,
    get_promoted_ops,
    train_to_epoch,
    check_csv_needs_header,
    write_raw_csv_row,
    aggregate_results
)

# =============================================================================
# Fixed-P Sampling
# =============================================================================

def sobol_sample_configs_fixed_p(
    op_name: str,
    fixed_p: float = 0.5,
    n_samples: int = 20,
    seed: int = 42,
) -> List[Tuple[float, float]]:
    """Sample magnitudes only, pairing them with a fixed probability."""
    
    # 1D Sobol sampling for Magnitude
    try:
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=1, scramble=True, seed=seed)
        samples = sampler.random(n_samples) # Shape (n, 1)
    except ImportError:
        np.random.seed(seed)
        samples = np.random.rand(n_samples, 1)
    
    # Get M bounds
    space = OP_SEARCH_SPACE.get(op_name, {"m": [0.0, 1.0]})
    m_min, m_max = space["m"]
    
    configs = []
    for m_unit in samples:
        m_val = m_unit[0]
        m = m_min + m_val * (m_max - m_min)
        configs.append((round(m, 4), fixed_p))
        
    return configs

# =============================================================================
# Modified ASHA Logic
# =============================================================================

def run_ablation(
    phase_a_csv: Path,
    baseline_csv: Path,
    output_dir: Path,
    fixed_p: float = 0.5,
    rungs: List[int] = [30, 80, 200],
    n_samples: int = 20,
    seed: int = 42,
    ops_filter: Optional[List[str]] = None
):
    print(f"Running Ablation: Fixed P={fixed_p}")
    ensure_dir(output_dir)
    
    # Load Phase A and Baseline
    # (Copied logic from main_phase_b to ensure standalone execution)
    phase_a_df = load_phase_a_results(phase_a_csv)
    base_acc, base_top5, base_loss = load_baseline_result(baseline_csv)
    
    promoted_ops = get_promoted_ops(phase_a_df, base_acc, base_top5, base_loss)
    
    # Filter ops if requested
    if ops_filter:
        promoted_ops = [op for op in promoted_ops if op in ops_filter]
        
    print(f"Promoted Ops: {promoted_ops}")
    
    # Generate Sampling
    all_configs = []
    for op in promoted_ops:
        configs = sobol_sample_configs_fixed_p(op, fixed_p, n_samples, seed)
        for m, p in configs:
            all_configs.append((op, m, p))
            
    print(f"Total Configs: {len(all_configs)}")
    
    # ASHA Loop
    device = get_device()
    raw_csv = output_dir / f"ablation_p{fixed_p}_raw.csv"
    summary_csv = output_dir / f"ablation_p{fixed_p}_summary.csv"
    
    active_configs = {i: (op, m, p, None) for i, (op, m, p) in enumerate(all_configs)}
    write_header = check_csv_needs_header(raw_csv)
    
    for rung_idx, target_epochs in enumerate(rungs):
        print(f"\n--- Rung {rung_idx+1}: {target_epochs} Epochs ---")
        rung_results = []
        
        for idx, (op, m, p, ckpt) in tqdm(active_configs.items()):
            try:
                result, new_ckpt = train_to_epoch(
                    op_name=op, magnitude=m, probability=p,
                    target_epochs=target_epochs, device=device,
                    checkpoint=ckpt, fold_idx=0, seed=seed,
                    num_workers=0  # Safe for MacOS MPS
                    # Note: We use fold 0 for search
                )
                
                # Write to raw CSV
                write_raw_csv_row(raw_csv, result, write_header)
                write_header = False
                
                rung_results.append((idx, result["val_acc"], new_ckpt))
                
            except Exception as e:
                print(f"Error: {e}")
                rung_results.append((idx, -1.0, None))
                
        # Selection for next rung
        if rung_idx < len(rungs) - 1:
            rung_results.sort(key=lambda x: x[1], reverse=True)
            n_keep = max(1, len(rung_results) // 2) # Halving
            survivors = rung_results[:n_keep]
            
            # Update active configs
            new_active = {}
            for idx, _, ckpt in survivors:
                op, m, p, _ = active_configs[idx]
                new_active[idx] = (op, m, p, ckpt)
            active_configs = new_active
            print(f"Kept {n_keep} configs.")

    # Aggregate
    agg_df = aggregate_results(raw_csv, summary_csv)
    print("\nTop Results:")
    print(agg_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed_p", type=float, default=0.5)
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()
    
    # Default to ColorJitter for efficient ablation (Proof of Concept)
    ops_filter = ["ColorJitter"]

    run_ablation(
        Path("outputs/phase_a_results.csv"),
        Path("outputs/baseline_result.csv"),
        Path("outputs/ablation"),
        fixed_p=args.fixed_p,
        n_samples=args.n_samples,
        ops_filter=ops_filter
    )
