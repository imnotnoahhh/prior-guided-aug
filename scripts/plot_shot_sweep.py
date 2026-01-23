#!/usr/bin/env python3
"""
Plot Shot Sweep Results

Generates three figures:
1. Accuracy vs Shot (line plot with std shading)
2. Fold Std vs Shot (bar plot)
3. Lower Bound vs Shot (line plot)

Usage:
    python scripts/plot_shot_sweep.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# Color palette
COLORS = {
    'Baseline': '#7f7f7f',      # Gray
    'RandAugment': '#d62728',   # Red
    'SAS': '#2ca02c',           # Green
}

MARKERS = {
    'Baseline': 'o',
    'RandAugment': 's',
    'SAS': '^',
}


def load_data(output_dir: Path) -> pd.DataFrame:
    """Load shot sweep summary data."""
    summary_path = output_dir / "shot_sweep_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    return pd.read_csv(summary_path)


def plot_accuracy_vs_shot(df: pd.DataFrame, output_dir: Path):
    """Plot Accuracy vs Shot with std shading."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    shots = sorted(df["shot"].unique())
    methods = ["Baseline", "RandAugment", "SAS"]
    
    for method in methods:
        subset = df[df["method"] == method].sort_values("shot")
        if len(subset) == 0:
            continue
        
        x = subset["shot"].values
        y = subset["mean_acc"].values
        std = subset["std_acc"].values
        
        # Plot line with markers
        ax.plot(x, y, 
                color=COLORS.get(method, 'gray'),
                marker=MARKERS.get(method, 'o'),
                markersize=8,
                linewidth=2,
                label=f"{method}")
        
        # Add std shading
        ax.fill_between(x, y - std, y + std,
                        color=COLORS.get(method, 'gray'),
                        alpha=0.2)
    
    ax.set_xlabel("Samples per Class (Shot)")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Accuracy vs Sample Size")
    ax.set_xticks(shots)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits with some padding
    y_min = df["mean_acc"].min() - df["std_acc"].max() - 2
    y_max = df["mean_acc"].max() + df["std_acc"].max() + 2
    ax.set_ylim(max(0, y_min), min(100, y_max))
    
    plt.tight_layout()
    output_path = output_dir / "figures" / "fig_shot_sweep_accuracy.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_std_vs_shot(df: pd.DataFrame, output_dir: Path):
    """Plot Fold Std vs Shot (bar plot)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    shots = sorted(df["shot"].unique())
    methods = ["Baseline", "RandAugment", "SAS"]
    
    x = np.arange(len(shots))
    width = 0.25
    
    for i, method in enumerate(methods):
        subset = df[df["method"] == method].sort_values("shot")
        if len(subset) == 0:
            continue
        
        stds = subset["std_acc"].values
        offset = (i - 1) * width
        
        bars = ax.bar(x + offset, stds,
                      width=width,
                      color=COLORS.get(method, 'gray'),
                      label=method,
                      edgecolor='black',
                      linewidth=0.5)
        
        # Add value labels on bars
        for bar, std in zip(bars, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f'{std:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Samples per Class (Shot)")
    ax.set_ylabel("Fold Standard Deviation (%)")
    ax.set_title("Variance vs Sample Size (Lower is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(shots)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "figures" / "fig_shot_sweep_std.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_lower_bound_vs_shot(df: pd.DataFrame, output_dir: Path):
    """Plot Lower Bound (Mean - Std) vs Shot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    shots = sorted(df["shot"].unique())
    methods = ["Baseline", "RandAugment", "SAS"]
    
    for method in methods:
        subset = df[df["method"] == method].sort_values("shot")
        if len(subset) == 0:
            continue
        
        x = subset["shot"].values
        y = subset["lower_bound"].values
        
        ax.plot(x, y,
                color=COLORS.get(method, 'gray'),
                marker=MARKERS.get(method, 'o'),
                markersize=8,
                linewidth=2,
                label=method)
    
    ax.set_xlabel("Samples per Class (Shot)")
    ax.set_ylabel("Lower Bound: Mean - Std (%)")
    ax.set_title("Worst-Case Performance vs Sample Size")
    ax.set_xticks(shots)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "figures" / "fig_shot_sweep_lower_bound.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined(df: pd.DataFrame, output_dir: Path):
    """Combined 2x1 plot for paper."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    shots = sorted(df["shot"].unique())
    methods = ["Baseline", "RandAugment", "SAS"]
    
    # Left: Accuracy with shading
    ax = axes[0]
    for method in methods:
        subset = df[df["method"] == method].sort_values("shot")
        if len(subset) == 0:
            continue
        
        x = subset["shot"].values
        y = subset["mean_acc"].values
        std = subset["std_acc"].values
        
        ax.plot(x, y,
                color=COLORS.get(method, 'gray'),
                marker=MARKERS.get(method, 'o'),
                markersize=8,
                linewidth=2,
                label=method)
        ax.fill_between(x, y - std, y + std,
                        color=COLORS.get(method, 'gray'),
                        alpha=0.2)
    
    ax.set_xlabel("Samples per Class")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("(a) Accuracy vs Sample Size")
    ax.set_xticks(shots)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Right: Std comparison
    ax = axes[1]
    x = np.arange(len(shots))
    width = 0.25
    
    for i, method in enumerate(methods):
        subset = df[df["method"] == method].sort_values("shot")
        if len(subset) == 0:
            continue
        
        stds = subset["std_acc"].values
        offset = (i - 1) * width
        
        ax.bar(x + offset, stds,
               width=width,
               color=COLORS.get(method, 'gray'),
               label=method,
               edgecolor='black',
               linewidth=0.5)
    
    ax.set_xlabel("Samples per Class")
    ax.set_ylabel("Fold Std (%)")
    ax.set_title("(b) Variance vs Sample Size")
    ax.set_xticks(x)
    ax.set_xticklabels(shots)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "figures" / "fig_shot_sweep_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("Loading data...")
    df = load_data(output_dir)
    print(f"Loaded {len(df)} summary rows")
    print(df.to_string())
    
    print("\nGenerating plots...")
    plot_accuracy_vs_shot(df, output_dir)
    plot_std_vs_shot(df, output_dir)
    plot_lower_bound_vs_shot(df, output_dir)
    plot_combined(df, output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
