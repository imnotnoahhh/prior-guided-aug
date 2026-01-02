
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def plot_strategic_collapse(csv_path: str, output_dir: str):
    """
    Plots the "Strategic Collapse" phenomenon:
    As the number of operations increases (greedy addition), 
    the variance (instability) increases, often outweighing accuracy gains.
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Calculate number of operations based on '+' count in op_name
    # "ColorJitter" -> 0 '+' -> 1 op
    # "ColorJitter+RandomPerspective" -> 1 '+' -> 2 ops
    df['n_ops'] = df['op_name'].apply(lambda x: x.count('+') + 1)
    
    # Group by n_ops and name to get mean/std across seeds first (per combo)
    # Then we want to show the BEST combo at each step or ALL combos?
    # The "Search" path follows the greedy choice.
    # Let's visualize the distribution of ALL attempted combinations at each depth.
    
    # Calculate Mean and Std of Val Accuracy for each (op_name, n_ops) pair across seeds
    grouped = df.groupby(['n_ops', 'op_name'])['val_acc'].agg(['mean', 'std']).reset_index()
    
    # Setup style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(8, 6))
    
    # Plot 1: Mean Accuracy vs N_Ops
    # We use a stripplot to show all candidates, and a line for the "Chosen" path (Best Mean)
    
    # Find best at each level (Greedy choice)
    best_per_level = grouped.loc[grouped.groupby('n_ops')['mean'].idxmax()]
    
    # Plot all candidates as grey dots
    sns.stripplot(
        data=grouped, 
        x='n_ops', 
        y='mean', 
        color='gray', 
        alpha=0.5, 
        jitter=0.1, 
        label='Candidates'
    )
    
    # Plot the "Greedy Path"
    plt.plot(
        best_per_level['n_ops'] - 1, # adjust for 0-indexing if needed, but stripplot uses categorical
        best_per_level['mean'], 
        marker='o', 
        color='#D32F2F', 
        linewidth=2, 
        markersize=8, 
        label='Greedy Path (Best Mean)'
    )
    
    # Add error bars to the Greedy Path to show Variance exploding
    plt.errorbar(
        x=best_per_level['n_ops'] - 1, 
        y=best_per_level['mean'], 
        yerr=best_per_level['std'], 
        fmt='none', 
        ecolor='#D32F2F', 
        capsize=5, 
        linewidth=2,
        label='Stability (Std Dev)'
    )
    
    plt.title('Strategic Collapse: Accuracy vs. Complexity', fontsize=14, pad=15)
    plt.xlabel('Number of Operations', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    
    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Filter duplicate labels
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower left')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'strategic_collapse.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    # Also print the stats for the paper
    print("\n--- Statistics for Paper ---")
    print(best_per_level[['n_ops', 'mean', 'std']])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="outputs/phase_c_history.csv")
    parser.add_argument("--output", type=str, default="outputs/figures")
    args = parser.parse_args()
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    plot_strategic_collapse(args.csv, args.output)
