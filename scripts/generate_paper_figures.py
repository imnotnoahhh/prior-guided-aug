import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

OUTPUT_DIR = 'outputs/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_stability_boxplot():
    print("Generating Stability Boxplot...")
    df = pd.read_csv('outputs/phase_d_results.csv')
    
    # Rename op_name to method for clarity
    if 'op_name' in df.columns:
        df.rename(columns={'op_name': 'method'}, inplace=True)
    
    # Filter methods
    methods = ['Baseline', 'RandAugment', 'Ours_optimal'] 
    df = df[df['method'].isin(methods)]
    
    # Order
    df['method'] = pd.Categorical(df['method'], categories=methods, ordered=True)
    
    plt.figure(figsize=(8, 6))
    
    # Custom color palette
    colors = {'Baseline': '#95a5a6', 'RandAugment': '#e74c3c', 'Ours_optimal': '#2ecc71'}
    
    sns.boxplot(x='method', y='val_acc', data=df, palette=colors, width=0.5)
    sns.stripplot(x='method', y='val_acc', data=df, color='black', alpha=0.6, jitter=0.1)
    
    plt.title('Stability Analysis: Training Variance (5 Folds)', fontsize=14, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)')
    plt.xlabel('')
    
    # Add Std Dev annotation
    stats = df.groupby('method')['val_acc'].agg(['mean', 'std'])
    for i, method in enumerate(methods):
        mean = stats.loc[method, 'mean']
        std = stats.loc[method, 'std']
        plt.text(i, mean + 1.5, f"Mean: {mean:.1f}%\n$ \\sigma $: {std:.2f}",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.ylim(36, 45)  # Zoom in to see variance
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_stability_boxplot.png'))
    plt.close()

def plot_search_space_heatmap():
    print("Generating Phase A Search Space Heatmap...")
    df = pd.read_csv('outputs/phase_a_results.csv')
    
    # Filter for the winning op: ColorJitter
    df_op = df[df['op_name'] == 'ColorJitter'].copy()
    
    plt.figure(figsize=(8, 6))
    
    sc = plt.scatter(df_op['probability'], df_op['magnitude'], 
                     c=df_op['val_acc'], cmap='viridis', s=100, edgecolors='black', alpha=0.8)
    
    plt.colorbar(sc, label='Valid Accuracy (%)')
    
    plt.title('Prior-Guided Search Space (ColorJitter)', fontsize=14, fontweight='bold')
    plt.xlabel('Probability (p)')
    plt.ylabel('Magnitude (m)')
    
    # Highlight the chosen point roughly (Phase C selected ~ m=0.25, p=0.42)
    plt.scatter([0.42], [0.25], s=200, facecolors='none', edgecolors='red', linewidth=2, label='Optimal Found')
    plt.legend(loc='upper left')
    
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_search_space_colorjitter.png'))
    plt.close()

def plot_complexity_tradeoff():
    print("Generating Complexity-Stability Trade-off...")
    
    # Data manually constructed from Phase D results
    data = {
        'Method': ['Baseline', 'Ours (Optimal)', 'RandAugment'],
        'Accuracy': [39.90, 40.74, 42.24],
        'Stability': [1.0/1.01, 1.0/0.78, 1.0/1.17], # Inverse Variance proxy
        'Std': [1.01, 0.78, 1.17],
        'Complexity': [1, 2, 8]  # Conceptual complexity score
    }
    
    plt.figure(figsize=(8, 6))
    
    colors = ['#95a5a6', '#2ecc71', '#e74c3c']
    
    # Bubble chart: X=Complexity, Y=Stability (1/Std), Size=Accuracy
    sizes = [(acc - 30) * 30 for acc in data['Accuracy']] # Scale for visibility
    
    plt.scatter(data['Complexity'], data['Std'], s=sizes, c=colors, alpha=0.7, edgecolors='black')
    
    # Add labels
    for i, method in enumerate(data['Method']):
        plt.text(data['Complexity'][i], data['Std'][i] + 0.05, 
                 f"{method}\nAcc: {data['Accuracy'][i]}%\nStd: {data['Std'][i]}", 
                 ha='center', fontweight='bold')
    
    # Arrow for Pareto frontier
    # plt.annotate('Pareto Efficient\n(Low Cost, High Stability)', xy=(2, 0.78), xytext=(4, 0.9),
    #              arrowprops=dict(facecolor='black', shrink=0.05))

    plt.title('The Complexity Gap: Stability vs. Complexity', fontsize=14, fontweight='bold')
    plt.ylabel('Instability (Standard Deviation) $\\downarrow$', fontsize=12)
    plt.xlabel('Algorithmic Complexity (Search Space Size) $\\rightarrow$', fontsize=12)
    
    plt.xticks([1, 2, 8], ['None', 'Single-Op', 'Multi-Op (N=2,M=9)'])
    plt.ylim(0.5, 1.5)
    plt.xlim(0, 10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_complexity_gap.png'))
    plt.close()

if __name__ == "__main__":
    try:
        plot_stability_boxplot()
        plot_search_space_heatmap()
        plot_complexity_tradeoff()
        print(f"Figures saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error generating figures: {e}")
