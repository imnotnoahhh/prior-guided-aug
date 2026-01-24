#!/usr/bin/env python3
"""
Statistical verification script for EUSIPCO 2026 paper.
This script verifies all key numerical claims in the paper against raw data.

Usage:
    python scripts/verify_statistics.py
"""

import numpy as np
from scipy.stats import wilcoxon
import pandas as pd
from pathlib import Path


def verify_table_ii():
    """Verify Table II (Main Results) values."""
    print("=" * 60)
    print("TABLE II VERIFICATION")
    print("=" * 60)
    
    # Raw fold data from phase_d_results.csv
    data = {
        'Baseline': [39.5, 40.5, 40.7, 40.5, 38.3],
        'Baseline-NoAug': [29.8, 30.7, 32.7, 28.3, 23.9],
        'RandAugment': [42.5, 41.6, 42.9, 40.6, 43.6],
        'Cutout': [37.5, 35.5, 36.8, 34.5, 37.0],
        'SAS': [40.1, 42.1, 40.6, 40.5, 40.4]
    }
    
    for method, folds in data.items():
        folds = np.array(folds)
        mean = np.mean(folds)
        std_sample = np.std(folds, ddof=1)  # sample std (used in paper)
        std_pop = np.std(folds, ddof=0)  # population std
        width = np.max(folds) - np.min(folds)
        cv = (std_sample / mean) * 100
        
        print(f"\n{method}:")
        print(f"  Folds: {folds.tolist()}")
        print(f"  Mean: {mean:.2f}%")
        print(f"  Std (sample, ddof=1): {std_sample:.4f}  <- Paper uses this")
        print(f"  Std (population, ddof=0): {std_pop:.4f}")
        print(f"  Width (max-min): {width:.1f}")
        print(f"  CV: {cv:.2f}%")


def verify_ablation_table():
    """Verify Ablation Table (Table V) values."""
    print("\n" + "=" * 60)
    print("ABLATION TABLE VERIFICATION")
    print("=" * 60)
    
    # Phase A only results from search_ablation_results.csv
    phase_a_only = [35.9, 34.1, 34.9, 35.2, 38.9]
    
    folds = np.array(phase_a_only)
    mean = np.mean(folds)
    std_sample = np.std(folds, ddof=1)  # sample std
    std_pop = np.std(folds, ddof=0)  # population std
    width = np.max(folds) - np.min(folds)
    cv = (std_pop / mean) * 100  # Paper uses population std for Phase A
    
    print(f"\nPhase A Only:")
    print(f"  Folds: {folds.tolist()}")
    print(f"  Mean: {mean:.2f}%")
    print(f"  Std (sample, ddof=1): {std_sample:.4f}")
    print(f"  Std (population, ddof=0): {std_pop:.4f}  <- Paper uses 1.65")
    print(f"  Width (max-min): {width:.1f} (Paper claims: 4.8)")
    print(f"  CV (using pop std): {cv:.2f}%")


def verify_wilcoxon_test():
    """Verify Wilcoxon signed-rank test p-value."""
    print("\n" + "=" * 60)
    print("WILCOXON SIGNED-RANK TEST")
    print("=" * 60)
    
    randaugment_folds = np.array([42.5, 41.6, 42.9, 40.6, 43.6])
    sas_folds = np.array([40.1, 42.1, 40.6, 40.5, 40.4])
    
    # Paired Wilcoxon signed-rank test
    stat, p_value = wilcoxon(randaugment_folds, sas_folds)
    
    print(f"\nRandAugment folds: {randaugment_folds.tolist()}")
    print(f"SAS folds: {sas_folds.tolist()}")
    print(f"Differences: {(randaugment_folds - sas_folds).tolist()}")
    print(f"\nWilcoxon statistic: {stat}")
    print(f"p-value: {p_value:.4f}")
    print(f"Paper claims: p=0.19")
    
    if abs(p_value - 0.19) < 0.02:
        print("✓ p-value matches paper claim (within rounding)")
    else:
        print(f"✗ p-value discrepancy: {p_value:.4f} vs 0.19")


def verify_lambda_threshold():
    """Verify the risk aversion threshold λ calculation."""
    print("\n" + "=" * 60)
    print("LAMBDA THRESHOLD VERIFICATION")
    print("=" * 60)
    
    # From Table II
    mu_ra = 42.24  # RandAugment mean
    sigma_ra = 1.17  # RandAugment std
    mu_sas = 40.74  # SAS mean
    sigma_sas = 0.78  # SAS std
    
    print(f"\nRandAugment: μ={mu_ra}%, σ={sigma_ra}")
    print(f"SAS: μ={mu_sas}%, σ={sigma_sas}")
    
    print(f"\nUtility function: U = μ - λσ")
    print(f"For SAS to outperform RandAugment:")
    print(f"  μ_SAS - λ·σ_SAS > μ_RA - λ·σ_RA")
    print(f"  {mu_sas} - λ·{sigma_sas} > {mu_ra} - λ·{sigma_ra}")
    print(f"  λ·({sigma_ra} - {sigma_sas}) > {mu_ra} - {mu_sas}")
    print(f"  λ·{sigma_ra - sigma_sas:.2f} > {mu_ra - mu_sas:.2f}")
    
    lambda_threshold = (mu_ra - mu_sas) / (sigma_ra - sigma_sas)
    print(f"  λ > {lambda_threshold:.2f}")
    print(f"\nPaper claims: λ > 3.85")


def verify_percentage_claims():
    """Verify percentage improvement claims."""
    print("\n" + "=" * 60)
    print("PERCENTAGE CLAIMS VERIFICATION")
    print("=" * 60)
    
    # Std values
    std_ra = 1.17
    std_sas = 0.78
    
    # CV values
    cv_ra = 2.77
    cv_sas = 1.91
    
    # Width values
    width_ra = 3.0
    width_sas = 2.0
    
    std_reduction = (std_ra - std_sas) / std_ra * 100
    cv_increase = (cv_ra - cv_sas) / cv_sas * 100
    width_increase = (width_ra - width_sas) / width_sas * 100
    
    print(f"\n33% std reduction:")
    print(f"  ({std_ra} - {std_sas}) / {std_ra} × 100 = {std_reduction:.1f}%")
    
    print(f"\n45% higher CV (RandAugment vs SAS):")
    print(f"  ({cv_ra} - {cv_sas}) / {cv_sas} × 100 = {cv_increase:.1f}%")
    
    print(f"\n50% wider range (RandAugment vs SAS):")
    print(f"  ({width_ra} - {width_sas}) / {width_sas} × 100 = {width_increase:.1f}%")
    
    # +50% higher std
    std_increase = (std_ra - std_sas) / std_sas * 100
    print(f"\n+50% higher std (RandAugment vs SAS):")
    print(f"  ({std_ra} - {std_sas}) / {std_sas} × 100 = {std_increase:.1f}%")


def verify_variance_difference():
    """Bootstrap test for variance difference between RandAugment and SAS."""
    print("\n" + "=" * 60)
    print("VARIANCE DIFFERENCE BOOTSTRAP TEST")
    print("=" * 60)
    
    randaugment_folds = np.array([42.5, 41.6, 42.9, 40.6, 43.6])
    sas_folds = np.array([40.1, 42.1, 40.6, 40.5, 40.4])
    
    # Observed std difference
    std_ra = np.std(randaugment_folds, ddof=1)
    std_sas = np.std(sas_folds, ddof=1)
    observed_diff = std_ra - std_sas
    
    print(f"\nObserved:")
    print(f"  RandAugment std: {std_ra:.4f}")
    print(f"  SAS std: {std_sas:.4f}")
    print(f"  Difference (RA - SAS): {observed_diff:.4f}")
    
    # Bootstrap confidence interval for std difference
    n_bootstrap = 10000
    np.random.seed(42)
    
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        ra_sample = np.random.choice(randaugment_folds, size=len(randaugment_folds), replace=True)
        sas_sample = np.random.choice(sas_folds, size=len(sas_folds), replace=True)
        
        std_ra_boot = np.std(ra_sample, ddof=1)
        std_sas_boot = np.std(sas_sample, ddof=1)
        bootstrap_diffs.append(std_ra_boot - std_sas_boot)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    ci_low = np.percentile(bootstrap_diffs, 2.5)
    ci_high = np.percentile(bootstrap_diffs, 97.5)
    
    # Proportion of bootstrap samples where RA has higher std
    prop_ra_higher = np.mean(bootstrap_diffs > 0)
    
    print(f"\nBootstrap Analysis (n={n_bootstrap}):")
    print(f"  95% CI for std difference: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  P(RA std > SAS std): {prop_ra_higher:.4f}")
    
    if ci_low > 0:
        print(f"  ✓ CI does not include 0: RandAugment has significantly higher variance")
    else:
        print(f"  ⚠ CI includes 0: difference may not be significant")


def verify_destructiveness_metrics():
    """Verify SSIM/LPIPS claims from destructiveness_metrics.csv."""
    print("\n" + "=" * 60)
    print("DESTRUCTIVENESS METRICS VERIFICATION")
    print("=" * 60)
    
    # Values from destructiveness_metrics.csv
    metrics = {
        'Baseline': {'SSIM': 0.198, 'LPIPS': 0.084},
        'RandAugment': {'SSIM': 0.147, 'LPIPS': 0.124},
        'SAS': {'SSIM': 0.196, 'LPIPS': 0.091}
    }
    
    for method, vals in metrics.items():
        print(f"\n{method}:")
        print(f"  SSIM: {vals['SSIM']:.3f}")
        print(f"  LPIPS: {vals['LPIPS']:.3f}")
    
    # 25% SSIM degradation
    ssim_baseline = metrics['Baseline']['SSIM']
    ssim_ra = metrics['RandAugment']['SSIM']
    ssim_degradation = (ssim_baseline - ssim_ra) / ssim_baseline * 100
    
    print(f"\n25% SSIM degradation (RandAugment vs Baseline):")
    print(f"  ({ssim_baseline} - {ssim_ra}) / {ssim_baseline} × 100 = {ssim_degradation:.1f}%")


def main():
    """Run all verifications."""
    print("\n" + "#" * 60)
    print("# EUSIPCO 2026 PAPER STATISTICAL VERIFICATION")
    print("#" * 60)
    
    verify_table_ii()
    verify_ablation_table()
    verify_wilcoxon_test()
    verify_lambda_threshold()
    verify_percentage_claims()
    verify_variance_difference()
    verify_destructiveness_metrics()
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
