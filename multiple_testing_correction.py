"""
A/A simulation: control and treatment are IDENTICAL (no true effect anywhere).
Run 5 metrics per experiment, repeat 2000 times, and measure how often at
least one metric falsely reaches p < 0.05 -- with and without correction.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

N_PER_ARM = 5000
N_METRICS = 5
N_SIMULATIONS = 2000
ALPHA = 0.05

def run_one_aa_experiment():
    """5 metrics, all with a TRUE null (control == treatment). Returns 5 p-values."""
    p_values = []
    # metric 1: conversion (proportion)
    c = np.random.binomial(1, 0.04, N_PER_ARM)
    t = np.random.binomial(1, 0.04, N_PER_ARM)
    _, p = stats.ttest_ind(c, t)
    p_values.append(p)
    # metrics 2-5: continuous metrics with varying noise, all true null
    for mu, sigma in [(50, 15), (5, 2), (0.3, 0.1), (120, 40)]:
        c = np.random.normal(mu, sigma, N_PER_ARM)
        t = np.random.normal(mu, sigma, N_PER_ARM)
        _, p = stats.ttest_ind(c, t)
        p_values.append(p)
    return np.array(p_values)

def benjamini_hochberg(p_values, alpha):
    m = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    passed = ranked <= thresholds
    if not passed.any():
        return np.zeros(m, dtype=bool)
    max_rank = np.max(np.where(passed)[0])
    reject = np.zeros(m, dtype=bool)
    reject[order[:max_rank + 1]] = True
    return reject

naive_any_fp = 0
bonferroni_any_fp = 0
bh_any_fp = 0
per_metric_naive_fp = np.zeros(N_METRICS)

for _ in range(N_SIMULATIONS):
    p_values = run_one_aa_experiment()
    per_metric_naive_fp += (p_values < ALPHA)
    if (p_values < ALPHA).any():
        naive_any_fp += 1
    bonferroni_alpha = ALPHA / N_METRICS
    if (p_values < bonferroni_alpha).any():
        bonferroni_any_fp += 1
    if benjamini_hochberg(p_values, ALPHA).any():
        bh_any_fp += 1

print(f"Simulations: {N_SIMULATIONS}, metrics per experiment: {N_METRICS}, true effect: NONE (A/A)\n")
print(f"Per-metric false positive rate (naive alpha=0.05 on each): "
      f"{[f'{x/N_SIMULATIONS*100:.1f}%' for x in per_metric_naive_fp]}")
print(f"\nFamily-wise false positive rate (>=1 of 5 metrics falsely 'significant'):")
print(f"  Naive (no correction):     {naive_any_fp/N_SIMULATIONS*100:.1f}%  "
      f"(expected without correction ~= 1-(1-0.05)^5 = {(1-(1-ALPHA)**N_METRICS)*100:.1f}%)")
print(f"  Bonferroni (alpha/5):      {bonferroni_any_fp/N_SIMULATIONS*100:.1f}%")
print(f"  Benjamini-Hochberg (FDR):  {bh_any_fp/N_SIMULATIONS*100:.1f}%")
