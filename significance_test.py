"""
Run the actual test: two-proportion z-test on conversion, then show why
a plain t-test on revenue-per-user can mislead you on skewed data.
"""
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

np.random.seed(42)

N_PER_ARM = 17923          # from sample_size_calculator.py, 15% MDE case
BASELINE_RATE = 0.04
TRUE_LIFT = 0.15           # treatment is genuinely 15% relatively better

# --- Metric 1: conversion (binary) ---
control_conv = np.random.binomial(1, BASELINE_RATE, N_PER_ARM)
treat_conv = np.random.binomial(1, BASELINE_RATE * (1 + TRUE_LIFT), N_PER_ARM)

count = np.array([control_conv.sum(), treat_conv.sum()])
nobs = np.array([N_PER_ARM, N_PER_ARM])
z_stat, p_value = proportions_ztest(count, nobs)

control_rate = control_conv.mean()
treat_rate = treat_conv.mean()
rel_lift = (treat_rate - control_rate) / control_rate

print("=== Conversion rate (two-proportion z-test) ===")
print(f"Control:   {count[0]:,} / {nobs[0]:,} = {control_rate*100:.3f}%")
print(f"Treatment: {count[1]:,} / {nobs[1]:,} = {treat_rate*100:.3f}%")
print(f"Observed relative lift: {rel_lift*100:.2f}%")
print(f"z = {z_stat:.4f}, p = {p_value:.6f}")
print(f"Significant at alpha=0.05: {p_value < 0.05}")

# --- Metric 2: revenue per user (continuous, zero-inflated, right-skewed) ---
# ~96% of users spend $0; the rest follow a lognormal spend distribution.
# Treatment nudges the spend distribution up slightly, not the zero-inflation rate.
def simulate_revenue(n, spend_prob, mu, sigma):
    spends = np.zeros(n)
    spenders = np.random.random(n) < spend_prob
    n_spenders = spenders.sum()
    spends[spenders] = np.random.lognormal(mu, sigma, n_spenders)
    return spends

control_rev = simulate_revenue(N_PER_ARM, spend_prob=0.04, mu=3.0, sigma=1.0)
treat_rev = simulate_revenue(N_PER_ARM, spend_prob=0.04, mu=3.08, sigma=1.0)

t_stat, t_p = stats.ttest_ind(control_rev, treat_rev, equal_var=False)

print("\n=== Revenue per user (Welch's t-test on raw, skewed data) ===")
print(f"Control mean:   ${control_rev.mean():.4f}  (median ${np.median(control_rev):.2f}, "
      f"skew {stats.skew(control_rev):.2f})")
print(f"Treatment mean: ${treat_rev.mean():.4f}  (median ${np.median(treat_rev):.2f}, "
      f"skew {stats.skew(treat_rev):.2f})")
print(f"t = {t_stat:.4f}, p = {t_p:.4f}")
print(f"Significant at alpha=0.05: {t_p < 0.05}")

# Bootstrap CI on the difference in means -- doesn't assume normality
np.random.seed(42)
N_BOOT = 10000
boot_diffs = np.empty(N_BOOT)
for i in range(N_BOOT):
    c_sample = np.random.choice(control_rev, size=N_PER_ARM, replace=True)
    t_sample = np.random.choice(treat_rev, size=N_PER_ARM, replace=True)
    boot_diffs[i] = t_sample.mean() - c_sample.mean()

ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
boot_p_like = 2 * min((boot_diffs < 0).mean(), (boot_diffs > 0).mean())

print("\n=== Same revenue data, bootstrap on the mean difference ===")
print(f"Observed mean diff: ${treat_rev.mean() - control_rev.mean():.4f}")
print(f"95% bootstrap CI: [${ci_low:.4f}, ${ci_high:.4f}]")
print(f"Two-sided bootstrap p-like value: {boot_p_like:.4f}")
print(f"CI excludes zero: {ci_low > 0 or ci_high < 0}")
