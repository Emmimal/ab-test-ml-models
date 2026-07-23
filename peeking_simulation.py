"""
A/A simulation again (no true effect), but this time we accumulate data
day by day for 30 days and check the p-value EVERY day (peeking) versus
checking it only once, at day 30 (fixed horizon).
"""
import numpy as np
from scipy import stats

np.random.seed(42)

DAILY_USERS_PER_ARM = 300
N_DAYS = 30
N_SIMULATIONS = 2000
ALPHA = 0.05
BASELINE_RATE = 0.10  # a higher-traffic binary metric so daily peeks are meaningful

peeked_any_significant = 0
fixed_horizon_significant = 0
day_first_crossed = []

for _ in range(N_SIMULATIONS):
    control_cum = np.array([], dtype=int)
    treat_cum = np.array([], dtype=int)
    crossed = False
    for day in range(1, N_DAYS + 1):
        control_cum = np.append(control_cum, np.random.binomial(1, BASELINE_RATE, DAILY_USERS_PER_ARM))
        treat_cum = np.append(treat_cum, np.random.binomial(1, BASELINE_RATE, DAILY_USERS_PER_ARM))
        count = np.array([control_cum.sum(), treat_cum.sum()])
        nobs = np.array([len(control_cum), len(treat_cum)])
        # two-proportion z-test, pooled
        p_pool = count.sum() / nobs.sum()
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / nobs[0] + 1 / nobs[1]))
        if se > 0:
            z = (count[0] / nobs[0] - count[1] / nobs[1]) / se
            p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        else:
            p_val = 1.0
        if not crossed and p_val < ALPHA:
            crossed = True
            day_first_crossed.append(day)
        if day == N_DAYS and p_val < ALPHA:
            fixed_horizon_significant += 1
    if crossed:
        peeked_any_significant += 1

print(f"Simulations: {N_SIMULATIONS}, true effect: NONE (A/A), {DAILY_USERS_PER_ARM}/arm/day, {N_DAYS} days\n")
print(f"False positive rate checking ONLY at day {N_DAYS} (fixed horizon): "
      f"{fixed_horizon_significant/N_SIMULATIONS*100:.1f}%")
print(f"False positive rate checking EVERY DAY and stopping at first p<0.05 (peeking): "
      f"{peeked_any_significant/N_SIMULATIONS*100:.1f}%")

if day_first_crossed:
    day_first_crossed = np.array(day_first_crossed)
    print(f"\nOf the {len(day_first_crossed)} simulations that crossed p<0.05 at some point:")
    print(f"  Median day of first crossing: {np.median(day_first_crossed):.0f}")
    print(f"  25th/75th percentile day: {np.percentile(day_first_crossed, 25):.0f} / "
          f"{np.percentile(day_first_crossed, 75):.0f}")
    print(f"  % that crossed within the first 10 days: "
          f"{(day_first_crossed <= 10).mean()*100:.1f}%")
