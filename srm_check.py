"""
Sample Ratio Mismatch: is the traffic split actually what you configured?
Chi-square goodness-of-fit test against the expected allocation.
A real bug pattern: the treatment page is slightly slower to instrument,
so a fraction of treatment users bounce before being logged -- silently
skewing the observed split even though assignment itself was correct.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

EXPECTED_RATIO = 0.5  # intended 50/50 split

def srm_test(n_control, n_treatment, expected_ratio=0.5):
    total = n_control + n_treatment
    expected_control = total * expected_ratio
    expected_treatment = total * (1 - expected_ratio)
    chi2, p_value = stats.chisquare(
        f_obs=[n_control, n_treatment],
        f_exp=[expected_control, expected_treatment]
    )
    return chi2, p_value

print("=== Scenario 1: clean 50/50 split, no logging issue ===")
n = 40000
assignments = np.random.binomial(1, 0.5, n)
n_control = (assignments == 0).sum()
n_treatment = (assignments == 1).sum()
chi2, p = srm_test(n_control, n_treatment)
print(f"Control: {n_control:,}, Treatment: {n_treatment:,}, ratio: {n_treatment/n:.4f}")
print(f"chi2 = {chi2:.4f}, p = {p:.4f} -> SRM detected: {p < 0.01}")

print("\n=== Scenario 2: 1% of treatment events silently dropped before logging ===")
assignments = np.random.binomial(1, 0.5, n)
n_control_raw = (assignments == 0).sum()
n_treatment_raw = (assignments == 1).sum()
drop_mask = np.random.random(n_treatment_raw) < 0.01
n_treatment_logged = n_treatment_raw - drop_mask.sum()
chi2, p = srm_test(n_control_raw, n_treatment_logged)
print(f"Control: {n_control_raw:,}, Treatment (post-drop): {n_treatment_logged:,}, "
      f"ratio: {n_treatment_logged/(n_control_raw+n_treatment_logged):.4f}")
print(f"chi2 = {chi2:.4f}, p = {p:.6f} -> SRM detected: {p < 0.01}")

print("\n=== Same 1% drop rate, but only 4,000 users (a test stopped early) ===")
n_small = 4000
assignments = np.random.binomial(1, 0.5, n_small)
n_control_raw = (assignments == 0).sum()
n_treatment_raw = (assignments == 1).sum()
drop_mask = np.random.random(n_treatment_raw) < 0.01
n_treatment_logged = n_treatment_raw - drop_mask.sum()
chi2, p = srm_test(n_control_raw, n_treatment_logged)
print(f"Control: {n_control_raw:,}, Treatment (post-drop): {n_treatment_logged:,}, "
      f"ratio: {n_treatment_logged/(n_control_raw+n_treatment_logged):.4f}")
print(f"chi2 = {chi2:.4f}, p = {p:.4f} -> SRM detected: {p < 0.01}")

print("\n=== How much scale or drop rate does it actually take to catch this? ===")
for n_test, drop_rate in [(40000, 0.01), (40000, 0.02), (40000, 0.03),
                           (200000, 0.01), (500000, 0.005), (500000, 0.01)]:
    assignments = np.random.binomial(1, 0.5, n_test)
    n_c = (assignments == 0).sum()
    n_t_raw = (assignments == 1).sum()
    drop_mask = np.random.random(n_t_raw) < drop_rate
    n_t = n_t_raw - drop_mask.sum()
    chi2, p = srm_test(n_c, n_t)
    print(f"N={n_test:,}, drop rate={drop_rate*100:.1f}%: observed ratio="
          f"{n_t/(n_c+n_t):.4f}, p={p:.4f}, detected={p < 0.01}")
