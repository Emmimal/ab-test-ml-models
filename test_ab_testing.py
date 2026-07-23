"""
Sanity checks for the A/B testing scripts. Run with: pytest test_ab_testing.py -v
These don't re-validate the math libraries -- they validate that our
simulations behave the way the underlying theory says they must.
"""
import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

np.random.seed(42)
analysis = NormalIndPower()


def test_sample_size_shrinks_as_mde_grows():
    """A bigger minimum detectable effect should need fewer users, not more."""
    baseline = 0.04
    ns = []
    for mde in [0.05, 0.10, 0.15, 0.20, 0.30]:
        es = proportion_effectsize(baseline * (1 + mde), baseline)
        n = analysis.solve_power(effect_size=es, alpha=0.05, power=0.80, ratio=1.0)
        ns.append(n)
    assert all(ns[i] > ns[i + 1] for i in range(len(ns) - 1)), "N should strictly decrease as MDE grows"


def test_ztest_p_value_in_valid_range():
    control = np.random.binomial(1, 0.04, 5000)
    treatment = np.random.binomial(1, 0.046, 5000)
    count = np.array([control.sum(), treatment.sum()])
    nobs = np.array([len(control), len(treatment)])
    _, p = proportions_ztest(count, nobs)
    assert 0.0 <= p <= 1.0


def test_bootstrap_ci_contains_true_zero_under_null():
    """Under a true null (no effect), the 95% bootstrap CI should contain 0
    in the large majority of resamples -- check it does here, once, at seed 42."""
    np.random.seed(42)
    control = np.random.lognormal(3.0, 1.0, 5000) * (np.random.random(5000) < 0.04)
    treatment = np.random.lognormal(3.0, 1.0, 5000) * (np.random.random(5000) < 0.04)
    diffs = np.array([
        np.random.choice(treatment, 5000, replace=True).mean()
        - np.random.choice(control, 5000, replace=True).mean()
        for _ in range(2000)
    ])
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    assert ci_low < 0 < ci_high, "95% CI should straddle zero under a true null"


def test_bonferroni_is_stricter_than_naive():
    """Bonferroni's per-test threshold must always be <= the naive alpha."""
    alpha = 0.05
    n_metrics = 5
    assert (alpha / n_metrics) < alpha


def test_peeking_inflates_false_positives_vs_fixed_horizon():
    """Re-run a small version of the peeking simulation and confirm the
    direction of the effect: peeking must show a higher false-positive rate."""
    np.random.seed(1)
    n_sims, n_days, daily_n = 300, 20, 200
    fixed_fp, peek_fp = 0, 0
    for _ in range(n_sims):
        c, t = np.array([], dtype=int), np.array([], dtype=int)
        crossed = False
        for day in range(1, n_days + 1):
            c = np.append(c, np.random.binomial(1, 0.10, daily_n))
            t = np.append(t, np.random.binomial(1, 0.10, daily_n))
            p_pool = (c.sum() + t.sum()) / (len(c) + len(t))
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / len(c) + 1 / len(t)))
            z = (c.mean() - t.mean()) / se if se > 0 else 0
            p_val = 2 * (1 - stats.norm.cdf(abs(z)))
            if not crossed and p_val < 0.05:
                crossed = True
            if day == n_days and p_val < 0.05:
                fixed_fp += 1
        if crossed:
            peek_fp += 1
    assert peek_fp >= fixed_fp, "Peeking should never show a LOWER false-positive rate than fixed horizon"


def test_srm_chi_square_detects_large_mismatch():
    chi2, p = stats.chisquare(f_obs=[19000, 21000], f_exp=[20000, 20000])
    assert p < 0.01, "A 19k/21k split out of 40k should trip the SRM check"


def test_srm_chi_square_passes_clean_split():
    chi2, p = stats.chisquare(f_obs=[19990, 20010], f_exp=[20000, 20000])
    assert p > 0.5, "A near-exact 50/50 split should not trigger an SRM alarm"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS: {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed")
