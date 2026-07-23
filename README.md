# ab-test-ml-models

Six scripts that answer the questions an ML A/B test actually needs answered — not
"is p < 0.05" in isolation, but which test fits which metric, how big a sample you
need, what happens when you track five metrics instead of one, and what checking
your dashboard every morning does to your false-positive rate.

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Every A/B testing tutorial explains the two-sample t-test and stops. This repo
simulates experiments with a known, planted ground truth — so instead of "trust
the p-value," you get to check whether the test's conclusion was actually right,
and where it wasn't.

Read the full write-up on Towards Data Science → [How to A/B Test Machine Learning Models the Right Way](https://emitechlogic.com/how-to-a-b-test-machine-learning-models-the-right-way/)

## What It Does

```
control / treatment data → statistical test → decision
        ↑                        ↑
  known ground truth      matched to metric type
  (planted in advance)    (proportion / skewed continuous)
```

Five independent experiments, one test suite:

| Script | Question it answers |
|---|---|
| `sample_size_calculator.py` | How many users per arm, and how many days, for a given MDE? |
| `significance_test.py` | Two-proportion z-test on conversion + bootstrap vs. t-test on skewed revenue |
| `multiple_testing_correction.py` | How much does tracking 5 metrics inflate false positives — with and without correction? |
| `peeking_simulation.py` | What does checking a dashboard daily do to your false-positive rate? |
| `srm_check.py` | Did the traffic actually split the way you configured it? |
| `test_ab_testing.py` | Validates the direction of every result above against the underlying theory |

## Installation

```bash
git clone https://github.com/Emmimal/ab-test-ml-models.git
cd ab-test-ml-models
pip install numpy scipy statsmodels
```

No other dependencies. Every script runs on the standard scientific Python stack.

## Quick Start

Run everything in one pass:

```bash
python run_all.py
```

Or run a single experiment:

```python
# significance_test.py demonstrates the core pattern:
# match the test to the metric, don't default to a t-test for everything.

import numpy as np
from statsmodels.stats.proportion import proportions_ztest

np.random.seed(42)
control = np.random.binomial(1, 0.04, 17923)     # proportion metric
treatment = np.random.binomial(1, 0.046, 17923)

count = np.array([control.sum(), treatment.sum()])
nobs = np.array([len(control), len(treatment)])
z_stat, p_value = proportions_ztest(count, nobs)
```

## Running the Test Suite

```bash
python test_ab_testing.py
```

Seven checks that validate direction, not just syntax — e.g. "peeking must show a
higher false-positive rate than fixed horizon," not just "the function runs."

| # | What it validates |
|---|---|
| 1 | Required sample size shrinks monotonically as MDE grows |
| 2 | z-test p-value stays in [0, 1] |
| 3 | Bootstrap CI contains zero under a true null |
| 4 | Bonferroni's per-test threshold is always stricter than the naive one |
| 5 | Peeking never shows a *lower* false-positive rate than fixed horizon |
| 6 | Chi-square SRM check fires on a large mismatch |
| 7 | Chi-square SRM check passes a clean 50/50 split |

## Configuration Reference

Each script exposes its parameters at the top of the file rather than behind a
config object — deliberately, so the assumption you're testing is visible on
the first read:

```python
# sample_size_calculator.py
BASELINE_RATE = 0.04            # current model's conversion rate
ALPHA = 0.05
POWER = 0.80
DAILY_TRAFFIC_PER_ARM = 2500

# multiple_testing_correction.py
N_METRICS = 5
N_SIMULATIONS = 2000

# peeking_simulation.py
DAILY_USERS_PER_ARM = 300
N_DAYS = 30
```

## Project Structure

```
ab-test-ml-models/
├── sample_size_calculator.py       # Power analysis, MDE sweep, alpha/power tradeoffs
├── significance_test.py            # z-test (proportion) + t-test/bootstrap (skewed continuous)
├── multiple_testing_correction.py  # Bonferroni + Benjamini-Hochberg vs. naive, on true A/A data
├── peeking_simulation.py           # Fixed horizon vs. daily peeking, on true A/A data
├── srm_check.py                    # Chi-square goodness-of-fit, clean split vs. drop-rate bugs
├── test_ab_testing.py              # 7 direction-of-effect checks across all five scripts
└── run_all.py                      # Runs everything above in sequence, one combined output
```

## Reproducibility

Every number in the write-up came from these scripts at `seed=42`, on Python 3.12.3
/ NumPy 2.4.4 / SciPy 1.17.1 / statsmodels 0.14.6. Independently re-run on a second,
unrelated machine (Windows, separate Python environment) during review — every
figure matched exactly, including the resampling-heavy multiple-testing and
peeking simulations. Total runtime varied by hardware (27s vs. 111s for the full
`run_all.py` pass); the statistical results didn't.

## When to Use This

Worth it when you have:
- An ML model candidate and a real traffic-splitting decision to make
- More than one metric you care about, especially if some are skewed or zero-inflated
- A team tempted to check results before the planned sample size is reached

Skip it when you have:
- A single clean binary metric and no multiple-comparisons problem — a plain
  two-proportion test and a fixed calendar date are enough
- An existing experimentation platform (Statsig, Optimizely) already handling
  sequential testing and SRM — this repo is for understanding what those
  platforms are doing under the hood, not replacing them

## Known Limitations

- `peeking_simulation.py` demonstrates *why* peeking inflates false positives; it
  does not implement a sequential/always-valid testing method as a fix. That's a
  separate, larger undertaking (see Johari et al., cited in the article).
- The bootstrap in `significance_test.py` resamples 10,000 times per run — fine
  for a one-off analysis, too slow to re-run per day in a live dashboard without
  optimization.
- `srm_check.py` only tests a two-arm 50/50 split. Multi-arm or unequal-ratio
  designs need the expected-frequency vector adjusted accordingly.
- Sample size calculations assume a fixed daily traffic figure. Real traffic is
  rarely constant across days of the week — treat the "days to run" column as a
  lower bound, not a guarantee.
