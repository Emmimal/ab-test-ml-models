"""
How long does an ML A/B test need to run?
Power analysis for a two-proportion test (conversion rate), swept across
a few realistic minimum detectable effects (MDE).
"""
import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

np.random.seed(42)

BASELINE_RATE = 0.04   # current model's conversion rate
ALPHA = 0.05
POWER = 0.80
DAILY_TRAFFIC_PER_ARM = 2500  # users per arm per day, split 50/50

analysis = NormalIndPower()

print(f"{'MDE (relative)':<16}{'Treatment rate':<16}{'Effect size (h)':<18}{'N per arm':<14}{'Days to run':<12}")
mdes = [0.05, 0.10, 0.15, 0.20, 0.30]
rows = []
for mde in mdes:
    treatment_rate = BASELINE_RATE * (1 + mde)
    effect_size = proportion_effectsize(treatment_rate, BASELINE_RATE)
    n_per_arm = analysis.solve_power(effect_size=effect_size, alpha=ALPHA, power=POWER,
                                       ratio=1.0, alternative='two-sided')
    n_per_arm = int(np.ceil(n_per_arm))
    days = int(np.ceil(n_per_arm / DAILY_TRAFFIC_PER_ARM))
    rows.append((mde, treatment_rate, effect_size, n_per_arm, days))
    print(f"{mde*100:<15.0f}%{treatment_rate*100:<15.2f}%{effect_size:<18.4f}{n_per_arm:<14,}{days:<12}")

print("\n--- Effect of alpha/power choices on the 15% MDE case ---")
mde = 0.15
treatment_rate = BASELINE_RATE * (1 + mde)
effect_size = proportion_effectsize(treatment_rate, BASELINE_RATE)
for alpha, power in [(0.05, 0.80), (0.05, 0.90), (0.01, 0.80), (0.10, 0.80)]:
    n_per_arm = int(np.ceil(analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power,
                                                    ratio=1.0, alternative='two-sided')))
    days = int(np.ceil(n_per_arm / DAILY_TRAFFIC_PER_ARM))
    print(f"alpha={alpha}, power={power}: N/arm={n_per_arm:,}, days={days}")
