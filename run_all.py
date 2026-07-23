"""
Runs every benchmark script in this repo, in order, and prints one combined
output -- same numbers you'd get running each file individually, seed=42
throughout. Also runs the test suite at the end.
"""
import subprocess
import sys
import time

SCRIPTS = [
    ("1. Sample size / test duration", "sample_size_calculator.py"),
    ("2. Significance testing (conversion + revenue)", "significance_test.py"),
    ("3. Multiple metrics correction (Bonferroni / BH)", "multiple_testing_correction.py"),
    ("4. Peeking / optional stopping", "peeking_simulation.py"),
    ("5. Sample Ratio Mismatch check", "srm_check.py"),
    ("6. Test suite", "test_ab_testing.py"),
]

total_start = time.time()

for title, script in SCRIPTS:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    start = time.time()
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(result.stdout, end="")
    if result.stderr:
        print("STDERR:", result.stderr)
    print(f"\n[{script} finished in {time.time() - start:.2f}s]")

print("\n" + "=" * 70)
print(f"All scripts finished in {time.time() - total_start:.2f}s total")
print("=" * 70)
