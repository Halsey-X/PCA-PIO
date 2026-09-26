# -*- coding: utf-8 -*-
"""
High-dimensional comparison at D=100 (manuscript Section "High-Dimensional
Experiments").

Runs the five algorithms on Sphere, Rosenbrock, Rastrigin, Ackley and Griewank
at D=100 under the unified protocol (N=30, Tmax=500, T1=400, 30 runs).

Usage:  python experiments/run_highdim.py   (writes results/highdim_d100.csv)
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pca_pio.benchmarks import HIGH_DIM_NAMES, get_function
from pca_pio.algorithms import ALGORITHMS, run_with_dim

D = 100
N = 30
TMAX = 500
T1 = 400
RUNS = 30


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "highdim_d100.csv")

    header = ["Function", "Algorithm", "Mean", "Std"] + [f"Run_{i}" for i in range(RUNS)]
    rows = []
    t0 = time.time()

    for fi, name in enumerate(HIGH_DIM_NAMES):
        spec = get_function(name)
        fn = spec["fn"]
        lo, hi = spec["lo"], spec["hi"]
        for alg in ALGORITHMS:
            runs = []
            for r in range(RUNS):
                seed = 200000 + fi * 100000 + r
                best = run_with_dim(fn, (lo, hi), D, seed, alg,
                                    N=N, Tmax=TMAX, T1=T1)
                runs.append(best)
            mean = float(sum(runs) / len(runs))
            std = (sum((v - mean) ** 2 for v in runs) / len(runs)) ** 0.5
            rows.append([name, alg, f"{mean:.6e}", f"{std:.6e}"] + [f"{v:.6e}" for v in runs])
            print(f"  {name:<12} {alg:<9} mean={mean:.4e} std={std:.4e}")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nWrote {out_path}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
