# -*- coding: utf-8 -*-
"""
12-classical-benchmark comparison at D=30 (manuscript Section "Results on
Basic Dimensions").

Runs PSO, GWO, PIO, CPIO and PCA-PIO on the 12 functions with the unified
protocol (N=30, Tmax=500, T1=400, 30 independent runs). The same random seed
is used for every algorithm within each paired (function, run) pair, exactly as
described in the manuscript's "Implementation details".

Usage:  python experiments/run_benchmarks.py   (writes results/benchmarks_d30.csv)
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pca_pio.benchmarks import FUNCTIONS
from pca_pio.algorithms import ALGORITHMS, run_with_dim

D = 30
N = 30
TMAX = 500
T1 = 400
RUNS = 30


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmarks_d30.csv")

    header = ["Function", "Algorithm", "Mean", "Std"] + [f"Run_{i}" for i in range(RUNS)]
    rows = []
    t0 = time.time()

    for fi, spec in enumerate(FUNCTIONS):
        name = spec["name"]
        fn = spec["fn"]
        lo, hi = spec["lo"], spec["hi"]
        for alg in ALGORITHMS:
            runs = []
            for r in range(RUNS):
                seed = 100000 + fi * 100000 + r  # same seed for every algorithm
                best = run_with_dim(fn, (lo, hi), D, seed, alg,
                                    N=N, Tmax=TMAX, T1=T1)
                runs.append(best)
            mean = float(sum(runs) / len(runs))
            std = (sum((v - mean) ** 2 for v in runs) / len(runs)) ** 0.5
            rows.append([name, alg, f"{mean:.6e}", f"{std:.6e}"] + [f"{v:.6e}" for v in runs])
            print(f"  {name:<14} {alg:<9} mean={mean:.4e} std={std:.4e}")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nWrote {out_path}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
